"""GPU Video Decoder Pool for parallel data loading with GPU acceleration.

This module provides a decoder pool that runs GPU video decoders in a separate process,
allowing DataLoader workers to benefit from GPU-accelerated video decoding without
each worker needing to initialize CUDA contexts.
"""

import os
from multiprocessing import Manager, Process, Queue
from typing import Optional

import fsspec
import torch


class GPUVideoDecoderPool:
    """Manages GPU video decoders in a dedicated process for multi-worker data loading.

    This pool allows multiple DataLoader workers to request GPU-decoded frames via IPC,
    avoiding CUDA initialization errors that occur when workers try to use CUDA directly.

    Usage:
        # Create pool in main process before DataLoader
        pool = GPUVideoDecoderPool(device="cuda")
        pool.start()

        # In dataset __getitem__, get worker ID
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        # Request frames
        frames = pool.decode_frames(video_path, frame_indices, worker_id)

        # Cleanup when done
        pool.shutdown()
    """

    def __init__(self, device="cuda", max_cache_size=100):
        """Initialize the GPU decoder pool.

        Args:
            device: CUDA device to use (e.g., "cuda", "cuda:0")
            max_cache_size: Maximum number of video decoders to cache
        """
        self.device = device
        self.max_cache_size = max_cache_size
        self.request_queue = None
        self.response_queues = {}
        self.process = None
        self._started = False
        self._manager = None

    def start(self):
        """Start the decoder pool process."""
        if self._started:
            return

        # Create manager for IPC
        self._manager = Manager()
        self.request_queue = self._manager.Queue(maxsize=1000)

        # Start decoder worker process
        self.process = Process(
            target=self._decoder_worker,
            args=(self.request_queue, self.device, self.max_cache_size),
            daemon=True
        )
        self.process.start()
        self._started = True

        # Give the process time to initialize CUDA
        import time
        time.sleep(0.5)

    def _decoder_worker(self, request_queue, device, max_cache_size):
        """Worker process that handles GPU decoding requests.

        This runs in a separate process and manages all GPU decoders.
        """
        try:
            from torchcodec.decoders import VideoDecoder
        except ImportError:
            print("ERROR: torchcodec not available in decoder worker")
            return

        decoder_cache = {}
        cache_order = []  # Track access order for LRU eviction

        print(f"GPU Decoder Pool started on {device}")

        while True:
            try:
                # Get request with timeout to check for shutdown
                try:
                    request = request_queue.get(timeout=0.1)
                except:
                    continue

                if request is None:  # Shutdown signal
                    print("GPU Decoder Pool shutting down...")
                    break

                worker_id, video_path, indices, response_queue_id = request

                # Get or create decoder
                video_path_str = str(video_path)
                if video_path_str not in decoder_cache:
                    # Evict oldest if cache is full
                    if len(decoder_cache) >= max_cache_size:
                        oldest_path = cache_order.pop(0)
                        decoder, file_handle = decoder_cache.pop(oldest_path)
                        file_handle.close()

                    # Create new decoder
                    try:
                        file_handle = fsspec.open(video_path_str).__enter__()
                        decoder = VideoDecoder(file_handle, device=device, seek_mode="approximate")
                        decoder_cache[video_path_str] = (decoder, file_handle)
                        cache_order.append(video_path_str)
                    except Exception as e:
                        # Send error response
                        response_queue_id.put((False, f"Failed to create decoder: {str(e)}"))
                        continue
                else:
                    # Move to end (most recently used)
                    cache_order.remove(video_path_str)
                    cache_order.append(video_path_str)

                decoder, _ = decoder_cache[video_path_str]

                # Decode frames
                try:
                    # Convert indices to tensor if needed
                    if not isinstance(indices, torch.Tensor):
                        indices = torch.tensor(indices, dtype=torch.int64)

                    frames_batch = decoder.get_frames_at(indices=indices)

                    # Transfer to CPU for IPC (shared memory would be faster but more complex)
                    frames_cpu = frames_batch.data.cpu()

                    # Send response
                    response_queue_id.put((True, frames_cpu))

                except Exception as e:
                    response_queue_id.put((False, f"Decoding error: {str(e)}"))

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in decoder worker: {e}")
                continue

        # Cleanup
        for decoder, file_handle in decoder_cache.values():
            try:
                file_handle.close()
            except:
                pass

    def decode_frames(self, video_path: str, indices, worker_id: int = 0):
        """Request frame decoding from the GPU pool.

        Args:
            video_path: Path to video file
            indices: Frame indices to decode (list or tensor)
            worker_id: ID of the requesting worker (for tracking)

        Returns:
            Decoded frames as CPU tensor (you can move to GPU in main process)
        """
        if not self._started:
            raise RuntimeError("GPU decoder pool not started. Call start() first.")

        # Create response queue for this worker if needed
        if worker_id not in self.response_queues:
            self.response_queues[worker_id] = self._manager.Queue(maxsize=10)

        response_queue = self.response_queues[worker_id]

        # Send decode request
        self.request_queue.put((worker_id, video_path, indices, response_queue))

        # Wait for response
        success, result = response_queue.get()

        if not success:
            raise RuntimeError(f"GPU decoding failed: {result}")

        return result

    def shutdown(self):
        """Shutdown the decoder pool and cleanup resources."""
        if self.process and self.process.is_alive():
            # Send shutdown signal
            try:
                self.request_queue.put(None, timeout=1.0)
            except:
                pass

            # Wait for graceful shutdown
            self.process.join(timeout=5)

            # Force terminate if still alive
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)

        # Cleanup manager
        if self._manager:
            try:
                self._manager.shutdown()
            except:
                pass

        self._started = False
        self.response_queues.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()
