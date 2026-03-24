"""Reserve GPUs by allocating most of their memory. Run with nohup to persist."""

import argparse
import signal
import sys
import time

import torch


def reserve(gpu_ids: list[int], fraction: float = 0.85):
    tensors = []
    for gid in gpu_ids:
        dev = torch.device(f"cuda:{gid}")
        free, total = torch.cuda.mem_get_info(dev)
        alloc_bytes = int(free * fraction)
        # Allocate as int8 so 1 element = 1 byte
        t = torch.empty(alloc_bytes, dtype=torch.int8, device=dev)
        used_gb = alloc_bytes / (1024 ** 3)
        print(f"GPU {gid}: reserved {used_gb:.1f} GiB / {total / (1024**3):.1f} GiB")
        tensors.append(t)
    return tensors


def main():
    parser = argparse.ArgumentParser(description="Reserve GPU memory")
    parser.add_argument("--gpus", nargs="+", type=int, required=True,
                        help="GPU indices to reserve")
    parser.add_argument("--fraction", type=float, default=0.85,
                        help="Fraction of free memory to allocate (default: 0.85)")
    args = parser.parse_args()

    tensors = reserve(args.gpus, args.fraction)

    def cleanup(sig, frame):
        print("\nReleasing GPUs...")
        tensors.clear()
        torch.cuda.empty_cache()
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    print(f"Holding GPUs {args.gpus}. Kill this process (or Ctrl+C) to release.")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
