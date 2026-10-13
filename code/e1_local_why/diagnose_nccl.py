"""Minimal two-rank NCCL sanity test."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    value = torch.tensor([float(rank + 1)], device="cuda")
    dist.all_reduce(value)
    torch.cuda.synchronize()
    print(f"rank={rank} sum={value.item()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
