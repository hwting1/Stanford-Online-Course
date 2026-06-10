from __future__ import annotations

import argparse
import timeit
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch.cuda import nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch


MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


@dataclass
class BenchmarkConfig:
    model_size: str = "small"
    mode: str = "full"
    warmup: int = 5
    steps: int = 10
    device: str = "cuda"
    dtype: str = "fp32"

    vocab_size: int = 10_000
    dataset_size: int = 100_000
    batch_size: int = 4
    context_length: int = 512

    memory_profile: bool = False
    memory_snapshot_path: str = "memory_snapshot.pickle"


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="CS336 Assignment 2 Benchmark Script")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "medium", "large", "xl", "10B"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["forward", "backward", "full"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--context-length", type=int, default=512)

    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Record CUDA memory history and dump a PyTorch memory snapshot.",
    )
    parser.add_argument(
        "--memory-snapshot-path",
        type=str,
        default="memory_snapshot.pickle",
        help="Output path for torch.cuda.memory._dump_snapshot.",
    )

    args = parser.parse_args()

    return BenchmarkConfig(
        device=args.device,
        model_size=args.model_size,
        mode=args.mode,
        dtype=args.dtype,
        warmup=args.warmup,
        steps=args.steps,
        context_length=args.context_length,
        memory_profile=args.memory_profile,
        memory_snapshot_path=args.memory_snapshot_path,
    )


def make_dataset(config: BenchmarkConfig) -> npt.NDArray[np.int64]:
    return np.random.randint(
        low=0,
        high=config.vocab_size,
        size=(config.dataset_size,),
        dtype=np.int64,
    )


def make_model(config: BenchmarkConfig) -> BasicsTransformerLM:
    model_kwargs = {
        "vocab_size": config.vocab_size,
        "context_length": config.context_length,
        **MODEL_CONFIGS[config.model_size],
    }

    model = BasicsTransformerLM(**model_kwargs)
    return model.to(config.device)


def sync_if_cuda(device: str) -> None:
    if "cuda" in device:
        torch.cuda.synchronize()


def get_autocast_context(config: BenchmarkConfig):
    if config.dtype == "bf16":
        if "cuda" not in config.device:
            raise ValueError("bf16 autocast benchmark should be run on CUDA.")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    if config.dtype == "fp32":
        return nullcontext()

    raise ValueError(f"Unknown dtype: {config.dtype}")


def run_step(
    config: BenchmarkConfig,
    model: torch.nn.Module,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    if config.mode == "forward":
        with torch.no_grad():
            with nvtx.range("forward"):
                with get_autocast_context(config):
                    _ = model(x)

    elif config.mode == "backward":
        optimizer.zero_grad(set_to_none=True)

        with nvtx.range("forward"):
            with get_autocast_context(config):
                logits = model(x)

        with nvtx.range("loss"):
            with get_autocast_context(config):
                loss = cross_entropy(logits, y)

        with nvtx.range("backward"):
            loss.backward()

    elif config.mode == "full":
        optimizer.zero_grad(set_to_none=True)

        with nvtx.range("forward"):
            with get_autocast_context(config):
                logits = model(x)

        with nvtx.range("loss"):
            with get_autocast_context(config):
                loss = cross_entropy(logits, y)

        with nvtx.range("backward"):
            loss.backward()

        with nvtx.range("optimizer"):
            optimizer.step()

    else:
        raise ValueError(f"Unknown mode: {config.mode}")


def timed_step(
    config: BenchmarkConfig,
    model: torch.nn.Module,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    sync_if_cuda(config.device)

    start = timeit.default_timer()

    run_step(config, model, optimizer, x, y)

    sync_if_cuda(config.device)

    end = timeit.default_timer()
    return end - start


def benchmark(config: BenchmarkConfig) -> tuple[float, float]:
    if config.dataset_size <= config.context_length:
        raise ValueError("dataset_size must be larger than context_length.")

    if config.memory_profile and "cuda" not in config.device:
        raise ValueError("Memory profiling requires CUDA.")

    dataset = make_dataset(config)
    model = make_model(config)
    optimizer = AdamW(model.parameters())

    model.train()

    with nvtx.range("warmup"):
        for _ in range(config.warmup):
            x, y = get_batch(
                dataset,
                config.batch_size,
                config.context_length,
                config.device,
            )
            run_step(config, model, optimizer, x, y)

    sync_if_cuda(config.device)

    times: list[float] = []

    if config.memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
        torch.cuda.reset_peak_memory_stats()

    try:
        with nvtx.range("benchmark"):
            for _ in range(config.steps):
                x, y = get_batch(
                    dataset,
                    config.batch_size,
                    config.context_length,
                    config.device,
                )

                elapsed = timed_step(
                    config,
                    model,
                    optimizer,
                    x,
                    y,
                )

                times.append(elapsed)

        sync_if_cuda(config.device)

        if config.memory_profile:
            torch.cuda.memory._dump_snapshot(config.memory_snapshot_path)

    finally:
        if config.memory_profile:
            torch.cuda.memory._record_memory_history(enabled=None)

    mean = float(np.mean(times))
    std = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0

    return mean, std


def main() -> None:
    config = parse_args()
    mean, std = benchmark(config)

    print(f"device               : {config.device}")
    print(f"model_size           : {config.model_size}")
    print(f"mode                 : {config.mode}")
    print(f"dtype                : {config.dtype}")
    print(f"warmup               : {config.warmup}")
    print(f"steps                : {config.steps}")
    print(f"batch_size           : {config.batch_size}")
    print(f"context_length       : {config.context_length}")
    print(f"memory_profile       : {config.memory_profile}")
    print(f"mean_time            : {mean:.6f} s")
    print(f"std_time             : {std:.6f} s")

    if config.memory_profile:
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_mib = peak_bytes / 1024**2
        print(f"peak_memory_allocated: {peak_mib:.2f} MiB")
        print(f"memory_snapshot_path : {config.memory_snapshot_path}")


if __name__ == "__main__":
    main()