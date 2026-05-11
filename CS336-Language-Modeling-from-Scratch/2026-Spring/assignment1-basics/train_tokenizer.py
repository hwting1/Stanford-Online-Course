from pathlib import Path
import argparse
import json
import os
import time
import resource

import wandb
from huggingface_hub import HfApi

from cs336_basics.tokenizer import train_bpe


SPECIAL_TOKENS = ["<|endoftext|>"]


def save_vocab(vocab, vocab_path: Path):
    serializable = {str(i): token.decode("latin1") for i, token in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)


def save_merges(merges, merges_path: Path):
    serializable = [[a.decode("latin1"), b.decode("latin1")] for a, b in merges]
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)


def get_rusage_memory_gb():
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return self_usage.ru_maxrss / (1024 ** 2), child_usage.ru_maxrss / (1024 ** 2)


def infer_dataset_name(input_path: str) -> str:
    name = Path(input_path).stem.lower()
    if "tiny" in name:
        return "tinystories"
    if "owt" in name or "openwebtext" in name:
        return "owt"
    return name.replace("_train", "").replace("-train", "")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output-root", default="tokenizers")
    parser.add_argument("--num-processes", type=int, default=4)

    parser.add_argument("--wandb-project", default="cs336-assignment-1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--disable-wandb", action="store_true")

    parser.add_argument("--hf-repo-id", default=None, help="e.g. username/cs336-tokenizers")
    parser.add_argument("--hf-repo-type", default="dataset")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--disable-hf-upload", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset_name or infer_dataset_name(args.input_path)
    run_name = f"{dataset_name}-bpe-vocab{args.vocab_size}"
    output_dir = Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.json"
    metadata_path = output_dir / "metadata.json"

    wandb_run = None
    if not args.disable_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            job_type="train-bpe-tokenizer",
            config={
                "input_path": args.input_path,
                "dataset_name": dataset_name,
                "vocab_size": args.vocab_size,
                "special_tokens": SPECIAL_TOKENS,
                "num_processes": args.num_processes,
                "output_dir": str(output_dir),
            },
        )

    print("=== Training BPE ===")
    print(f"Input: {args.input_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Output dir: {output_dir}")
    print(f"Process ID: {os.getpid()}")

    start_time = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        num_processes=args.num_processes,
    )

    elapsed = time.perf_counter() - start_time
    self_peak_gb, children_peak_gb = get_rusage_memory_gb()
    estimated_peak_gb = max(self_peak_gb, children_peak_gb)

    longest_id, longest_token = max(vocab.items(), key=lambda item: len(item[1]))

    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)

    metrics = {
        "dataset_name": dataset_name,
        "input_path": args.input_path,
        "vocab_size": len(vocab),
        "target_vocab_size": args.vocab_size,
        "num_merges": len(merges),
        "training_time_seconds": elapsed,
        "peak_rss_main_gb": self_peak_gb,
        "peak_rss_children_gb": children_peak_gb,
        "estimated_peak_rss_gb": estimated_peak_gb,
        "longest_token_id": longest_id,
        "longest_token_length_bytes": len(longest_token),
        "longest_token_repr": repr(longest_token),
        "vocab_file_mb": vocab_path.stat().st_size / (1024 ** 2),
        "merges_file_mb": merges_path.stat().st_size / (1024 ** 2),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if wandb_run is not None:
        wandb.log(metrics)

        artifact = wandb.Artifact(
            name=run_name,
            type="bpe-tokenizer",
            metadata=metrics,
        )
        artifact.add_file(str(vocab_path), name="vocab.json")
        artifact.add_file(str(merges_path), name="merges.json")
        artifact.add_file(str(metadata_path), name="metadata.json")
        wandb_run.log_artifact(artifact)

    if args.hf_repo_id and not args.disable_hf_upload:
        api = HfApi()
        api.create_repo(
            repo_id=args.hf_repo_id,
            repo_type=args.hf_repo_type,
            private=args.hf_private,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=args.hf_repo_id,
            repo_type=args.hf_repo_type,
            folder_path=str(output_dir),
            path_in_repo=run_name,
            commit_message=f"Add {run_name}",
        )

    print("Done!")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()