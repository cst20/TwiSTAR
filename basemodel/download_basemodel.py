#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download_model(repo_id: str, target_subdir: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    target_dir = base_dir / target_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HF model into TwiSTAR/basemodel")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace repo id (e.g., Qwen/Qwen3.5-4B)",
    )
    parser.add_argument(
        "--target_subdir",
        type=str,
        default="Qwen3-1-7B",
        help="Target subdir under basemodel/ (e.g., Qwen3.5-4B)",
    )
    args = parser.parse_args()

    out_dir = download_model(repo_id=args.repo_id, target_subdir=args.target_subdir)
    print(f"✅ Downloaded {args.repo_id} -> {out_dir}")


if __name__ == "__main__":
    main()
