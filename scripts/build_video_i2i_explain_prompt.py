#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper entry for prompt builder.

This keeps compatibility with running from the TwiSTAR folder:
  python scripts/build_video_i2i_explain_prompt.py --input_json sample.json

The actual implementation lives at:
  /opt/tiger/LLRM_eval/scripts/build_video_i2i_explain_prompt.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "scripts" / "build_video_i2i_explain_prompt.py"
    if not target.exists():
        raise SystemExit(f"未找到目标脚本: {target}")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
