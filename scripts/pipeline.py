"""一键流水线：词典 → 实体抽取对比 → 三元组对比 → 知识图谱与性能比较。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["step1_dictionary.py", "step2_ner_compare.py", "step3_re_compare.py", "step4_kg.py"]


def main():
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    for step in STEPS:
        print(f"\n===== {step} =====", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / step)], env=env)
        if r.returncode != 0:
            print(f"!!! {step} failed with code {r.returncode}", flush=True)
            return r.returncode
    print("\nALL STEPS DONE. 报告见 outputs/ 目录。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
