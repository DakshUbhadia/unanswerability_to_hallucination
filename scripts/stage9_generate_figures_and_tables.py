#!/usr/bin/env python3
"""Run Stage 9: figures, final tables, executive summary, and README results block (CPU only)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG
from src.plotting_and_report import run_stage9, verify_stage9_outputs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="accepted for runner symmetry; Stage 9 always re-renders")
    a = p.parse_args(argv)
    for path in (CFG.stage5_out_path, CFG.stage6_out_path, CFG.stage8_out_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run the earlier stages first")
    if not CFG.stage7_out_path.exists():
        print("[Stage 9] Stage 7 output missing: Figure 1 capability panels will be empty", flush=True)
    summary = run_stage9()
    verify_stage9_outputs()
    print(f"[Stage 9] confirmed: {summary['confirmed']}")
    print(f"[Stage 9] interpretation: {summary['interpretation']['overall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
