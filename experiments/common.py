import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--dev", default=str(ROOT / "data" / "segment_paces_data.csv"))
    p.add_argument("--berlin", default=str(ROOT / "data" / "berlin_marathon_2021_2024.csv"))
    p.add_argument("--wall", default=str(ROOT / "data" / "wall_labels_data.csv"))
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max-train", type=int, default=150000)
    p.add_argument("--n-estimators", type=int, default=250)
    p.add_argument("--target", choices=["residual", "absolute"], default="residual",
                   help="base-learner target: finish minus constant-pace projection (default) or finish time")
    p.add_argument("--quick", action="store_true", help="smoke run: tiny subsample, 1 seed")
    return p


def maybe_path(p):
    return p if p and Path(p).exists() else None
