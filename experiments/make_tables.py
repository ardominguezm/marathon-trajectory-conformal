"""Aggregate raw result CSVs into compact markdown/LaTeX tables (mean +- sd over seeds)."""
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
T = R / "tables"; T.mkdir(exist_ok=True)


def ms(g, cols):
    m, s = g[cols].mean(), g[cols].std(ddof=1).fillna(0)
    return pd.DataFrame({c: [f"{m[c]:.3f} ({s[c]:.3f})"] for c in cols})


def write(df, name):
    df.to_csv(T / f"{name}.csv")
    (T / f"{name}.md").write_text(df.to_markdown())
    (T / f"{name}.tex").write_text(df.to_latex(escape=True))


for proto in ["temporal", "exchangeable"]:
    f = R / f"trajectory_summary_{proto}.csv"
    if f.exists():
        d = pd.read_csv(f)
        cols = ["coverage_mean", "coverage_min_ckpt", "joint_coverage", "width_mean", "interval_score_mean"]
        write(d.groupby("method").apply(lambda g: ms(g, cols).iloc[0]), f"trajectory_{proto}")
f = R / "baselines_full.csv"
if f.exists():
    d = pd.read_csv(f)
    t = d.groupby(["checkpoint", "method"])[["coverage", "width_mean", "interval_score", "rmse_point"]].mean().round(3)
    write(t, "baselines")
f = R / "transport_full.csv"
if f.exists():
    d = pd.read_csv(f)
    t = d.groupby(["target", "checkpoint", "scheme", "mode"]).coverage.mean().unstack("scheme").round(3)
    write(t, "transport_coverage")
    j = d.groupby(["target", "scheme", "mode"]).joint_coverage.mean().unstack("scheme").round(3)
    write(j, "transport_joint_coverage")
print("tables written to", T)
