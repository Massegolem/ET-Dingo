#!/usr/bin/env python3
"""Summarise dingo_pipe outdirs: mean inference time and IS efficiency."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD = 0.01  # 1 % sample efficiency


def sample_efficiency(result_file: Path) -> float:
    """IS efficiency = n_eff / N, with n_eff = (sum w)^2 / sum w^2."""
    try:  # preferred: let dingo do it
        from dingo.gw.result import Result
        r = Result(file_name=str(result_file))
        return float(r.n_eff / len(r.samples))
    except Exception:
        import h5py  # fallback: read the weights directly
        with h5py.File(result_file, "r") as f:
            w = np.asarray(f["samples/weights"])
        return float(w.sum() ** 2 / (w ** 2).sum() / len(w))


def analyse_outdir(outdir: Path) -> dict | None:
    row = {"outdir": outdir.name, "t_sampling": np.nan, "t_is": np.nan,
           "efficiency": np.nan}

    # timing files (identified via the "stage" key, since the filenames overlap)
    for tf in outdir.rglob("*_timing.json"):
        t = json.loads(tf.read_text())
        if t["stage"] == "sampling":
            row["t_sampling"] = t["total_seconds"]
        elif t["stage"] == "importance_sampling":
            row["t_is"] = t["total_seconds"]

    # IS result file (exclude the plain proposal-sampling result)
    res = [p for p in outdir.rglob("*_importance_sampling.hdf5")]
    if res:
        row["efficiency"] = sample_efficiency(res[0])

    return row


def main(root: str):
    outdirs = sorted(p for p in Path(root).glob("outdir_*") if p.is_dir())
    if not outdirs:
        sys.exit(f"No outdir_* folders found in {root}")

    df = pd.DataFrame([analyse_outdir(o) for o in outdirs])
    df["t_total"] = df["t_sampling"] + df["t_is"]  # NaN if either stage missing
    df["eff_above_1pct"] = df["efficiency"] > THRESHOLD

    pd.set_option("display.width", 160)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    n = len(df)
    n_done = df["efficiency"].notna().sum()
    n_good = df["eff_above_1pct"].sum()
    print("\n=== Summary ===")
    print(f"outdirs found:                 {n}")
    print(f"with IS result:                {n_done}")
    print(f"IS efficiency > 1 %:           {n_good} / {n_done} "
          f"({100 * n_good / max(n_done, 1):.1f} %)")
    print(f"mean sampling time (GPU):      {df['t_sampling'].mean():.1f} s")
    print(f"mean importance-sampling time: {df['t_is'].mean():.1f} s")
    print(f"mean total inference time:     {df['t_total'].mean():.1f} s "
          f"(median {df['t_total'].median():.1f} s)")

    df.to_csv("outdir_summary.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")