"""
Batch-compare one or more DINGO importance-sampling runs against GWTC PE
release files for many events, producing one overlaid corner plot per event
with the DINGO effective sample size / importance-sampling efficiency of
every run annotated on the plot.

Supported GWTC release layouts (all in one flat folder)
--------------------------------------------------------
GWTC-2.1 / GWTC-3 (O1-O3):
    IGWN-GWTC2p1-v2-<EVENT>_PEDataRelease_mixed_cosmo.h5
    IGWN-GWTC2p1-v2-<EVENT>_PEDataRelease_mixed_nocosmo.h5
    IGWN-GWTC3p0-v3-<EVENT>_PEDataRelease_mixed_cosmo.h5
    IGWN-GWTC3p0-v3-<EVENT>_PEDataRelease_mixed_nocosmo.h5
GWTC-4.1 / GWTC-5.0 (O4a/O4b):
    IGWN-GWTC4p1-<hash>-<EVENT>-combined_PEDataRelease.hdf5
    IGWN-GWTC5p0-<hash>-<EVENT>-combined_PEDataRelease.hdf5

DINGO results (one subfolder per event, one base folder per run):
    <dingo_base>/outdir_<EVENT>/result/<EVENT>_data0_<id>_importance_sampling.hdf5
    (the *_part0/1/2/3.hdf5 chunk files are ignored automatically)

The FIRST run in `dingo_runs` defines the event list (its outdir_GW* folders).
Further runs are looked up by the same event name in their own base folder.

Output:
    <output_dir>/<EVENT>_corner_comparison.png   (one per event)
    <output_dir>/summary.csv                     (per-event bookkeeping)

Requires: h5py, numpy, pandas, corner, matplotlib
"""

import json
import os
import re
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import corner
import matplotlib
matplotlib.use("Agg")  # safe for headless/HPC batch runs
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# --------------------------------------------------------------------------
# 1. Generic HDF5 posterior loader (handles both DINGO and GWTC layouts)
# --------------------------------------------------------------------------

def _search(h5obj, required_keys):
    """Recursively search an open h5py File/Group for the first dataset or
    group of 1-D datasets that contains all `required_keys`. Returns
    (kind, obj, path) or None. (Used for DINGO files.)"""
    for name, obj in h5obj.items():
        path = f"{h5obj.name}/{name}".lstrip("/")
        if isinstance(obj, h5py.Dataset):
            fields = obj.dtype.names
            if fields and all(k in fields for k in required_keys):
                return "dataset", obj, path
        elif isinstance(obj, h5py.Group):
            keys = set(obj.keys())
            if required_keys.issubset(keys):
                return "group", obj, path
            found = _search(obj, required_keys)
            if found is not None:
                return found
    return None


def load_posterior(path, group_path=None,
                    required_keys=("chirp_mass", "mass_ratio")):
    """
    Load a posterior-sample table from an HDF5 file into a pandas
    DataFrame, regardless of whether it's stored as a single compound
    dataset (DINGO / pesummary "posterior_samples" style) or a group of 1-D
    datasets. Returns (DataFrame, group_path_used).
    """
    required = set(required_keys)
    with h5py.File(path, "r") as f:
        if group_path is not None:
            obj = f[group_path]
            kind = "dataset" if isinstance(obj, h5py.Dataset) else "group"
            target, used_path = obj, group_path
        else:
            found = _search(f, required)
            if found is None:
                raise ValueError(
                    f"Could not auto-locate a posterior table containing "
                    f"{required_keys} in '{path}'."
                )
            kind, target, used_path = found

        if kind == "dataset":
            data = target[()]
            df = pd.DataFrame({n: data[n] for n in data.dtype.names})
        else:
            df = pd.DataFrame({
                k: target[k][()] for k in target.keys()
                if isinstance(target[k], h5py.Dataset) and target[k].ndim == 1
            })
    return df, used_path


def _has_mass_params(keys):
    keys = set(keys)
    return ({"chirp_mass", "mass_ratio"} <= keys) or ({"mass_1", "mass_2"} <= keys)


def list_posterior_groups(path):
    """Return every '<...>/posterior_samples' table in a GWTC file that
    contains mass parameters, e.g. ['C01:IMRPhenomXPHM/posterior_samples',
    'C01:Mixed/posterior_samples', ...]."""
    found = []

    def visit(name, obj):
        if not name.endswith("posterior_samples"):
            return
        if isinstance(obj, h5py.Dataset):
            keys = obj.dtype.names or ()
        else:
            keys = obj.keys()
        if _has_mass_params(keys):
            found.append(name)

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return found


def find_gwtc_posterior_group(path, preferred_names):
    """
    Pick the posterior_samples table to use from a GWTC file.

    1. Walk `preferred_names` (e.g. "C01:Mixed", "C01:IMRPhenomXPHM", ...)
       and return the first one that exists in this file.
    2. Otherwise fall back to any group with "mixed" in its name.
    3. Otherwise take the first posterior_samples table that has mass
       parameters (a note is printed, since this differs per event).
    """
    available = list_posterior_groups(path)
    for name in preferred_names:
        candidate = f"{name}/posterior_samples"
        if candidate in available:
            return candidate
    mixed = [a for a in available if "mixed" in a.lower()]
    if mixed:
        print(f"\n    (fallback to '{mixed[0]}'; available: {available})", end=" ")
        return mixed[0]
    if available:
        print(f"\n    (fallback to '{available[0]}'; available: {available})", end=" ")
        return available[0]
    raise ValueError(f"No usable posterior_samples group found in {path}")


def describe_gwtc_file(path):
    """Debug helper: print all posterior tables and their columns."""
    for g in list_posterior_groups(path):
        with h5py.File(path, "r") as f:
            obj = f[g]
            cols = obj.dtype.names if isinstance(obj, h5py.Dataset) else list(obj.keys())
        print(f"{g}: {len(cols)} columns\n  {list(cols)}")


def get_weights(df):
    """Return normalized importance-sampling weights if present,
    otherwise None (equal-weight posterior)."""
    if "weights" in df.columns:
        w = df["weights"].to_numpy(dtype=float)
        return w / w.sum()
    return None


def effective_sample_size(raw_weights):
    """
    Kish effective sample size for (possibly unnormalized) importance
    weights w_i:   n_eff = (sum w_i)^2 / sum(w_i^2)
    Returns (n_eff, efficiency) where efficiency = n_eff / len(w).
    """
    w = np.asarray(raw_weights, dtype=float)
    n_eff = (w.sum() ** 2) / np.sum(w ** 2)
    efficiency = n_eff / len(w)
    return float(n_eff), float(efficiency)


# --------------------------------------------------------------------------
# 2. Derive quantities to make the parameterizations comparable
# --------------------------------------------------------------------------

def add_derived_quantities(df):
    """Make sure chirp_mass, mass_ratio, mass_1, mass_2 and chi_eff are all
    present (derived from whatever is available), using whichever spin
    columns exist."""
    df = df.copy()
    cols = set(df.columns)

    # (mass_1, mass_2) -> (chirp_mass, mass_ratio)   [needed for some older files]
    if {"mass_1", "mass_2"} <= cols:
        m1_, m2_ = df["mass_1"].to_numpy(), df["mass_2"].to_numpy()
        if "chirp_mass" not in cols:
            df["chirp_mass"] = (m1_ * m2_) ** 0.6 / (m1_ + m2_) ** 0.2
        if "mass_ratio" not in cols:
            df["mass_ratio"] = m2_ / m1_

    # (chirp_mass, mass_ratio) -> (mass_1, mass_2)
    if "mass_1" not in df.columns and {"chirp_mass", "mass_ratio"} <= set(df.columns):
        mc, q = df["chirp_mass"].to_numpy(), df["mass_ratio"].to_numpy()
        total_mass = mc * (1.0 + q) ** 1.2 / q ** 0.6
        df["mass_1"] = total_mass / (1.0 + q)
        df["mass_2"] = total_mass * q / (1.0 + q)

    if "chi_eff" not in df.columns:
        m1, m2 = df.get("mass_1"), df.get("mass_2")
        if m1 is not None and m2 is not None:
            if {"chi_1", "chi_2"} <= set(df.columns):
                s1z, s2z = df["chi_1"].to_numpy(), df["chi_2"].to_numpy()
            elif {"spin_1z", "spin_2z"} <= set(df.columns):
                s1z, s2z = df["spin_1z"].to_numpy(), df["spin_2z"].to_numpy()
            elif {"a_1", "cos_tilt_1", "a_2", "cos_tilt_2"} <= set(df.columns):
                s1z = df["a_1"].to_numpy() * df["cos_tilt_1"].to_numpy()
                s2z = df["a_2"].to_numpy() * df["cos_tilt_2"].to_numpy()
            elif {"a_1", "tilt_1", "a_2", "tilt_2"} <= set(df.columns):
                s1z = df["a_1"].to_numpy() * np.cos(df["tilt_1"].to_numpy())
                s2z = df["a_2"].to_numpy() * np.cos(df["tilt_2"].to_numpy())
            else:
                s1z = s2z = None
            if s1z is not None:
                df["chi_eff"] = (m1.to_numpy() * s1z + m2.to_numpy() * s2z) / (m1 + m2).to_numpy()
    return df


# --------------------------------------------------------------------------
# 3. Overlay corner plot (N datasets, ESS annotation per DINGO run)
# --------------------------------------------------------------------------

def make_comparison_corner(datasets, params, outfile, title=None,
                            annotations=None):
    """
    datasets    : list of dicts {"df", "w", "label", "color"}; any number >= 2
    annotations : list of (text, color) tuples, drawn top-right below legend
    """
    arrays = [d["df"][params].to_numpy() for d in datasets]

    ranges = []
    for i in range(len(params)):
        lo = min(np.nanmin(a[:, i]) for a in arrays)
        hi = max(np.nanmax(a[:, i]) for a in arrays)
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ranges.append((lo - pad, hi + pad))

    fig = None
    for d, data in zip(datasets, arrays):
        kwargs = dict(
            weights=d["w"], range=ranges, color=d["color"],
            plot_datapoints=False, plot_density=False,
            fill_contours=True, levels=(0.5, 0.9),
            hist_kwargs={"density": True},
        )
        if fig is None:
            fig = corner.corner(data, labels=params, **kwargs)
        else:
            corner.corner(data, fig=fig, **kwargs)

    legend_handles = [
        mlines.Line2D([], [], color=d["color"], label=d["label"])
        for d in datasets
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=30,
               frameon=False, bbox_to_anchor=(0.98, 0.98))

    if title:
        fig.suptitle(title, fontsize=50, y=1.02)

    # One annotation block per DINGO run, in the colour of its contours.
    # y starts below the legend (legend height grows with the number of entries).
    y = 0.98 - 0.045 * len(datasets) - 0.02
    for text, color in (annotations or []):
        fig.text(0.98, y, text, fontsize=30, ha="right", va="top",
                 color=color, transform=fig.transFigure)
        y -= 0.07 * (text.count("\n") + 1)

    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# 4. File discovery helpers
# --------------------------------------------------------------------------

DEFAULT_RELEASE_PRIORITY = ["GWTC5p0", "GWTC4p1", "GWTC3p0", "GWTC2p1"]
_RELEASE_RE = re.compile(r"IGWN-(GWTC\d+p\d+)")


def find_dingo_file(outdir):
    """Find the combined *_importance_sampling.hdf5 file inside
    outdir_<EVENT>/result/ (the _partN.hdf5 chunk files don't match this
    exact suffix and are skipped automatically)."""
    result_dir = Path(outdir) / "result"
    if not result_dir.is_dir():
        return None
    candidates = sorted(result_dir.glob("*_importance_sampling.hdf5"))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"  Note: multiple importance_sampling.hdf5 files in {result_dir}, "
              f"using {candidates[0].name}")
    return candidates[0]


def _classify_gwtc_file(path):
    """Return (release, variant) for a GWTC PE sample file, else None.
    variant is 'combined' (GWTC-4/5), or 'cosmo' / 'nocosmo' (GWTC-2.1/3).
    Skymap stats, tarballs, notebooks, PESummaryTable and leftover tmp*
    download files are rejected."""
    name = path.name
    if path.suffix not in (".h5", ".hdf5") or name.startswith("tmp"):
        return None
    m = _RELEASE_RE.search(name)
    if m is None:
        return None
    release = m.group(1)
    if "combined_PEDataRelease" in name:
        return release, "combined"
    if "_mixed_nocosmo" in name:
        return release, "nocosmo"
    if "_mixed_cosmo" in name:
        return release, "cosmo"
    return None


def find_gwtc_file(gwtc_dir, event, variant="nocosmo",
                    release_priority=DEFAULT_RELEASE_PRIORITY):
    """Find the GWTC PE release file for an event, e.g. GW190521_030229 or
    GW230601_224134. Returns (path, release) or None.

    - GWTC-2.1/3 events: picks the `variant` file ("nocosmo" or "cosmo").
    - If an event appears in several releases, the release earliest in
      `release_priority` wins.
    """
    cands = []
    for p in sorted(Path(gwtc_dir).glob(f"*{event}*")):
        cls = _classify_gwtc_file(p)
        if cls is None:
            continue
        release, var = cls
        if var in ("cosmo", "nocosmo") and var != variant:
            continue
        cands.append((p, release))
    if not cands:
        return None

    def rank(item):
        rel = item[1]
        return release_priority.index(rel) if rel in release_priority else len(release_priority)

    cands.sort(key=rank)  # stable sort: ties keep alphabetical order
    return cands[0]


def extract_event_name(outdir_path):
    """outdir_GW230601_224134 -> GW230601_224134"""
    name = Path(outdir_path).name
    return name[len("outdir_"):] if name.startswith("outdir_") else name


def _slug(label):
    """'DINGO (T4 low)' -> 'dingo_t4_low' (used as summary.csv column prefix)."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower()


def read_timing(outdir):
    """Read per-stage wall times from the dingo_pipe *_timing.json files in
    an outdir. Stages are identified via the "stage" key (filenames overlap).
    Returns (t_sampling, t_is, t_total) in seconds; NaN where missing
    (t_total is NaN if either stage is missing)."""
    t_sampling = t_is = np.nan
    for tf in Path(outdir).rglob("*_timing.json"):
        try:
            t = json.loads(tf.read_text())
        except (OSError, ValueError):
            continue
        if t.get("stage") == "sampling":
            t_sampling = t.get("total_seconds", np.nan)
        elif t.get("stage") == "importance_sampling":
            t_is = t.get("total_seconds", np.nan)
    return float(t_sampling), float(t_is), float(t_sampling + t_is)


def _fmt_time(seconds):
    """Compact human-readable duration for plot annotations."""
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    if seconds < 100:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.1f} min"


def load_dingo_run(outdir):
    """Load one DINGO IS result. Returns a dict with df, normalized weights,
    n_eff, efficiency, timings and the file path -- or None if no file exists."""
    dingo_file = find_dingo_file(outdir)
    if dingo_file is None:
        return None
    df, _ = load_posterior(dingo_file)
    t_sampling, t_is, t_total = read_timing(outdir)
    w_raw = df["weights"].to_numpy(dtype=float) if "weights" in df.columns else None
    if w_raw is not None:
        n_eff, efficiency = effective_sample_size(w_raw)
        w = w_raw / w_raw.sum()
    else:
        n_eff, efficiency, w = float(len(df)), 1.0, None
    return {
        "df": add_derived_quantities(df), "w": w,
        "n_eff": n_eff, "efficiency": efficiency,
        "n_samples": len(df), "file": dingo_file,
        "t_sampling": t_sampling, "t_is": t_is, "t_total": t_total,
    }


# --------------------------------------------------------------------------
# 5. Batch driver
# --------------------------------------------------------------------------

def batch_compare(dingo_runs, gwtc_dir, output_dir,
                   gwtc_group_priority, params_wanted,
                   gwtc_variant="nocosmo",
                   release_priority=DEFAULT_RELEASE_PRIORITY,
                   require_all_runs=False):
    """
    dingo_runs       : list of (label, base_dir, color) -- first entry defines
                       the event list; `color` may be omitted (label, base_dir).
    require_all_runs : if True, events missing in any run are skipped;
                       if False (default), the missing run is simply left out
                       of that event's plot (noted in summary.csv).
    """
    os.makedirs(output_dir, exist_ok=True)

    default_colors = ["C0", "C2", "C3", "C4", "C5"]  # C1 is reserved for GWTC
    runs = []
    for i, r in enumerate(dingo_runs):
        label, base = r[0], r[1]
        color = r[2] if len(r) > 2 else default_colors[i % len(default_colors)]
        runs.append({"label": label, "base": Path(base), "color": color,
                     "slug": _slug(label)})
    if len({r["slug"] for r in runs}) != len(runs):
        raise ValueError("DINGO run labels must be unique.")

    outdirs = sorted(runs[0]["base"].glob("outdir_GW*"))
    if not outdirs:
        raise ValueError(f"No outdir_GW* folders found under {runs[0]['base']}")

    rows = []
    for outdir in outdirs:
        event = extract_event_name(outdir)
        print(f"[{event}] ", end="")
        row = {"event": event}

        # ---- load every DINGO run for this event -------------------------
        loaded = []          # (run, result-dict) for runs that have data
        missing = []
        for i, run in enumerate(runs):
            ev_dir = run["base"] / f"outdir_{event}"
            res = load_dingo_run(ev_dir) if ev_dir.is_dir() else None
            if res is None:
                missing.append(run["label"])
                continue
            loaded.append((run, res))
            s = run["slug"]
            row[f"{s}_file"] = str(res["file"])
            row[f"{s}_n_samples"] = res["n_samples"]
            row[f"{s}_n_eff"] = res["n_eff"]
            row[f"{s}_sampling_efficiency"] = res["efficiency"]
            row[f"{s}_t_sampling_s"] = res["t_sampling"]
            row[f"{s}_t_is_s"] = res["t_is"]
            row[f"{s}_t_total_s"] = res["t_total"]
        row["runs_missing"] = ",".join(missing)

        if runs[0]["label"] in missing:
            print("SKIP (no importance_sampling.hdf5 for the first run)")
            row["status"] = "skipped_no_dingo_file"
            rows.append(row)
            continue
        if missing and require_all_runs:
            print(f"SKIP (missing runs: {missing})")
            row["status"] = "skipped_missing_run"
            rows.append(row)
            continue
        if missing:
            print(f"(note: no result for {missing}) ", end="")

        # ---- GWTC --------------------------------------------------------
        hit = find_gwtc_file(gwtc_dir, event, variant=gwtc_variant,
                             release_priority=release_priority)
        if hit is None:
            print("SKIP (no matching GWTC file)")
            row["status"] = "skipped_no_gwtc_file"
            rows.append(row)
            continue
        gwtc_file, release = hit
        row["gwtc_file"] = str(gwtc_file)
        row["gwtc_release"] = release

        try:
            gwtc_group = find_gwtc_posterior_group(gwtc_file, gwtc_group_priority)
            df_g, _ = load_posterior(gwtc_file, group_path=gwtc_group)
            df_g = add_derived_quantities(df_g)
            w_g = get_weights(df_g)
            row["gwtc_group_used"] = gwtc_group

            # parameters present in ALL plotted datasets
            all_dfs = [res["df"] for _, res in loaded] + [df_g]
            params = [p for p in params_wanted
                      if all(p in d.columns for d in all_dfs)]
            dropped = [p for p in params_wanted if p not in params]
            if len(params) < 2:
                print(f"SKIP (fewer than 2 shared params: {params})")
                row["status"] = "skipped_too_few_params"
                rows.append(row)
                continue

            datasets, annotations = [], []
            for run, res in loaded:
                datasets.append({"df": res["df"], "w": res["w"],
                                 "label": run["label"], "color": run["color"]})
                annotations.append((
                    f"{run['label']}\n"
                    f"$N_\\mathrm{{eff}}$ = {res['n_eff']:.0f} / {res['n_samples']}\n"
                    f"IS efficiency = {res['efficiency']:.1%}\n"
                    f"$t_\\mathrm{{total}}$ = {_fmt_time(res['t_total'])} "
                    f"({_fmt_time(res['t_sampling'])} + {_fmt_time(res['t_is'])})",
                    run["color"],
                ))
            datasets.append({"df": df_g, "w": w_g,
                             "label": f"{release} PE", "color": "C1"})

            outfile = Path(output_dir) / f"{event}_corner_comparison.png"
            fig = make_comparison_corner(
                datasets, params, outfile=str(outfile),
                title=event, annotations=annotations,
            )
            plt.close(fig)

            eff_str = ", ".join(
                f"{run['label']}: n_eff={res['n_eff']:.0f} ({res['efficiency']:.1%}), "
                f"t={_fmt_time(res['t_total'])}"
                for run, res in loaded
            )
            print(f"OK ({eff_str}; {len(params)} params, {release}, "
                  f"gwtc_group={gwtc_group})")
            row.update({
                "status": "ok",
                "n_params_compared": len(params),
                "params_dropped": ",".join(dropped),
                "plot_file": str(outfile),
            })
            rows.append(row)
        except Exception as e:
            print(f"ERROR ({e})")
            traceback.print_exc()
            row.update({"status": "error", "error": str(e)})
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = Path(output_dir) / "summary.csv"
    summary.to_csv(summary_path, index=False)
    n_ok = (summary["status"] == "ok").sum() if "status" in summary else 0
    print(f"\nDone: {n_ok}/{len(summary)} events plotted. Summary -> {summary_path}")
    return summary


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # --- edit these paths ---------------------------------------------
    # (label shown in legend / annotation, base folder containing outdir_GW*/result/, [colour])
    # The FIRST entry defines which events are processed.
    DINGO_RUNS = [
        ("DINGO (low sample)", "/home/tpausch/ET-Dingo/LIGO/O4/T4_low_sample"),
        ("DINGO (high sample)",           "/home/tpausch/ET-Dingo/LIGO/O4/T4"),
    ]
    GWTC_DIR = "/scratch/tpausch/zenodo"
    OUTPUT_DIR = "/home/tpausch/ET-Dingo/LIGO/O4/T4_low_sample/compare_results_2runs"
    # ---------------------------------------------------------------------

    # GWTC-2.1 / GWTC-3 ship two files per event: "nocosmo" or "cosmo".
    # (Ignored for GWTC-4/5 files.)
    GWTC_VARIANT = "nocosmo"

    # If an event shows up in several releases, earlier entries win.
    RELEASE_PRIORITY = ["GWTC5p0", "GWTC4p1", "GWTC3p0", "GWTC2p1"]

    # Priority order of GWTC approximant groups (first one present in a given
    # file wins). If none match, the script falls back to any group with
    # "mixed" in its name, then to the first posterior_samples table found
    # (a note is printed in that case; see gwtc_group_used in summary.csv).
    GWTC_GROUP_PRIORITY = [
        # GWTC-2.1 / GWTC-3
        "C01:IMRPhenomXPHM", "C01:Mixed", "C01:SEOBNRv4PHM",
        # GWTC-4 / GWTC-5
        "C00:IMRPhenomXO4a", "C00:IMRPhenomXPHM-SpinTaylor", "C00:Mixed+XO4a", "C00:Mixed",
        "C00:SEOBNRv5PHM", "C00:NRSur7dq4",
    ]

    # Shared detector-frame parameters to plot (dropped automatically per
    # event if not present in ALL plotted datasets -- see params_dropped in summary.csv)
    PARAMS_WANTED = [
        "chirp_mass", "mass_ratio", "mass_1", "mass_2",
        "chi_eff", "theta_jn", "phase",
        "ra", "dec", "psi", "luminosity_distance",
    ]

    batch_compare(DINGO_RUNS, GWTC_DIR, OUTPUT_DIR,
                  GWTC_GROUP_PRIORITY, PARAMS_WANTED,
                  gwtc_variant=GWTC_VARIANT,
                  release_priority=RELEASE_PRIORITY,
                  require_all_runs=False)
