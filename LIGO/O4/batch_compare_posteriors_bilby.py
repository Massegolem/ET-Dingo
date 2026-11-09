"""
Batch-compare DINGO importance-sampling posteriors against GWTC PE release
files (and, if present, bilby_pipe results) for many events, producing one
overlaid corner plot per event with the DINGO effective sample size /
importance-sampling efficiency annotated on the plot.

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

DINGO results (one subfolder per event):
    <dingo_base>/outdir_<EVENT>/result/<EVENT>_data0_<id>_importance_sampling.hdf5
    (the *_part0/1/2/3.hdf5 chunk files are ignored automatically)

bilby_pipe results (one subfolder per event):
    <bilby_base>/outdir_bilby_<EVENT>/result/*_result.hdf5
    If several result files exist, a "*merge*" file is preferred.

Only events for which ALL THREE results are found (DINGO, bilby, GWTC) are
plotted. Corner plot layering, back to front: DINGO, bilby, GWTC.

Output:
    <output_dir>/<EVENT>_corner_comparison.png   (one per event)
    <output_dir>/summary.csv                     (per-event bookkeeping)

Requires: h5py, numpy, pandas, corner, matplotlib
Optional: bilby (used to read the bilby result files; if it is not
          importable, a plain h5py fallback is used)
"""

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
    (kind, obj, path) or None."""
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
    dataset or a group of 1-D datasets. Returns (DataFrame, group_path_used).
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


def load_bilby_posterior(path):
    """Load the posterior of a bilby(_pipe) result file as a DataFrame.

    Preferred: bilby.result.read_in_result (robust across bilby versions).
    Fallback (bilby not installed): generic h5py search for a table that
    contains chirp_mass and mass_ratio.
    """
    try:
        import bilby
    except ImportError:
        df, _ = load_posterior(path)
        return df.reset_index(drop=True)

    result = bilby.result.read_in_result(filename=str(path))
    return result.posterior.reset_index(drop=True)


def _has_mass_params(keys):
    keys = set(keys)
    return ({"chirp_mass", "mass_ratio"} <= keys) or ({"mass_1", "mass_2"} <= keys)


def list_posterior_groups(path):
    """Return every '<...>/posterior_samples' table in a GWTC file that
    contains mass parameters."""
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

    1. Walk `preferred_names` and return the first one that exists.
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

    # (mass_1, mass_2) -> (chirp_mass, mass_ratio)
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
# 3. Overlay corner plot (any number of datasets)
# --------------------------------------------------------------------------

def make_comparison_corner(datasets, params, outfile, title=None, annotation=None):
    """
    datasets: list of dicts with keys 'df', 'weights' (or None), 'label'
    and optionally 'color'. Drawn in the order given (first = back,
    last = front).
    """
    all_data = [d["df"][params].to_numpy() for d in datasets]

    ranges = []
    for i in range(len(params)):
        lo = min(np.nanmin(x[:, i]) for x in all_data)
        hi = max(np.nanmax(x[:, i]) for x in all_data)
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ranges.append((lo - pad, hi + pad))

    fig = None
    handles = []
    for k, (d, data) in enumerate(zip(datasets, all_data)):
        color = d.get("color", f"C{k}")
        kwargs = dict(
            weights=d["weights"], range=ranges, color=color,
            plot_datapoints=False, plot_density=False,
            fill_contours=True, levels=(0.5, 0.9),
            hist_kwargs={"density": True},
        )
        if fig is None:
            fig = corner.corner(data, labels=params, **kwargs)
        else:
            corner.corner(data, fig=fig, **kwargs)
        handles.append(mlines.Line2D([], [], color=color, label=d["label"]))

    fig.legend(handles=handles, loc="upper right", fontsize=30,
               frameon=False, bbox_to_anchor=(0.98, 0.98))

    if title:
        fig.suptitle(title, fontsize=50, y=1.02)
    if annotation:
        fig.text(0.98, 0.90 - 0.04 * (len(datasets) - 2), annotation,
                 fontsize=30, ha="right", va="top", transform=fig.transFigure)

    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# 4. File discovery helpers
# --------------------------------------------------------------------------

DEFAULT_RELEASE_PRIORITY = ["GWTC5p0", "GWTC4p1", "GWTC3p0", "GWTC2p1"]
_RELEASE_RE = re.compile(r"IGWN-(GWTC\d+p\d+)")


def find_dingo_file(outdir):
    """Find the combined *_importance_sampling.hdf5 file inside
    outdir_<EVENT>/result/ (the _partN.hdf5 chunk files are skipped)."""
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


def find_bilby_file(bilby_base, event, prefix="outdir_bilby_"):
    """Find the bilby_pipe result file for an event, or None if there is none.

    Looks in <bilby_base>/<prefix><event>/result/ for *_result.hdf5 and
    prefers a merged result file ("merge" in the name) if several exist.
    """
    result_dir = Path(bilby_base) / f"{prefix}{event}" / "result"
    if not result_dir.is_dir():
        return None
    candidates = sorted(result_dir.glob("*_result.hdf5"))
    if not candidates:
        return None
    merged = [c for c in candidates if "merge" in c.name]
    chosen = merged[0] if merged else candidates[0]
    if len(candidates) > 1:
        print(f"\n    Note: multiple bilby result files in {result_dir}, "
              f"using {chosen.name}", end=" ")
    return chosen


def _classify_gwtc_file(path):
    """Return (release, variant) for a GWTC PE sample file, else None.
    variant is 'combined' (GWTC-4/5), or 'cosmo' / 'nocosmo' (GWTC-2.1/3)."""
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
    """Find the GWTC PE release file for an event. Returns (path, release)
    or None. If an event appears in several releases, the release earliest
    in `release_priority` wins."""
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


# --------------------------------------------------------------------------
# 5. Batch driver
# --------------------------------------------------------------------------

def batch_compare(dingo_base, gwtc_dir, output_dir,
                   gwtc_group_priority, params_wanted,
                   gwtc_variant="nocosmo",
                   release_priority=DEFAULT_RELEASE_PRIORITY,
                   bilby_base=None, bilby_prefix="outdir_bilby_"):
    """
    bilby_base: folder containing <bilby_prefix><EVENT>/result/. Defaults to
    dingo_base. Events missing the DINGO, bilby or GWTC result are skipped.
    """
    if bilby_base is None:
        bilby_base = dingo_base
    os.makedirs(output_dir, exist_ok=True)
    # "outdir_GW*" does not match "outdir_bilby_GW*", so bilby folders are
    # not mistaken for DINGO folders.
    outdirs = sorted(Path(dingo_base).glob("outdir_GW*"))
    if not outdirs:
        raise ValueError(f"No outdir_GW* folders found under {dingo_base}")

    rows = []
    for outdir in outdirs:
        event = extract_event_name(outdir)
        print(f"[{event}] ", end="")

        dingo_file = find_dingo_file(outdir)
        if dingo_file is None:
            print("SKIP (no importance_sampling.hdf5 found)")
            rows.append({"event": event, "status": "skipped_no_dingo_file"})
            continue

        hit = find_gwtc_file(gwtc_dir, event, variant=gwtc_variant,
                             release_priority=release_priority)
        if hit is None:
            print("SKIP (no matching GWTC file)")
            rows.append({"event": event, "status": "skipped_no_gwtc_file",
                         "dingo_file": str(dingo_file)})
            continue
        gwtc_file, release = hit

        bilby_file = find_bilby_file(bilby_base, event, prefix=bilby_prefix)
        if bilby_file is None:
            print("SKIP (no bilby result file found)")
            rows.append({"event": event, "status": "skipped_no_bilby_file",
                         "dingo_file": str(dingo_file), "gwtc_file": str(gwtc_file),
                         "gwtc_release": release})
            continue

        try:
            df_a, _ = load_posterior(dingo_file)
            gwtc_group = find_gwtc_posterior_group(gwtc_file, gwtc_group_priority)
            df_b, _ = load_posterior(gwtc_file, group_path=gwtc_group)

            w_a_raw = df_a["weights"].to_numpy(dtype=float) if "weights" in df_a.columns else None
            if w_a_raw is not None:
                n_eff, efficiency = effective_sample_size(w_a_raw)
                w_a = w_a_raw / w_a_raw.sum()
            else:
                n_eff, efficiency = float(len(df_a)), 1.0
                w_a = None
            w_b = get_weights(df_b)

            df_a = add_derived_quantities(df_a)
            df_b = add_derived_quantities(df_b)

            df_c = load_bilby_posterior(bilby_file)
            w_c = get_weights(df_c)  # bilby posterior: equal weights -> None
            df_c = add_derived_quantities(df_c)
            n_bilby = len(df_c)

            # Draw order = back to front: DINGO, bilby, GWTC
            datasets = [
                {"df": df_a, "weights": w_a, "label": "DINGO (IS)", "color": "C0"},
                {"df": df_c, "weights": w_c, "label": "bilby (dynesty)", "color": "C2"},
                {"df": df_b, "weights": w_b, "label": f"{release} PE", "color": "C1"},
            ]

            # Only plot parameters present in ALL datasets
            params = [p for p in params_wanted
                      if all(p in d["df"].columns for d in datasets)]
            dropped = [p for p in params_wanted if p not in params]
            if len(params) < 2:
                print(f"SKIP (fewer than 2 shared params: {params})")
                rows.append({"event": event, "status": "skipped_too_few_params",
                             "dingo_file": str(dingo_file), "gwtc_file": str(gwtc_file),
                             "gwtc_release": release,
                             "bilby_file": str(bilby_file)})
                continue

            outfile = Path(output_dir) / f"{event}_corner_comparison.png"
            annotation = (
                f"$N_\\mathrm{{eff}}$ = {n_eff:.0f} / {len(df_a)}\n"
                f"IS efficiency = {efficiency:.1%}"
            )
            annotation += f"\nbilby: {n_bilby} samples"
            fig = make_comparison_corner(
                datasets, params, outfile=str(outfile),
                title=event, annotation=annotation,
            )
            plt.close(fig)

            print(f"OK (n_eff={n_eff:.0f}, eff={efficiency:.1%}, "
                  f"{len(params)} params, {release}, gwtc_group={gwtc_group}, "
                  f"bilby={n_bilby} samples)")
            rows.append({
                "event": event, "status": "ok",
                "dingo_file": str(dingo_file), "gwtc_file": str(gwtc_file),
                "gwtc_release": release, "gwtc_group_used": gwtc_group,
                "bilby_file": str(bilby_file),
                "n_dingo_samples": len(df_a), "n_eff": n_eff,
                "sampling_efficiency": efficiency,
                "n_bilby_samples": n_bilby,
                "n_params_compared": len(params),
                "params_dropped": ",".join(dropped),
                "plot_file": str(outfile),
            })
        except Exception as e:
            print(f"ERROR ({e})")
            traceback.print_exc()
            rows.append({"event": event, "status": "error", "error": str(e),
                         "dingo_file": str(dingo_file), "gwtc_file": str(gwtc_file),
                         "gwtc_release": release,
                         "bilby_file": str(bilby_file)})

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
    DINGO_BASE = "/home/tpausch/ET-Dingo/LIGO/O4/T4"   # contains outdir_GW*/result/
    BILBY_BASE = DINGO_BASE                             # contains outdir_bilby_GW*/result/
    GWTC_DIR = "/scratch/tpausch/zenodo"
    OUTPUT_DIR = "/home/tpausch/ET-Dingo/LIGO/O4/T4/compare_results/bilby"
    # ---------------------------------------------------------------------

    # GWTC-2.1 / GWTC-3 ship two files per event: "nocosmo" or "cosmo".
    # (Ignored for GWTC-4/5 files.)
    GWTC_VARIANT = "nocosmo"

    # If an event shows up in several releases, earlier entries win.
    RELEASE_PRIORITY = ["GWTC5p0", "GWTC4p1", "GWTC3p0", "GWTC2p1"]

    # Priority order of GWTC approximant groups (first one present in a given
    # file wins). If none match, the script falls back to any group with
    # "mixed" in its name, then to the first posterior_samples table found.
    GWTC_GROUP_PRIORITY = [
        # GWTC-2.1 / GWTC-3
        "C01:IMRPhenomXPHM", "C01:Mixed", "C01:SEOBNRv4PHM",
        # GWTC-4 / GWTC-5
        "C00:IMRPhenomXO4a", "C00:IMRPhenomXPHM-SpinTaylor", "C00:Mixed+XO4a", "C00:Mixed",
        "C00:SEOBNRv5PHM", "C00:NRSur7dq4",
    ]

    # Shared detector-frame parameters to plot (dropped automatically per
    # event if not present in all datasets -- see params_dropped in summary.csv)
    PARAMS_WANTED = [
        "chirp_mass", "mass_ratio", "mass_1", "mass_2",
        "chi_eff", "theta_jn", "phase",
        "ra", "dec", "psi", "luminosity_distance",
    ]

    batch_compare(DINGO_BASE, GWTC_DIR, OUTPUT_DIR,
                  GWTC_GROUP_PRIORITY, PARAMS_WANTED,
                  gwtc_variant=GWTC_VARIANT,
                  release_priority=RELEASE_PRIORITY,
                  bilby_base=BILBY_BASE)