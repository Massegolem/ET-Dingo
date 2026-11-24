#!/usr/bin/env python
"""
Build a frequency-domain template bank on a (detector-frame chirp mass, mass ratio)
grid for chirp-mass template matching.

Templates: zero spin, face-on (iota = 0), phase 0, fixed distance. Amplitude, phase and
coalescence time are maximised over at matching time, so none of them need a grid axis.
Each template is stored UNWHITENED and L2-normalised (amplitude is free anyway), on the
same frequency grid as the network's data domain, so that data and templates can be
combined directly:   rho(t) ~ IFFT( d * conj(h) / asd^2 ).

Two ways to choose the chirp-mass grid:
  * fixed   : log-uniform with constant fractional spacing (--dlnmc)           [fast, default]
  * adaptive: step in ln(Mc) as large as possible while neighbouring templates
              keep a match >= --min-match for every q (--adaptive)             [slow, minutes-hours]

Example
-------
python build_template_bank.py \
    --out /scratch/tpausch/template_bank/bank_stage1.h5 \
    --mc-min 50 --mc-max 1000 --q-min 0.1 --q-max 1.0 --n-q 6 \
    --f-min 5 --f-max 256 --delta-f 0.125 --f-ref 20 \
    --asd /scratch/tpausch/ASD/asds_ET_T8.hdf5 \
    --dlnmc 0.03 --workers 16 --check 30

(Numbers above are placeholders: take them from your stage-1 prior and data domain.)
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool

import h5py
import lal
import lalsimulation as lalsim
import numpy as np

DIST_MPC = 1000.0  # arbitrary; amplitude is normalised away


# --------------------------------------------------------------------------------------
# Waveforms
# --------------------------------------------------------------------------------------
def mc_q_to_m1m2(mc, q):
    """Detector-frame chirp mass and q = m2/m1 <= 1  ->  m1, m2 in Msun."""
    m1 = mc * (1.0 + q) ** 0.2 * q ** -0.6
    return m1, q * m1


def fd_template(mc, q, cfg):
    """One-sided complex FD template on the data-domain grid, L2-normalised."""
    m1, m2 = mc_q_to_m1m2(mc, q)
    hp, _ = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,                 # zero spins
        DIST_MPC * 1e6 * lal.PC_SI,
        0.0, 0.0,                                     # iota = 0 (face-on), phi_ref = 0
        0.0, 0.0, 0.0,                                # longAscNodes, ecc, meanPerAno
        cfg["delta_f"], cfg["f_min"], cfg["f_max"], cfg["f_ref"],
        lal.CreateDict(),
        lalsim.GetApproximantFromString(cfg["approximant"]),
    )
    h = np.zeros(cfg["n_full"], dtype=complex)
    n = min(cfg["n_full"], hp.data.data.size)
    h[:n] = hp.data.data[:n]
    h[: cfg["i_min"]] = 0.0                           # nothing below f_min
    h = h[cfg["i0"]:]                                 # truncate to the grid used for storage
    return h / np.sqrt(np.sum(np.abs(h) ** 2))


def _gen(args_pair, cfg):
    return fd_template(args_pair[0], args_pair[1], cfg)


# --------------------------------------------------------------------------------------
# Matching (used for adaptive spacing and for the check)
# --------------------------------------------------------------------------------------
def match(a, b, w, pad=4):
    """max over time and phase of the normalised overlap <a|b>, weight w = 1/asd^2."""
    z = np.fft.ifft(a * np.conj(b) * w, n=pad * a.size) * (pad * a.size)
    return np.abs(z).max() / np.sqrt(np.sum(np.abs(a) ** 2 * w) * np.sum(np.abs(b) ** 2 * w))


def adaptive_mc_grid(cfg, q_grid, w, mc_min, mc_max, min_match, max_dlnmc):
    grid = [mc_min]
    while grid[-1] < mc_max:
        mc0 = grid[-1]
        h0 = [fd_template(mc0, q, cfg) for q in q_grid]

        def ok(step):
            return all(
                match(h0[j], fd_template(mc0 * np.exp(step), q, cfg), w) >= min_match
                for j, q in enumerate(q_grid)
            )

        lo, hi = 1e-3, max_dlnmc
        if ok(hi):
            step = hi
        else:
            for _ in range(8):  # bisection in ln(Mc)
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ok(mid) else (lo, mid)
            step = lo
        grid.append(mc0 * np.exp(step))
        print(f"  Mc = {grid[-1]:9.2f}  (dlnMc = {step:.4f})", flush=True)
    grid[-1] = mc_max
    return np.array(grid)


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------
def load_bank(path):
    """Returns templates (N_mc, N_q, F), mc, q, frequencies, attrs."""
    with h5py.File(path, "r") as f:
        return f["templates"][:], f["mc"][:], f["q"][:], f["frequencies"][:], dict(f.attrs)


def load_weight(asd_file, asd_index, cfg):
    """w = 1/asd^2 on the storage grid (mean over detectors); flat weights if no file."""
    n_store = cfg["n_full"] - cfg["i0"]
    if asd_file is None:
        print("WARNING: no ASD given, using flat weights for spacing/check.")
        return np.ones(n_store)
    from dingo.gw.noise.asd_dataset import ASDDataset

    asds = ASDDataset(file_name=asd_file).asds
    w = 0.0
    for det, arr in asds.items():
        a = np.asarray(arr)[asd_index]
        if a.size == n_store:
            pass
        elif a.size == cfg["n_full"]:
            a = a[cfg["i0"]:]
        else:
            raise ValueError(
                f"ASD '{det}' has {a.size} bins, expected {n_store} (f>=f_min) or "
                f"{cfg['n_full']} (full grid). Check f_min/f_max/delta_f."
            )
        w = w + 1.0 / a ** 2
    return w / len(asds)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mc-min", type=float, required=True)
    ap.add_argument("--mc-max", type=float, required=True)
    ap.add_argument("--q-min", type=float, default=0.1)
    ap.add_argument("--q-max", type=float, default=1.0)
    ap.add_argument("--n-q", type=int, default=6)
    ap.add_argument("--f-min", type=float, required=True)
    ap.add_argument("--f-max", type=float, required=True)
    ap.add_argument("--delta-f", type=float, required=True)
    ap.add_argument("--f-ref", type=float, default=20.0)
    ap.add_argument("--approximant", default="IMRPhenomXPHM")
    ap.add_argument("--asd", default=None, help="ASD hdf5 (dingo ASDDataset); used for spacing + check")
    ap.add_argument("--asd-index", type=int, default=0)
    ap.add_argument("--dlnmc", type=float, default=0.03, help="fixed grid: fractional step in Mc")
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--min-match", type=float, default=0.97)
    ap.add_argument("--max-dlnmc", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", type=int, default=0, help="test the bank on N random off-grid (Mc, q)")
    args = ap.parse_args()

    n_full = int(round(args.f_max / args.delta_f)) + 1
    i_min = int(round(args.f_min / args.delta_f))
    cfg = dict(approximant=args.approximant, f_min=args.f_min, f_max=args.f_max,
               delta_f=args.delta_f, f_ref=args.f_ref, n_full=n_full, i_min=i_min, i0=0)

    # Storage grid: truncated to f >= f_min if the ASD file is (dingo's data domain usually
    # is), otherwise the full 0..f_max grid.
    if args.asd is not None:
        from dingo.gw.noise.asd_dataset import ASDDataset

        n_asd = np.asarray(next(iter(ASDDataset(file_name=args.asd).asds.values()))).shape[-1]
        if n_asd == n_full - i_min:
            cfg["i0"] = i_min
    freqs = (cfg["i0"] + np.arange(n_full - cfg["i0"])) * args.delta_f
    w = load_weight(args.asd, args.asd_index, cfg)

    q_grid = np.geomspace(args.q_min, args.q_max, args.n_q)

    # ---- chirp-mass grid -------------------------------------------------------------
    if args.adaptive:
        print("Adaptive chirp-mass spacing ...")
        mc_grid = adaptive_mc_grid(cfg, q_grid, w, args.mc_min, args.mc_max,
                                   args.min_match, args.max_dlnmc)
    else:
        n_mc = int(np.ceil(np.log(args.mc_max / args.mc_min) / args.dlnmc)) + 1
        mc_grid = np.geomspace(args.mc_min, args.mc_max, n_mc)
    print(f"Bank: {len(mc_grid)} Mc x {len(q_grid)} q = {len(mc_grid) * len(q_grid)} templates, "
          f"{freqs.size} frequency bins")

    # ---- generate --------------------------------------------------------------------
    pairs = [(mc, q) for mc in mc_grid for q in q_grid]
    with Pool(args.workers) as pool:
        out = pool.map(partial(_gen, cfg=cfg), pairs, chunksize=4)
    T = np.array(out).reshape(len(mc_grid), len(q_grid), -1).astype(np.complex64)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("templates", data=T, compression="gzip", compression_opts=4)
        f["mc"], f["q"], f["frequencies"] = mc_grid, q_grid, freqs
        f.attrs.update(dict(
            approximant=args.approximant, f_min=args.f_min, f_max=args.f_max,
            delta_f=args.delta_f, f_ref=args.f_ref, i0=cfg["i0"],
            spins="zero", inclination=0.0, phi_ref=0.0,
            normalisation="L2 over stored bins, unwhitened",
            mass_convention="detector-frame chirp mass [Msun], q=m2/m1<=1",
            min_match=args.min_match if args.adaptive else -1.0,
            grid="adaptive" if args.adaptive else f"fixed dlnMc={args.dlnmc}",
        ))
    print(f"Saved {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    # ---- check: fitting factor at random off-grid points -------------------------------
    if args.check:
        rng = np.random.default_rng(0)
        flat = T.reshape(-1, T.shape[-1]).astype(complex)
        ff = []
        for _ in range(args.check):
            mc = np.exp(rng.uniform(np.log(args.mc_min), np.log(args.mc_max)))
            q = np.exp(rng.uniform(np.log(args.q_min), np.log(args.q_max)))
            h = fd_template(mc, q, cfg)
            ff.append(max(match(h, b, w) for b in flat))
        ff = np.array(ff)
        print(f"Fitting factor over {args.check} random points: "
              f"min {ff.min():.3f}, 5th pct {np.percentile(ff, 5):.3f}, median {np.median(ff):.3f}")


if __name__ == "__main__":
    main()
