"""
Schnelles Template Matching zur Chirp-Massen-Abschaetzung (ET-HF-IMBH-Modell).

Unterschied zum alten Notebook
------------------------------
* Die Bank wird EINMAL erzeugt (Polarisation h_plus bei fester Referenzdistanz,
  phase=0, geocent_time=0) und als Tensor gespeichert. Pro Injektion gibt es
  danach keinen Wellenform-Aufruf mehr, nur GPU-Filterung.
* Die Extrinsik wird NICHT mehr aus der Injektion uebernommen (blind):
    - Amplitude/Phase/Polarisation: pro Detektorkanal eine freie komplexe Amplitude
    - Koaleszenzzeit: Maximierung ueber das Prior-Fenster +-(0.1 s + Detektorlaufzeit)
    - Inklination: kleine Nuisance-Achse (theta_jn) in der Bank
  Statistik:  rho^2(t) = sum_c |z_c(t)|^2 / (h|h)_c ,  z_c(t) = (d_c|h)(t)
* Chirp-Masse: Profil ueber das Maximum aller Nuisance-Parameter pro Chirp-Massen-Gitterpunkt,
  danach Parabel-Verfeinerung in log(Mc).

Annahmen (bitte einmal mit sanity_check pruefen):
* Dingo-WaveformGenerator ist unter `generator.waveform_generator` erreichbar und
  `generate_hplus_hcross(params)` liefert {"h_plus", "h_cross"}.
* Datenarrays pro Detektor sind 1D komplex, Laenge 2049 (0..f_max) oder 2001 (f_min..f_max).
"""
from __future__ import annotations

import multiprocessing as mp
import re
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
@dataclass
class BankConfig:
    # Domain (Modell-Metadaten)
    f_min: float = 6.0
    f_max: float = 256.0
    delta_f: float = 0.125
    # Prior (Modell-Metadaten)
    mc_min: float = 75.0
    mc_max: float = 1200.0
    m1_range: tuple = (80.0, 4500.0)
    m2_range: tuple = (30.0, 1500.0)
    d_l_ref: float = 1000.0
    # Gitter
    dlog_mc: float = 0.03                      # ~3 % Schritt in Mc (log-Gitter, falls mc_step None)
    mc_step: float = None                      # z.B. 10.0 -> aequidistantes Mc-Gitter in M_sun
    mass_ratios: tuple = (0.125, 0.2, 0.35, 0.6, 1.0)
    spins: tuple = (-0.6, 0.0, 0.6)            # chi-Werte (Prior: [-0.9, 0.9])
    spin_mode: str = "pairs"                   # "pairs": alle (chi_1, chi_2) aus spins^2
                                               # "equal": nur chi_1 = chi_2 (deckt chi_eff ab, viel billiger)
    theta_jn: tuple = (0.6, 1.5708)            # Nuisance-Achse Inklination
    phases: tuple = (0.0,)                     # Referenzphase. Nur bei Higher Modes (XHM/XPHM)
                                               # relevant, dann z.B. np.linspace(0, 2*np.pi, 6, endpoint=False)

    @property
    def n_full(self) -> int:
        return int(round(self.f_max / self.delta_f)) + 1

    @property
    def i_min(self) -> int:
        return int(round(self.f_min / self.delta_f))

    @property
    def n_band(self) -> int:
        return self.n_full - self.i_min

    @classmethod
    def from_metadata(cls, metadata: dict, **overrides) -> "BankConfig":
        """Liest Domain und Prior-Grenzen aus den Dingo-Modell-Metadaten."""
        ds = metadata["dataset_settings"]
        dom = ds["domain"]
        kw = dict(f_min=float(dom["f_min"]), f_max=float(dom["f_max"]),
                  delta_f=float(dom["delta_f"]))

        def bounds(s):
            m = re.search(r"minimum=([0-9.eE+\-_]+),\s*maximum=([0-9.eE+\-_]+)", str(s))
            return (float(m.group(1).replace("_", "")), float(m.group(2).replace("_", "")))

        pr = ds["intrinsic_prior"]
        kw["mc_min"], kw["mc_max"] = bounds(pr["chirp_mass"])
        kw["m1_range"] = bounds(pr["mass_1"])
        kw["m2_range"] = bounds(pr["mass_2"])
        kw["d_l_ref"] = float(pr.get("luminosity_distance", 1000.0))
        kw.update(overrides)
        return cls(**kw)


def component_masses(mc, q):
    """(m1, m2) mit q = m2/m1 <= 1."""
    m1 = mc * (1.0 + q) ** 0.2 / q ** 0.6
    return m1, q * m1


def make_grid(cfg: BankConfig) -> dict:
    """Gitter ueber den Prior-Support; Constraints auf mass_1/mass_2 werden angewendet."""
    if cfg.mc_step:
        n_mc = int(np.floor((cfg.mc_max - cfg.mc_min) / cfg.mc_step + 1e-9)) + 1
        mc_axis = cfg.mc_min + np.arange(n_mc) * cfg.mc_step
    else:
        n_mc = int(np.floor(np.log(cfg.mc_max / cfg.mc_min) / cfg.dlog_mc)) + 1
        mc_axis = cfg.mc_min * np.exp(np.arange(n_mc) * cfg.dlog_mc)
        mc_axis = mc_axis[mc_axis <= cfg.mc_max * (1 + 1e-9)]

    rows = []
    for i, mc in enumerate(mc_axis):
        if cfg.spin_mode == "equal":
            spin_pairs = [(c, c) for c in cfg.spins]
        elif cfg.spin_mode == "pairs":
            spin_pairs = list(product(cfg.spins, cfg.spins))
        else:
            raise ValueError(f"spin_mode muss 'pairs' oder 'equal' sein, nicht {cfg.spin_mode!r}")
        for q, (c1, c2), th, ph in product(cfg.mass_ratios, spin_pairs, cfg.theta_jn,
                                           cfg.phases):
            m1, m2 = component_masses(mc, q)
            if not (cfg.m1_range[0] <= m1 <= cfg.m1_range[1]
                    and cfg.m2_range[0] <= m2 <= cfg.m2_range[1]):
                continue
            rows.append((i, mc, q, c1, c2, th, ph))
    arr = np.array(rows, dtype=float)
    return dict(mc_index=arr[:, 0].astype(np.int64), chirp_mass=arr[:, 1],
                mass_ratio=arr[:, 2], chi_1=arr[:, 3], chi_2=arr[:, 4],
                theta_jn=arr[:, 5], phase=arr[:, 6], mc_axis=mc_axis)


# --------------------------------------------------------------------------- #
# Frequenzachse
# --------------------------------------------------------------------------- #
def crop_to_band(arr, cfg: BankConfig):
    """Akzeptiert Laenge n_full (0..f_max) oder n_band (f_min..f_max); gibt n_band zurueck."""
    arr = np.asarray(arr).reshape(-1)
    if arr.shape[0] == cfg.n_full:
        return arr[cfg.i_min:]
    if arr.shape[0] == cfg.n_band:
        return arr
    raise ValueError(f"Array-Laenge {arr.shape[0]} passt weder zu n_full={cfg.n_full} "
                     f"noch zu n_band={cfg.n_band}")


# --------------------------------------------------------------------------- #
# Bank-Erzeugung
# --------------------------------------------------------------------------- #
_WORKER_FN = None


def _work(params):
    try:
        return _WORKER_FN(params)
    except Exception as exc:  # einzelne Fehlschlaege nicht die ganze Bank kippen lassen
        return exc


def make_dingo_h0(generator, cfg: BankConfig):
    """Wrapper um dingo: h_plus(f) fuer intrinsische Parameter (phase=0, t_c=0, d_L=ref)."""
    wfg = generator.waveform_generator

    def gen(p):
        m1, m2 = component_masses(p["chirp_mass"], p["mass_ratio"])
        params = dict(
            chirp_mass=float(p["chirp_mass"]), mass_ratio=float(p["mass_ratio"]),
            mass_1=float(m1), mass_2=float(m2),
            chi_1=float(p["chi_1"]), chi_2=float(p["chi_2"]),
            theta_jn=float(p["theta_jn"]), phase=float(p.get("phase", 0.0)),
            luminosity_distance=cfg.d_l_ref, geocent_time=0.0,
        )
        pol = wfg.generate_hplus_hcross(params)
        return pol["h_plus"]

    return gen


def build_bank(cfg: BankConfig, generate_h0, n_workers: int = 1, verbose: bool = True):
    """
    generate_h0(params_dict) -> komplexes Array (Laenge n_full oder n_band).
    Gibt (h [N, n_band] complex64, grid-dict) zurueck.
    """
    global _WORKER_FN
    grid = make_grid(cfg)
    names = ("chirp_mass", "mass_ratio", "chi_1", "chi_2", "theta_jn", "phase")
    plist = [{k: grid[k][i] for k in names} for i in range(len(grid["chirp_mass"]))]
    if verbose:
        print(f"Bank: {len(plist):,} Templates, {len(grid['mc_axis'])} Mc-Gitterpunkte")

    _WORKER_FN = generate_h0
    if n_workers > 1:
        with mp.get_context("fork").Pool(n_workers) as pool:
            out = pool.map(_work, plist, chunksize=16)
    else:
        out = [_work(p) for p in plist]

    keep, rows = [], []
    for i, o in enumerate(out):
        if isinstance(o, Exception) or o is None:
            continue
        h = crop_to_band(o, cfg)
        if np.all(np.isfinite(h)) and np.any(h != 0):
            keep.append(i)
            rows.append(h.astype(np.complex64))
    if len(keep) < len(plist):
        print(f"WARNUNG: {len(plist) - len(keep)} Templates verworfen (Fehler/NaN/0).")
    keep = np.array(keep)
    grid = {k: (v[keep] if k != "mc_axis" else v) for k, v in grid.items()}
    return np.stack(rows), grid


def _peak(prof, axis, dlog, step=None):
    """argmax und Parabel-Verfeinerung eines Profils: log-Gitter (dlog) oder lineares (step)."""
    i = int(np.argmax(prof))
    delta = 0.0
    if 0 < i < len(prof) - 1:
        y0, y1, y2 = prof[i - 1], prof[i], prof[i + 1]
        den = y0 - 2 * y1 + y2
        if np.isfinite(den) and den < 0:
            delta = float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))
    if step:
        return i, float(axis[i] + delta * step)
    return i, float(axis[i] * np.exp(delta * dlog))


class TemplateBank:
    def __init__(self, h, grid, cfg: BankConfig, device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Rohwellenformen sind ~1e-24: |h|^2 laeuft in float32 unter. Pro Template auf
        # Einheitsnorm normieren (in float64). rho^2 ist von der Template-Amplitude unabhaengig.
        h = np.asarray(h).astype(np.complex128)
        h /= np.sqrt(np.sum(np.abs(h) ** 2, axis=1, keepdims=True))
        self.h = torch.as_tensor(h, dtype=torch.complex64, device=self.device)   # [N, F]
        self.h2 = (self.h.real ** 2 + self.h.imag ** 2)                           # [N, F]
        self.grid = grid
        self.mc_index = torch.as_tensor(grid["mc_index"], device=self.device)
        self.mc_axis = np.asarray(grid["mc_axis"])
        self.n_mc = len(self.mc_axis)
        # log-Prior-Gewicht jeder Gitterzelle (Prior uniform in Mc, q, chi; sin(theta_jn)).
        # Log-Mc-Gitter: Zellbreite ~ Mc. q-/chi-Gitter sollten aequidistant sein.
        lw = np.zeros(len(grid["chirp_mass"]))
        if not cfg.mc_step:
            lw += np.log(np.asarray(grid["chirp_mass"]))
        lw += np.log(np.clip(np.sin(np.asarray(grid["theta_jn"])), 1e-3, None))
        self.log_w = torch.as_tensor(lw, dtype=torch.float32, device=self.device)
        # Gesamtmassen-Profil (Diagnose: Merger-Ringdown legt M fest, Mc ist entartet)
        m1, m2 = component_masses(np.asarray(grid["chirp_mass"]), np.asarray(grid["mass_ratio"]))
        mt = m1 + m2
        lo = float(mt.min())
        n_m = int(np.floor(np.log(mt.max() / lo) / cfg.dlog_mc)) + 1
        self.m_axis = lo * np.exp(np.arange(n_m) * cfg.dlog_mc)
        idx = np.clip(np.round(np.log(mt / lo) / cfg.dlog_mc).astype(np.int64), 0, n_m - 1)
        self.m_index = torch.as_tensor(idx, device=self.device)
        self.n_m = n_m

    def __len__(self):
        return self.h.shape[0]

    def save(self, path):
        torch.save(dict(h=self.h.cpu(), grid=self.grid, cfg=asdict(self.cfg)), path)

    @classmethod
    def load(cls, path, device="cuda"):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = BankConfig(**blob["cfg"])
        return cls(blob["h"].numpy(), blob["grid"], cfg, device=device)


# --------------------------------------------------------------------------- #
# Matched filter
# --------------------------------------------------------------------------- #
class ChirpMassMatcher:
    """
    oversample : Zeitaufloesung dt = 1 / (oversample * f_max * 2 ...) -> 4 gibt ~1 ms
    t_window   : Suchfenster fuer die Koaleszenzzeit [s]: Prior +-0.1 s + Detektorlaufzeit
    chunk      : Templates pro GPU-Block (Speicher ~ chunk * C * M * 8 Byte)
    """

    def __init__(self, bank: TemplateBank, oversample: int = 4,
                 t_window: float = 0.125, chunk: int = 256):
        self.bank, self.cfg, self.dev = bank, bank.cfg, bank.device
        self.chunk = chunk
        self.M = int(oversample * (self.cfg.n_full - 1))
        self.dt = 1.0 / (self.M * self.cfg.delta_f)
        self.K = int(t_window / self.dt)

    # -- Daten vorbereiten ---------------------------------------------------
    def _prep(self, data: dict, asds: dict):
        dets = sorted(set(data) & set(asds))
        if not dets:
            raise ValueError(f"Keine gemeinsamen Detektoren: {list(data)} / {list(asds)}")
        d = np.stack([crop_to_band(data[k], self.cfg) for k in dets]).astype(np.complex128)
        a = np.stack([crop_to_band(asds[k], self.cfg) for k in dets]).astype(np.float64)
        ok = np.isfinite(a) & (a > 0) & np.isfinite(d.real) & np.isfinite(d.imag)
        if not ok.any():
            raise ValueError("Keine gueltigen Frequenzbins (ASD/Daten nicht endlich oder <= 0)")
        # d und ASD gemeinsam auf O(1) skalieren (rho^2 ~ d^2/asd^2 ist invariant),
        # sonst underflowt ASD^2 ~ 1e-48 in float32.
        scale = np.median(a[ok])
        a = np.where(ok, a / scale, 1.0)
        d = np.where(ok, d / scale, 0.0)
        inv_psd = np.where(ok, 4.0 * self.cfg.delta_f / a ** 2, 0.0)
        d = torch.as_tensor(d, dtype=torch.complex64, device=self.dev)
        inv_psd = torch.as_tensor(inv_psd, dtype=torch.float32, device=self.dev)
        return d, inv_psd, dets

    # -- rho^2 pro Template --------------------------------------------------
    def _filter_block(self, hc, w, inv_psd):
        """hc [n,F] Templates, w = d*inv_psd [C,F] -> (rho2 max ueber t [n], log L marg. [n])."""
        C, F = w.shape
        i0 = self.cfg.i_min
        n = hc.shape[0]
        h2 = hc.real ** 2 + hc.imag ** 2
        norm = (h2 @ inv_psd.T).clamp_min(1e-30)              # [n, C]  (h|h)_c
        X = hc.conj().unsqueeze(1) * w.unsqueeze(0)           # [n, C, F]
        Xp = torch.zeros(n, C, self.M, dtype=torch.complex64, device=self.dev)
        Xp[..., i0:i0 + F] = X
        z = self.M * torch.fft.ifft(Xp, dim=-1)               # z_c(t), komplex
        zw = torch.cat([z[..., :self.K + 1], z[..., -self.K:]], dim=-1)
        p = ((zw.real ** 2 + zw.imag ** 2) / norm.unsqueeze(-1)).sum(1)   # [n, T]
        # Marginal-Likelihood: freie komplexe Amplitude pro Kanal analytisch integriert
        # (Faktor 1/(h|h)_c), Koaleszenzzeit numerisch marginalisiert.
        logL = torch.logsumexp(0.5 * p, dim=-1) - torch.log(norm).sum(1)
        return p.amax(-1), logL

    @torch.no_grad()
    def rho2(self, data: dict, asds: dict, return_logL: bool = False):
        d, inv_psd, _ = self._prep(data, asds)
        w = d * inv_psd                                       # [C, F]
        N = len(self.bank)
        out = torch.empty(N, device=self.dev)
        logL = torch.empty(N, device=self.dev)
        for s in range(0, N, self.chunk):
            out[s:s + self.chunk], logL[s:s + self.chunk] = self._filter_block(
                self.bank.h[s:s + self.chunk], w, inv_psd)
        return (out, logL) if return_logL else out

    # -- Chirp-Massen-Schaetzer ---------------------------------------------
    @torch.no_grad()
    def estimate(self, data: dict, asds: dict, temperature: float = 1.0,
                 n_top: int = 5) -> dict:
        rho2, logL = self.rho2(data, asds, return_logL=True)
        if not torch.isfinite(rho2).all():
            raise FloatingPointError(
                f"{int((~torch.isfinite(rho2)).sum())} nicht-endliche rho2-Werte")
        n_mc = self.bank.n_mc
        prof = torch.full((n_mc,), -float("inf"), device=self.dev)
        prof = prof.scatter_reduce(0, self.bank.mc_index, rho2, reduce="amax")
        prof = prof.cpu().numpy()
        axis = self.bank.mc_axis
        i, mc_ref = _peak(prof, axis, self.cfg.dlog_mc, step=self.cfg.mc_step)

        profm = torch.full((self.bank.n_m,), -float("inf"), device=self.dev)
        profm = profm.scatter_reduce(0, self.bank.m_index, rho2, reduce="amax").cpu().numpy()
        _, mt_ref = _peak(profm, self.bank.m_axis, self.cfg.dlog_mc)

        # Likelihood-artig gewichteter Mittelwert in log(Mc)
        wgt = np.exp(-(prof.max() - prof) / (2.0 * temperature))
        mc_w = float(np.exp(np.sum(wgt * np.log(axis)) / np.sum(wgt)))

        # Marginale Posterior p(Mc|d): Summe (nicht Maximum) ueber q, chi, theta_jn, phase,
        # gewichtet mit dem Prior. Robuster entlang der Mc-q-Entartung als das Profil.
        lp = logL + self.bank.log_w
        wt = torch.exp(lp - lp.max())
        post = torch.zeros(n_mc, device=self.dev).scatter_add(0, self.bank.mc_index, wt)
        post = (post / post.sum()).cpu().numpy()
        cdf = np.cumsum(post)
        lax = np.log(axis)
        mc_q = np.exp(np.interp([0.05, 0.5, 0.95], cdf, lax))
        mc_mean = float(np.exp(np.sum(post * lax)))

        # bestes Template der n_top besten Mc-Gitterpunkte (Startpunkte fuer LocalRefiner)
        top = []
        for ii in np.argsort(prof)[::-1][:n_top]:
            if not np.isfinite(prof[ii]):
                break
            sel = torch.nonzero(self.bank.mc_index == int(ii)).squeeze(1)
            jj = int(sel[torch.argmax(rho2[sel])])
            top.append((jj, float(rho2[jj])))

        j = int(torch.argmax(rho2))
        g = self.bank.grid
        return dict(mc_argmax=float(axis[i]), mc_refined=mc_ref, mc_weighted=mc_w,
                    mc_median=float(mc_q[1]), mc_mean=mc_mean,
                    mc_q05=float(mc_q[0]), mc_q95=float(mc_q[2]),
                    mt_refined=mt_ref,
                    snr_best=float(np.sqrt(prof[i])),
                    q_best=float(g["mass_ratio"][j]), chi_1_best=float(g["chi_1"][j]),
                    chi_2_best=float(g["chi_2"][j]), theta_jn_best=float(g["theta_jn"][j]),
                    phase_best=float(g["phase"][j]) if "phase" in g else 0.0,
                    profile=prof, posterior=post, top_templates=top)


# --------------------------------------------------------------------------- #
# Auswertung ueber ein Injektionsverzeichnis (Format wie im alten Notebook)
# --------------------------------------------------------------------------- #
def load_injection(path):
    import pickle
    with open(path, "rb") as f:
        payload = pickle.load(f)
    event = payload["strain_data"]
    asds = event.get("asd", event.get("asds"))
    if asds is None:
        raise KeyError(f"Kein ASD im Event. Keys: {list(event.keys())}")
    return payload["theta_true"], event["waveform"], asds


def evaluate_injections(matcher: ChirpMassMatcher, injection_dir, max_injections=None,
                        kernel_halfwidth: float = 45.0, temperature: float = 1.0,
                        refiner=None, n_workers: int = 1):
    import pandas as pd
    files = sorted(Path(injection_dir).glob("inj_*.pkl"))
    if max_injections:
        files = files[:max_injections]
    rows, jobs = [], []
    for k, fn in enumerate(files):
        theta, data, asds = load_injection(fn)
        est = matcher.estimate(data, asds, temperature=temperature)
        if refiner is not None:
            jobs.append(refiner.prepare(data, asds, est))
        for key in ("profile", "posterior", "top_templates"):
            est.pop(key)
        mc_true = float(theta["chirp_mass"])
        rows.append(dict(file=fn.name, chirp_mass_true=mc_true,
                         luminosity_distance=theta.get("luminosity_distance", np.nan), **est))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(files)}", end="\r")
    if refiner is not None:
        print(f"Bank-Stufe fertig, verfeinere {len(jobs)} Injektionen mit {n_workers} Workern ...")
        for row, r in zip(rows, refine_many(refiner, jobs, n_workers=n_workers)):
            if isinstance(r, Exception):
                print(f"WARNUNG: Refinement fehlgeschlagen fuer {row['file']}: {r!r}")
                continue
            row.update(r)
    df = pd.DataFrame(rows)
    for col in [c for c in ("mc_argmax", "mc_refined", "mc_weighted", "mc_median",
                            "mc_mean", "mc_local") if c in df]:
        df[f"err_{col}"] = df[col] - df["chirp_mass_true"]
        df[f"inside_{col}"] = df[f"err_{col}"].abs() <= kernel_halfwidth
    return df


@torch.no_grad()
def sanity_check(matcher: ChirpMassMatcher, generator, theta_true: dict, asds: dict = None):
    """
    Rauschfreie Injektion: Fitting-Faktor FF = rho_best / rho_opt sollte nahe 1
    liegen (>~0.9) und mc_refined nahe am wahren Wert. Prueft zugleich Konventionen
    (Frequenzachse, Zeitfenster, h_plus-API).

    generator.signal() liefert nur die Wellenform, keinen ASD. Der ASD kommt aus
    `asds` oder aus `generator.asd` (dict Detektor -> Array), siehe Notebook.
    """
    if asds is None:
        asds = getattr(generator, "asd", None)
    if not asds:
        raise ValueError("Kein ASD: asds=... uebergeben oder generator.asd setzen.")
    sig = generator.signal(theta_true)
    data = sig["waveform"]
    est = matcher.estimate(data, asds)
    d, inv_psd, _ = matcher._prep(data, asds)
    rho_opt = float(torch.sqrt(((d.real ** 2 + d.imag ** 2) * inv_psd).sum()))
    print(f"Mc_true={theta_true['chirp_mass']:.1f}  Mc_refined={est['mc_refined']:.1f}  "
          f"FF={est['snr_best'] / rho_opt:.3f}  rho_opt={rho_opt:.1f}")
    return est, rho_opt


# --------------------------------------------------------------------------- #
# Lokale kontinuierliche Verfeinerung (zweite Stufe)
# --------------------------------------------------------------------------- #
def _rho2_np(h, w, inv_psd, i0, M, K):
    """rho^2 (max ueber t) fuer EIN Template, reines numpy -> laeuft in Worker-Prozessen ohne CUDA."""
    from scipy import fft as sfft
    F = h.shape[0]
    h = h / np.sqrt(np.sum(np.abs(h) ** 2))            # skaleninvariant; vermeidet Underflow
    norm = np.maximum((np.abs(h) ** 2) @ inv_psd.T, 1e-300)         # [C]
    Xp = np.zeros((w.shape[0], M), dtype=np.complex128)
    Xp[:, i0:i0 + F] = h.conj()[None, :] * w
    z = M * sfft.ifft(Xp, axis=-1, workers=1)
    zw = np.concatenate([z[:, :K + 1], z[:, -K:]], axis=-1)
    return float(((np.abs(zw) ** 2) / norm[:, None]).sum(0).max())


class LocalRefiner:
    """
    Kontinuierliche Optimierung (Nelder-Mead ueber log Mc, q, chi_1, chi_2, theta_jn[, phase])
    ausgehend von den besten Bank-Templates. Die Kosten sind fast nur Wellenform-Aufrufe
    (CPU), daher: wenige Startpunkte, grobe Abbruchtoleranz, Parallelisierung ueber Injektionen.

    generate_h0  : wie fuer build_bank, z.B. make_dingo_h0(generator, cfg)
    n_starts     : max. Startpunkte (beste Templates der besten Mc-Gitterpunkte)
    restarts     : Nelder-Mead wird so oft vom eigenen Ergebnis neu gestartet (frischer
                   Simplex); verhindert vorzeitiges Festlaufen entlang der Mc-q-Entartung
    start_drho2  : Startpunkte mit rho2 < rho2_best - start_drho2 werden uebersprungen
    fatol        : Abbruch, wenn sich rho^2 im Simplex um weniger aendert (rho^2-Einheiten;
                   0.05 ist fein genug; das Ergebnis haengt kaum davon ab)
    fit_theta    : theta_jn mitoptimieren (bei Higher Modes/XPHM wichtig, ~30 % teurer)
    fit_phase    : Referenzphase mitoptimieren (bei Higher Modes noetig, sonst harmlos)
    """

    def __init__(self, matcher: ChirpMassMatcher, generate_h0, n_starts: int = 2,
                 start_drho2: float = 30.0, maxiter: int = 300, fatol: float = 0.05,
                 xatol: float = 1e-3, restarts: int = 3, q_min: float = None,
                 chi_max: float = 0.9, fit_phase: bool = True, fit_theta: bool = True):
        self.m, self.gen = matcher, generate_h0
        self.restarts = restarts
        self.n_starts, self.start_drho2 = n_starts, start_drho2
        self.maxiter, self.fatol, self.xatol = maxiter, fatol, xatol
        g = matcher.bank.grid
        self.q_min = q_min if q_min is not None else float(np.min(g["mass_ratio"]))
        self.chi_max = chi_max
        self.fit_phase = fit_phase
        self.fit_theta = fit_theta

    # -- GPU-Teil (Hauptprozess): Startpunkte + Daten als numpy ----------------
    def prepare(self, data, asds, est: dict) -> dict:
        d, inv_psd, _ = self.m._prep(data, asds)
        w = (d * inv_psd).cpu().numpy().astype(np.complex128)
        inv_psd = inv_psd.cpu().numpy().astype(np.float64)
        g = self.m.bank.grid
        top = est["top_templates"]                      # [(index, rho2), ...] absteigend
        best = top[0][1]
        starts, fixed = [], []
        for j, r2 in top[:self.n_starts]:
            if r2 < best - self.start_drho2:
                break
            x0 = [np.log(g["chirp_mass"][j]), g["mass_ratio"][j], g["chi_1"][j], g["chi_2"][j]]
            if self.fit_theta:
                x0.append(g["theta_jn"][j])
            if self.fit_phase:
                x0.append(g["phase"][j] if "phase" in g else 0.0)
            starts.append(np.array(x0, dtype=float))
            fixed.append(dict(theta_jn=float(g["theta_jn"][j]),
                              phase=float(g["phase"][j]) if "phase" in g else 0.0))
        return dict(w=w, inv_psd=inv_psd, starts=starts, fixed=fixed, rho2_bank=best)

    # -- CPU-Teil (parallelisierbar) -------------------------------------------
    def _unpack(self, x, fixed):
        p = dict(chirp_mass=float(np.exp(x[0])), mass_ratio=x[1], chi_1=x[2], chi_2=x[3],
                 theta_jn=fixed["theta_jn"], phase=fixed["phase"])
        k = 4
        if self.fit_theta:
            p["theta_jn"] = x[k]; k += 1
        if self.fit_phase:
            p["phase"] = x[k] % (2 * np.pi)
        if p["mass_ratio"] > 1.0:   # q > 1 = vertauschte Massen: gleiche Wellenform, gleiches Mc
            p["mass_ratio"] = 1.0 / p["mass_ratio"]
            p["chi_1"], p["chi_2"] = p["chi_2"], p["chi_1"]
        return p

    def _in_prior(self, p) -> bool:
        cfg = self.m.cfg
        mc, q = p["chirp_mass"], p["mass_ratio"]
        if not (cfg.mc_min <= mc <= cfg.mc_max and self.q_min <= q <= 1.0
                and abs(p["chi_1"]) <= self.chi_max and abs(p["chi_2"]) <= self.chi_max
                and 0 <= p["theta_jn"] <= np.pi):
            return False
        m1, m2 = component_masses(mc, q)
        return (cfg.m1_range[0] <= m1 <= cfg.m1_range[1]
                and cfg.m2_range[0] <= m2 <= cfg.m2_range[1])

    def _objective(self, x, fixed, job, counter):
        cfg = self.m.cfg
        p = self._unpack(x, fixed)
        if not self._in_prior(p):
            return 0.0
        counter[0] += 1
        try:
            h = crop_to_band(self.gen(p), cfg).astype(np.complex128)
        except Exception:
            return 0.0
        nrm = np.sqrt(np.sum(np.abs(h) ** 2))
        if not np.isfinite(nrm) or nrm == 0:
            return 0.0
        return -_rho2_np(h / nrm, job["w"], job["inv_psd"], cfg.i_min, self.m.M, self.m.K)

    def optimize(self, job: dict) -> dict:
        import time
        from scipy.optimize import minimize
        t0, counter = time.time(), [0]
        best_x, best_fixed, best_r2 = None, None, -np.inf
        steps = np.array([0.05, 0.1, 0.3, 0.3] + ([0.3] if self.fit_theta else [])
                         + ([0.6] if self.fit_phase else []))
        for x0, fixed in zip(job["starts"], job["fixed"]):
            x = np.array(x0, dtype=float)
            for _ in range(self.restarts):
                simplex = np.vstack([x] + [x + np.eye(len(x))[k] * steps[k]
                                           for k in range(len(x))])
                res = minimize(self._objective, x, args=(fixed, job, counter),
                               method="Nelder-Mead",
                               options=dict(maxiter=self.maxiter, initial_simplex=simplex,
                                            xatol=self.xatol, fatol=self.fatol, adaptive=True))
                x = res.x
            if -res.fun > best_r2:
                best_x, best_fixed, best_r2 = x, fixed, -res.fun
        if best_r2 < job["rho2_bank"]:            # Optimierer hat nichts gebracht
            best_x, best_r2 = job["starts"][0], job["rho2_bank"]
            best_fixed = job["fixed"][0]
        p = self._unpack(best_x, best_fixed)
        m1, m2 = component_masses(p["chirp_mass"], p["mass_ratio"])
        return dict(mc_local=p["chirp_mass"], q_local=float(p["mass_ratio"]),
                    chi_1_local=float(p["chi_1"]), chi_2_local=float(p["chi_2"]),
                    mt_local=float(m1 + m2), snr_local=float(np.sqrt(max(best_r2, 0.0))),
                    n_wf_calls=counter[0], t_refine=time.time() - t0)

    def refine(self, data, asds, est: dict = None) -> dict:
        if est is None:
            est = self.m.estimate(data, asds)
        return self.optimize(self.prepare(data, asds, est))


_REFINER = None


def _refine_work(job):
    try:
        return _REFINER.optimize(job)
    except Exception as exc:
        return exc


def refine_many(refiner: LocalRefiner, jobs: list, n_workers: int = 1) -> list:
    """Optimiert viele vorbereitete Jobs parallel (fork; Worker nutzen nur CPU/numpy)."""
    global _REFINER
    _REFINER = refiner
    if n_workers > 1:
        with mp.get_context("fork").Pool(n_workers) as pool:
            return pool.map(_refine_work, jobs, chunksize=1)
    return [_refine_work(j) for j in jobs]


# --------------------------------------------------------------------------- #
# Direkter LALSimulation-Aufruf (umgeht den Dingo-Wrapper, ~1 ms statt ~200 ms)
# --------------------------------------------------------------------------- #
class LALWaveform:
    """
    h_plus ueber SimInspiralChooseFDWaveform mit denselben Argumenten wie Dingo
    (uniformes Gitter, delta_f, f_min, f_max, f_ref, lal_params=None). Nur alignierte Spins,
    wie im Dingo-Prior. Drop-in fuer generate_h0 in build_bank und LocalRefiner.
    Konventionen einmal mit check_lal_vs_dingo pruefen.

    approximant : Name ("IMRPhenomXPHM") oder LAL-ID (101 im Dingo-Log = IMRPhenomXPHM)
    """

    def __init__(self, cfg: BankConfig, approximant, f_ref: float = 20.0):
        import lal
        import lalsimulation as ls
        self.lal, self.ls, self.cfg = lal, ls, cfg
        self.approx = (ls.GetApproximantFromString(approximant) if isinstance(approximant, str)
                       else int(approximant))
        self.name = ls.GetStringFromApproximant(self.approx)
        self.f_ref = float(f_ref)
        self.dist = cfg.d_l_ref * 1e6 * lal.PC_SI

    @classmethod
    def from_generator(cls, generator, cfg: BankConfig):
        wfg = generator.waveform_generator
        approx = getattr(wfg, "approximant", None)
        if approx is None:
            approx = getattr(wfg, "approximant_str")
        return cls(cfg, approx, f_ref=float(getattr(wfg, "f_ref", 20.0)))

    def __call__(self, p: dict) -> np.ndarray:
        cfg = self.cfg
        m1, m2 = component_masses(float(p["chirp_mass"]), float(p["mass_ratio"]))
        hp, _ = self.ls.SimInspiralChooseFDWaveform(
            m1 * self.lal.MSUN_SI, m2 * self.lal.MSUN_SI,
            0.0, 0.0, float(p["chi_1"]), 0.0, 0.0, float(p["chi_2"]),
            self.dist, float(p["theta_jn"]), float(p.get("phase", 0.0)), 0.0, 0.0, 0.0,
            cfg.delta_f, cfg.f_min, cfg.f_max, self.f_ref, None, self.approx)
        return crop_to_band(hp.data.data[:cfg.n_full], cfg).astype(np.complex128)


def check_lal_vs_dingo(generator, cfg: BankConfig, lalwf: LALWaveform, n: int = 10,
                       seed: int = 0, asd=None):
    """
    Vergleicht Dingo-h_plus mit dem direkten LAL-Aufruf fuer zufaellige Parameter.
    'match' ist ueber Zeit und Phase maximiert und sollte > 0.9999 sein; 'roh' (ohne
    Maximierung) nahe 1 heisst, dass auch Zeit- und Phasenkonvention identisch sind.
    Bei SNR ~150 kostet ein Mismatch von 1e-3 schon Delta rho^2 ~ 45.
    """
    rng = np.random.default_rng(seed)
    dingo_gen = make_dingo_h0(generator, cfg)
    inv = (np.ones(cfg.n_band) if asd is None else 1.0 / crop_to_band(asd, cfg) ** 2)
    M = 4 * (cfg.n_full - 1)
    out = []
    while len(out) < n:
        p = dict(chirp_mass=float(np.exp(rng.uniform(np.log(cfg.mc_min), np.log(cfg.mc_max)))),
                 mass_ratio=float(rng.uniform(0.125, 1.0)), chi_1=float(rng.uniform(-0.9, 0.9)),
                 chi_2=float(rng.uniform(-0.9, 0.9)),
                 theta_jn=float(np.arccos(rng.uniform(-1, 1))),
                 phase=float(rng.uniform(0, 2 * np.pi)))
        m1, m2 = component_masses(p["chirp_mass"], p["mass_ratio"])
        if not (cfg.m1_range[0] <= m1 <= cfg.m1_range[1]
                and cfg.m2_range[0] <= m2 <= cfg.m2_range[1]):
            continue
        hd = crop_to_band(dingo_gen(p), cfg).astype(np.complex128)
        hl = lalwf(p)
        nd, nl = np.sum(np.abs(hd) ** 2 * inv), np.sum(np.abs(hl) ** 2 * inv)
        r2 = _rho2_np(hl, (hd * inv)[None, :], inv[None, :], cfg.i_min, M, M // 2 - 1)
        mt = float(np.sqrt(r2 / nd))
        raw = float(abs(np.sum(hd.conj() * hl * inv)) / np.sqrt(nd * nl))
        out.append(mt)
        print(f"Mc={p['chirp_mass']:7.1f} q={p['mass_ratio']:.2f} theta_jn={p['theta_jn']:.2f}  "
              f"match={mt:.6f}  roh={raw:.6f}  Amplitudenverh.={np.sqrt(nl / nd):.4f}")
    return np.array(out)
