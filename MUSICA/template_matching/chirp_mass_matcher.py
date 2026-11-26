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
    dlog_mc: float = 0.03                      # ~3 % Schritt in Mc
    mass_ratios: tuple = (0.125, 0.2, 0.35, 0.6, 1.0)
    spins: tuple = (-0.6, 0.0, 0.6)            # chi-Werte (Prior: [-0.9, 0.9])
    spin_mode: str = "pairs"                   # "pairs": alle (chi_1, chi_2) aus spins^2
                                               # "equal": nur chi_1 = chi_2 (deckt chi_eff ab, viel billiger)
    theta_jn: tuple = (0.6, 1.5708)            # Nuisance-Achse Inklination

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
        for q, (c1, c2), th in product(cfg.mass_ratios, spin_pairs, cfg.theta_jn):
            m1, m2 = component_masses(mc, q)
            if not (cfg.m1_range[0] <= m1 <= cfg.m1_range[1]
                    and cfg.m2_range[0] <= m2 <= cfg.m2_range[1]):
                continue
            rows.append((i, mc, q, c1, c2, th))
    arr = np.array(rows, dtype=float)
    return dict(mc_index=arr[:, 0].astype(np.int64), chirp_mass=arr[:, 1],
                mass_ratio=arr[:, 2], chi_1=arr[:, 3], chi_2=arr[:, 4],
                theta_jn=arr[:, 5], mc_axis=mc_axis)


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
            theta_jn=float(p["theta_jn"]), phase=0.0,
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
    names = ("chirp_mass", "mass_ratio", "chi_1", "chi_2", "theta_jn")
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
    @torch.no_grad()
    def rho2(self, data: dict, asds: dict) -> torch.Tensor:
        d, inv_psd, _ = self._prep(data, asds)
        C, F = d.shape
        i0 = self.cfg.i_min
        w = d * inv_psd                                       # [C, F]
        norm = (self.bank.h2 @ inv_psd.T).clamp_min(1e-30)    # [N, C]  (h|h)_c
        N = len(self.bank)
        out = torch.empty(N, device=self.dev)
        for s in range(0, N, self.chunk):
            hc = self.bank.h[s:s + self.chunk]
            n = hc.shape[0]
            X = hc.conj().unsqueeze(1) * w.unsqueeze(0)       # [n, C, F]
            Xp = torch.zeros(n, C, self.M, dtype=torch.complex64, device=self.dev)
            Xp[..., i0:i0 + F] = X
            z = self.M * torch.fft.ifft(Xp, dim=-1)           # z_c(t), komplex
            zw = torch.cat([z[..., :self.K + 1], z[..., -self.K:]], dim=-1)
            p = (zw.real ** 2 + zw.imag ** 2) / norm[s:s + n].unsqueeze(-1)
            out[s:s + n] = p.sum(1).amax(-1)
        return out

    # -- Chirp-Massen-Schaetzer ---------------------------------------------
    @torch.no_grad()
    def estimate(self, data: dict, asds: dict, temperature: float = 1.0) -> dict:
        rho2 = self.rho2(data, asds)
        if not torch.isfinite(rho2).all():
            raise FloatingPointError(
                f"{int((~torch.isfinite(rho2)).sum())} nicht-endliche rho2-Werte")
        n_mc = self.bank.n_mc
        prof = torch.full((n_mc,), -float("inf"), device=self.dev)
        prof = prof.scatter_reduce(0, self.bank.mc_index, rho2, reduce="amax")
        prof = prof.cpu().numpy()
        axis = self.bank.mc_axis
        i = int(np.argmax(prof))

        # Parabel in log(Mc) um das Maximum
        delta = 0.0
        if 0 < i < n_mc - 1:
            y0, y1, y2 = prof[i - 1], prof[i], prof[i + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                delta = float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))
        mc_ref = float(axis[i] * np.exp(delta * self.cfg.dlog_mc))

        # Likelihood-artig gewichteter Mittelwert in log(Mc)
        wgt = np.exp(-(prof.max() - prof) / (2.0 * temperature))
        mc_w = float(np.exp(np.sum(wgt * np.log(axis)) / np.sum(wgt)))

        j = int(torch.argmax(rho2))
        g = self.bank.grid
        return dict(mc_argmax=float(axis[i]), mc_refined=mc_ref, mc_weighted=mc_w,
                    snr_best=float(np.sqrt(prof[i])),
                    q_best=float(g["mass_ratio"][j]), chi_1_best=float(g["chi_1"][j]),
                    chi_2_best=float(g["chi_2"][j]), theta_jn_best=float(g["theta_jn"][j]),
                    profile=prof)


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
                        kernel_halfwidth: float = 45.0, temperature: float = 1.0):
    import pandas as pd
    files = sorted(Path(injection_dir).glob("inj_*.pkl"))
    if max_injections:
        files = files[:max_injections]
    rows = []
    for k, fn in enumerate(files):
        theta, data, asds = load_injection(fn)
        est = matcher.estimate(data, asds, temperature=temperature)
        est.pop("profile")
        mc_true = float(theta["chirp_mass"])
        rows.append(dict(file=fn.name, chirp_mass_true=mc_true,
                         luminosity_distance=theta.get("luminosity_distance", np.nan), **est))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(files)}", end="\r")
    df = pd.DataFrame(rows)
    for col in ("mc_argmax", "mc_refined", "mc_weighted"):
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
