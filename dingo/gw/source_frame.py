"""
Constraints on source-frame masses.

A source-frame mass is m_source = m_detector / (1 + z), with z the redshift at the
luminosity distance. In Dingo the detector-frame masses live in the waveform dataset
(intrinsic prior) while the luminosity distance is drawn at train time (extrinsic
prior), so a source-frame constraint couples the two. This module implements the
constraint once, so that training and inference use exactly the same prior:

* at train time the waveform dataset is thinned with acceptance probability
  P(d allowed | masses), and the luminosity distance is drawn from the extrinsic
  prior truncated to the allowed range. Together this samples exactly from
  p(intrinsic) p(d) 1[constraint] / Z;
* at inference time the same constraint is added to the prior as a bilby
  Constraint, with a conversion function that generates the source-frame masses.

Train settings (``data`` section)::

    source_frame_mass_constraints:
      mass_1_source:
        maximum: 100.0
      # optional: any key of mass_1_source, mass_2_source, total_mass_source,
      # chirp_mass_source with minimum and/or maximum
      # cosmology: Planck15   # optional, bilby default if omitted
      # z_max: 50.0           # optional, see below

Distance and redshift are related through a lookup table, which must cover the
luminosity_distance prior. By default its upper redshift is chosen automatically:
starting from z = 1 it is doubled until the table reaches twice the prior's maximum
distance (the margin keeps posterior samples that leak slightly beyond the prior
inside the table). Setting z_max overrides this.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from bilby.core.prior import Constraint

SOURCE_FRAME_MASS_KEYS = (
    "mass_1_source",
    "mass_2_source",
    "total_mass_source",
    "chirp_mass_source",
)


def detector_frame_masses(parameters) -> Dict[str, np.ndarray]:
    """Detector-frame mass_1, mass_2, total_mass, chirp_mass (float64 arrays) from
    (chirp_mass, mass_ratio) or (mass_1, mass_2). mass_ratio = m2 / m1 <= 1."""
    if "chirp_mass" in parameters and "mass_ratio" in parameters:
        mc = np.asarray(parameters["chirp_mass"], dtype=np.float64)
        q = np.asarray(parameters["mass_ratio"], dtype=np.float64)
        total = mc * (1.0 + q) ** 1.2 / q**0.6
        m1 = total / (1.0 + q)
        m2 = q * m1
    elif "mass_1" in parameters and "mass_2" in parameters:
        m1 = np.asarray(parameters["mass_1"], dtype=np.float64)
        m2 = np.asarray(parameters["mass_2"], dtype=np.float64)
        total = m1 + m2
        mc = (m1 * m2) ** 0.6 / total**0.2
    else:
        raise KeyError(
            "Source-frame mass constraints need either (chirp_mass, mass_ratio) or "
            f"(mass_1, mass_2) among the parameters; got {list(parameters.keys())}."
        )
    return {"mass_1": m1, "mass_2": m2, "total_mass": m1 + m2, "chirp_mass": mc}


class SourceFrameMassConstraints:
    """Bounds on source-frame masses, and the distance-redshift relation to apply
    them. Redshift and distance are interpolated on a fixed grid, so training and
    inference agree exactly."""

    def __init__(
        self,
        bounds: Dict[str, dict],
        cosmology: Optional[str] = None,
        max_distance: Optional[float] = None,
        z_max: Optional[float] = None,
        num_grid: int = 4000,
    ):
        """
        Parameters
        ----------
        bounds : dict
            {key: {"minimum": ..., "maximum": ...}} for source-frame mass keys.
        cosmology : str, optional
            Passed to bilby.gw.cosmology.get_cosmology; bilby's default if None.
        max_distance : float, optional
            Maximum of the luminosity_distance prior (Mpc). Used to choose the
            redshift range of the lookup table when z_max is not given.
        z_max : float, optional
            Explicit upper redshift of the lookup table; overrides max_distance.
        num_grid : int
            Number of log-spaced redshift grid points.
        """
        if not bounds:
            raise ValueError("No source-frame mass constraints given.")
        self.bounds = {}
        for key, b in bounds.items():
            if key not in SOURCE_FRAME_MASS_KEYS:
                raise ValueError(
                    f"Unknown source-frame constraint {key!r}; allowed keys are "
                    f"{SOURCE_FRAME_MASS_KEYS}."
                )
            if not isinstance(b, dict) or not set(b) <= {"minimum", "maximum"} or not b:
                raise ValueError(
                    f"Constraint {key!r} must be a dict with 'minimum' and/or "
                    f"'maximum', got {b!r}."
                )
            minimum = float(b.get("minimum", 0.0))
            maximum = float(b.get("maximum", np.inf))
            if not 0.0 <= minimum < maximum:
                raise ValueError(f"Invalid bounds for {key!r}: [{minimum}, {maximum}].")
            self.bounds[key] = (minimum, maximum)

        self.cosmology = cosmology
        self.z_max_setting = None if z_max is None else float(z_max)
        self.num_grid = int(num_grid)

        from bilby.gw.cosmology import get_cosmology

        cosmo = get_cosmology(cosmology)
        if z_max is not None:
            if not z_max > 0:
                raise ValueError(f"z_max must be positive, got {z_max}.")
            self.z_max = float(z_max)
        elif max_distance is not None:
            self.z_max = self._automatic_z_max(cosmo, max_distance)
        else:
            raise ValueError(
                "Either z_max or the maximum of the luminosity_distance prior "
                "(max_distance) is needed to build the redshift table."
            )
        self._z = np.concatenate([[0.0], np.geomspace(1e-6, self.z_max, self.num_grid)])
        self._d = cosmo.luminosity_distance(self._z).to("Mpc").value

    #: Distance margin of the automatic table range, and its largest allowed redshift.
    DISTANCE_MARGIN = 2.0
    Z_LIMIT = 1.0e4

    @classmethod
    def _automatic_z_max(cls, cosmo, max_distance) -> float:
        """Smallest z = 2^k (k >= 0) whose luminosity distance is at least
        DISTANCE_MARGIN * max_distance. Deterministic, so training and inference
        build the same table from the same prior."""
        max_distance = float(max_distance)
        if not np.isfinite(max_distance) or max_distance <= 0:
            raise ValueError(
                f"Cannot choose the redshift table range automatically for a "
                f"luminosity_distance prior with maximum {max_distance}; set z_max in "
                f"source_frame_mass_constraints."
            )
        target = cls.DISTANCE_MARGIN * max_distance
        z = 1.0
        while cosmo.luminosity_distance(z).to("Mpc").value < target:
            z *= 2.0
            if z > cls.Z_LIMIT:
                raise ValueError(
                    f"luminosity_distance prior maximum {max_distance} Mpc needs a "
                    f"redshift above {cls.Z_LIMIT}; check the prior or set z_max."
                )
        return z

    @classmethod
    def from_settings(
        cls, settings: Optional[dict], max_distance: Optional[float] = None
    ):
        """Build from the train-settings entry; returns None if it is absent/empty.

        max_distance is the maximum of the luminosity_distance prior, used to size
        the redshift table unless the settings give z_max."""
        if not settings:
            return None
        settings = dict(settings)
        cosmology = settings.pop("cosmology", None)
        z_max = settings.pop("z_max", None)
        return cls(
            settings, cosmology=cosmology, max_distance=max_distance, z_max=z_max
        )

    @property
    def grid_max_distance(self) -> float:
        """Largest luminosity distance (Mpc) covered by the redshift grid."""
        return float(self._d[-1])

    def redshift(self, luminosity_distance):
        """Redshift at the given luminosity distance(s); NaN beyond the table.

        NaN (rather than an error) lets prior evaluation proceed on posterior samples
        that leak far outside the prior: their source-frame masses are NaN, which
        fails every Constraint, so they get zero prior weight. Inside the prior the
        table is always sufficient (checked when sampling)."""
        d = np.asarray(luminosity_distance, dtype=np.float64)
        return np.interp(d, self._d, self._z, right=np.nan)

    def luminosity_distance(self, redshift):
        """Inverse of redshift(); +inf beyond the grid (i.e. not reachable)."""
        z = np.asarray(redshift, dtype=np.float64)
        return np.interp(z, self._z, self._d, right=np.inf)

    def source_frame_masses(self, parameters) -> Dict[str, np.ndarray]:
        """Redshift and all source-frame masses for a sample containing masses and
        luminosity_distance."""
        z = self.redshift(parameters["luminosity_distance"])
        masses = detector_frame_masses(parameters)
        out = {"redshift": z}
        for k, m in masses.items():
            out[f"{k}_source"] = m / (1.0 + z)
        return out

    def distance_bounds(self, parameters) -> Tuple[np.ndarray, np.ndarray]:
        """Allowed luminosity-distance interval [lo, hi] for each set of detector-frame
        masses. m_det / (1+z) <= max  <=>  z >= m_det / max - 1  (lower distance
        bound); m_det / (1+z) >= min  <=>  z <= m_det / min - 1 (upper bound)."""
        masses = detector_frame_masses(parameters)
        shape = np.shape(masses["mass_1"])
        lo = np.zeros(shape)
        hi = np.full(shape, np.inf)
        for key, (minimum, maximum) in self.bounds.items():
            m_det = masses[key[: -len("_source")]]
            if np.isfinite(maximum):
                z_lo = np.maximum(m_det / maximum - 1.0, 0.0)
                lo = np.maximum(lo, self.luminosity_distance(z_lo))
            if minimum > 0.0:
                z_hi = m_det / minimum - 1.0
                d_hi = np.where(
                    z_hi < 0.0, -np.inf, self.luminosity_distance(np.maximum(z_hi, 0.0))
                )
                hi = np.minimum(hi, d_hi)
        return lo, hi

    def _cdf_interval(self, distance_prior, parameters):
        if np.isfinite(distance_prior.maximum) and (
            distance_prior.maximum > self.grid_max_distance
        ):
            raise ValueError(
                f"luminosity_distance prior extends to {distance_prior.maximum} Mpc, "
                f"beyond the redshift table ({self.grid_max_distance:.0f} Mpc, "
                f"z_max={self.z_max}); increase z_max or remove it to choose the "
                f"range automatically."
            )
        lo, hi = self.distance_bounds(parameters)
        lo = np.clip(lo, distance_prior.minimum, distance_prior.maximum)
        hi = np.clip(hi, distance_prior.minimum, distance_prior.maximum)
        f_lo = np.asarray(distance_prior.cdf(lo), dtype=np.float64)
        f_hi = np.asarray(distance_prior.cdf(hi), dtype=np.float64)
        f_hi = np.where(hi > lo, np.maximum(f_hi, f_lo), f_lo)
        return f_lo, f_hi

    def acceptance_probability(self, distance_prior, parameters) -> np.ndarray:
        """P(constraint satisfied | masses) when d ~ distance_prior."""
        f_lo, f_hi = self._cdf_interval(distance_prior, parameters)
        return np.clip(f_hi - f_lo, 0.0, 1.0)

    def sample_distance(self, distance_prior, parameters) -> np.ndarray:
        """Draw d ~ distance_prior truncated to the allowed range (inverse-CDF)."""
        f_lo, f_hi = self._cdf_interval(distance_prior, parameters)
        if np.any(f_hi <= f_lo):
            raise ValueError(
                "Some waveforms admit no luminosity distance satisfying the "
                "source-frame mass constraints. The waveform dataset should have been "
                "thinned in build_dataset(); check that the constraints in the train "
                "settings match the ones used there."
            )
        u = np.random.uniform(f_lo, f_hi)
        return np.asarray(distance_prior.rescale(u), dtype=np.float64)

    def prior_constraints(self) -> Dict[str, Constraint]:
        """bilby Constraint priors to add to a PriorDict for inference."""
        return {
            key: Constraint(minimum=minimum, maximum=maximum, name=key)
            for key, (minimum, maximum) in self.bounds.items()
        }

    def to_settings(self) -> dict:
        settings = {
            k: {
                **({"minimum": mn} if mn > 0 else {}),
                **({"maximum": mx} if np.isfinite(mx) else {}),
            }
            for k, (mn, mx) in self.bounds.items()
        }
        if self.cosmology is not None:
            settings["cosmology"] = self.cosmology
        if self.z_max_setting is not None:
            settings["z_max"] = self.z_max_setting
        return settings
