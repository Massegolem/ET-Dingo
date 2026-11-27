"""Source-frame mass constraints: exactness of the training-time sampling (dataset
thinning + truncated distance) against brute-force rejection from the joint prior,
and consistency of the inference-time prior with the same constraint."""

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from bilby.gw.prior import BBHPriorDict
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.prior import BBHExtrinsicPriorDict, build_prior_with_defaults
from dingo.gw.source_frame import SourceFrameMassConstraints, detector_frame_masses
from dingo.gw.training.train_builders import apply_source_frame_mass_constraints
from dingo.gw.transforms import SampleExtrinsicParameters

CONSTRAINTS = {"total_mass_source": {"maximum": 100.0}}
DISTANCE_PRIOR = (
    "bilby.core.prior.Uniform(minimum=100.0, maximum=6000.0, name='luminosity_distance')"
)
DATA_SETTINGS = {
    "extrinsic_prior": {"luminosity_distance": DISTANCE_PRIOR},
    "source_frame_mass_constraints": CONSTRAINTS,
}
DOMAIN = {"type": "UniformFrequencyDomain", "f_min": 20.0, "f_max": 32.0, "delta_f": 1.0}


def _intrinsic(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "chirp_mass": rng.uniform(20.0, 80.0, n),
            "mass_ratio": rng.uniform(0.25, 1.0, n),
            "luminosity_distance": 100.0,
            "geocent_time": 0.0,
        }
    )


def _wfd(parameters, tmp_path=None):
    n = len(parameters)
    num_bins = int(DOMAIN["f_max"] / DOMAIN["delta_f"]) + 1
    # Encode the row index in the waveform so that the selection can be checked.
    pol = np.repeat(np.arange(n, dtype=float)[:, None], num_bins, axis=1) + 0j
    polarizations = {"h_plus": pol, "h_cross": pol}
    if tmp_path is None:
        return WaveformDataset(
            dictionary={
                "settings": {"domain": DOMAIN},
                "parameters": parameters,
                "polarizations": polarizations,
            },
            precision="double",
        )
    WaveformDataset(
        dictionary={
            "settings": {"domain": DOMAIN},
            "parameters": parameters,
            "polarizations": polarizations,
        }
    ).to_file(tmp_path / "wfd.hdf5")
    return WaveformDataset(
        file_name=str(tmp_path / "wfd.hdf5"),
        precision="double",
        leave_waveforms_on_disk=True,
    )


def test_distance_bounds_enforce_constraint():
    sf = SourceFrameMassConstraints(
        {"mass_1_source": {"maximum": 60.0}, "chirp_mass_source": {"minimum": 10.0}},
        z_max=20.0,
    )
    params = _intrinsic(2000, 1)
    lo, hi = sf.distance_bounds(params)
    for frac in (0.0, 0.5, 1.0):
        ok = np.isfinite(hi) & (hi > lo)
        d = lo[ok] + frac * (np.minimum(hi[ok], 1e5) - lo[ok])
        sub = {k: params[k].to_numpy()[ok] for k in params}
        sub["luminosity_distance"] = d
        m = sf.source_frame_masses(sub)
        assert np.all(m["mass_1_source"] <= 60.0 * (1 + 1e-6))
        assert np.all(m["chirp_mass_source"] >= 10.0 * (1 - 1e-6))


def test_training_distribution_matches_joint_rejection():
    """Thinning + truncated distance == rejection sampling of the joint prior."""
    np.random.seed(0)
    n = 40_000
    d_prior = BBHExtrinsicPriorDict({"luminosity_distance": DISTANCE_PRIOR})[
        "luminosity_distance"
    ]
    sf = SourceFrameMassConstraints(CONSTRAINTS, max_distance=d_prior.maximum)

    # Dingo's construction.
    wfd = _wfd(_intrinsic(n, 2))
    apply_source_frame_mass_constraints(wfd, DATA_SETTINGS)
    params = {k: wfd.parameters[k].to_numpy() for k in wfd.parameters}
    d_dingo = sf.sample_distance(d_prior, params)
    mt_dingo = detector_frame_masses(params)["total_mass"]

    # Brute force: independent draws, reject the joint sample.
    ref = _intrinsic(n, 3)
    d_ref = d_prior.sample(n)
    m_src = detector_frame_masses(ref)["total_mass"] / (1 + sf.redshift(d_ref))
    keep = m_src <= 100.0
    mt_ref = detector_frame_masses(ref)["total_mass"][keep]

    assert ks_2samp(d_dingo, d_ref[keep]).pvalue > 1e-3
    assert ks_2samp(mt_dingo, mt_ref).pvalue > 1e-3
    m_dingo = mt_dingo / (1 + sf.redshift(d_dingo))
    assert np.all(m_dingo <= 100.0 * (1 + 1e-6))


@pytest.mark.parametrize("on_disk", [False, True])
def test_select_samples(tmp_path, on_disk):
    params = _intrinsic(50, 4)
    wfd = _wfd(params, tmp_path if on_disk else None)
    keep = np.array([1, 4, 7, 30, 49])
    wfd.select_samples(keep)
    assert len(wfd) == len(keep)
    for i, k in enumerate(keep):
        sample = wfd[i]
        assert np.isclose(sample["parameters"]["chirp_mass"], params["chirp_mass"][k])
        assert np.allclose(sample["waveform"]["h_plus"][int(DOMAIN["f_min"]) :], k)
    # Nested selection composes with the existing index map.
    wfd.select_samples([0, 3])
    assert np.isclose(wfd[1]["parameters"]["chirp_mass"], params["chirp_mass"][30])


def test_sample_extrinsic_parameters_batched_and_unbatched():
    np.random.seed(1)
    transform = SampleExtrinsicParameters(
        {"luminosity_distance": DISTANCE_PRIOR, "ra": 0.0, "dec": 0.0, "psi": 0.0,
         "geocent_time": 0.0},
        CONSTRAINTS,
    )
    sf = transform.source_frame
    params = _intrinsic(500, 5)
    params = params[sf.acceptance_probability(transform.prior["luminosity_distance"], params) > 0]
    batch = transform({"parameters": {k: params[k].to_numpy() for k in params}})
    d = batch["extrinsic_parameters"]["luminosity_distance"]
    assert d.dtype == np.float32 and d.shape == (len(params),)
    total = detector_frame_masses(params)["total_mass"]
    assert np.all(total / (1 + sf.redshift(d)) <= 100.0 * (1 + 1e-4))

    single = transform({"parameters": params.iloc[0].to_dict()})
    assert isinstance(single["extrinsic_parameters"]["luminosity_distance"], float)
    assert transform.reproduction_dict["source_frame_mass_constraints"] == CONSTRAINTS


def test_inference_prior_evaluates_constraint():
    settings = {
        "chirp_mass": "bilby.core.prior.Uniform(minimum=20.0, maximum=80.0)",
        "mass_ratio": "bilby.core.prior.Uniform(minimum=0.25, maximum=1.0)",
        "luminosity_distance": DISTANCE_PRIOR,
    }
    sample = {
        "chirp_mass": np.array([70.0, 70.0]),
        "mass_ratio": np.array([1.0, 1.0]),
        # total ~161 Msun detector frame: z~0.02 -> violates; z~1 -> fine.
        "luminosity_distance": np.array([100.0, 6000.0]),
    }
    constrained = build_prior_with_defaults(settings, CONSTRAINTS)
    assert "total_mass_source" in constrained
    np.testing.assert_array_equal(constrained.evaluate_constraints(sample), [0, 1])
    # Survives copy() and the plain prior is unchanged.
    np.testing.assert_array_equal(constrained.copy().evaluate_constraints(sample), [0, 1])
    plain = build_prior_with_defaults(settings)
    assert type(plain) is BBHPriorDict
    np.testing.assert_array_equal(plain.evaluate_constraints(sample), [1, 1])
    # Sampling (e.g. injections) respects the constraint.
    draws = constrained.sample(2000)
    m = constrained.source_frame.source_frame_masses(draws)
    assert np.all(m["total_mass_source"] <= 100.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"mass_source": {"maximum": 10.0}},
        {"mass_1_source": 80.0},
        {"mass_1_source": {"maximum": 10.0, "minimum": 20.0}},
        {"mass_1_source": {"max": 10.0}},
        {"mass_1_source": {"maximum": 10.0}, "z_max": -1.0},
    ],
)
def test_invalid_settings(bad):
    with pytest.raises(ValueError):
        SourceFrameMassConstraints.from_settings(bad, max_distance=6000.0)


@pytest.mark.parametrize("max_distance", [1000.0, 6000.0, 5.0e5, 3.0e6])
def test_automatic_table_covers_prior(max_distance):
    """The table reaches twice the prior maximum, including ET-scale priors far
    beyond the old fixed z_max = 20 (~2.3e5 Mpc)."""
    sf = SourceFrameMassConstraints.from_settings(CONSTRAINTS, max_distance=max_distance)
    assert sf.grid_max_distance >= 2.0 * max_distance
    # Smallest power of two: half the range would not suffice.
    assert sf.z_max == 1.0 or sf.luminosity_distance(sf.z_max / 2) < 2.0 * max_distance
    # Table stays accurate against astropy across the range.
    from bilby.gw.cosmology import get_cosmology

    z = np.array([0.001, 0.05, 0.4, 0.9]) * sf.z_max
    exact = get_cosmology(None).luminosity_distance(z).to("Mpc").value
    np.testing.assert_allclose(sf.luminosity_distance(z), exact, rtol=1e-4)
    np.testing.assert_allclose(sf.redshift(exact), z, rtol=1e-4)


def test_et_scale_prior_trains_and_evaluates():
    """End to end with a distance prior beyond z = 20: thinning, truncated sampling
    and the inference prior all work and agree on the redshift table."""
    np.random.seed(2)
    prior_str = (
        "bilby.core.prior.Uniform(minimum=1.0e3, maximum=5.0e5, "
        "name='luminosity_distance')"
    )
    settings = {
        "extrinsic_prior": {"luminosity_distance": prior_str},
        "source_frame_mass_constraints": {"total_mass_source": {"maximum": 20.0}},
    }
    wfd = _wfd(_intrinsic(2000, 7))
    apply_source_frame_mass_constraints(wfd, settings)
    transform = SampleExtrinsicParameters(
        {"luminosity_distance": prior_str}, settings["source_frame_mass_constraints"]
    )
    params = {k: wfd.parameters[k].to_numpy() for k in wfd.parameters}
    d = transform({"parameters": params})["extrinsic_parameters"]["luminosity_distance"]
    assert d.max() > 2.3e5  # beyond the old table
    prior = build_prior_with_defaults(
        {
            "chirp_mass": "bilby.core.prior.Uniform(minimum=20.0, maximum=80.0)",
            "mass_ratio": "bilby.core.prior.Uniform(minimum=0.25, maximum=1.0)",
            "luminosity_distance": prior_str,
        },
        settings["source_frame_mass_constraints"],
    )
    # Identical tables at train and inference time.
    np.testing.assert_array_equal(prior.source_frame._z, transform.source_frame._z)
    sample = {**params, "luminosity_distance": d.astype(np.float64)}
    sample = {k: sample[k] for k in ("chirp_mass", "mass_ratio", "luminosity_distance")}
    assert np.all(prior.evaluate_constraints(sample) == 1)


def test_explicit_z_max_overrides_and_is_checked():
    d_prior = BBHExtrinsicPriorDict({"luminosity_distance": DISTANCE_PRIOR})[
        "luminosity_distance"
    ]
    sf = SourceFrameMassConstraints.from_settings(
        {**CONSTRAINTS, "z_max": 50.0}, max_distance=d_prior.maximum
    )
    assert sf.z_max == 50.0 and sf.to_settings()["z_max"] == 50.0
    too_small = SourceFrameMassConstraints.from_settings(
        {**CONSTRAINTS, "z_max": 0.5}, max_distance=d_prior.maximum
    )
    with pytest.raises(ValueError, match="increase z_max"):
        too_small.acceptance_probability(d_prior, _intrinsic(10, 0))


def test_infinite_prior_needs_explicit_z_max():
    with pytest.raises(ValueError, match="set z_max"):
        SourceFrameMassConstraints.from_settings(CONSTRAINTS, max_distance=np.inf)
    sf = SourceFrameMassConstraints.from_settings(
        {**CONSTRAINTS, "z_max": 10.0}, max_distance=np.inf
    )
    assert sf.z_max == 10.0


def test_samples_beyond_table_get_zero_prior_weight():
    """Posterior samples leaking far outside the prior must not crash importance
    sampling; they fail the constraint instead."""
    prior = build_prior_with_defaults(
        {
            "chirp_mass": "bilby.core.prior.Uniform(minimum=20.0, maximum=80.0)",
            "mass_ratio": "bilby.core.prior.Uniform(minimum=0.25, maximum=1.0)",
            "luminosity_distance": DISTANCE_PRIOR,
        },
        CONSTRAINTS,
    )
    far = 10 * prior.source_frame.grid_max_distance
    sample = {
        "chirp_mass": np.array([30.0, 30.0]),
        "mass_ratio": np.array([1.0, 1.0]),
        "luminosity_distance": np.array([3000.0, far]),
    }
    np.testing.assert_array_equal(prior.evaluate_constraints(sample), [1, 0])
