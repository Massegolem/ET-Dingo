from copy import deepcopy

from bilby.gw.prior import BBHPriorDict
from bilby.gw.conversion import (
    fill_from_fixed_priors,
    convert_to_lal_binary_black_hole_parameters,
)
from bilby.core.prior import Uniform, Sine, Cosine

from dingo.gw.source_frame import SourceFrameMassConstraints

import numpy as np
from typing import Dict, Set, Any, Optional
import warnings

# Silence INFO and WARNING messages from bilby
import logging

logging.getLogger("bilby").setLevel("ERROR")


class BBHExtrinsicPriorDict(BBHPriorDict):
    """
    This class is the same as BBHPriorDict except that it does not require mass parameters.

    It also contains a method for estimating the standardization parameters.

    TODO:
        * Add support for zenith/azimuth
        * Defaults?
    """

    def default_conversion_function(self, sample):
        out_sample = fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lal_binary_black_hole_parameters(out_sample)

        # The previous call sometimes adds phi_jl, phi_12 parameters. These are
        # not needed so they can be deleted.
        if "phi_jl" in out_sample.keys():
            del out_sample["phi_jl"]
        if "phi_12" in out_sample.keys():
            del out_sample["phi_12"]

        return out_sample

    def mean_std(self, keys=([]), sample_size=50000, force_numerical=False):
        """
        Calculate the mean and standard deviation over the prior.

        Parameters
        ----------
        keys: list(str)
            A list of desired parameter names
        sample_size: int
            For nonanalytic priors, number of samples to use to estimate the
            result.
        force_numerical: bool (False)
            Whether to force a numerical estimation of result, even when
            analytic results are available (useful for testing)

        Returns dictionaries for the means and standard deviations.

        TODO: Fix for constrained priors. Shouldn't be an issue for extrinsic parameters.
        """
        mean = {}
        std = {}

        if not force_numerical:
            # First try to calculate analytically (works for standard priors)
            estimation_keys = []
            for key in keys:
                p = self[key]
                # A few analytic cases
                if isinstance(p, Uniform):
                    mean[key] = (p.maximum + p.minimum) / 2.0
                    std[key] = np.sqrt((p.maximum - p.minimum) ** 2.0 / 12.0).item()
                elif isinstance(p, Sine) and p.minimum == 0.0 and p.maximum == np.pi:
                    mean[key] = np.pi / 2.0
                    std[key] = np.sqrt(0.25 * (np.pi**2) - 2).item()
                elif (
                    isinstance(p, Cosine)
                    and p.minimum == -np.pi / 2
                    and p.maximum == np.pi / 2
                ):
                    mean[key] = 0.0
                    std[key] = np.sqrt(0.25 * (np.pi**2) - 2).item()
                else:
                    estimation_keys.append(key)
        else:
            estimation_keys = keys

        # For remaining parameters, estimate numerically
        if len(estimation_keys) > 0:
            samples = self.sample_subset(keys, size=sample_size)
            samples = self.default_conversion_function(samples)
            for key in estimation_keys:
                if key in samples.keys():
                    mean[key] = np.mean(samples[key]).item()
                    std[key] = np.std(samples[key]).item()

        return mean, std


default_extrinsic_dict = {
    "dec": "bilby.core.prior.Cosine(minimum=-np.pi/2, maximum=np.pi/2, name='dec')",
    "ra": 'bilby.core.prior.Uniform(minimum=0., maximum=2*np.pi, boundary="periodic", name="ra")',
    "geocent_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1, name='geocent_time')",
    "psi": 'bilby.core.prior.Uniform(minimum=0.0, maximum=np.pi, boundary="periodic", name="psi")',
    "luminosity_distance": "bilby.core.prior.Uniform(minimum=100.0, maximum=6000.0, name='luminosity_distance')",
}

default_intrinsic_dict = {
    "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0, name='mass_1')",
    "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0, name='mass_2')",
    "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0, name='mass_ratio')",
    "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0, name='chirp_mass')",
    "luminosity_distance": 1000.0,
    "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi, name='theta_jn')",
    "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic", name="phase")',
    "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99, name='a_1')",
    "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99, name='a_2')",
    "tilt_1": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi, name='tilt_1')",
    "tilt_2": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi, name='tilt_2')",
    "phi_12": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic", name="phi_12")',
    "phi_jl": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic", name="phi_jl")',
    "lambda_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=5000.0, name='lambda_1')",
    "lambda_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=5000.0, name='lambda_2')",
    "geocent_time": 0.0,
}

default_inference_parameters = [
    "chirp_mass",
    "mass_ratio",
    "phase",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "luminosity_distance",
    "geocent_time",
    "ra",
    "dec",
    "psi",
]


class BBHSourceFramePriorDict(BBHPriorDict):
    """
    BBHPriorDict whose conversion function also generates the redshift and the
    source-frame masses, so that Constraint priors on mass_1_source, mass_2_source,
    total_mass_source or chirp_mass_source are actually evaluated. (With the plain
    BBHPriorDict they would be silently ignored: bilby skips constraint keys that the
    conversion function does not produce.)
    """

    def __init__(self, dictionary=None, source_frame=None, **kwargs):
        self.source_frame = source_frame
        super().__init__(dictionary=dictionary, **kwargs)

    def default_conversion_function(self, sample):
        out_sample = super().default_conversion_function(sample)
        if self.source_frame is not None and "luminosity_distance" in out_sample:
            out_sample.update(self.source_frame.source_frame_masses(out_sample))
        return out_sample

    def copy(self):
        return self.__class__(dictionary=dict(self), source_frame=self.source_frame)


def build_prior_with_defaults(
    prior_settings: Dict[str, str],
    source_frame_mass_constraints: Optional[dict] = None,
):
    """
    Generate BBHPriorDict based on dictionary of prior settings,
    allowing for default values.

    Parameters
    ----------
    prior_settings: Dict
        A dictionary containing prior definitions for intrinsic parameters
        Allowed values for each parameter are:
            * 'default' to use a default prior
            * a string for a custom prior, e.g.,
               "Uniform(minimum=10.0, maximum=80.0, name=None, latex_label=None, unit=None, boundary=None)"

    source_frame_mass_constraints: dict, optional
        The ``source_frame_mass_constraints`` entry of the train settings. If given,
        the corresponding Constraint priors are added, and the returned prior
        generates source-frame masses to evaluate them.

    Depending on the particular prior choices the dimensionality of a
    parameter sample obtained from the returned GWPriorDict will vary.
    """

    full_prior_settings = deepcopy(prior_settings)
    for k, v in prior_settings.items():
        if v == "default":
            full_prior_settings[k] = default_intrinsic_dict[k]

    if not source_frame_mass_constraints:
        return BBHPriorDict(full_prior_settings)

    prior = BBHSourceFramePriorDict(full_prior_settings)
    if "luminosity_distance" not in prior:
        raise ValueError(
            "source_frame_mass_constraints require a luminosity_distance prior."
        )
    # Size the redshift table from the same distance prior as at train time, so the
    # table (and hence every redshift) is identical in training and inference.
    prior.source_frame = SourceFrameMassConstraints.from_settings(
        source_frame_mass_constraints,
        max_distance=prior["luminosity_distance"].maximum,
    )
    prior.update(prior.source_frame.prior_constraints())
    return prior


def split_off_extrinsic_parameters(theta):
    """
    Split theta into intrinsic and extrinsic parameters.

    Parameters
    ----------
    theta: dict
        BBH parameters. Includes intrinsic parameters to be passed to waveform
        generator, and extrinsic parameters for detector projection.

    Returns
    -------
    theta_intrinsic: dict
        BBH intrinsic parameters.
    theta_extrinsic: dict
        BBH extrinsic parameters (includes calibration parameters).
    """
    extrinsic_parameters = ["geocent_time", "luminosity_distance", "ra", "dec", "psi"]
    theta_intrinsic = {}
    theta_extrinsic = {}
    for k, v in theta.items():
        if k in extrinsic_parameters or "recalib" in k:
            theta_extrinsic[k] = v
        else:
            theta_intrinsic[k] = v
    # set fiducial values for time and distance
    theta_intrinsic["geocent_time"] = 0
    theta_intrinsic["luminosity_distance"] = 100
    return theta_intrinsic, theta_extrinsic
