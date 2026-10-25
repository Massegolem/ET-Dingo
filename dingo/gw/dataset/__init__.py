from .generate_dataset import generate_dataset, generate_parameters_and_polarizations
from .utils import merge_datasets
from .waveform_dataset import *

import bilby.gw.detector as _det
import bilby.gw.detector.networks as _net

if not getattr(_net.get_empty_interferometer, "_et_patched", False):
    _orig = _net.get_empty_interferometer

    def _get_empty_interferometer(name):
        # "ET-1", "ET-2", "ET-3" -> the matching corner of the ET triangle
        if name.startswith("ET-") and name != "ET-":
            for ifo in _orig("ET-"):      # iterating the triangle gives ET-1..3
                if ifo.name == name:
                    return ifo
        return _orig(name)

    _get_empty_interferometer._et_patched = True
    _net.get_empty_interferometer = _get_empty_interferometer
    _det.get_empty_interferometer = _get_empty_interferometer