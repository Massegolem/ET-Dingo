import os
import numpy as np
import pandas as pd
import lal
import GWFish.modules as gw
from joblib import Parallel, delayed

inj_df = pd.read_hdf('injections/BPOP_imbh_injections_20years.hdf5')

waveform_model = 'IMRPhenomXPHM'
f_ref = 20.

detectors = ['ET_HF']
yaml_file = 'my_detectors'
ConfigDet = os.path.join(
    yaml_file + '.yaml'
)

network = gw.detection.Network(
    detector_ids=detectors,
    detection_SNR=(0., 8.),
    config=ConfigDet
)

def compute_snr(chunk):
    m1 = chunk['mass_1_source'].to_numpy(dtype=float)
    m2 = chunk['mass_2_source'].to_numpy(dtype=float)
    mass_ratio = np.maximum(m1, m2) / np.minimum(m1, m2)
    valid_q = mass_ratio <= 1000
    M_source = m1 + m2
    M_detector = M_source * (
        1.0 + chunk['redshift'].to_numpy(dtype=float)
    )

    chi_eff = (
        m1 * chunk['a_1'].to_numpy()
        * np.cos(chunk['tilt_1'].to_numpy())
        + m2 * chunk['a_2'].to_numpy()
        * np.cos(chunk['tilt_2'].to_numpy())
    ) / M_source
    # IMRPhenomX max frequency (M*f); 0.33 for near-extremal spins, 0.30 otherwise
    # See https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomX_internals.c#L503
    Mf_cut = np.where(chi_eff > 0.99, 0.33, 0.30)
    f_cut = Mf_cut / (lal.MTSUN_SI * M_detector)
    f_min = float(np.min(network.detectors[0].frequencyvector))
    in_band = f_cut > f_min
    valid = in_band & valid_q

    # Missing numeric SNRs are represented by NaN, rather than zero.
    snr_values = np.full(len(chunk), np.nan, dtype=float)
    if valid.any():
        snr = gw.utilities.get_snr(
            chunk.loc[valid], network, waveform_model, f_ref
        )
        snr_column = network.detectors[0].name
        snr_values[valid] = snr[snr_column].to_numpy()

    return snr_values, in_band

# Split into chunks for parallel processing
n_jobs = os.cpu_count() or 1
chunks = [
    inj_df.iloc[row_indices].copy()
    for row_indices in np.array_split(np.arange(len(inj_df)), n_jobs * 4)
    if row_indices.size > 0
]

results = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(compute_snr)(chunk) for chunk in chunks
)

if results:
    snr_all = np.concatenate([snr for snr, _ in results])
    in_band_all = np.concatenate([in_band for _, in_band in results])
else:
    snr_all = np.empty(0, dtype=float)
    in_band_all = np.empty(0, dtype=bool)

df_binaries = inj_df.copy()
df_binaries['ET_in_band'] = in_band_all
df_binaries['SNR'] = snr_all
# None is stored as NaN in a numeric pandas column and in HDF5.
df_binaries.loc[~df_binaries['ET_in_band'], 'SNR'] = None

df_binaries.to_hdf('BPOP_imbh_injections_20years_SNR_ET_Delta_HF.hdf5', key='df', mode='w')

