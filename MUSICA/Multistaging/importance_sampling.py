import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import GWFish.modules as gw
import gc
import torch
from dingo.core.posterior_models.build_model import build_model_from_kwargs
import pandas as pd
from dingo.gw.inference.sampler import GWComposedSampler
import dingo.gw.injection as injection
from dingo.gw.noise.asd_dataset import ASDDataset
import shutil



os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use GPU 1
os.environ["taskset -c"] = "0-48"  # use CPU cores 0-48
waveform_model = 'IMRPhenomXPHM'
f_ref = 20.
detectors = ['ET']
network = gw.detection.Network(detector_ids = detectors, detection_SNR = (0., 8.))






def importance_sampling(
    name,
    injection_name,
    stage,
    importance_sampling_processes=12,
):

    pm = build_model_from_kwargs(
        filename=f"/scratch/tpausch/models/{name}/model_latest.pt",
        device="cuda",
        load_training_info=False,
    )

    results_dir = f"/scratch/tpausch/test_data/{stage}/results"
    injections_dir = (
        f"/scratch/tpausch/test_data/{injection_name}/injections"
    )

    os.makedirs(results_dir, exist_ok=True)

    injection_files = sorted(os.listdir(injections_dir))

    ess=[]
    for fname in injection_files:  

        # --------------------------------------------------------
        # Load injection
        # --------------------------------------------------------

        with open(f"{injections_dir}/{fname}", "rb") as f:
            payload = pickle.load(f)

        theta_true = payload["theta_true"]
        strain_data = payload["strain_data"]

        # --------------------------------------------------------
        # Run DINGO sampler
        # --------------------------------------------------------

        sampler = GWComposedSampler.from_model(pm, strain_data)

        sampler.run_sampler(
            num_samples=5_000,
            batch_size=5_000,
        )

        result = sampler.to_result()

        # --------------------------------------------------------
        # Check that the proposal density is available
        # --------------------------------------------------------

        if "log_prob" not in result.samples.columns:
            raise RuntimeError(
                f"Cannot importance-sample {fname}: "
                "the samples do not contain log_prob. "
                "Density recovery is required for density-free samplers."
            )

        # --------------------------------------------------------
        # Recover marginalized phase parameters if necessary
        # --------------------------------------------------------

        # Use this when phase and/or psi were marginalized out
        # during training, rather than fixed to specific values.
        if (
            "phase" not in result.samples.columns
            or "psi" not in result.samples.columns
        ):
            result.sample_proposal_extensions(
                synthetic_parameters_kwargs={
                    "num_processes": importance_sampling_processes,
                }
            )

        # --------------------------------------------------------
        # Importance sampling
        # --------------------------------------------------------

        result.importance_sample(
            num_processes=importance_sampling_processes,
        )

        # --------------------------------------------------------
        # Importance-sampling diagnostics
        # --------------------------------------------------------

        result.print_summary()

        log_evidence = float(result.log_evidence)
        n_eff = float(result.n_eff)
        sample_efficiency = float(result.sample_efficiency)
        ess.append(sample_efficiency)

        # --------------------------------------------------------
        # Calculate network SNR
        # --------------------------------------------------------

        parameters = pd.DataFrame(
            {
                key: np.array([np.asarray(val).item()])
                for key, val in theta_true.items()
            }
        )

        snr_df = gw.utilities.get_snr(
            parameters,
            network,
            waveform_model,
            f_ref,
        )

        network_snr = float(snr_df["network"].iloc[0])

        # --------------------------------------------------------
        # Store result, including importance-sampling diagnostics
        # --------------------------------------------------------

        result_payload = {
            "theta_true": theta_true,
            "result": result,
            "network_snr": network_snr,
            "log_evidence": log_evidence,
            "n_eff": n_eff,
            "sample_efficiency": sample_efficiency,
        }

        outname = fname.replace("inj_", "result_")
        outpath = f"{results_dir}/{outname}"

        with open(outpath, "wb") as f:
            pickle.dump(result_payload, f)

        print(f"Saved {outpath}")
        print(f"  Network SNR:       {network_snr:.2f}")
        print(f"  Log evidence:      {log_evidence:.3f}")
        print(f"  Effective samples: {n_eff:.1f}")
        print(f"  Sample efficiency: {sample_efficiency:.4f}")

    print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
    print(torch.cuda.memory_reserved() / 1e9, "GB reserved")

    del pm
    gc.collect()
    torch.cuda.empty_cache()

    return 0

stage_2_models=[#"50_125",
                    #"75_150",
                    #"100_175",
                    #"125_200",
                    #"125_275",
                    #"175_325",
                    #"225_375",
                    #"275_425",
                    #"325_475",
                    #"375_525",
                    "400_625",
                    "475_700",
                    "525_825",
                    #"600_975",
                    "750_1000"
                    ]


for model_name in stage_2_models:
    importance_sampling(
    name="stage1",
    injection_name=model_name,
    stage="stage1_importance_sampling",
    )

    importance_sampling(
    name=model_name,
    injection_name=model_name,
    stage="stage2_importance_sampling",
    )