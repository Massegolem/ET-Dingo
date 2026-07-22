#!/usr/bin/env bash

# imbh_T2_injection_test_data0_1126259462-391_generation
# PARENTS 
# CHILDREN imbh_T2_injection_test_data0_1126259462-391_sampling imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part0 imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part1 imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part2 imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part3
if [[ "imbh_T2_injection_test_data0_1126259462-391_generation" == *"$1"* ]]; then
    echo "Running: /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_generation outdir_imbh_T2_test/imbh_T2_injection_test_config_complete.ini --label imbh_T2_injection_test_data0_1126259462-391_generation --idx 0 --trigger-time 1126259462.391 --outdir outdir_imbh_T2_test"
    /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_generation outdir_imbh_T2_test/imbh_T2_injection_test_config_complete.ini --label imbh_T2_injection_test_data0_1126259462-391_generation --idx 0 --trigger-time 1126259462.391 --outdir outdir_imbh_T2_test
fi

# imbh_T2_injection_test_data0_1126259462-391_sampling
# PARENTS imbh_T2_injection_test_data0_1126259462-391_generation
# CHILDREN imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part0 imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part1 imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part2 imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part3
if [[ "imbh_T2_injection_test_data0_1126259462-391_sampling" == *"$1"* ]]; then
    echo "Running: /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_sampling outdir_imbh_T2_test/imbh_T2_injection_test_config_complete.ini --label imbh_T2_injection_test_data0_1126259462-391_sampling --event-data-file outdir_imbh_T2_test/data/imbh_T2_injection_test_data0_1126259462-391_generation_event_data.hdf5 --outdir outdir_imbh_T2_test"
    /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_sampling outdir_imbh_T2_test/imbh_T2_injection_test_config_complete.ini --label imbh_T2_injection_test_data0_1126259462-391_sampling --event-data-file outdir_imbh_T2_test/data/imbh_T2_injection_test_data0_1126259462-391_generation_event_data.hdf5 --outdir outdir_imbh_T2_test
fi

# imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part0
# PARENTS imbh_T2_injection_test_data0_1126259462-391_sampling imbh_T2_injection_test_data0_1126259462-391_generation
# CHILDREN imbh_T2_injection_test_data0_1126259462-391_importance_sampling_merge
if [[ "imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part0" == *"$1"* ]]; then
    echo "Running: /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_importance_sampling outdir_imbh_T2_test/imbh_T2_injection_test_config_complete.ini --outdir outdir_imbh_T2_test --label imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part0 --proposal-samples-file outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_sampling_part0.hdf5 --event-data-file outdir_imbh_T2_test/data/imbh_T2_injection_test_data0_1126259462-391_generation_event_data.hdf5"
    /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_importance_sampling outdir_imbh_T2_test/imbh_T2_injection_test_config_complete.ini --outdir outdir_imbh_T2_test --label imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part0 --proposal-samples-file outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_sampling_part0.hdf5 --event-data-file outdir_imbh_T2_test/data/imbh_T2_injection_test_data0_1126259462-391_generation_event_data.hdf5
fi

# imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part1
# PARENTS imbh_T2_injection_test_data0_1126259462-391_sampling imbh_T2_injection_test_data0_1126259462-391_generation
# CHILDREN imbh_T2_injection_test_data0_1126259462-391_importance_sampling_merge
if [[ "imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part1" == *"$1"* ]]; then
    echo "Running: /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_importance_sampling outdir_imbh_T2_test/imbh_T2_injection_test_config_complete.ini --outdir outdir_imbh_T2_test --label imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part1 --proposal-samples-file outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_sampling_part1.hdf5 --event-data-file outdir_imbh_T2_test/data/imbh_T2_injection_test_data0_1126259462-391_generation_event_data.hdf5"
    /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_importance_sampling outdir_imbh_T2_test/imbh_T2_injection_test_config_complete.ini --outdir outdir_imbh_T2_test --label imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part1 --proposal-samples-file outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_sampling_part1.hdf5 --event-data-file outdir_imbh_T2_test/data/imbh_T2_injection_test_data0_1126259462-391_generation_event_data.hdf5
fi

# imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part2
# PARENTS imbh_T2_injection_test_data0_1126259462-391_sampling imbh_T2_injection_test_data0_1126259462-391_generation
# CHILDREN imbh_T2_injection_test_data0_1126259462-391_importance_sampling_merge
if [[ "imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part2" == *"$1"* ]]; then
    echo "Running: /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_importance_sampling outdir_imbh_T2_test/imbh_T2_injection_test_config_complete.ini --outdir outdir_imbh_T2_test --label imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part2 --proposal-samples-file outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_sampling_part2.hdf5 --event-data-file outdir_imbh_T2_test/data/imbh_T2_injection_test_data0_1126259462-391_generation_event_data.hdf5"
    /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_importance_sampling outdir_imbh_T2_test/imbh_T2_injection_test_config_complete.ini --outdir outdir_imbh_T2_test --label imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part2 --proposal-samples-file outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_sampling_part2.hdf5 --event-data-file outdir_imbh_T2_test/data/imbh_T2_injection_test_data0_1126259462-391_generation_event_data.hdf5
fi

# imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part3
# PARENTS imbh_T2_injection_test_data0_1126259462-391_sampling imbh_T2_injection_test_data0_1126259462-391_generation
# CHILDREN imbh_T2_injection_test_data0_1126259462-391_importance_sampling_merge
if [[ "imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part3" == *"$1"* ]]; then
    echo "Running: /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_importance_sampling outdir_imbh_T2_test/imbh_T2_injection_test_config_complete.ini --outdir outdir_imbh_T2_test --label imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part3 --proposal-samples-file outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_sampling_part3.hdf5 --event-data-file outdir_imbh_T2_test/data/imbh_T2_injection_test_data0_1126259462-391_generation_event_data.hdf5"
    /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_importance_sampling outdir_imbh_T2_test/imbh_T2_injection_test_config_complete.ini --outdir outdir_imbh_T2_test --label imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part3 --proposal-samples-file outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_sampling_part3.hdf5 --event-data-file outdir_imbh_T2_test/data/imbh_T2_injection_test_data0_1126259462-391_generation_event_data.hdf5
fi

# imbh_T2_injection_test_data0_1126259462-391_importance_sampling_merge
# PARENTS imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part0 imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part1 imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part2 imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part3
# CHILDREN imbh_T2_injection_test_data0_1126259462-391_importance_sampling_plot
if [[ "imbh_T2_injection_test_data0_1126259462-391_importance_sampling_merge" == *"$1"* ]]; then
    echo "Running: /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_result --result outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part0.hdf5 outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part1.hdf5 outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part2.hdf5 outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part3.hdf5 --outdir outdir_imbh_T2_test/result --label imbh_T2_injection_test_data0_1126259462-391_importance_sampling_merge --extension hdf5 --merge"
    /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_result --result outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part0.hdf5 outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part1.hdf5 outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part2.hdf5 outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_importance_sampling_part3.hdf5 --outdir outdir_imbh_T2_test/result --label imbh_T2_injection_test_data0_1126259462-391_importance_sampling_merge --extension hdf5 --merge
fi

# imbh_T2_injection_test_data0_1126259462-391_importance_sampling_plot
# PARENTS imbh_T2_injection_test_data0_1126259462-391_importance_sampling_merge
# CHILDREN 
if [[ "imbh_T2_injection_test_data0_1126259462-391_importance_sampling_plot" == *"$1"* ]]; then
    echo "Running: /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_plot --label imbh_T2_injection_test_data0_1126259462-391_importance_sampling_plot --result outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_importance_sampling.hdf5 --outdir outdir_imbh_T2_test/result --corner --weights --log_probs"
    /home/tpausch/.conda/envs/ET-DINGO/bin/dingo_pipe_plot --label imbh_T2_injection_test_data0_1126259462-391_importance_sampling_plot --result outdir_imbh_T2_test/result/imbh_T2_injection_test_data0_1126259462-391_importance_sampling.hdf5 --outdir outdir_imbh_T2_test/result --corner --weights --log_probs
fi

