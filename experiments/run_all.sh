#!/usr/bin/env bash
# Reproduces every result file in ../results. Expects the input files in ../data (see data/README.md).
# Roughly two hours on two CPU cores.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../results/logs
# Joint validity, Mondrian ablation, subgroup coverage at 30 km (Table 2, Figure 2, Table S1)
python run_trajectory.py --protocol temporal     --seeds 3 2>&1 | tee ../results/logs/trajectory_temporal.log
python run_trajectory.py --protocol exchangeable --seeds 3 2>&1 | tee ../results/logs/trajectory_exchangeable.log
# Absolute-target ablation
python run_trajectory.py --protocol temporal --seeds 3 --target absolute 2>&1 | tee ../results/logs/trajectory_temporal_absolute.log
# Learner comparison with equal training size (Table 4) and the 150,000-row run of the absolute target
python run_baselines.py --seeds 2 --max-train 60000 --max-train-dl 60000 2>&1 | tee ../results/logs/baselines.log
python run_baselines.py --seeds 2 --out ../results/baselines_150k 2>&1 | tee ../results/logs/baselines_150k.log
mv ../results/baselines_150k/baselines_full.csv ../results/baselines_full_unequal150k.csv && rmdir ../results/baselines_150k
# Goal-time alerts, rank-normalized band, example trajectories (Table 3, Figure 1, rows of Table 2)
python run_decision.py --seeds 2 2>&1 | tee ../results/logs/decision.log
# Joint coverage by subgroup (Table S2)
python run_joint_subgroups.py --seeds 2 2>&1 | tee ../results/logs/joint_subgroups.log
# Transfer across events (Table 5, Figure 3) and causal same-day recalibration (Table 6)
python run_transport.py --seeds 2 2>&1 | tee ../results/logs/transport.log
python run_sameday.py   --seeds 2 2>&1 | tee ../results/logs/sameday.log
# Computational cost
python run_timing.py
# Summary tables in results/tables
python make_tables.py
