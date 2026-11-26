# Conformal prediction intervals for machine learning forecasts of marathon finish times across checkpoints and races

Code and result files for the paper of the same title by Andy Domínguez-Monterroza (manuscript under review).

The study evaluates split-conformal prediction intervals for in-race marathon finish-time forecasts on 531,100
finishers of the Boston, Chicago, New York and Berlin marathons (2021 to 2024). It compares per-checkpoint calibration
with jointly valid bands across the nine timing checkpoints, translates coverage into the reliability of goal-time
alerts, compares learners and target parameterizations, and tests whether calibration transfers across events and
editions.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest -q          # unit tests, including a numerical check of the joint-validity result
```

Python 3.10 or later. The main dependencies are numpy, pandas, scikit-learn and LightGBM.

## Data

The data are per-runner split times from the official results websites of the World Marathon Majors (Boston,
Chicago, New York and Berlin). They are public but are not redistributed here. Place the input files in `data/` as described in
[`data/README.md`](data/README.md).

## Reproducing the results

```bash
bash experiments/run_all.sh   # about two hours on two CPU cores
```

Every result file in `results/` is written by one of the scripts below; `experiments/make_tables.py` summarizes them in
`results/tables/`.

| Paper element | Script | Result files |
|---|---|---|
| Table 2 (per-checkpoint, Bonferroni, constant- and local-scale bands, Mondrian), Figure 2 | `run_trajectory.py --protocol temporal` and `--protocol exchangeable` | `trajectory_summary_*.csv`, `trajectory_by_checkpoint_*.csv` |
| Table 2 (rank-normalized bands), Table 3 (goal-time alerts), Figure 1 | `run_decision.py` | `decision_bands_full.csv`, `decision_alerts_full.csv`, `example_trajectories_full.csv` |
| Table 4 (learners on 60,000 rows) | `run_baselines.py --max-train 60000 --max-train-dl 60000` | `baselines_full.csv` |
| Absolute target with 150,000 rows | `run_baselines.py` (default sizes) | `baselines_full_unequal150k.csv` |
| Absolute-target ablation of the trajectory results | `run_trajectory.py --target absolute` | `trajectory_*_temporal_absolute.csv` |
| Table 5 (transfer across events) | `run_transport.py` | `transport_full.csv` |
| Table S3 (causal same-day recalibration) | `run_sameday.py` | `sameday_full.csv` |
| Table S1 (coverage at 30 km by subgroup) | `run_trajectory.py` | `trajectory_slices_*.csv` |
| Table S2 (joint coverage by subgroup) | `run_joint_subgroups.py` | `joint_by_subgroup.csv` |
| Computational cost | `run_timing.py` | `timing.csv` |
| Table 6 and Table S4 (simulation study) | `run_simulation.py --reps 20` | `simulation/sim_*.csv` |

Every script accepts `--quick` for a one-minute smoke run on a subsample.

## Repository layout

```
src/mtc/        data.py         loading, cleaning of Berlin, checkpoint features, splits
                models.py       LightGBM quantile models and locally weighted learners
                conformal.py    per-checkpoint, Bonferroni, constant-scale, rank-normalized and Mondrian calibration
                online.py       causal same-day recalibration and adaptive conformal inference
                metrics.py      coverage, joint coverage, width, interval score
experiments/    one script per experiment, run_all.sh, make_tables.py
results/        result files used in the paper
tests/          unit tests
data/           input files (not tracked)
```

## Citation

If you use this code, please cite the paper (reference to be completed on publication) and this repository; see
[`CITATION.cff`](CITATION.cff).

## License

MIT, see [`LICENSE`](LICENSE).
