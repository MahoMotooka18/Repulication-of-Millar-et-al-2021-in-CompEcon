# Replication of Millar et al. (2021) in CompEcon

This repository contains code and artifacts to replicate experiments from
Millar et al. (2021) as implemented for a computational economics course.
It includes two main projects (Section 4 and Section 5), configuration files,
and example outputs/plots.

**Reference**

The Section 5 implementation aligns with the official replication code and
uses the `Main_KS.ipynb` notebook from `https://github.com/marcmaliar/deep-learning-euler-method-krusell-smith`
as an implementation reference.

## Repository structure

- `Lab_Section4_ConsumptionSaving/`: Section 4 (consumption-saving) models,
  training code, and evaluation utilities.
- `Lab_Section5_Krusell_and_Smith_1998/`: Section 5 (Krusell and Smith 1998)
  models, training code, and reporting utilities.
- `configs/`: YAML configuration files for running experiments.
- `outputs/`: Example outputs (metrics, plots, checkpoints, debug logs).
- `note.md`: Technical summary of the paper and computational setup.
- `references/`: Reference materials, including the paper PDF.
  - `references/MMW_2021_JME.pdf`

## Setup

Use Python 3.9+ and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Submission checklist

- Paper PDF: `references/MMW_2021_JME.pdf`
- Summary note: `note.md`
- `README.md` with workflow and mapping to equations/figures
- `requirements.txt`
- All `.py` scripts used in the replication
- Execution files:
  - `Lab_Section4_ConsumptionSaving/main_section4.py`
  - `Lab_Section5_Krusell_and_Smith_1998/main_section5.py`

## Running the Section 4 project

Run the consumption-saving experiment with the default config:

```bash
python Lab_Section4_ConsumptionSaving/main_section4.py \
  --config configs/section4.yaml \
  --device cpu
```

Notes:
- Use `configs/section4_smoke.yaml` for a quick smoke test.
- Outputs are written under `outputs/section4/` with a timestamped folder.
- Set `--device cuda` if you have a CUDA-enabled GPU.

Output files (Section 4):
- `config.yaml`: configuration snapshot used for the run.
- `metrics*.csv`: training and evaluation metrics (overall, per objective).
- `plots/`: training curve figures (PNG).
- `checkpoints/`: saved model weights (`.pt`).
- `debug/`: warnings, validation failures, and diagnostics (`.log`, `.jsonl`).

Output folders (Section 4):
- `outputs/section4/<run_timestamp>/`: top-level run directory.
- `outputs/section4/<run_timestamp>/plots/`: training curves.
- `outputs/section4/<run_timestamp>/checkpoints/`: model checkpoints.
- `outputs/section4/<run_timestamp>/debug/`: debug logs and diagnostics.

## Running the Section 5 project

Run the Krusell and Smith (1998) experiment with the default config:

```bash
python Lab_Section5_Krusell_and_Smith_1998/main_section5.py \
  --config configs/section5.yaml \
  --device cpu
```

Notes:
- Outputs are written under `outputs/section5/` (metrics, plots, tables,
  checkpoints, and debug snapshots).
- Set `--device cuda` if you have a CUDA-enabled GPU.

Output files (Section 5):
- `metrics_*.csv`: objective-specific metrics (Euler, Bellman, lifetime reward).
- `plots/`: result figures (PNG) for each objective.
- `tables/` and `comparison/`: CSV tables and summary comparison figures.
- `checkpoints/`: saved model weights (`.pt`).
- `debug/`: input and policy snapshots plus consistency checks.

Output folders (Section 5):
- `outputs/section5/`: top-level output directory.
- `outputs/section5/plots/`: objective-specific plots (with subfolders).
- `outputs/section5/tables/`: objective-specific property tables.
- `outputs/section5/comparison/`: cross-objective comparison tables/figures.
- `outputs/section5/checkpoints/`: model checkpoints.
- `outputs/section5/debug/`: debug snapshots and checks.

## Mapping to paper (equations, algorithms, figures)

Section 4 (Consumption-Saving):
- Training loop, logging, output layout: `Lab_Section4_ConsumptionSaving/main_section4.py`
- Model equations, feasibility, and shock process: `Lab_Section4_ConsumptionSaving/model_consumption_saving.py`
- Policy parameterization (consumption share, multiplier, value): `Lab_Section4_ConsumptionSaving/nn_policy.py`
- Objectives (lifetime reward, Euler residuals, Bellman residuals): `Lab_Section4_ConsumptionSaving/objectives.py`
- Evaluation metrics: `Lab_Section4_ConsumptionSaving/evaluator.py`
- Plots for training curves: `Lab_Section4_ConsumptionSaving/plot_section4.py`

Section 5 (Krusell-Smith 1998):
- Training loop, outputs, and debug artifacts: `Lab_Section5_Krusell_and_Smith_1998/main_section5.py`
- Model equations (production, prices, transitions): `Lab_Section5_Krusell_and_Smith_1998/model_ks1998.py`
- Policy parameterization and distribution inputs: `Lab_Section5_Krusell_and_Smith_1998/nn_policy_ks.py`
- Objectives (lifetime reward, Euler residuals, Bellman residuals): `Lab_Section5_Krusell_and_Smith_1998/objectives_ks.py`
- Evaluation metrics and simulation statistics: `Lab_Section5_Krusell_and_Smith_1998/evaluator_ks.py`
- Policy scaling utilities: `Lab_Section5_Krusell_and_Smith_1998/policy_utils_ks.py`
- Reporting (tables/summary stats): `Lab_Section5_Krusell_and_Smith_1998/report_ks.py`

## Configuration
Both projects read their settings from YAML files in `configs/`. You can
duplicate a config file and adjust hyperparameters, paths, and training
settings to run alternative experiments.

## Implementation notes

### KS regression diagnostic in Section 5

The **Krusell-Smith (KS) regression** (`ln(K_{t+1}) = ξ_0 + ξ_1·ln(K_t) + ξ_2·ln(Z_t)`)
is computed in `evaluator_ks.py:compute_statistics()` for diagnostic and validation purposes only.

**Status of implementation:**
The original implementation attempted to support all three objectives (lifetime-reward, Euler, Bellman)
as described in Maliar et al. (2021, Sections 5–7). 

However, during development, the lifetime-reward objective exhibited numerical instability during the KS regression calculation (least-squares computation). 

Since this regression is **not part of the core training algorithm** (it is purely diagnostic), the implementation safely **skips the KS regression when objective_name is 'lifetime_reward'**.

**Technical rationale:**
- The KS regression is used only for validation and monitoring of the law of motion approximation.

- The three training objectives (lifetime-reward, Euler, Bellman) do not depend on regression outputs.

- Skipping the regression for lifetime-reward training eliminates numerical instability without affecting algorithm correctness.

- For Euler and Bellman objectives, the regression is computed normally.


**YAML Parameters**

Section 4 (`configs/section4.yaml`)
- `seed`: random seed for reproducibility.
- `model.gamma`: CRRA coefficient.
- `model.beta`: discount factor.
- `model.r`: gross interest rate.
- `model.rho`: AR(1) income persistence.
- `model.sigma`: income shock standard deviation.
- `model.horizon`: finite horizon used for evaluation.
- `training.objective`: objective list (`lifetime_reward`, `euler`, `bellman`).
- `training.network_sizes`: hidden-layer width grid.
- `training.num_epochs`: total training epochs.
- `training.batch_size`: minibatch size.
- `training.learning_rate`: optimizer learning rate.
- `training.wealth_range`: sampling bounds for wealth `[w_min, w_max]`.
- `training.eval_interval`: evaluation frequency (epochs).
- `training.nu`: weight on FB residual.
- `training.nu_h`: weight on multiplier matching.
- `plotting.smoothing_window`: moving-average window for plots.
- `plotting.show_raw`: whether to overlay raw curves.
- `debug.enabled`: enable debug logging.
- `debug.interval`: debug logging interval (epochs).
- `output_dir`: output base directory.
- `device`: `cpu` or `cuda`.

Section 5 (`configs/section5.yaml`)
- `seed`: random seed for reproducibility.
- `model.gamma`: CRRA coefficient (log utility when 1.0).
- `model.beta`: discount factor.
- `model.alpha`: capital share in production.
- `model.delta`: depreciation rate.
- `model.rho_y`: idiosyncratic income persistence.
- `model.sigma_y`: idiosyncratic income volatility.
- `model.rho_z`: aggregate TFP persistence.
- `model.sigma_z`: aggregate TFP volatility.
- `model.enforce_bounds`: enforce policy/wealth bounds from reference code.
- `model.use_log_shock_shift`: apply log-shock shift used in the reference.
- `model.num_agents`: default agent count (overridden by `training.agent_counts`).
- `model.horizon`: finite horizon used for evaluation.
- `training.objectives`: objective list (`lifetime_reward`, `euler`, `bellman`).
- `training.agent_counts`: grid of agent counts for experiments.
- `training.hidden_size`: hidden-layer width.
- `training.distribution_features`: number of distribution inputs.
- `training.num_epochs`: total training epochs.
- `training.simulation_length`: simulation length for training data.
- `training.train_every`: train every N simulated periods.
- `training.train_points`: number of simulated points per update.
- `training.pretrain_value_iters`: Bellman pretraining iterations.
- `training.learning_rate`: optimizer learning rate.
- `training.batch_size`: minibatch size.
- `training.eval_interval`: evaluation frequency (epochs).
- `training.eval_horizon`: evaluation simulation horizon.
- `training.w_training_sampling.enabled`: enable broader wealth sampling.
- `training.nu`: weight on FB residual.
- `training.nu_h`: weight on multiplier matching.
- `plotting.smoothing_window`: moving-average window for plots.
- `plotting.show_raw`: whether to overlay raw curves.
- `plotting.w_plot_max`: x-axis max for wealth plots.
- `policy_output_types.default`: default policy output type.
- `policy_output_types.lifetime_reward`: output type for lifetime-reward objective.
- `policy_output_types.euler`: output type for Euler objective.
- `policy_output_types.bellman`: output type for Bellman objective.
- `normalization.w_normalized`: enable wealth normalization.
- `normalization.w_scale`: wealth scale.
- `normalization.w_shift`: wealth shift.
- `normalization.c_normalized`: enable consumption normalization.
- `normalization.c_scale`: consumption scale.
- `normalization.c_shift`: consumption shift.
- `mismatch_checks.curvature_threshold`: curvature mismatch threshold.
- `mismatch_checks.overlap_threshold`: overlap threshold.
- `mismatch_checks.share_variation_threshold`: consumption-share variation threshold.
- `mismatch_checks.wealth_range_threshold`: wealth range mismatch threshold.
- `mismatch_checks.wealth_quantile_bounds`: acceptable wealth quantile bounds.
- `input_scaling.enabled`: enable steady-state input scaling.
- `input_scaling.w_min`: minimum wealth for scaling.
- `input_scaling.w_max_multiplier`: max wealth multiplier for scaling.
- `debug.enabled`: enable debug logging.
- `debug.interval`: debug logging interval (epochs).
- `output_dir`: output base directory.
- `comparison.agent_count`: agent count used in cross-objective comparison.
- `device`: `cpu` or `cuda`.

Other config files (`configs/section4_smoke.yaml`) follow the same schema and adjust only selected
values for smoke tests or alternative experiments.