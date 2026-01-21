# Replication of Millar et al. (2021) in CompEcon

This repository contains code and artifacts to replicate experiments from
Millar et al. (2021) as implemented for a computational economics course.
It includes two main projects (Section 4 and Section 5), configuration files,
and example outputs/plots.

## Repository structure

- `Lab_Section4_ConsumptionSaving/`: Section 4 (consumption-saving) models,
  training code, and evaluation utilities.
- `Lab_Section5_Krusell_and_Smith_1998/`: Section 5 (Krusell and Smith 1998)
  models, training code, and reporting utilities.
- `configs/`: YAML configuration files for running experiments.
- `outputs/`: Example outputs (metrics, plots, checkpoints, debug logs).
- `section4_math.md`, `section5_math.md`: Mathematical notes for each section.
- `MMW_2021_JME.pdf`: Reference paper PDF.

## Setup

Use Python 3.9+ and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Section 4 project

Run the consumption-saving experiment with the default config:

```bash
python Lab_Section4_ConsumptionSaving/run_section4_experiment.py \
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
python Lab_Section5_Krusell_and_Smith_1998/train_ks_experiment.py \
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

## Configuration

Both projects read their settings from YAML files in `configs/`. You can
duplicate a config file and adjust hyperparameters, paths, and training
settings to run alternative experiments.
