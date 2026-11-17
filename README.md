# TimeSeries: gas-composition forecasting

RNN, LSTM and GRU forecasting of CO, CO2, CH4 and H2 from gasification measurements. This repository organizes the **NEW strict forecasting protocol** used for the revised experiments. It does not reproduce the historical June 2025 training run.

## Repository layout

```text
.
├── reproduce.py             # Inspect, prepare, checkpoint inference, evaluation and plots
├── src/
│   ├── pipeline.py          # Dataset, fixed model definitions, fitting and cache checks
│   └── audit_raw.py         # Original raw parser, preserved byte for byte
├── scripts/                 # Training launchers and Linux CUDA environment setup
├── configs/                 # Recorded protocol and 120 model configurations
├── metadata/                # Record identities, splits, hashes and provenance
├── requirements/            # Training and optional archived-analysis dependencies
├── requirements.txt         # Data preparation, evaluation and plotting dependencies
├── data/raw/                # Local measurements; excluded from Git
├── results/                 # Local generated outputs; excluded from Git
├── permissioned_artifacts/  # Separately supplied checkpoints/caches; excluded from Git
├── archive/                 # Original paper-analysis source and release documentation
└── docs/                    # Data placement, reproduction and reorganization notes
```

`configs/protocol.json` and `configs/model_configs.csv` describe the recorded experiment. The current implementation uses fixed settings in `src/pipeline.py`; these files are reference metadata, not runtime configuration switches.

## Installation

Use Python 3.11. The recorded training stack was Python 3.11.18, PyTorch 2.5.1+cu124, NumPy 2.2.6 and pandas 2.2.2 on four RTX A5000 GPUs; see `metadata/environment_training.json` for the archived record.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python reproduce.py inspect
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. The Linux GPU setup helper below manages its own `.venvs/` environment instead. `inspect` lists missing records without fitting or requiring CUDA. Measurements, fitted weights, observation caches and manuscript documents are not included.

## Data and preparation

Place the unchanged source files under `data/raw/`, preserving each relative `source_file` path in `metadata/data_manifest.csv`. See [data placement](docs/DATA.md). The parser excludes pyrolysis records and the clockless spreadsheet from fitting according to the fixed protocol.

```bash
python reproduce.py prepare --output results/prepared
```

Preparation verifies the recorded split, target timestamps and scaler digests. Output directories must be empty. Different source bytes, file roles or normalization constitute a different experiment.

## Training on four GPUs

Preview assignments without reading data or starting training:

```bash
python scripts/run_4gpu.py --gpus 0,1,2,3 --plan-only --job reproduced_forecasting
```

Set up the Linux environment, then explicitly start the training job:

```bash
GPU_IDS=0,1,2,3 bash scripts/setup_env.sh
GPU_IDS=0,1,2,3 bash scripts/run_4gpu.sh --resume --job reproduced_forecasting
```

The launcher runs device checks and small smoke fits before full fitting. GPU 0/1/2/3 receives leads 1/3/5/10 seconds. Outputs go to `results/reproduced_forecasting/`; terminal and worker logs show progress. Use the same job and `--resume` only for matching provenance. This directory reorganization did not start a training run.

## Checkpoint inference and evaluation

With a separately supplied completed job (containing `data_contract/` and `full/`):

```bash
python reproduce.py infer --job-dir permissioned_artifacts/training --model gru --lead 10 --seed 20261113 --device cpu --output results/gru10_replay
python reproduce.py evaluate --job-dir permissioned_artifacts/training --output results/evaluation
python reproduce.py plot --evaluation-dir results/evaluation --output results/plots
```

For a newly completed job, use `--job-dir results/reproduced_forecasting`. Inference verifies the saved checkpoint/scaler and compares predictions with the stored arrays (default tolerance `1e-5`). Evaluation checks all 120 completed fits. See [reproduction details](docs/REPRODUCIBILITY.md) and [validation boundaries](docs/VALIDATION.md).

## Experiment scope

- Inputs: 11 process channels and four gas histories; 30 consecutive rows ending at prediction origin.
- Targets: four gases at actual leads of 1, 3, 5 and 10 seconds; identical common target rows across models and leads.
- Fixed file split: 10 train / 4 validation / 2 test. Scaling uses only unique finite, clocked training observations.
- One recurrent layer, hidden size 64, linear four-output head; batch 64; Adam `lr=0.001`, `weight_decay=0.0001`; standardized L1 loss; at most 200 epochs; validation patience 20.
- Three models × four leads × ten optimization seeds (`20261113`–`20261122`) = 120 fits.
- Reported equal-record summaries average two record-specific scores within each seed, then summarize ten seeds with sample SD. Seeds do not represent independent physical experiments.

The two test records were historically inspected and held out from NEW fitting/tuning. Analyzer latency, physical concentration basis and some laboratory metadata still require laboratory confirmation. Negative observations/spikes and unconstrained predictions are retained.

## Development

```bash
gh repo clone NSLab-CUK/TimeSeries
cd TimeSeries
```

This directory is the repository root. `.gitignore` excludes raw data, generated outputs, checkpoints, local environments and credentials. Before committing changes, inspect `git status --short` and `git diff --cached --stat` to review the upload set.

The code is provided without raw measurements or fitted artifacts. See [LICENSE_PENDING.md](LICENSE_PENDING.md) for the recorded license status; an open-source license has not been assigned.
