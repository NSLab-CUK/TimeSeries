# Time-Series Soft Sensing of Syngas Evolution during Fluidized-Bed Gasification of Waste Masks

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&amp;logoColor=white&amp;style=flat-square" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&amp;logoColor=white&amp;style=flat-square" alt="PyTorch 2.5.1" />
  <img src="https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&amp;logoColor=white&amp;style=flat-square" alt="CUDA 12.4" />
</p>

**TimeSeries** is a PyTorch implementation of RNN, LSTM, and GRU models for forecasting syngas composition from measured gasification time series. The implementation is maintained by the [Network Science Lab at The Catholic University of Korea](https://nslab-cuk.github.io/).

## 1. Overview

Syngas composition changes as fuel feed, reactor conditions, and gas-phase reactions evolve during fluidized-bed gasification. This study uses reactor measurements from waste-mask gasification under air, CO₂, O₂/steam, and steam conditions to evaluate short-horizon predictions of **CO, CO₂, CH₄, and H₂**.

The forecast starts at time *t* using 30 consecutive one-second observations through *t*. Each observation contains 11 logged reactor-sensor channels and four gas concentration measurements. A single recurrent layer encodes the sequence, and a linear head maps its final hidden state to the four gas concentrations at *t* + *h*. A separate model is fitted for each lead time *h* ∈ {1, 3, 5, 10} s.

<p align="center">
  <img src="docs/architecture.svg" width="100%" alt="Thirty one-second observations with 15 features feed a 64-unit RNN, LSTM, or GRU. A linear head predicts CO, CO2, CH4, and H2 at the selected lead time." />
</p>

The paper's revised evaluation holds out two entire experimental records and compares the models with persistence, which carries the last available gas measurement forward. The relative performance varies by gas and lead time. Real-time use depends on whether the recorded gas measurements are available at the forecast origin.

## 2. Reproducibility

### Datasets

The protocol uses a fixed **10 training / 4 validation / 2 test** split by experimental record. Timestamp gaps or nonfinite selected measurements break the input sequences. Standardization uses training observations only. Every model and lead is evaluated at the same eligible test timestamps.

Preparing or replaying the experiment requires authorized access to the raw measurements or a completed job, respectively. For preparation, place all 17 gasification candidates under `data/raw/` using the relative paths from the [data manifest](metadata/data_manifest.csv). One candidate lacks an absolute clock and is excluded during preparation. See the [data placement guide](docs/DATA.md) for the directory structure and file checks.

### Requirements and environment setup

The recorded training environment used Python 3.11.18, PyTorch 2.5.1 with CUDA 12.4, and four NVIDIA RTX A5000 GPUs. The [environment record](metadata/environment_training.json) gives the complete per-worker details.

```bash
gh repo clone NSLab-CUK/TimeSeries
cd TimeSeries

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For preprocessing and checkpoint inference on CPU, install PyTorch with the CPU wheel index. The Linux GPU launcher below has its own CUDA environment setup.

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

### Quick checks and data preparation

The first two commands work from a fresh clone without experimental records or GPUs. `inspect` lists the expected files. `--plan-only` shows the four-worker assignment without starting training.

```bash
python reproduce.py inspect
python scripts/run_4gpu.py --gpus 0,1,2,3 --plan-only --job syngas_forecasting
```

With authorized records in place, verify the raw file hashes, split, target timestamps, and training-only scaler. Use an empty output directory.

```bash
python reproduce.py prepare --output results/prepared
```

### Train on four GPUs

Run these commands from the repository root on Linux with the raw records in place. GPUs 0–3 handle leads 1, 3, 5, and 10 s, respectively. The launcher checks devices and runs short trial fits before the full experiment. Progress appears in the terminal and worker logs under `results/syngas_forecasting/`.

```bash
GPU_IDS=0,1,2,3 bash scripts/setup_env.sh
GPU_IDS=0,1,2,3 bash scripts/run_4gpu.sh --job syngas_forecasting
```

To continue an interrupted run of this checkout, repeat the launch command with `--resume` and the same job name. Use a new job name when source files or settings change.

### Replay checkpoints, evaluate, and plot

Place the authorized completed job under `permissioned_artifacts/training/`, with its `data_contract/` cache and `full/` training outputs intact. For a job trained using the commands above, use `--job-dir results/syngas_forecasting` instead.

```bash
python reproduce.py infer --job-dir permissioned_artifacts/training \
  --model gru --lead 10 --seed 20261113 --device cpu --output results/gru10_replay
python reproduce.py evaluate --job-dir permissioned_artifacts/training \
  --output results/evaluation
python reproduce.py plot --evaluation-dir results/evaluation --output results/plots
```

`infer` compares the selected checkpoint's predictions with saved predictions. `evaluate` recomputes MAE, MSE, RMSE, and R² for all 120 fits and scores persistence on the same targets. `plot` generates the standard comparison plots from the resulting tables. See [reproduction details](docs/REPRODUCIBILITY.md) and the [recorded checks](docs/VALIDATION.md).

### Training settings and aggregation

| Setting | Value |
| --- | --- |
| Architecture | One recurrent layer, 64 hidden units, four-output linear head |
| Input length | 30 consecutive observations |
| Leads | 1, 3, 5, and 10 seconds |
| Optimization | Adam with learning rate 0.001 and weight decay 0.0001, batch size 64 |
| Loss and stopping | L1 on standardized outputs, at most 200 epochs, validation patience 20 |
| Runs | 3 architectures × 4 leads × 10 seeds = 120 fits |

For each seed, the two test records receive equal weight when averaging the record-level scores. The paper reports the mean and sample standard deviation across the ten seeds. The seed repeats measure training variability within the fixed split.

The implementation for the revised experiment is in [`src/pipeline.py`](src/pipeline.py). [`configs/protocol.json`](configs/protocol.json) and [`configs/model_configs.csv`](configs/model_configs.csv) record the settings and fit assignments. The older analysis scripts are documented in [`archive/`](archive/README.md).

## 3. Reference and citation

Chae Eun Hyun, Ji Young Nam, Seo Yeon Ahn, Junhee Cho, Myung Won Seo, Dong Hyun Lee, O-Joun Lee, and Jun Young Kim. *Time series Soft Sensing of Syngas Evolution during Fluidized bed Gasification of Waste Masks Using Recurrent Neural Networks*. Manuscript.

For work using this implementation, cite the manuscript and include the [TimeSeries repository](https://github.com/NSLab-CUK/TimeSeries) and the commit used.

See the [license status](LICENSE_PENDING.md) for code reuse.
