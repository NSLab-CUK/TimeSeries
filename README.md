# Time-Series Soft Sensing of Syngas Evolution during Fluidized-Bed Gasification of Waste Masks

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&amp;logoColor=white&amp;style=flat-square" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&amp;logoColor=white&amp;style=flat-square" alt="PyTorch 2.5.1" />
  <img src="https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&amp;logoColor=white&amp;style=flat-square" alt="CUDA 12.4" />
</p>

Implementation of RNN, LSTM, and GRU models for short-horizon syngas forecasting, maintained by the [Network Science Lab at The Catholic University of Korea](https://nslab-cuk.github.io/).

## 1. Overview

The study forecasts **CO, CO₂, CH₄, and H₂** during bubbling fluidized-bed gasification of waste-mask pellets. The experiments cover air, CO₂, O₂/steam, and steam gasification.

<p align="center">
  <img src="https://raw.githubusercontent.com/NSLab-CUK/Timeseries-Soft-Sensing-of-Syngas-Evolution/80e69c0ff30ce5a5266282da26084c1f8e20d76d/docs/gasification-apparatus.png" width="820" alt="Gasification setup with pellet feeder, gas and steam supply, preheater, fluidized-bed reactor, cyclone, activated-carbon reactor, condenser and chiller, and gas analyzer." />
</p>
<p align="center"><em>Fig. 1. Experimental apparatus for waste-mask gasification.</em></p>

A screw feeder delivers pellets into a fluidized bed of silica sand. Mass-flow controllers regulate the gas supply, and a peristaltic pump supplies water for steam-containing experiments. Temperature and pressure are monitored at several reactor heights. The cyclone removes entrained solids, and the downstream gas analyzer measures product-gas concentrations after gas conditioning.

The forecast uses 30 consecutive one-second observations ending at time *t*, with 11 selected reactor signals and four gas histories per observation. A recurrent layer with 64 hidden units maps the input window to a final state. A linear head forecasts the four concentrations at *t + h* for a lead time *h* of 1, 3, 5, or 10 seconds. Real-time use depends on when the analyzer readings become available.

<p align="center">
  <img src="https://raw.githubusercontent.com/NSLab-CUK/Timeseries-Soft-Sensing-of-Syngas-Evolution/80e69c0ff30ce5a5266282da26084c1f8e20d76d/docs/architecture.svg" width="100%" alt="Thirty observations with 15 features feed a 64-unit RNN, LSTM, or GRU, followed by a linear head with four outputs at the selected lead time." />
</p>

The experiment fits 3 architectures × 4 leads × 10 seeds on a fixed 10/4/2 train/validation/test split. The two test records represent air and CO₂ gasification. Persistence carries the last gas values at *t* forward and uses the same targets as the neural models. Among the neural models, RNN had the lowest mean H₂ RMSE at 1 and 3 seconds and GRU at 5 and 10 seconds. Persistence gave lower mean H₂ RMSE at 10 seconds and lower mean CO₂ RMSE at all four leads.

## 2. Reproduce the experiment

### Environment and input files

```bash
gh repo clone NSLab-CUK/Timeseries-Soft-Sensing-of-Syngas-Evolution
cd Timeseries-Soft-Sensing-of-Syngas-Evolution
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

The CPU environment supports preparation, checkpoint replay, evaluation, and plotting. The GPU training launcher sets up its own Linux environment with PyTorch 2.5.1 and CUDA 12.4.

Training from raw data requires the 17 authorized gasification candidates. Put each file at `data/raw/<source_file>` using the exact relative path in the [input manifest](metadata/data_manifest.csv). One clockless spreadsheet is excluded during preparation. The [locked split](metadata/split_manifest.csv) identifies the 16 included records and their source hashes. The parser splits each record at gaps and invalid observations, and fits the scaler on training observations only.

### Check inputs and train

```bash
python reproduce.py inspect
python scripts/run_4gpu.py --gpus 0,1,2,3 --plan-only --job syngas_forecasting
python reproduce.py prepare --output results/prepared
```

`inspect` and `--plan-only` work in a fresh clone. `prepare` requires the source records and verifies the split, target timestamps, and scaler. Use a new or empty output directory. The commands below independently prepare the training job and assign one lead to each GPU.

```bash
GPU_IDS=0,1,2,3 bash scripts/setup_env.sh
GPU_IDS=0,1,2,3 bash scripts/run_4gpu.sh --job syngas_forecasting
```

The job and worker logs go under `results/syngas_forecasting/`. Resume an interrupted job with the same command and `--resume` from the same source revision. Use a new job name after changing the source, inputs, or training settings.

### Replay, score, and plot

A completed job needs `data_contract/` and `full/`. The commands below use a newly trained job. For an authorized archived job, set `JOB=permissioned_artifacts/training` after placing the completed job there.

```bash
JOB=results/syngas_forecasting
python reproduce.py infer --job-dir "$JOB" --model gru --lead 10 --seed 20261113 \
  --device cpu --output results/gru10_replay
python reproduce.py evaluate --job-dir "$JOB" --output results/evaluation
python reproduce.py plot --evaluation-dir results/evaluation --output results/plots
```

`infer` compares checkpoint output with the saved predictions. `evaluate` recalculates MAE, MSE, RMSE, and R² for the 120 fits and the persistence baseline. Each seed gives equal weight to the two test records. The summary reports the mean and sample standard deviation over ten seeds. `plot` generates the four-gas RMSE comparison from the evaluated tables.

The repository implements the revised forecasting experiment. Archived raw measurements, prepared caches, and trained weights require authorized access. New outputs are written locally under `results/`. The plotting command generates the evaluation comparison independently of the manuscript layout.

## 3. Reference

Chae Eun Hyun, Ji Young Nam, Seo Yeon Ahn, Junhee Cho, Myung Won Seo, Dong Hyun Lee, O-Joun Lee, and Jun Young Kim. *Time series Soft Sensing of Syngas Evolution during Fluidized bed Gasification of Waste Masks Using Recurrent Neural Networks*. Manuscript.

When citing the implementation, include the repository URL and commit used. The code license is pending author approval.
