# Executed NEW recurrent gas-composition forecasting

Author-review staging; no repository was created and nothing was uploaded. This is the executed NEW replacement protocol, not a reproduction of the historical 250613 results. Code-license and data-access approval remain pending (LICENSE_PENDING.md).

## Included and excluded

Included: exact executed model/sample-builder source and parser; launcher/environment helpers; explicit 10/4/2 file roles and raw-file hashes; 120 configuration rows; artifact-path/hash links; standalone checkpoint inference/evaluation; persistence and figure-generation source. The manifest is metadata, not measurements. Relative source paths and experiment IDs are necessary to preserve the declared split. The scalar-value data are not included.

Not included: raw CSV/XLSX measurements, observation caches, scaler arrays, fitted weights, prediction arrays, Office source documents, SSH keys, machine credentials, or private absolute paths. Metadata stores scaler digests only. This ZIP alone cannot train or infer without the corresponding permissioned data/artifacts. A different dataset or split is a new experiment.

## Exact protocol and ordering

Inputs: FNC-1,FNC-2,FNC-3,TI-2,TI-3,TI-4,TI-5,PI-2,PI-3,PI-4,PI-5,CO,CO2,CH4,H2. Outputs: CO,CO2,CH4,H2. Thirty consecutive rows end at origin t (29-s span). Target is t+h for actual h=1,3,5,10 s. No future row/mask is supplied. Common target offsets begin at segment row 39 for all leads. Invalid selected values/clocks or any non-1-s step break segments. Negative values/spikes and unconstrained predictions remain unchanged.

The raw parser finds the header among the first twelve rows, reads the calendar date from metadata and constructs time from Hour/Min/Sec. A >12-h backward clock change is treated as a midnight rollover; other discontinuities break segments. A clockless spreadsheet is excluded without inventing timestamps. File roles/prefixes are fixed in pipeline.py and metadata/split_manifest.csv. Normalization uses population standard deviations from unique finite clocked training rows only (32,234 rows). The test list has two historically inspected records held out from NEW fitting/tuning; ten seeds are optimization repetitions, not ten partitions or independent physical tests. Analyzer availability/latency and physical concentration basis remain laboratory questions.

RNN/LSTM/GRU: one recurrent layer, hidden64, linear4 head, batch64, Adam lr0.001, weight_decay0.0001, betas(0.9,0.999), eps1e-8, standardized four-output L1, max200 epochs, validation patience20 and minimum-validation checkpoint. Seeds20261113..20261122 give120 fits. No revised Optuna search. Actual lengths26..195 epochs. metadata/model_configs.csv and protocol.json are authoritative.

## Linux environment and training (explicit opt-in)

The actual full execution used Python3.11.18, torch2.5.1+cu124, NumPy2.2.6, pandas2.2.2, openpyxl3.1.5, four RTX A5000 GPUs. The Python3.12 note in the preserved historical requirements comment refers to development/smoke, not the full execution. setup_env.sh installs project-local Python3.11 and the specified CUDA stack; it does not start fitting. NVIDIA driver/device availability is required for full training. Bitwise identity across other hardware/software is not guaranteed.

Unpack into a new directory and run from its JCOU_code_release root. Obtain permissioned source records first, at the project-relative paths in revision/JCOU_D_26_00869_20261113/data_manifest.csv (all seventeen candidates, including the clockless spreadsheet). Keep source filenames/bytes unchanged.

```bash
cd JCOU_code_release
GPU_IDS=0,1,2,3 bash revision/JCOU_D_26_00869_resolution/reexperiment/setup_env.sh
source .venvs/jcou_revision_py311/bin/activate
python reproduce.py inspect
python reproduce.py prepare --output local_checks/prepared
GPU_IDS=0,1,2,3 bash revision/JCOU_D_26_00869_resolution/reexperiment/run_4gpu.sh --plan-only --job reproduced_forecasting
```

The prepare check verifies source hashes, complete split, all common targets and exact scaler digests without fitting. Only after these checks, to intentionally train a NEW120-fit run:

```bash
GPU_IDS=0,1,2,3 bash revision/JCOU_D_26_00869_resolution/reexperiment/run_4gpu.sh --resume --job reproduced_forecasting
```

Outputs: revision/JCOU_D_26_00869_resolution/reexperiment/server_runs/reproduced_forecasting/. GPU0/1/2/3 receives h1/3/5/10. Terminal/console.log and workers/gpu_*/console.log contain live progress. Resume refuses changed provenance. Do not substitute a new run for the archived execution evidence.

## Reuse permissioned completed artifacts; no training

For archived-checkpoint verification, separately obtain the permissioned full training artifact bundle, preserving its training/data_contract and training/full layout. The public code ZIP does not contain it. Set JOB to that training directory or the completed server job. Cache/source/scaler hashes and COMPLETED.json files connect each fit to its source and targets. metadata/expected_fit_artifacts.csv records the1920 previously audited file hashes.

```bash
JOB="$(pwd)/permissioned_artifacts/training"
python reproduce.py infer --job-dir "$JOB" --model gru --lead 10 --seed 20261113 --device cuda:3 --output local_checks/gru10_replay
python reproduce.py evaluate --job-dir "$JOB" --output local_checks/evaluation
python -m pip install -r requirements_reproduce.txt
python reproduce.py plot --evaluation-dir local_checks/evaluation --output local_checks/plots
```

Inference uses the saved best checkpoint/scaler, raw-scale truths and identical target rows; default comparison tolerance1e-5 is declared. Same-device archived replay had difference0; CPU/other-device rounding may differ. Evaluation verifies all120 completed fit links and produces960 record/model/lead/seed/gas scores. Each fitted seed first averages its two record-specific metrics; mean and sample SD(ddof=1) are over the ten fitted seeds. Persistence is the last input-row gas value at t, unchanged at t+h, with the same targets, units and averaging. No fitting, clipping, future/target value or best-case filtering is used. Record-specific outcomes can differ from equal-record means; no significance inference follows from these SD bars.

## Exact paper plots and limitations

The original executed paper plotting source is retained at revision/JCOU_D_26_00869_posttraining_20261113/scripts/analyze_and_plot.py. To regenerate the exact paper panels, also obtain the internal permissioned posttraining analysis subset at that same relative path: metrics_per_record_seed_gas.csv, metrics_pooled_targets_per_seed.csv, training_run_audit.csv, persistence_baseline.csv; training/data_contract/{common_target_index.csv,observations_cache.npz,observations_cache_records.json,split_manifest.csv}; first-seed pred.npy for all12 model/lead combinations and its true.npy; all120 training_log.csv; and figures/Table1_logged_examples.png. These dependencies are deliberately absent from this public staging. Run the analysis script from the release root after restoring that subset. The standalone plot command above needs only freshly evaluated CSV outputs and creates a transparent comparison without Office dependencies. Word/PPT generation additionally requires the permissioned original documents and Windows Office; reviewed PDFs remain in the internal delivery.

The final-closeout graphical abstract and Table1 display differ from the preceding plotting snapshot only by qualified mean-performance wording and numeric-string formatting. The final Table1 source remains internal because it contains measurements. Figures1/3 and physical metadata require experimental-author confirmation and are not code-reproduction evidence.

## Publication status

New revision repository: not created. Upload: not performed. Sharing URL: unavailable/pending author code-license and data-access decision. Do not publish the separate internal review ZIP or full artifact archive as a substitute for this data-free code staging. The historical Gasification_temp repository is a different code family and is not the release of these120 fits.
