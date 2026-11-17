# Reproduction and compatibility

`reproduce.py` supports `inspect`, `prepare`, `infer`, `evaluate` and `plot`; run each command with `--help` for arguments. Preparing raw records and replaying saved artifacts are distinct operations. `infer` and `evaluate` can read a completed job's prepared observation cache without restoring the raw records.

The raw parser at `src/audit_raw.py` is preserved byte for byte because completed data contracts verify its SHA-256. Its historical standalone `main()` is an archive-auditing workflow requiring additional evidence; use `reproduce.py prepare` for supported preparation in this layout.

The model, sample construction, split logic, loss and optimizer are retained. `src/pipeline.py` has path adaptations and explicit LF serialization for prepared-contract CSV files, so its source hash differs from the archived training source. **Do not resume an archived in-progress job using this reorganized source.** Use a new job name. The original source remains in the surrounding workspace's release staging; `metadata/source_reorganization.csv` records original and reorganized hashes for each copied file.

`metadata/expected_fit_artifacts.csv` refers to archived completed fits. New fitting may differ numerically with hardware/software and produces new hashes. `reproduce.py evaluate` checks each supplied fit's own completion record. Recorded configurations are read-only documentation of the fixed protocol; this change did not introduce a hyperparameter search or configurable training interface.

Evaluation computes MAE, MSE, RMSE and R2 per record/model/lead/seed/gas. Within each seed, each of the two records receives equal weight. The summary averages ten seed scores and uses sample SD (`ddof=1`). Persistence uses the last observed gas values at the prediction origin, with identical targets and averaging. No clipping, best-case filtering or fitting is applied to persistence.

The `plot` command generates PNG/PDF/SVG comparisons from evaluated CSV files. Historical exact paper-generation scripts are separately preserved in `archive/paper_analysis/`; they require the internal posttraining analysis subset and are not supported as standalone commands in the new layout. The original release README there describes that old layout; use the root README for current commands.

No training artifacts or raw data are required to view the GPU assignment plan. Full fitting is explicitly invoked by the training launcher and requires the corresponding raw records and devices.
