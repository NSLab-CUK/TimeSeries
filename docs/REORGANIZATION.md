# Source reorganization

The source for this copy is the original workspace's `revision/JCOU_D_26_00869_posttraining_20261113/code_release/`. Original manuscripts, figures, measurements and completed experiments remain outside this repository.

| Previous release location | Current repository location |
| --- | --- |
| `revision/JCOU_D_26_00869_resolution/reexperiment/pipeline.py` | `src/pipeline.py` |
| `revision/JCOU_D_26_00869_20261113/scripts/audit_raw.py` | `src/audit_raw.py` |
| Training and environment launchers under `reexperiment/` | `scripts/` |
| `revision/JCOU_D_26_00869_20261113/data_manifest.csv` | `metadata/data_manifest.csv` |
| `metadata/protocol.json`, `metadata/model_configs.csv` | `configs/` |
| Paper-generation scripts under the posttraining revision | `archive/paper_analysis/` |
| Raw records at the old release root | Local `data/raw/` (not included) |
| `reexperiment/server_runs/<job>/` | Local `results/<job>/` |

Relative imports, parser checks, source snapshots, shell environment paths, raw-record lookups and output locations were adjusted together. Prepared-contract CSV files explicitly use LF newlines so Windows emits the same target-index bytes as the archived Linux execution. The original parser, split metadata and recorded configurations are unchanged. `.gitattributes` preserves their bytes when Git checks out the repository; other text files use LF.

The original release manifest under `archive/` describes the old staging only. `metadata/source_reorganization.csv` maps copied source files to their current locations and hashes. `metadata/repository_manifest.csv` inventories the final release files and hashes, excluding itself. Validation details are in [VALIDATION.md](VALIDATION.md).

The GitHub repository is `NSLab-CUK/TimeSeries`. Only the verified data-free upload files were copied into its checkout. Surrounding manuscripts, measurements and measurement-bearing validation outputs are excluded. No license was selected by the directory reorganization.
