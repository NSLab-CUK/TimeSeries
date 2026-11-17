# Directory-layout validation

Validated on 2026-11-17 using the existing Windows environment (Python 3.12, PyTorch 2.5.1+cu124, NumPy 2.2.6, pandas 2.2.2). This environment was used to check relocation; the archived full-training environment remains separately recorded in `metadata/environment_training.json`.

| Check | Result |
| --- | --- |
| Python syntax | All 10 copied Python source files parsed |
| CLI entry points | Inspection, all reproduction help commands, worker/server help and GPU planning succeeded from outside the repository directory |
| Missing-data inspection | Correctly reports all 17 candidates absent from the upload directory |
| Four-GPU scheduling | 120 planned fits, 30 per device; leads 1/3/5/10 mapped to devices 0/1/2/3 |
| Original parser | SHA-256 unchanged |
| Model and training functions | AST comparison identical except declared filesystem paths and prepared-CSV LF serialization |
| Original raw-record hashes | All 16 included records matched recorded SHA-256 values |
| Split and common targets | Exact 10/4/2 split; 31,670 train, 11,622 validation and 7,266 test targets |
| Target-index bytes | Exact archived SHA-256 after making prepared CSV newlines platform independent |
| Training-only scaler | All four array digests matched the archived values exactly |
| Forward computation | RNN, LSTM and GRU accepted a real `(30, 15)` sample and produced four finite outputs |
| Linux shell syntax | Both setup and training wrappers passed `bash -n` |
| Git exclusions | Raw records, outputs, fitted weights, environments, caches and credential examples excluded |
| Git byte preservation | Parser and metadata have text conversion disabled |

Real-data preparation used the original workspace through a temporary in-memory lookup override. Raw files were not copied into this directory. Measurement-bearing validation outputs are retained in the surrounding workspace's `revision/github_repository_validation_20261117/`, outside the upload directory. A data-free summary is in `metadata/layout_validation.json`.

No optimizer steps, full training, checkpoint replay, complete 120-fit evaluation, GPU allocation or environment installation were performed for this reorganization. Forward checks verify shape and execution only, not model accuracy. These checks were run against the initial upload staging before copying it into the `NSLab-CUK/TimeSeries` checkout. GitHub delivery is separate from the model validation described here.
