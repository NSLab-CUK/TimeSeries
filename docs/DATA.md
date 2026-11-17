# Data placement and identity

The release contains record metadata, not scalar measurements. Copy unchanged source records into:

```text
data/raw/<source_file from metadata/data_manifest.csv>
```

For example:

```text
data/raw/논문에 사용한 데이터_남지영_최종/20210504_mask pellet air gasificaion.csv
```

The manifest lists 21 source records, including pyrolysis. The fixed preparation considers 17 gasification candidates, excludes one spreadsheet without an absolute clock and uses the remaining 16 records in a 10/4/2 split. Restore all 17 gasification candidates, including the excluded spreadsheet, to reproduce the exclusion audit. `metadata/split_manifest.csv` contains the 16 included records and their SHA-256 digests. Do not rename records or edit their bytes.

`source_file` entries remain identical to the recorded experiment; only their lookup base changed from the release root to `data/raw/`. Split-manifest bytes, common target rows, parser bytes and scaler digests are preserved by the reorganization.

The raw parser searches the first 12 rows for headers, obtains the date from metadata and constructs timestamps from Hour/Min/Sec. A backward change greater than 12 hours is a midnight rollover. Other timing discontinuities or nonfinite selected features break segments. No timestamps are invented for the clockless spreadsheet. Scaling uses finite clocked training rows once each, rather than repeated windows.

Run `python reproduce.py inspect` to list missing source paths, and `python reproduce.py prepare --output results/prepared` to check the restored sources against the recorded protocol. Neither command fits models. Keep generated measurement-bearing files under `results/` or another ignored local directory.
