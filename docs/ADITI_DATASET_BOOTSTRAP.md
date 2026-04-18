# ADITI 4.0 Dataset Bootstrap

This workspace now includes an execution script to fetch open/public SSA datasets and stage them under `data/aditi4`.

## Run Dataset Download

```bash
cd ssis
bash scripts/download_aditi_datasets.sh
```

Optional environment flags:

- `DOWNLOAD_FIGSHARE_FULL=1`: pulls SpaceTrack-TimeSeries large files via Figshare metadata.
- `DOWNLOAD_SPLID_DATA=1`: pulls SPLID challenge zip from the Dropbox link discovered in the devkit.
- `SPACETRACK_ID` and `SPACETRACK_PASSWORD`: enables authenticated Space-Track API pulls.
- `TRACSS_EPHEMERIS_URL` and `TRACSS_CDM_URL`: enables TraCSS large file pulls from approved URLs.

Status output:

- `data/aditi4/manifests/download_status.tsv`
- `data/aditi4/manifests/download_report.md`

## Run Full-Start Training

```bash
cd ssis
bash scripts/run_aditi_fullstart.sh
```

By default this launcher uses:

- downloaded CelesTrak TLEs when present,
- COMSOC covariance/weather/EOP if external equivalents are not staged,
- downloaded CelesTrak SATCAT when present.

Override inputs with environment variables:

- `ADITI_XPTLE`
- `ADITI_COV`
- `ADITI_WEATHER`
- `ADITI_EOP`
- `ADITI_CATALOG`
- `OUTPUT_DIR`
- `ADITI_STRICT=1` to enforce strict training mode.

Quality gate controls (enabled by default):

- `ENFORCE_QUALITY=1` enables post-run checks.
- `QUALITY_TARGET_F1=0.92` sets maneuver minimum F1 threshold.
- `QUALITY_REQUIRE_PASS_VERDICT=0` keeps maneuver-quality-first acceptance.
- `QUALITY_REQUIRE_PASS_VERDICT=1` additionally requires overall report verdict `pass`.
