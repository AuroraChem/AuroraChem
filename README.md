# AuroraChem

[TODO: one-paragraph description and paper citation]

## Installation

Requires Python >= 3.12.

```bash
pip install -r requirements.txt
```

`aurora/` is a modified copy of Microsoft Aurora 1.8.0 (see `aurora/README.md`). Do not install `microsoft-aurora` separately.

## Data

Set `data_dir` and `output_dir` in the config files. `data_dir` should contain:

```
data_dir/
├── GEOS_Chem/   [TODO]
├── era5_interp/ [TODO]
├── emissions/   [TODO]
├── MERRA2/      [TODO: only if met_type: "merra2"]
└── TROPOMI/     [TODO: only for TROPOMI nudging]
```

[TODO: data sources and download instructions]

## Usage

Run from the repository root:

```bash
python -m scripts.aurora_train --config configs/aurora_train_config.yml
python -m scripts.aurora_eval --config configs/aurora_eval_config.yml
```

Training downloads the pretrained Aurora weights from Hugging Face on first run.

## Trained weights

[TODO: Zenodo DOI and link]

## License

MIT; see `LICENSE`. `aurora/` is MIT-licensed by Microsoft Corporation; see `aurora/LICENSE.txt`.
