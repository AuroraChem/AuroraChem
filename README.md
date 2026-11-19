# AuroraChem

**AuroraChem** is a fine-tuned version of Microsoft's [Aurora](https://github.com/microsoft/aurora) atmospheric foundation model that simulates global atmospheric chemistry and meteorology together. It predicts OH, Cl, NO, NO<sub>2</sub>, O<sub>3</sub>, CO, and <sup>14</sup>CO along with temperature, humidity, winds, and geopotential height on a 4°×5° grid with 13 pressure levels, in 6-hour steps. It is designed as the forward model for joint chemical and meteorological data assimilation.

**Documentation:** https://aurorachem.github.io

---

## Features

- Joint chemical and meteorological state: an observation of either one can update the other
- Seven chemical species relevant to the oxidative capacity of the troposphere (OH, Cl, NO, NO<sub>2</sub>, O<sub>3</sub>, CO, <sup>14</sup>CO)
- Stable year-long rollouts; a one-year rollout takes about 12 minutes on a single GPU
- Direct insertion of meteorological fields, chemical fields, or TROPOMI-like NO<sub>2</sub> during a rollout
- Differentiable, so it provides gradients for variational data assimilation without an adjoint
- Fine-tuned from the pretrained 1.3 billion-parameter Aurora model

---

## Installation

Requires Python >= 3.12 and a CUDA-capable GPU.

```bash
git clone https://github.com/AuroraChem/AuroraChem.git
cd AuroraChem
pip install -r requirements.txt
```

`aurora/` is a modified copy of Microsoft Aurora 1.8.0 (see `aurora/README.md`). Do not install `microsoft-aurora` separately. Run all scripts from the repository root so that this copy is imported.

---

## Repository structure

```
AuroraChem/
├── aurora/                       # Modified copy of Microsoft Aurora 1.8.0 (MIT)
│   ├── model/                    # Encoder, Swin Transformer backbone, decoder
│   └── rollout.py                # Free and direct-insertion rollouts
├── configs/
│   ├── aurora_train_config.yml   # Fine-tuning settings
│   └── aurora_eval_config.yml    # Rollout and evaluation settings
├── scripts/
│   ├── aurora_train.py           # Fine-tune Aurora on GEOS-Chem and ERA5 fields
│   └── aurora_eval.py            # One-step forecasts and rollouts
├── src/
│   ├── data/                     # Batch construction
│   ├── training/                 # Loss functions, learning rate schedule, early stopping
│   ├── utils/                    # Config loading and utilities
│   └── visualization/            # Plotting constants
└── requirements.txt
```

---

## Data

Set `data_dir` and `output_dir` in the config files. All fields must be on the 4°×5° grid, and 3D fields on the 13 Aurora pressure levels (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, and 50 hPa). `data_dir` should contain:

```
data_dir/
├── GEOS_Chem/   # Chemical fields: <gc_version>/time_avg/4x5/<year>/<species>_filled_<year>_interp<scheme>.nc
├── era5_interp/ # ERA5 meteorology and static fields: <year>/<date>-atmospheric_interp<scheme>.nc,
│                #   <year>/<date>-surface-level_interp<scheme>.nc, <year>/static_<year>_interp<scheme>.nc
├── emissions/   # CAMS monthly emissions: ammonia_<year>.nc, carbon-monoxide_<year>.nc,
│                #   nitrogen-oxides_<year>.nc, sulphur-dioxide_<year>.nc
├── MERRA2/      # Only if met_type: "merra2"
└── TROPOMI/     # Only for TROPOMI nudging: tropomi_no2_mask_2019_6hourly.nc
```

| Category | Fields | Source |
|---|---|---|
| Chemical | OH, Cl, NO, NO<sub>2</sub>, O<sub>3</sub>, CO, <sup>14</sup>CO | GEOS-Chem v14.5.3 (FETCH<sub>4</sub> simulation) |
| Meteorological | T, q, U, V, Z on pressure levels | ERA5 |
| Surface meteorology | 2 m T, 10 m U and V, mean sea level pressure | ERA5 |
| Emissions (input only) | Monthly anthropogenic CO, NO<sub>x</sub>, NH<sub>3</sub>, SO<sub>2</sub> | [CAMS global emission inventories](https://ads.atmosphere.copernicus.eu/datasets/cams-global-emission-inventories) |
| Static (input only) | Land-sea mask, soil type, topography | ERA5 |

[TODO: data sources and download instructions]

---

## Usage

Run from the repository root:

```bash
python -m scripts.aurora_train --config configs/aurora_train_config.yml
python -m scripts.aurora_eval --config configs/aurora_eval_config.yml
```

Training downloads the pretrained Aurora weights from Hugging Face on first run.

### Training

`aurora_train` fine-tunes all weights on the years in `training.years` with early stopping. The new chemistry layers use `chem_lr` and the pretrained weights use `met_lr`. The weights and normalization statistics are saved to `output_dir/Weights/<run_name>_best_weights.pth` and `output_dir/Weights/<run_name>_norm_stats.npz`.

### Rollouts and direct insertion

`aurora_eval` loads the weights for `run_name` from `output_dir/Weights/` and runs from `start` to `end`. The `rollout` block sets the rollout type and how often fields are inserted. `nudge_int` is the number of 6-hour steps between insertions, so `nudge_int: 20` inserts every 5 days.

```yaml
rollout:
    rollout_bool: True
    rollout_type: "nudged_met"
    nudge_int: 20
    met_nudge_fields: ["t", "q", "u", "v", "z"]
    chem_nudge_fields: ["OH", "NO", "NO2", "Cl", "O3", "CO", "C14O"]
```

| `rollout_type` | Fields inserted |
|---|---|
| `free` | None |
| `nudged` | All meteorological and chemical fields |
| `nudged_met` | Meteorological fields only |
| `nudged_chem` | Chemical fields only |
| `nudged_partial` | The fields listed in `met_nudge_fields` and `chem_nudge_fields` |
| `nudged_tropomi_no2` | NO<sub>2</sub> profiles in grid cells with valid TROPOMI pixels |
| `nudged_tropomi_no2_ak` | NO<sub>2</sub> columns computed with a TROPOMI averaging kernel |

Set `one_step.one_step_bool: True` to also evaluate one-step (6-hour) forecasts.

### Output

Rollouts are written as netCDF files to `output_dir`, named `<run_name>_<start>_<end>_<rollout type>.nc`.

---

## Trained weights

[TODO: Zenodo DOI and link]

Place `<run_name>_best_weights.pth` and `<run_name>_norm_stats.npz` in `output_dir/Weights/` and set `run_name` in `configs/aurora_eval_config.yml` to match.

---

## Citation

If you use AuroraChem in your research, please cite:

> [Kim, E. J., Mei, E. J., Ivatt, P. D., Murray, L. T., and Turner, A. J. (2026). Enabling joint chemical and meteorological data assimilation with an atmospheric foundation model, arXiv:2611.24020.](https://arxiv.org/abs/2611.24020)

Please also cite Aurora:

> Bodnar, C., Bruinsma, W. P., Lucic, A., Stanley, M., Allen, A., Brandstetter, J., et al. (2025). A foundation model for the Earth system. *Nature*, 641, 1180–1187. [https://doi.org/10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y)

---

## Support

This work is part of the [FETCH<sub>4</sub> project](https://fetch4.github.io/), and support was provided by Schmidt Sciences.

---

## License

MIT; see `LICENSE`. `aurora/` is MIT-licensed by Microsoft Corporation; see `aurora/LICENSE.txt`.

---

## Contact

Questions? Open an issue on [GitHub](https://github.com/AuroraChem/AuroraChem) or contact:

- **Eliot Kim** (lead developer): ejkim1@uw.edu
- **Alex Turner**: turneraj@uw.edu
