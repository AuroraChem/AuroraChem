# Modified Aurora

This folder is a modified copy of Microsoft's Aurora model, version 1.8.0
(`microsoft-aurora` on PyPI/conda-forge; https://github.com/microsoft/aurora),
distributed under the MIT License in `LICENSE.txt`.

Files modified from the 1.8.0 release:

- `__init__.py`
- `_version.py`
- `rollout.py`
- `model/aurora.py`
- `model/compat.py`
- `model/decoder.py`
- `model/encoder.py`
- `model/patchembed.py`
- `model/posencoding.py`
- `model/swin3d.py`

Run the scripts from the repository root (e.g. `python -m scripts.aurora_train ...`)
so that this copy is imported instead of any installed `microsoft-aurora` package.
