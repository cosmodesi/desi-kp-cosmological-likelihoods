# Full-Shape (or ShapeFit) Cobaya likelihoods

## Download the likelihood files

Run:
```python
python download.py --data-dir ./data/likelihood
```
To download the likelihood hdf5 files.

## Power spectrum (+ post-reconstruction BAO) likelihood

Specify the path to the ```data/likelihood``` directory in the entry ```data_dir``` of ```desi_fs_bao_all.yaml```.
To generate tracer-specific likelihoods, with and without post-reconstruction BAO, run
```python
python generate_fs_bao.py
```
To remove these files, run with the option ``--clean``.

An example [Cobaya](https://github.com/CobayaSampler/cobaya) configuration file is provided as ```test_fs_bao_all.yaml```.

## (Compressed) ShapeFit likelihood

As above, but files contain "shapefit" instead of "fs" in their name.

## Requirements

- [lsstypes](https://github.com/adematti/lsstypes) (can be easily replaced with direct h5py use)
- [cosmoprimo](https://github.com/cosmodesi/cosmoprimo) (mostly for BAO filtering)