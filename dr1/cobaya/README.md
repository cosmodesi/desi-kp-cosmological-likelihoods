# Full-Shape (or ShapeFit) Cobaya likelihoods

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

- [lsstypes](https://github.com/adematti/lsstypes) (can be esaily removed)
- [cosmoprimo](https://github.com/cosmodesi/cosmoprimo)  (mostly for BAO filtering)