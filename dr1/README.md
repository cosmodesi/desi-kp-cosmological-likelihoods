# DR1 Full-Shape and BAO Likelihoods

## Content

- The notebook `read_clustering_statistics.ipynb` demonstrates how to read the DR1 Full-Shape and BAO clustering statistics.
- The script `prepare_fiducial_likelihood.py` illustrates how to construct the final likelihoods — including scale cuts and systematic contributions — from the raw data vector, window matrix, and covariance matrix.
- The `cobaya/` directory provides a [Cobaya](https://github.com/CobayaSampler/cobaya) implementation of the Full-Shape + BAO likelihoods, including download of the necessary likelihood files

---

## Downloads

As described [here](https://data.desi.lbl.gov/doc/releases/dr1/vac/full-shape-bao-clustering/), the clustering products are available under:
```
https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-bao-clustering/v1.0/
```
Clustering products are available for DESI data, EZ mocks (used to compute the covariance) and AbacusSummit mocks.
However, in most cases, as for the notebook `read_clustering_statistics.ipynb`, you will only need the DESI `data/` products only, which contains all data vectors, window and covariance matrices.
If you just need to evaluate the DESI Full-Shape (and BAO) likelihood, the `data/likelihood` products only are necessary (see `cobaya/` directory).

### Globus

See [DESI Public Data with Globus](https://data.desi.lbl.gov/doc/access/#bulk-download-with-globus).
In a nutshell:

#### Install Globus CLI

```
pip install globus-cli
```

#### Endpoint

If you would like to download data on your personal computer, install [Globus Connect Personal](https://www.globus.org/globus-connect-personal).
Provide a Collection name (usually `your name`). Then you should be able to find your id `your id` with:
```
globus endpoint search [your name]
```

#### Transfer

```
globus transfer 6b4e1f6a-e600-11ed-9b9b-c9bb788c490e:/dr1/vac/dr1/full-shape-bao-clustering/v1.0/data  [your id]:/path/to/final/directory --recursive
```
To check the task list:
```
globus task list
```
and the progress:
```
globus task show [task id]
```