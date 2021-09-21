# The Impact of Stationarity, Regularity, and Context on the Predictability of Individual Human Mobility

This repository contains the code for our paper titled [The Impact of Stationarity, Regularity, and Context on the Predictability of Individual Human Mobility](https://dl.acm.org/doi/10.1145/3459625), published on ACM Transactions on Spatial Algorithms and Systems, in 2021. 

The Jupyter notebooks show our experiments for the main results in the paper.

## Datasets

The datasets used in the paper have basically four columns: 

- `device_id`: identifier of the user.
- `timestamp`: the date/time of when the user's location was observed.
- `lat`: the latitude of the user's location at the moment of the observation.
- `lon`: the longitude of the user's location at the moment of the observation.

## Scripts

The script `data_processing.py` reads the dataset, converts each lat/lon to a unique identifier, computes the metrics described in the paper, and generates a CSV file with all that data. The notebook `data_processing.ipynb` has the same contents as the script `data_processing.py`, but has more comments on each processing step. Each of the scripts described below reads the CSV file generated in the preprocessing step.

The notebook `data_visualization.ipynb` contains the code used to generate most of the plots in the paper, as well as to compute some aggregate statistics about our datasets.

The notebook `regression.ipynb` contains the code for the regression models that we built in the paper, as well as plots based on those models.


## Citation

Here's the bibtex for citing the paper, in case you find it useful:

```
@article{Teixeira:2021,
    author = {Teixeira, Douglas Do Couto and Viana, Aline Carneiro and Almeida, Jussara M. and Alvim, Mrio S.},
    title = {The Impact of Stationarity, Regularity, and Context on the Predictability of Individual Human Mobility},
    year = {2021},
    issue_date = {June 2021},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {7},
    number = {4},
    issn = {2374-0353},
    url = {https://doi.org/10.1145/3459625},
    doi = {10.1145/3459625},
    journal = {ACM Trans. Spatial Algorithms Syst.},
    month = jun,
    articleno = {19},
    numpages = {24},
    keywords = {predictability, entropy estimators, Human mobility, contextual information}
}
```