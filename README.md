# Descriptor-based AI benchmark for BBB permeability prediction

This repository accompanies the study *Benchmarking descriptor-based AI approaches for predicting the BBB permeability of drug-like xenobiotics*.

This repository provides the processed molecular features, predefined feature subsets, and code required to reproduce the main findings of the study, including model benchmarking, model refitting with cross-validated hyperparameter optimization, and independent external validation.

## Structure

```text
requirements.txt
bbb_benchmark/
├── resources/
│   └── feature_sets.csv
├── benchmark.py
├── ensemble_feature_selection.py
├── models.py
├── utils.py
└── validation.py
data/
├── alldesFP_8159.csv
└── alldesFP_test_data.xlsx

```
## Data
Molecular preprocessing and descriptor calculation steps relying on proprietary software, such as LigPrep and AlvaDesc, are not redistributed as executable code.
The molecular features computed through RDKit, AlvaDesc, Mordred, and MACCS fingerprints are stored in the training feature matrix:
```text
data/alldesFP_8159.csv
```
The training feature matrix is stored using Git Large File Storage (Git LFS) because of its file size.


The external-validation matrix is included as:

```text
data/alldesFP_test_data.xlsx
```
The computed ACF, Ens-log2 and Ens-RFA feature subsets are provided in `bbb_benchmark/resources/feature_sets.csv`.

`bbb_benchmark/ensemble_feature_selection.py` contains the routine used to combine Chi-square, information-gain, Random-Forest-importance and Relief rankings.

`bbb_benchmark/models.py` contains the machine-learning models, hyperparameter grids, and cross-validation settings used in the workflow.

`bbb_benchmark/benchmark.py` evaluates the predefined model configurations using five-fold cross-validated hyperparameter optimization.

`bbb_benchmark/validation.py` performs the independent external validation using the alldesFP representation with a correlation threshold of 0.8, the Ens-log2 feature-selection strategy, and the 10-feature Random Forest configuration selected for the final refit.

## Setup

Python 3.12 and scikit-learn 1.5.0 are recommended for consistency with the environment used in the study.

```bash
python -m pip install -r requirements.txt
```
