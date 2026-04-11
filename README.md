# Graphene-seq

Code and source data used for graphene-based optically clear in situ sequencing
and electrophysiology analyses.

## Repository Layout

- `notebooks/`: Figure and supplementary figure notebooks.
- `src/graphene_electro_seq_analysis/`: Analysis utilities used by notebooks.
- `SourceData/`: Preprocessed data and source data tables.

## Requirements

- Python 3.8
- Jupyter Notebook
- Common packages: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scanpy`, `anndata`, `scikit-learn`

Install dependencies in your preferred environment and run notebooks in order.

## Usage

1. Create and activate a Python 3.8 environment.
2. Launch Jupyter Notebook from the repository root.
3. Open a notebook in `notebooks/` and run cells top to bottom.

## External Code

The file `src/graphene_electro_seq_analysis/importrhdutilities.py` is from:
<https://github.com/Intan-Technologies/load-rhd-notebook-python>
