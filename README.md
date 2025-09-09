# Developing a machine learning model to map new-build gentrification: A mixed-methods approach
This repository accompaniees the paper:

**"Developing a machine learning model to map new-build gentrification: A mixed-methods approach"**

It provides code, processed data, and reproducible workflows for the study of new-build gentrification in Philadelphia using Street View Imagery, permit data, and community-informed protocols.

## Contents
- `notebooks/` – PCA analysis, focus group protocol integration, data preparation, model training, validation
- `src/` – Core Python modules (data, models, training, utils)
- `data/` – Scripts and metadata for dataset creation (not raw imagery)
- `docs/` – figures, supplementary materials

## Getting Started
```bash
git clone https://github.com/<your-org>/gentrification-newbuild-ml.git
cd gentrification-newbuild-ml
pip install -r requirements.txt
```
## Starting traning
```bash
python run_train.py
```
## Evaluation
```bash
python run_eval.py
