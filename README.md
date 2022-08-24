# P4_Data_Science_OpenClassrooms

## Context
https://www.energystar.gov/buildings/benchmark/analyze_benchmarking_results

## Dataset

- Create a folder **dataset**
- Download the csv at this address : https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P4/2016_Building_Energy_Benchmarking.csv

Source  : https://data.seattle.gov/dataset/2016-Building-Energy-Benchmarking/2bpz-gwpy

## Libraries
Install the python libraries with the same version :

```bash
pip install -r requirements.txt
```

pip freeze | findstr scikit-learn

## Execution
You can run the notebook **P4_01_preprocess.ipynb** if you want to have a detailed explanation of the steps and cleaning and exploration of the raw dataset.
If you want to fet the cleaned dataset, you can run the script **preprocess.py** directly.

```bash
python preprocess.py
```

## Methodology :

Problematic :

1) Get data
2) Clean dataset
We have 3160 buildings that will be used to predict their Energy consumption and Greenhouse Gas emissions.

3) Explore data (graphs, understand dataset)

4) Scientific modelling
5) Model evaluation

#### Modelling : 
features / variables to cross ??

### Baselines

Tester les modèles suivants : regression linéaire (avec différentes régularisation : Ridge, Lasso, Elastic), Random Forest, XGboost
Penser à comparer les performances des différents modèles : utiliser la MAE
Penser également à optimiser les hyper paramètres de chaque modèle via GridSearch.