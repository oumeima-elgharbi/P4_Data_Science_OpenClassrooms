# Project 4 Data Science OpenClassrooms

## Supervised Learning : Regression

Author : Oumeima EL GHARBI.

Date : August, September 2022.

### Context

For the city of Seattle, we want to predict the CO2 emissions of buildings.
We want to use these buildings properties (such as the geographical position, the type of building, ...) to predict the
energy use and CO2 emissions.

1) Predict Total Energy
2) Predict CO2
3) Predict CO2 using ENERGY STAR Score

This is supervised machine learning problem, more precisely a Regression problem.

We have tried these models :

- baseline : Dummy Regressor
- baseline : Linear model / Linear Regression
- Linear model / Ridge Regression
- Linear model / LASSO
- Linear model / Elastic Net
- Ensemble methods / Random Forest Regressor
- Ensemble methods / XGBoost

We optimized the hyper-parameters with a GridSearchCV.

The results were all Cross-Validated a second time (with 5 folds).

### Dataset folder

- Create a folder **dataset**

- Create a folder **dataset/source**
- Create a folder **dataset/cleaned**

- Create a folder **dataset/energy**
- Create a folder **dataset/CO2**
- Create a folder **dataset/CO2_ENERGYSTARScore**

- Download the csv at this address in the folder **source** :
  https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P4/2016_Building_Energy_Benchmarking.csv

Source  : https://data.seattle.gov/dataset/2016-Building-Energy-Benchmarking/2bpz-gwpy

### Model folders

- Create a folder **model**

- Create a folder **model/energy**
- Create a folder **model/CO2**
- Create a folder **model/CO2_ENERGYSTARScore**

Copy/Paste the folder **model** in each **Experiment** folder or just in the folder **Experiment_1_log**

### Libraries

Install the python libraries with the same version :

```bash
pip install -r requirements.txt
```

pip freeze | findstr scikit-learn

### Execution

We will present the results from the first experiment.

You can run the notebook **P4_01_cleaning.ipynb** if you want to have a detailed explanation of the steps and cleaning
and exploration of the raw dataset.
If you want to get the cleaned dataset, you can run the script **cleaning.py** directly.

NB : the execution of each *prediction notebook* can take around 5 minutes for a compter with 16 Go RAM.

#### 1) Run the preprocessing notebooks

```bash
run P4_01_cleaning.ipynb
run P4_02_exploratory_data_analysis.ipynb
run P4_03_feature_engineering_energy.ipynb
```

Or,

```bash
python cleaning.py 
python exploration.py
run P4_03_feature_engineering.ipynb
```

#### 2) Run the predictions notebooks

```bash
run P4_04_prediction_energy.ipynb
run P4_05_prediction_CO2.ipynb
run P4_06_prediction_CO2_ENERGYSTARScore.ipynb
```