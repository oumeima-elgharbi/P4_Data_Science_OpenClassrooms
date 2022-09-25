# Project 4 Data Science OpenClassrooms 
## Supervised Learning : Regression

Author : Oumeima EL GHARBI.

Date : August, September 2022.

### Context

For the city of Seattle, we want to predict the CO2 emissions of buildings.
We have energy information such as the electricity, steam, natural gas used by each of these buildings.
We want to use these buildings properties (such as the geographical position, the type of building, ...) to predict the energy usage.
Using these energy predictions, we will predict the CO2.

This is supervised machine learning problem, more precisely a Regression problem.

We have tried these models :

- baseline : Dummy Regressor
- baseline : Linear model / Linear Regression
- Linear model / Ridge Regression
- Linear model / LASSO
- Linear model / Elastic Net
- Ensemble methods / Random Forest Regressor
- Ensemble methods / XGBoost

We optimized the hyperparameters with a GridSearchCV.

The results were all Cross-Validated a second time (with 5 folds).

### Idea 

While trying to predict the CO2, we tried three methods :

1) Experiment 1 : predict the energy used by the buildings and use it to predict the CO2.
    **This first experiment was the best, and it is the one presented for the evaluation of the project.**

3) Experiment 3 : predict the total energy of the buildings. Then, to predict the CO2, we use all the buildings' properties, the ratio of electricity, steam and gas and the predicted total energy.
    This third experiment produced the best result, however to use the energy ratio to predict the CO2, depending on the interpretation of the project wording could be considered as data leak or not.

### Dataset folder

- Create a folder **dataset**

- Create a folder **dataset/source**
- Create a folder **dataset/cleaned**

- Create a folder **dataset/energy**
- Create a folder **dataset/CO2**
- Create a folder **dataset/CO2_ENERGYSTARScore**

- Download the csv at this address in the folder **source** : https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P4/2016_Building_Energy_Benchmarking.csv

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

You can run the notebook **P4_01_cleaning.ipynb** if you want to have a detailed explanation of the steps and cleaning and exploration of the raw dataset.
If you want to get the cleaned dataset, you can run the script **cleaning.py** directly.

NB : the execution of each *prediction notebook* can take around 5 minutes for a compter with 16 Go RAM.

#### 1) Run the energy preprocessing notebooks

```bash
python cleaning.py 
python exploration.py
run P4_03_feature_engineering.ipynb
```

Or,

```bash
run P4_01_cleaning.ipynb
run P4_02_exploratory_data_analysis.ipynb
run P4_03_feature_engineering_energy.ipynb
```

#### 2) Run the predictions notebooks

```bash
run P4_04_prediction_SiteEnergyUseWN.ipynb
run P4_05_prediction_CO2.ipynb
run P4_06_prediction_CO2_ENERGYSTARScore.ipynb
```