import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cleaning import save_dataset_csv

# to compute time of pipeline
from time import time, strftime, gmtime

import warnings

warnings.filterwarnings(action="ignore")

input_path = "./dataset/cleaned/"
input_filename = "data_cleaned.csv"

output_path = "./dataset/cleaned/"
output_filename = "data_exploration.csv"

# Set file name
input_dataset_file = "{}{}".format(input_path, input_filename)
output_dataset_file = "{}{}".format(output_path, output_filename)


def display_boxplot_per_feature(data_frame, all_features, category):
    """

    :param data_frame:
    :param all_features: (list) a list of features to plot (column names, numeric variables)
    :param category: (string) category to make different plots
    :return:
    """
    # to make the graphs bigger
    sns.set(rc={'figure.figsize': (12, 6)})

    for i, feature in enumerate(all_features):
    # liste triée des zones selon médiane des valeurs
        order = data_frame.groupby(category)[feature].median().sort_values(ascending=False).index
        plt.figure(i)
        sns.boxplot(data=data_frame, x=feature, y=category, order=order)


def box_categorical(df, col_categorical, col_numeric):
    """
    Input : dataframe, colonne d'une variable qualitative, colonne d'une variable quantitative
    output : boxplot de la variable quantitative en fonction de la variable qualitative
    """
    # Largeur du graphe en fonction du nombre de modalités de col_categorical
    xsize = min(round(df[col_categorical].drop_duplicates().shape[0]/2), 20)

    # liste triée des zones selon médiane des valeurs
    order = df.groupby(col_categorical)[col_numeric].median().sort_values(ascending=False).index

    # Graphique
    fig, ax = plt.subplots(figsize=(xsize,4))
    bp = sns.boxplot(x=col_categorical, y=col_numeric, data=df, order=order, showfliers=False)
    bp.set_xticklabels(bp.get_xticklabels(), rotation=45)
    bp.set_title(col_categorical)
    plt.xticks(rotation=90)
    plt.show()


def get_features_skewness(data_frame):
    """

    :param data_frame:
    :return:
    """
    print("___Getting features with skewness greater than 2___")
    df = data_frame.copy()
    numerical_features = df.select_dtypes(include='number').columns
    skew_features = []

    for feature in numerical_features:
        # if the absolute value is greater than 2, we will need to do a log transformation
        if abs(df[feature].skew()) > 2:
            skew_features.append(feature)

    return skew_features


# Graphs to see the effect of log transformation
def compute_log_effect_on_skewness(data_frame, features):
    """

    :param data_frame:
    :param features: (list)
    :return: (DataFrame)
    """
    df = data_frame.copy()
    all_log_transformations = pd.DataFrame({})

    for feature in features:
        test_ln = np.log(1 + df[feature])
        test_log2 = np.log2(1 + df[feature])
        test_log10 = np.log10(1 + df[feature])

        test_df = pd.DataFrame({"feature": [feature],
                                "ln": [abs(test_ln.skew())],
                                "log-2": [abs(test_log2.skew())],
                                "log-10": [abs(test_log10.skew())]})

        all_log_transformations = pd.concat([all_log_transformations, test_df])

    return all_log_transformations


def log_transformation(data_frame, features):
    """
    np.log = neperian log (ln)

    :param data_frame:
    :param features:
    :return:
    """
    print("___Logarithmic transformation of features___")
    df = data_frame.copy()
    print('Log-transformation of the variables to predict.')
    for feature in features:
        # we name the new variable
        log_feature = "Log-{}".format(feature)
        # we add the transformed variable to our dataframe
        df[log_feature] = np.log(1 + df[feature])  # we add 1 in case the feature = 0
    return df


def log_transformation_based_on_skewness(df):
    """


    """
    print("___Log transformation pipeline___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    print("These are the features that need a transformation :")
    skew_features = get_features_skewness(data_frame)
    print(skew_features)
    log_skew_features = ["Log-{}".format(feature) for feature in skew_features]

    print("Before :")
    density(data_frame[skew_features])

    data_frame = log_transformation(data_frame, skew_features)
    # display(data_frame)
    density(data_frame[log_skew_features])

    print("After :", data_frame.shape)
    return data_frame


# From Jérémy Fasy
# Grille des courbes de densité
def density(df, lines=7, cols=4):
    """
    Input : dataframe, lignes, colonnes
    Output : grille des courbes de densités des variables numériques du dataframe
    """
    print("___Density distribution___")
    df = df.select_dtypes(include='number').copy()

    fig, ax = plt.subplots(lines, cols, figsize=(min(15, cols * 3), lines * 2))

    for i, val in enumerate(df.columns.tolist()):
        bp = sns.distplot(df[val], hist=False, ax=ax[i // cols, i % cols], kde_kws={'shade': True})
        bp.set_title("skewness : " + str(round(df[val].skew(), 1)), fontsize=12)
        bp.set_yticks([])
        imax = i

    for i in range(imax + 1, lines * cols):
        ax[i // cols, i % cols].axis('off')

    plt.tight_layout()
    plt.show()


######################################

global columns_to_categorize
columns_to_categorize = ["OSEBuildingID", "BuildingType", "PrimaryPropertyType", "Neighborhood", "ZipCode", "CouncilDistrictCode",
                         "LargestPropertyUseType", "SecondLargestPropertyUseType", "ThirdLargestPropertyUseType"]


def load_data_types(dataset_file, columns):
    """
    O)
    :param dataset_file: (string)
    :param columns: (list)

    """
    print("___Loading dataset___")
    category_types = {column: 'object' for column in columns}

    # Load raw data
    print("This dictionary will be used when reading the csv file to assign a type to categorical features :",
          category_types)
    dataset = pd.read_csv(dataset_file, dtype=category_types)

    print("Initial shape :", dataset.shape)
    return dataset


def correlation_matrix(df, width=15, height=15):
    # we create a dataframe with all the numerical variables
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    numeric_columns.remove("ENERGYSTARScore")

    df_to_corr = df[numeric_columns]

    # we assign the type float to all the values of the matrix
    df_to_corr = df_to_corr.astype(float)
    corr = df_to_corr.corr(method='pearson')
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    print("We display here the correlation matrix without options to justify the display below.")
    plt.figure(figsize=(width, height))

    # sns.heatmap(corr, annot=True, vmin=-1, cmap='coolwarm')
    sns.heatmap(corr, center=0, cmap=sns.color_palette("RdBu_r", 7), linewidths=1,
                annot=True, annot_kws={"size": 9}, fmt=".02f")

    plt.title('Correlation matrix', fontsize=18)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    plt.show()


"""
As we can see both the correlation coefficients give the positive correlation value for Girth and Height of the trees but the value given by them is slightly different because Pearson correlation coefficients measure the linear relationship between the variables while Spearman correlation coefficients measure only monotonic relationships, relationship in which the variables tend to move in the same/opposite direction but not necessarily at a constant rate whereas the rate is constant in a linear relationship.
"""


def delete_correlated_variables(df):
    """


    """
    print("___Removing correlated features___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    # Suppression des variables
    # low_correlated = ['NumberofBuildings', 'YearBuilt']
    # high_correlated = ['Electricity(kWh)', 'NaturalGas(therms)', 'PropertyGFABuilding(s)', 'LargestPropertyUseTypeGFA']
    high_correlated = ['Electricity(kWh)', 'NaturalGas(therms)']

    target_linked = ['GHGEmissionsIntensity', 'SiteEnergyUseWN(kBtu)',
                     'SiteEUI(kBtu/sf)', 'SiteEUIWN(kBtu/sf)', 'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)']

    # to_drop = list(set(high_correlated + target_linked))
    to_drop = high_correlated + target_linked
    data_frame = data_frame.drop(columns=to_drop, axis=1)

    print("After :", data_frame.shape)
    return data_frame


def exploration_pipeline():
    """

    """
    print("_____Starting exploration pipeline_____")
    cleaned_data = load_data_types(input_dataset_file, columns_to_categorize)

    print("___Correlation matrix___")
    correlation_matrix(cleaned_data)
    data_v1 = delete_correlated_variables(cleaned_data)
    correlation_matrix(data_v1, width=8, height=8)



    print("___Boxplot categorical features / target___")
    #features_to_predict = ["TotalGHGEmissions", "TotalEnergy(kBtu)", "Electricity(kBtu)", "NaturalGas(kBtu)", "SteamUse(kBtu)"]

    # Variable BuildingType / émission CO2
    box_categorical(data_v1, "BuildingType")
    #display_boxplot_per_feature(data_v1, all_features=features_to_predict, category="BuildingType")

    # Variable PrimaryPropertyType" / émission CO2
    box_categorical(data_v1, "PrimaryPropertyType")
    #display_boxplot_per_feature(data_v1, all_features=features_to_predict, category="PrimaryPropertyType")

    # Variable Neighborhood / émission CO2
    box_categorical(data_v1, "Neighborhood")
    #display_boxplot_per_feature(data_v1, all_features=features_to_predict, category="Neighborhood")


    data_v2 = log_transformation_based_on_skewness(data_v1)

    print("___Keeping only relevant features___") # "CouncilDistrictCode",
    prediction_features = ["Neighborhood", "BuildingType", "PrimaryPropertyType", "ENERGYSTARScore",
                           "YearBuilt",
                           'Log-NumberofBuildings',
                           'Log-NumberofFloors',
                           'Log-PropertyGFATotal',
                           'Log-PropertyGFAParking',
                           'Log-SecondLargestPropertyUseTypeGFA',
                           'Log-ThirdLargestPropertyUseTypeGFA']
    # , "LargestPropertyUseType", "SecondLargestPropertyUseType", "ThirdLargestPropertyUseType",
    # 'Log-PropertyGFABuilding(s)', 'Log-LargestPropertyUseTypeGFA',

    target_features = [
     'Log-TotalEnergy(kBtu)',
     'Log-SteamUse(kBtu)',
     'Log-Electricity(kBtu)',
     'Log-NaturalGas(kBtu)',
     'Log-TotalGHGEmissions']

    data_v3 = data_v2[prediction_features + target_features]


    print("Final shape of cleaned dataset before feature engineering : ", data_v3.shape)
    data_v3 = data_v3.reset_index(drop=True) #####
    save_dataset_csv(data_v3, output_dataset_file)
    print("_____End of exploration pipeline_____")


if __name__ == '__main__':
    if __name__ == '__main__':
        # Starting time
        t0 = time()
        exploration_pipeline()
        # End of pipeline time
        t1 = time()
        print("computing time : {:8.6f} sec".format(t1 - t0))
        print("computing time : " + strftime('%H:%M:%S', gmtime(t1 - t0)))
