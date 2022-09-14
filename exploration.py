import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# to compute time of pipeline
from time import time, strftime, gmtime

import warnings
warnings.filterwarnings(action="ignore")

input_path = "./dataset/cleaned/"
input_filename = "data_2016_tryout_1.csv"

output_path = "./dataset/cleaned/"
output_filename = "data_2016_tryout_2.csv"

# Set file name
input_dataset_file = "{}{}".format(input_path, input_filename)
output_dataset_file = "{}{}".format(output_path, output_filename)


def display_boxplot_per_feature(data_frame, x_all_features, y_category):
    """

    :param data_frame:
    :param x_all_features: (list) a list of features to plot (column names, numeric variables)
    :param y_category: (string) category to make different plots
    :return:
    """
    # to make the graphs bigger
    sns.set(rc={'figure.figsize': (12, 6)})
    for i, feature in enumerate(x_all_features):
        plt.figure(i)
        sns.boxplot(data=data_frame, x=feature, y=y_category)


def display_distribution_per_feature(data_frame, all_features, nb_bins):
    """

    :param data_frame:
    :param all_features:
    :param nb_bins:
    :return:
    """
    for column in all_features:
        plt.figure(figsize=(12, 6))
        plt.title('Distribution of : ' + column)
        sns.histplot(data_frame[column], bins=nb_bins)

def get_features_skewness(data_frame):
    """

    :param data_frame:
    :return:
    """
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
    base = 10 then log

    :param data_frame:
    :param features:
    :param base: (by default we use the natural log : ln)
    :return:
    """
    df = data_frame.copy()
    print('Log-transformation of the variables to predict.')
    for feature in features:
        # we name the new variable
        log_feature = "Log-{}".format(feature)
        # we add the transformed variable to our dataframe
        df[log_feature] = np.log(1 + df[feature])  # we add 1 in case the feature = 0
    return df


# From Jérémy Fasy
# Grille des courbes de densité
def densite(df, lines=7, cols=4):
    """
    Input : dataframe, lignes, colonnes
    Output : grille des courbes de densités des variables numériques du dataframe
    """
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


# Description du dataframe
def extrems(df):
    """
    Input : dataframe
    output : description du dataframe
    """
    return df.describe().loc[['min', '25%', '50%', '75%', 'max'], :]


######################################

global columns_to_categorize
columns_to_categorize = ["BuildingType", "PrimaryPropertyType", "Neighborhood", "ZipCode", "CouncilDistrictCode", "LargestPropertyUseType", "SecondLargestPropertyUseType", "ThirdLargestPropertyUseType"]


def load_data_types(dataset_file, columns):
    """
    O)
    :param dataset_file: (string)
    :param columns: (list)

    """
    print("___Loading raw dataset___")
    category_types = {column: 'category' for column in columns}

    # Load raw data
    print("This dictionary will be used when reading the csv file to assign a type to categorical features :", category_types)
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

    #sns.heatmap(corr, annot=True, vmin=-1, cmap='coolwarm')
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
    print("___Dropping buildings with missing values___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    # Suppression des variables
    #low_correlated = ['NumberofBuildings', 'YearBuilt']
    #high_correlated = ['Electricity(kWh)', 'NaturalGas(therms)', 'PropertyGFABuilding(s)', 'LargestPropertyUseTypeGFA']
    high_correlated = ['Electricity(kWh)', 'NaturalGas(therms)']

    target_linked = ['GHGEmissionsIntensity', 'SiteEnergyUseWN(kBtu)',
                     'SiteEUI(kBtu/sf)', 'SiteEUIWN(kBtu/sf)', 'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)']

    #to_drop = list(set(high_correlated + target_linked))
    to_drop = high_correlated + target_linked
    data_frame = data_frame.drop(columns=to_drop, axis=1)

    print("After :", data_frame.shape)
    return data_frame




def exploration_pipeline():
    """

    """
    cleaned_data = load_data_types(input_dataset_file, columns_to_categorize)

    densite()
    log_transformation()

    correlation_matrix(cleaned_data)
    dataset_v1 = delete_correlated_variables(cleaned_data)
    correlation_matrix(dataset_v1, width=8, height=8)







if __name__ == '__main__':
    pass
