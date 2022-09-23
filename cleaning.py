import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# to compute time of pipeline
from time import time, strftime, gmtime

import warnings

warnings.filterwarnings(action="ignore")

input_path = "./dataset/source/"
input_filename = "2016_Building_Energy_Benchmarking.csv"

output_path = "./dataset/cleaned/"
output_filename = "data_cleaned.csv"

# Set file name
raw_dataset_file = "{}{}".format(input_path, input_filename)
cleaned_dataset_file = "{}{}".format(output_path, output_filename)


def display_barplot_na(df, number_features):
    """

    """
    data_nan = df.isna().sum().sort_values(ascending=False).head(number_features)
    plt.figure(figsize=(10, 8))
    plt.title('Proportion of NaN per feature (%)')
    sns.barplot(x=(100 * data_nan.values / df.shape[0]), y=data_nan.index)


def display_countplot(df, feature):
    """

    """
    print(
        "This is an histogram that presents the distribution of the values of the variable {} by counting them.".format(
            feature))
    # to make the graphs bigger
    sns.set(rc={'figure.figsize': (12, 8)})
    sns.countplot(data=df, x=feature)  # countnplot is for discrete variable / categories here.


#############################

def load_data(dataset_file):
    """
    Step 0)
    :param dataset_file: (string)

    :return:
    """
    print("___Loading raw dataset___")

    # Load raw data
    dataset = pd.read_csv(dataset_file)

    print("Initial shape :", dataset.shape)
    return dataset


def dropping_non_relevant_columns(df):
    """
    1)
    """
    print("___Dropping non relevant columns___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    columns_to_drop = ["DataYear", "PropertyName", "ListOfAllPropertyUseTypes", "Address", "City", "State",
                       "TaxParcelIdentificationNumber", "YearsENERGYSTARCertified", "DefaultData",
                       "Comments", "Latitude", "Longitude", "Outlier"]
    print("These are the columns to remove :", columns_to_drop)

    data_frame = data_frame.drop(columns=columns_to_drop)

    print("After :", data_frame.shape)
    return data_frame


def filling_property_use_type(df):
    """
    Step 2)
    :param df: (DataFrame)

    """
    print("___Filling Second/ThirdLargestPropertyUseType___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    display_barplot_na(data_frame, 10)

    # Inputting missing values for SecondLargestPropertyUseType (we use Parking as the default type)
    _filter = data_frame['PropertyGFAParking'] > 0
    # if the value for parking is positive, then we say this is a Parking
    data_frame.loc[_filter, 'SecondLargestPropertyUseType'].fillna('Parking', inplace=True)
    # we put the value for parking
    data_frame.loc[_filter, 'SecondLargestPropertyUseTypeGFA'].fillna(data_frame.loc[_filter, 'PropertyGFAParking'],
                                                                      inplace=True)

    data_frame['SecondLargestPropertyUseType'].fillna('No use', inplace=True)
    data_frame['SecondLargestPropertyUseTypeGFA'].fillna(0, inplace=True)

    # Imputation des valeurs manquantes pour la construction tertiaire
    data_frame['ThirdLargestPropertyUseType'].fillna('No use', inplace=True)
    data_frame['ThirdLargestPropertyUseTypeGFA'].fillna(0, inplace=True)

    print("After :", data_frame.shape)
    display_barplot_na(data_frame, 10)
    return data_frame


def drop_buildings_subset_nan(df, features_to_check):
    """

    :param data_frame:
    :param features_to_check:
    :return:
    """
    print("___Dropping buildings with missing values___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    data_frame = data_frame.dropna(subset=features_to_check)
    # Deleting rows containing NaN
    print("After cleaning missing values, the file contains {} rows et {} columns.".format(data_frame.shape[0],
                                                                                           data_frame.shape[1]))
    print("Remaining missing values : " + str(data_frame.isnull().sum().sum()))

    print("After :", data_frame.shape)
    return data_frame


def assign_type(df):
    """
    Converts datatype
    """
    print("___Assigning datatype___")
    data_frame = df.copy()
    # Checking datatype for each feature
    print("Before :", data_frame.dtypes)

    # Qualitative variables => object type
    col_object = ['OSEBuildingID', 'CouncilDistrictCode', 'ZipCode']
    for col in col_object:
        data_frame[col] = data_frame[col].astype('object')

    # Quantitative variables => numerical type
    col_int = ['NumberofFloors', 'NumberofBuildings']
    for col in col_int:
        data_frame[col] = data_frame[col].astype(np.int64)

    # col_float = ['Latitude', 'Longitude']
    # for col in col_float:
    #   data_frame[col] = data_frame[col].astype(np.float64)

    print("After :", data_frame.dtypes)
    return data_frame


def capitalize_categorical_variables(df):
    """
    maps categorical values
    """
    print("___Capitalization standardization___")
    data_frame = df.copy()
    # Checking datatype for each feature
    print("Before :", data_frame.shape)

    # Capitalization standardization
    cols_non_numeric = data_frame.select_dtypes(exclude='number').columns
    print("We capitalize these columns :", cols_non_numeric)
    data_frame[cols_non_numeric] = data_frame[cols_non_numeric].apply(lambda x: x.astype(str).str.capitalize())
    print("We change delridge neighborhoods to delridge manually.")
    replace_value_for_a_feature(data_frame, "Neighborhood", "Delridge neighborhoods", "Delridge")

    print("After :", data_frame.shape)
    return data_frame


def convert_type_and_map(df):
    """
    4) Converts datatype and maps categorical values

    """
    print("___Assigning datatype___")
    data_frame = df.copy()
    # Checking datatype for each feature
    print("Before :", data_frame.dtypes)

    # Qualitative variables => object type
    col_object = ['OSEBuildingID', 'CouncilDistrictCode', 'ZipCode']
    for col in col_object:
        data_frame[col] = data_frame[col].astype('object')

    # Quantitative variables => numerical type
    col_int = ['NumberofFloors', 'NumberofBuildings']
    for col in col_int:
        data_frame[col] = data_frame[col].astype(np.int64)

    # col_float = ['Latitude', 'Longitude']
    # for col in col_float:
    #    data_frame[col] = data_frame[col].astype(np.float64)

    print("___Capitalization standardization___")
    # Capitalization standardization
    cols_non_numeric = data_frame.select_dtypes(exclude='number').columns
    print("We capitalize these columns :", cols_non_numeric)
    data_frame[cols_non_numeric] = data_frame[cols_non_numeric].apply(lambda x: x.astype(str).str.capitalize())
    print("We change delridge neighborhoods to delridge manually.")
    replace_value_for_a_feature(data_frame, "Neighborhood", "Delridge neighborhoods", "Delridge")

    print("After :", data_frame.dtypes)
    return data_frame


def dropping_negative_values(df):
    """
    5)
    """
    print("___Dropping buildings with negative values___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    # List of numerical columns
    cols_numeric = data_frame.select_dtypes(include='number').columns
    print("We have", len(cols_numeric) - 1, "numerical columns without the ENERGYSTARScore.")

    # We only keep the rows / buildings without negative values
    df_numeric = data_frame.loc[:, [x for x in cols_numeric if x not in ['ENERGYSTARScore']]]
    data_frame = data_frame[
        (df_numeric.select_dtypes(include='number') >= 0).all(axis=1)].copy()  # all to get all columns

    print("After :", data_frame.shape)
    return data_frame


def verify_PropertyGFA(df):
    """
    6.1)
    :param data_frame:
    :return:

    :UC: 100% filled df[["PropertyGFATotal", "PropertyGFABuilding(s)", "PropertyGFAParking"]]
    """
    print("___Checking PropertyGFATotal___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)  ####
    data_frame = data_frame[["PropertyGFATotal", "PropertyGFABuilding(s)", "PropertyGFAParking"]]

    print("Starting checking")
    for index, row in data_frame.iterrows():
        if row["PropertyGFATotal"] != row["PropertyGFABuilding(s)"] + row["PropertyGFAParking"]:
            print("This building doesn't have a right PropertyGFATotal : ", index, row)
    print("End of checking")
    print("After :", data_frame.shape)  ###


def keep_compliant(df):
    """
    7) drops Outlier and ComplianceStatus
    """
    print("___Keeping compliant buildings only___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    # On ne garde que les observations avec le status "Compliant" (conforme)
    data_frame = data_frame[data_frame['ComplianceStatus'] == 'Compliant']

    print("We delete the columns ComplianceStatus.")
    data_frame = drop_selected_features(data_frame, ["ComplianceStatus"])

    print("After :", data_frame.shape)
    return data_frame


def verify_PropertyGFA(data_frame):
    """

    :param data_frame:
    :return:

    :UC: 100% filled df[["PropertyGFATotal", "PropertyGFABuilding(s)", "PropertyGFAParking"]]
    """
    print("___Verifying propertyGFATotal___")
    df = data_frame.copy()
    df = df[["PropertyGFATotal", "PropertyGFABuilding(s)", "PropertyGFAParking"]]
    for index, row in df.iterrows():
        if row["PropertyGFATotal"] != row["PropertyGFABuilding(s)"] + row["PropertyGFAParking"]:
            print("HERE : ", index, row)
    print("End of checking.")


def compute_total_energy(df):
    """
    8***** CORRECT
    """
    print("___Computing Total Energy___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    # 1) We add a column that computes the difference between the Total Energy and Electricity, SteamUse and NaturalGas
    data_frame["RemainingEnergy(kBtu)"] = data_frame["SiteEnergyUse(kBtu)"] - data_frame["SteamUse(kBtu)"] - data_frame[
        "Electricity(kBtu)"] - data_frame["NaturalGas(kBtu)"]

    # 2) we take the absolute value of the difference and round up to the superior unit.
    data_frame["RemainingEnergy(%)"] = round(
        abs(data_frame["RemainingEnergy(kBtu)"] / data_frame["SiteEnergyUse(kBtu)"] * 100),
        1)  # abs adds 5 buildings

    # data_frame = data_frame.sort_values(by="RemainingEnergy(%)", ascending=False)
    display(data_frame)
    print("Shape :", data_frame.shape)

    # 3) Suppression des observations avec total énergie qui est significativement inférieur à la somme des composantes
    # ce qui correspond à "Others(kbtu)" inférieur à 0,

    # On supprime les observations avec des consommations d'autres énergie inférieures à 0 ==> 794 lignes
    to_drop1 = data_frame["RemainingEnergy(kBtu)"] < 0

    # On ne supprime que si le montant inférieur à 0 est significatif par rapport à la consommation totale ==> 37 lignes
    to_drop2 = np.abs(data_frame["RemainingEnergy(kBtu)"]) > (0.001 * data_frame['SiteEnergyUse(kBtu)'])
    #print(to_drop2)

    to_drop3 = np.abs(data_frame["RemainingEnergy(kBtu)"]) > (0.01 * data_frame['SiteEnergyUse(kBtu)'])
    #print(to_drop3)

    # suppression des consommations d'autres énérgie négatives lorsque significatives ==> 5 buildings
    to_drop = to_drop1 & to_drop2
    data_frame = data_frame[~to_drop3]

    # Imputation à 0 des consommations d'autres énérgie négatives lorsque non significatives
    # data_frame.loc[data_frame["RemainingEnergy(kBtu)"] < 0, "RemainingEnergy(kBtu)"] = 0

    # Renomme la variable de consommation totale d'énergie
    data_frame = data_frame.rename(columns={"SiteEnergyUse(kBtu)": "TotalEnergy(kBtu)"})
    data_frame = data_frame.drop(columns=["RemainingEnergy(kBtu)", "RemainingEnergy(%)"])

    print("After :", data_frame.shape)
    return data_frame


def compute_ratio_energy(df):
    """

    """
    print("___Computing ratio___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    data_frame["Ratio_Electricity"] = data_frame["Electricity(kBtu)"] / data_frame["TotalEnergy(kBtu)"]

    data_frame["Ratio_Steam"] = data_frame["SteamUse(kBtu)"] / data_frame["TotalEnergy(kBtu)"]

    data_frame["Ratio_Gas"] = data_frame["NaturalGas(kBtu)"] / data_frame["TotalEnergy(kBtu)"]

    data_frame["Ratio_Steam+Gas"] = (data_frame["NaturalGas(kBtu)"] + data_frame["SteamUse(kBtu)"]) / data_frame[
        "TotalEnergy(kBtu)"]

    data_frame["Ratio_Other"] = (data_frame["TotalEnergy(kBtu)"] - data_frame["NaturalGas(kBtu)"] - data_frame[
        "SteamUse(kBtu)"] - data_frame["Electricity(kBtu)"]) / data_frame["TotalEnergy(kBtu)"]

    data_frame["Ratio_Steam+Gas+Other"] = (data_frame["TotalEnergy(kBtu)"] - data_frame["Electricity(kBtu)"]) / \
                                          data_frame["TotalEnergy(kBtu)"]

    print("After :", data_frame.shape)
    return data_frame


def transforming_building_type(df):
    """
    8
    """
    print("___Transforming building type___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)
    print(data_frame["BuildingType"].value_counts())

    data_frame.loc[data_frame.BuildingType == "Nonresidential wa", "BuildingType"] = "Nonresidential"

    print("After :", data_frame.shape)
    print(data_frame["BuildingType"].value_counts())
    return data_frame


def save_dataset_csv(data_frame, path):
    """
    9)

    """
    print("___Saving cleaned dataset___")
    # Save
    data_frame.to_csv(path, index=False)


# functions :


def replace_value_for_a_feature(data_frame, feature, old_value, new_value):
    """

    :param data_frame:
    :param feature:
    :param old_value:
    :param new_value:
    :return:
    """
    # Applying the condition
    data_frame.loc[data_frame[feature] == old_value, feature] = new_value


def drop_selected_features(data_frame, list_features_to_drop):
    """

    :param data_frame:
    :param list_features_to_drop:
    :return:
    """
    df = data_frame.drop(columns=list_features_to_drop)
    return df


def cleaning_pipeline():
    """

    """
    print("_____Starting cleaning pipeline_____")
    raw_dataset = load_data(raw_dataset_file)

    # data_v1 = input_zipcode(raw_dataset)  # here because inputting needs all buildings
    data_v1 = dropping_non_relevant_columns(raw_dataset)

    display_countplot(data_v1, feature="ComplianceStatus")
    data_v2 = keep_compliant(data_v1)
    data_v3 = filling_property_use_type(data_v2)

    # data_v4 = dropping_missing_values(data_v3)
    features_without_EnergyStarScore = data_v3.columns.tolist()
    features_without_EnergyStarScore.remove("ENERGYSTARScore")
    print(features_without_EnergyStarScore)
    data_v4 = drop_buildings_subset_nan(data_v3, features_without_EnergyStarScore)
    display_barplot_na(data_v4, 10)

    # data_v5 = convert_type_and_map(data_v4)
    data_v5 = assign_type(data_v4)
    data_v6 = capitalize_categorical_variables(data_v5)
    data_v7 = transforming_building_type(data_v6)

    data_v8 = dropping_negative_values(data_v7)  # removes one building
    verify_PropertyGFA(data_v8)  # data_v8 = input_propertyGFATotal(data_v7)

    data_v9 = compute_total_energy(data_v8)
    data_v10 = compute_ratio_energy(data_v9)

    print(data_v10.info())
    save_dataset_csv(data_v10, cleaned_dataset_file)
    print("_____End of cleaning pipeline_____")


if __name__ == '__main__':
    # Starting time
    t0 = time()
    cleaning_pipeline()
    # End of pipeline time
    t1 = time()
    print("computing time : {:8.6f} sec".format(t1 - t0))
    print("computing time : " + strftime('%H:%M:%S', gmtime(t1 - t0)))
