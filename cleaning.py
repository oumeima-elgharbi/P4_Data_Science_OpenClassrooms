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
output_filename = "data_2016_tryout_1.csv"

# Set file name
raw_dataset_file = "{}{}".format(input_path, input_filename)
cleaned_dataset_file = "{}{}".format(output_path, output_filename)


def display_barplot_na(df):
    """

    """
    data_nan = df.isna().sum().sort_values(ascending=False).head(10)
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


def input_zipcode(df):
    """
    :UC: the dataframe must be raw
    """
    columns = df.columns.tolist()
    assert "Address" in columns and "ZipCode" in columns, "You need Address and ZipCode."

    print("___Inputting missing ZipCode___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    all_zipcode = data_frame["ZipCode"].unique().tolist()
    print("We have :", len(all_zipcode), "unique zipcodes.")

    # DataFrame with 16 missing ZipCodes
    zipcode_na_df = data_frame[data_frame["ZipCode"].isna()]
    print("We have :", zipcode_na_df.shape[0], "missing ZipCodes.")

    # We make a list with the address of the building for which the ZipCode is missing.
    zipcode_na_list_address = zipcode_na_df["Address"].tolist()
    zipcode_na_list = [[i, ""] for i in zipcode_na_list_address]

    # This is the list of zipcodes for each of the 16 missing zipcode. We found it on searching on internet using the Address
    correct_zipcode = [98125, 98144, 98117, 98125, 98107, 98117, 98119, 98112, 98122, 98118, 98126, 98108, 98104, 98119,
                       98108, 98108]
    # print(len(right_zipcode))

    for i, zipcode in enumerate(correct_zipcode):
        zipcode_na_list[i][1] = zipcode
    # print(zipcode_na_list)

    print("We replace the missing ZipCodes by their correct value.")
    # zipcode_na_df.index
    # We iterate on the index of the buildings for which the ZipCode is missing
    for i, index in enumerate(zipcode_na_df.index):
        data_frame.at[index, "ZipCode"] = correct_zipcode[i]

    # Verification
    display(data_frame[data_frame["ZipCode"].isna()])
    print("We have :", zipcode_na_df.shape[0], "missing ZipCodes.")
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

    display_barplot_na(data_frame)

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
    display_barplot_na(data_frame)
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
    df_positive = data_frame.loc[:, [x for x in cols_numeric if x not in ['ENERGYSTARScore']]]
    data_frame = data_frame[
        (df_positive.select_dtypes(include='number') >= 0).all(axis=1)].copy()  # all to get all columns

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

    #data_frame = data_frame.sort_values(by="RemainingEnergy(%)", ascending=False)
    print(data_frame)
    print("Shape :", data_frame.shape)

    # 3) Suppression des observations avec total énergie qui est significativement inférieur à la somme des composantes
    # ce qui correspond à "Others(kbtu)" inférieur à 0,

    # On supprime les observations avec des consommations d'autres énergie inférieures à 0 ==> 794 lignes
    to_drop1 = data_frame["RemainingEnergy(kBtu)"] < 0

    # On ne supprime que si le montant inférieur à 0 est significatif par rapport à la consommation totale ==> 37 lignes
    to_drop2 = np.abs(data_frame["RemainingEnergy(kBtu)"]) > (0.001 * data_frame['SiteEnergyUse(kBtu)'])
    print(to_drop2)

    to_drop3 = np.abs(data_frame["RemainingEnergy(kBtu)"]) > (0.01 * data_frame['SiteEnergyUse(kBtu)'])
    print(to_drop3)

    # suppression des consommations d'autres énérgie négatives lorsque significatives ==> 5 buildings
    to_drop = to_drop1 & to_drop2
    data_frame = data_frame[~to_drop3]

    # Imputation à 0 des consommations d'autres énérgie négatives lorsque non significatives
    #data_frame.loc[data_frame["RemainingEnergy(kBtu)"] < 0, "RemainingEnergy(kBtu)"] = 0

    # Renomme la variable de consommation totale d'énergie
    data_frame = data_frame.rename(columns={"SiteEnergyUse(kBtu)": "TotalEnergy(kBtu)"})
    data_frame = data_frame.drop(columns=["RemainingEnergy(kBtu)", "RemainingEnergy(%)"])

    print("After :", data_frame.shape)
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

    #data_v1 = input_zipcode(raw_dataset)  # here because inputting needs all buildings
    data_v2 = dropping_non_relevant_columns(raw_dataset)

    #data_v5 = test(data_v2)
    #'''

    display_countplot(data_v2, feature="ComplianceStatus") ###
    data_v3 = keep_compliant(data_v2) ##
    data_v4 = filling_property_use_type(data_v3) ###

    # data_v4 = dropping_missing_values(data_v3)
    features_with_nan = data_v4.columns.tolist()
    features_with_nan.remove("ENERGYSTARScore")
    print(features_with_nan)
    data_v5 = drop_buildings_subset_nan(data_v4, features_with_nan)
    display_barplot_na(data_v5)
   # '''

    data_v6 = convert_type_and_map(data_v5)
    data_v7 = dropping_negative_values(data_v6)  # removes one building

    verify_PropertyGFA(data_v7)
    # data_v8 = input_propertyGFATotal(data_v7)

    data_v8 = compute_total_energy(data_v7)

    # data_v8 = transforming_building_type(data_v7)
    data_v8.loc[data_v8.BuildingType == "Nonresidential wa", "BuildingType"] = "Nonresidential"

    print(data_v8.info())
    save_dataset_csv(data_v8, cleaned_dataset_file)
    print("_____End of cleaning pipeline_____")


def test(df):
    # Imputation des valeurs manquantes
    display_barplot_na(df)
    df_clean = df.copy()

    # Colonnes à ne pas garder, trop de données manquantes et indépendantes de la problématique
    to_drop = ['ENERGYSTARScore']
    df_clean = df_clean.drop(to_drop, axis=1).copy()

    # Imputation des valeurs manquantes pour la construction secondaire (parking par défaut)
    filtre = df_clean['PropertyGFAParking'] > 0
    df_clean.loc[filtre, 'SecondLargestPropertyUseType'].fillna('Parking', inplace=True)
    df_clean.loc[filtre, 'SecondLargestPropertyUseTypeGFA'].fillna(df_clean.loc[filtre, 'PropertyGFAParking'],
                                                                   inplace=True)

    df_clean['SecondLargestPropertyUseType'].fillna('No use', inplace=True)
    df_clean['SecondLargestPropertyUseTypeGFA'].fillna(0, inplace=True)

    # Imputation des valeurs manquantes pour la construction tertiaire
    df_clean['ThirdLargestPropertyUseType'].fillna('No use', inplace=True)
    df_clean['ThirdLargestPropertyUseTypeGFA'].fillna(0, inplace=True)

    display_barplot_na(df_clean)
    # Suppression des lignes contenant des NA
    df_clean.dropna(inplace=True)

    df_types = df_clean.join(df['ENERGYSTARScore'])
    return df_types

if __name__ == '__main__':
    # Starting time
    t0 = time()
    cleaning_pipeline()
    # End of pipeline time
    t1 = time()
    print("computing time : {:8.6f} sec".format(t1 - t0))
    print("computing time : " + strftime('%H:%M:%S', gmtime(t1 - t0)))

#%%
