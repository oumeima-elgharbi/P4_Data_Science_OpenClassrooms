import pandas as pd
import numpy as np

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
    print("___Filling Second/ThirdLargestPropertyUseType___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    columns_to_drop = ["DataYear", "PropertyName", "ListOfAllPropertyUseTypes", "Address", "City", "State",
                       "TaxParcelIdentificationNumber", "YearsENERGYSTARCertified", "DefaultData",
                       "Comments", "Latitude", "Longitude"]
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

    # Inputting missing values for SecondLargestPropertyUseType (we use Parking as the default type)
    _filter = data_frame['PropertyGFAParking'] > 0
    # if the value for parking is positive, then we say this is a Parking
    data_frame.loc[_filter, 'SecondLargestPropertyUseType'].fillna('Parking', inplace=True)
    # we put the value for parking
    data_frame.loc[_filter, 'SecondLargestPropertyUseTypeGFA'].fillna(data_frame.loc[_filter, 'PropertyGFAParking'], inplace=True)

    data_frame['SecondLargestPropertyUseType'].fillna('No use', inplace=True)
    data_frame['SecondLargestPropertyUseTypeGFA'].fillna(0, inplace=True)

    # Imputation des valeurs manquantes pour la construction tertiaire
    data_frame['ThirdLargestPropertyUseType'].fillna('No use', inplace=True)
    data_frame['ThirdLargestPropertyUseTypeGFA'].fillna(0, inplace=True)

    print("After :", data_frame.shape)
    return data_frame


def dropping_missing_values(df):
    """
    Step 2)
    :param df: (DataFrame)

    """
    print("___Dropping buildings with missing values___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    # Suppression des lignes contenant des NA
    # data_frame.dropna(inplace=True)
    data_frame = data_frame.dropna()

    # Valeurs manquantes restantes
    print("After cleaning missing values, the file contains {} rows et {} columns.".format(data_frame.shape[0],
                                                                                           data_frame.shape[1]))
    print("Remaining missing values : " + str(data_frame.isnull().sum().sum()))

    print("After :", data_frame.shape)
    return data_frame


def convert_type(df, raw_data_frame):
    """
    3 ***********

    """
    print("___Assigning datatype___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    # Variables qualitatives => type objet
    col_object = ['OSEBuildingID', 'CouncilDistrictCode']

    # Contrôle du type des variables
    df_types = data_frame.join(raw_data_frame['ENERGYSTARScore'])
    print(df_types.dtypes)

    for col in col_object:
        df_types[col] = df_types[col].astype('object')

    # Variables quantitatives => type numérique
    col_int = ['NumberofFloors']
    col_float = ['Latitude', 'Longitude']

    for col in col_int:
        df_types[col] = df_types[col].astype(np.int64)

    for col in col_float:
        df_types[col] = df_types[col].astype(np.float64)

    # Uniformisation de la capitalisation
    cols_non_numeric = df_types.select_dtypes(exclude='number').columns
    df_types[cols_non_numeric] = df_types[cols_non_numeric].apply(lambda x: x.astype(str).str.capitalize())

    print("After :", df_types.shape)
    return df_types


def dropping_negative_values(df):
    """
    4
    """
    print("___Dropping buildings with negative values___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    # Liste des colonnes numériques
    cols_numeric = data_frame.select_dtypes(include='number').columns

    # On ne garde que les observations sans valeurs négatives
    df_positive = data_frame.loc[:, [x for x in cols_numeric if x not in ['Longitude', 'Latitude', 'ENERGYSTARScore']]]
    data_frame = data_frame[(df_positive.select_dtypes(include='number') >= 0).all(axis=1)].copy()

    # Valeurs négatives
    print("Après nettoyage des valeurs négatives, le fichier comporte {} lignes".format(data_frame.shape[0]))

    print("After :", data_frame.shape)
    return data_frame


def input_propertyGFATotal(df):
    """
    5
    """
    print("___Inputting correct propertyGFATotal after removing negative values above___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    # On imputele vrai total à la surface totale
    data_frame['PropertyGFATotal'] = data_frame['PropertyGFAParking'] + data_frame['PropertyGFABuilding(s)']

    print("After :", data_frame.shape)
    return data_frame


def keep_compliant(df):
    """
    6
    """
    print("___Keeping compliant buildings only___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    # On ne garde que les observations avec le status "Compliant" (conforme)
    data_frame = data_frame[data_frame['ComplianceStatus'] == 'Compliant']

    print("After :", data_frame.shape)
    return data_frame


def compute_total_energy(df):
    """
    7*****
    """
    print("___Computing Total Energy___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    # Création d'une variable pour les autres sources d'énérgie
    data_frame['Others(kBtu)'] = data_frame['SiteEnergyUse(kBtu)'] - data_frame['SteamUse(kBtu)'] - data_frame[
        'Electricity(kBtu)'] - data_frame['NaturalGas(kBtu)']

    print("Shape : 3219, 44", data_frame.shape)

    # Suppression des observations avec total énergie qui est significativement inférieur à la somme des composantes
    # ce qui correspond à "Others(kbtu)" inférieur à 0,
    # car elle vient d'être calculée par différence entre le total énérgie et ses 3 premières composantes

    # On supprime les observations avec des consommations d'autres énergie inférieures à 0 ==> 823 lignes
    to_drop1 = data_frame['Others(kBtu)'] < 0

    # On ne supprime que si le montant inférieur à 0 est significatif par rapport à la consommation totale ==> 36 lignes
    to_drop2 = np.abs(data_frame['Others(kBtu)']) > (0.001 * data_frame['SiteEnergyUse(kBtu)'])

    # suppression des consommations d'autres énérgie négatives lorsque significatives
    to_drop = to_drop1 & to_drop2
    data_frame = data_frame[~to_drop]

    # Imputation à 0 des consommations d'autres énérgie négatives lorsque non significatives
    data_frame.loc[data_frame['Others(kBtu)'] < 0, 'Others(kBtu)'] = 0

    # Renomme la variable de consommation totale d'énergie
    data_frame = data_frame.rename(columns={'SiteEnergyUse(kBtu)': 'Total_energy'})

    print("After :", data_frame.shape)
    return data_frame





def save_dataset_csv(data_frame, path):
    """
    9)

    """
    print("___Saving cleaned dataset___")
    # Save
    data_frame.to_csv(path, index=False)


def cleaning_pipeline():
    """

    """
    print("_____Starting cleaning pipeline_____")
    raw_dataset = load_data(raw_dataset_file)
    
    dropping_non_relevant_columns()

    data_v1 = filling_property_use_type(raw_dataset)
    data_v2 = dropping_missing_values(data_v1)
    data_v3 = convert_type(data_v2, raw_dataset)
    data_v4 = dropping_negative_values(data_v3)
    data_v5 = input_propertyGFATotal(data_v4)
    data_v6 = keep_compliant(data_v5)
    data_v7 = compute_total_energy(data_v6)
    #data_v8 = transforming_building_type(data_v7)

    save_dataset_csv(data_v7, cleaned_dataset_file)
    print("_____End of cleaning pipeline_____")


# Verification functions :

def verify_PropertyGFA(data_frame):
    """

    :param data_frame:
    :return:

    :UC: 100% filled df[["PropertyGFATotal", "PropertyGFABuilding(s)", "PropertyGFAParking"]]
    """
    df = data_frame.copy()
    df = df[["PropertyGFATotal", "PropertyGFABuilding(s)", "PropertyGFAParking"]]

    print("Starting checking")
    for index, row in df.iterrows():
        if row["PropertyGFATotal"] != row["PropertyGFABuilding(s)"] + row["PropertyGFAParking"]:
            print("HERE : ", index, row)
    print("End of checking.")


def verify_min_value(data_frame, all_features, ceiling):
    """

    :param data_frame:
    :param all_features:
    :param ceiling:
    :return:
    """
    features_with_negative_values = []
    print("Starting checking")

    for energy_feature in all_features:
        energy_min = data_frame[energy_feature].min()
        if energy_min < ceiling:
            print("For this feature :", energy_feature, "we have this negative value :", energy_min)
            features_with_negative_values.append(energy_feature)

    print("End of checking.")
    return features_with_negative_values


# Cleaning pipeline
def preprocess_features(raw_data):
    """

    :param raw_data:
    :return:
    """
    print("Starting cleaning pipeline.")
    print("Initial shape :", raw_data.shape)

    # Applying the condition
    # print("The minimum for the number of buildings is 0 which is not possible, we correct that by replacing 0 by 1.")
    # replace_value_for_a_feature(raw_data, "NumberofBuildings", 0, 1)

    columns_to_drop = ["DataYear", "PropertyName", "ListOfAllPropertyUseTypes", "Address", "City", "State",
                       "TaxParcelIdentificationNumber", "YearsENERGYSTARCertified", "DefaultData",
                       "Comments", "Latitude", "Longitude"]
    all_data_v1 = drop_selected_features(raw_data, columns_to_drop)
    print("v1:", all_data_v1.shape)

    # all_data_v2 = fill_property_use_type_GFA(all_data_v1)
    all_data_v2 = filling_property_use_type(all_data_v1)
    print("v2:", all_data_v2.shape)

    all_data_v3 = drop_outliers_based_on_dataset(all_data_v2)
    print("v3:", all_data_v3.shape)

    features_with_nan = ["SiteEUI(kBtu/sf)", "SiteEUIWN(kBtu/sf)"]
    all_data_v4 = drop_buildings_subset_nan(all_data_v3, features_with_nan)
    all_data_v4 = fill_nan_column_by_value(all_data_v4, "ENERGYSTARScore", -1)
    print("v4:", all_data_v4.shape)

    all_data_v5 = removing_outliers(all_data_v4, "SourceEUIWN(kBtu/sf)", 0, less_than_or_equal=False)
    print("v5:", all_data_v5.shape)

    print(
        "The computed remaining energy is the absolute value of the percentage of total energy that remaining energy has.")
    print(
        "The computation of the remaining energy is based on a hypothesis that the site energy is the sum of electricity, steam and natural gas.")
    all_data_v6 = compute_TotalEnergy(all_data_v5).sort_values(by="RemainingEnergy(%)", ascending=False)
    print("v6:", all_data_v6.shape)

    all_data_v7 = removing_outliers(all_data_v6, "RemainingEnergy(%)", 0.01, less_than_or_equal=True)
    print("v7:", all_data_v7.shape)

    columns_to_categorize = ["DataYear", "BuildingType", "PrimaryPropertyType", "ZipCode", "CouncilDistrictCode",
                             "YearBuilt", "LargestPropertyUseType", "SecondLargestPropertyUseType",
                             "ThirdLargestPropertyUseType"]
    all_data_v7 = assign_type_column(all_data_v7, columns_to_categorize, "category")
    print("We have changed the type of the categorical features to 'category'.")

    # We reset the index
    all_data_vf = all_data_v7.reset_index(drop=True)
    print("Final shape of our dataset : ", all_data_vf.shape)

    print("End of pipeline.")
    return all_data_vf


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


def verify_PropertyGFA(data_frame):
    """

    :param data_frame:
    :return:

    :UC: 100% filled df[["PropertyGFATotal", "PropertyGFABuilding(s)", "PropertyGFAParking"]]
    """
    df = data_frame.copy()
    df = df[["PropertyGFATotal", "PropertyGFABuilding(s)", "PropertyGFAParking"]]
    for index, row in df.iterrows():
        if row["PropertyGFATotal"] != row["PropertyGFABuilding(s)"] + row["PropertyGFAParking"]:
            print("HERE : ", index, row)
    print("End of checking.")


def fill_property_use_type_GFA(data_frame):
    # 1) dropna for LargestPropertyUseTypeGFA

    rows_to_drop = data_frame[data_frame["LargestPropertyUseTypeGFA"].isna()]
    index_to_drop = rows_to_drop.index

    # 2) we drop all the rows for which we have a NaN for LargestPropertyUseTypeGFA
    df = data_frame.copy()
    df = df.drop(index=index_to_drop)

    # 3) We fill the NaN for SecondLargestPropertyUseTypeGFA and ThirdLargestPropertyUseTypeGFA
    df["SecondLargestPropertyUseTypeGFA"] = df["SecondLargestPropertyUseTypeGFA"].fillna(0)
    df["ThirdLargestPropertyUseTypeGFA"] = df["ThirdLargestPropertyUseTypeGFA"].fillna(0)

    print("This won't work if you set the category before this function.")
    df["SecondLargestPropertyUseType"] = df["SecondLargestPropertyUseType"].fillna("None")
    df["ThirdLargestPropertyUseType"] = df["ThirdLargestPropertyUseType"].fillna("None")

    return df


def drop_outliers_based_on_dataset(data_frame):
    """

    :param data_frame:
    :return:
    """
    df = data_frame.copy()

    # ComplianceStatus
    print("Before removing the non compliant buildings :", df.shape)
    non_compliant_status = ['Error - Correct Default Data', 'Missing Data', 'Non-Compliant']
    df = df[~df["ComplianceStatus"].isin(non_compliant_status)]
    print("After removing the non compliant buildings :", df.shape)

    # Outliers
    print("Before removing the 32 outliers :", df.shape)
    df = df[df["Outlier"].isna()]
    print("After removing the outliers :", df.shape)

    print("We delete the columns ComplianceStatus and Outlier.")
    df = drop_selected_features(df, ["ComplianceStatus", "Outlier"])

    return df


def removing_outliers(data_frame, feature, ceiling, less_than_or_equal=True):
    """

    :param data_frame:
    :param energy_feature:
    :return:
    """
    if less_than_or_equal:
        df = data_frame[data_frame[feature] <= ceiling]
    else:
        df = data_frame[data_frame[feature] >= ceiling]
    return df


def drop_buildings_subset_nan(data_frame, features_to_check):
    """

    :param data_frame:
    :param features_to_check:
    :return:
    """
    df = data_frame.dropna(subset=features_to_check)
    return df


def fill_nan_column_by_value(data_frame, column, value):
    """

    :param data_frame:
    :return:
    """
    df = data_frame.copy()
    df[column] = df[column].fillna(value)
    return df


def compute_TotalEnergy(data_frame):
    """

    :param data_frame:
    :return:

    :UC: 100% filled df[["SiteEnergyUse(kBtu)", "Electricity(kBtu)", "SteamUse(kBtu)", "NaturalGas(kBtu)"]]
    """
    df = data_frame.copy()
    # 1) We had a column that computes the difference between the Total Energy and Electricity, SteamUse and NaturalGas
    df["RemainingEnergy(kBtu)"] = df["SiteEnergyUse(kBtu)"] - (
            df["Electricity(kBtu)"] + df["SteamUse(kBtu)"] + df["NaturalGas(kBtu)"])
    # 2) we take the absolute vaue of the difference and round up to the superior unit.
    df["RemainingEnergy(%)"] = round(abs(df["RemainingEnergy(kBtu)"] / df["SiteEnergyUse(kBtu)"] * 100),
                                     1)  # abs adds 5 buildings

    return df


def assign_type_column(data_frame, columns, new_type):
    """

    :param data_frame:
    :param columns:
    :param new_type:
    :return:
    """
    # Assigning a new type to the columns selected
    df = data_frame.copy()
    df[columns] = df[columns].astype(new_type)
    return df


def map_neighborhoods(data_frame):
    """

    :param data_frame:
    :return:
    """
    df = data_frame.copy()
    mapper = {}
    for neighborhood in df["Neighborhood"].unique().tolist():
        # we change it to the same string in lower case
        mapper[neighborhood] = neighborhood.lower()
        # special cases
        if neighborhood.lower() == "delridge neighborhoods":
            mapper[neighborhood] = "delridge"
    print(mapper)
    for old_neighborhood in mapper.keys():
        replace_value_for_a_feature(df, "Neighborhood", old_neighborhood, mapper[old_neighborhood])

    return df


property_use_types_columns = ['SecondLargestPropertyUseType',
                              'LargestPropertyUseType',
                              'ThirdLargestPropertyUseType',
                              'PrimaryPropertyType']

usetype_dict = {'Retail Store': 'Retail',
                'Supermarket/Grocery Store': 'Retail',
                'Repair Services (Vehicle, Shoe, Locksmith, etc)': 'Retail',
                'Automobile Dealership': 'Retail',
                'Convenience Store without Gas Station': 'Retail',
                'Personal Services': 'Retail',
                'Enclosed Mall': 'Retail',
                'Strip Mall': 'Retail',
                'Wholesale Club/Supercenter': 'Retail',
                'Other - Mall': 'Retail',
                'Supermarket / Grocery Stor': 'Retail',

                'Food Sales': 'Leisure',
                'Restaurant': 'Leisure',
                'Other - Restaurant/Bar': 'Leisure',
                'Food Service': 'Leisure',
                'Worship Facility': 'Leisure',
                'Other - Recreation': 'Leisure',
                'Other - Entertainment/Public Assembly': 'Leisure',
                'Performing Arts': 'Leisure',
                'Bar/Nightclub': 'Leisure',
                'Movie Theater': 'Leisure',
                'Museum': 'Leisure',
                'Social/Meeting Hall': 'Leisure',
                'Fitness Center/Health Club/Gym': 'Leisure',
                'Lifestyle Center ': 'Leisure',
                'Fast Food Restaurant': 'Leisure',

                'Multifamily Housing': 'Hotel/Senior Care/Housing',
                'Other - Lodging/Residential': 'Hotel/Senior Care/Housing',
                'Residence Hall/Dormitory': 'Hotel/Senior Care/Housing',
                'Hotel': 'Hotel/Senior Care/Housing',
                'Senior Care Community': 'Hotel/Senior Care/Housing',
                'Residential Care Facility': 'Hotel/Senior Care/Housing',
                'High-Rise Multifamily': 'Hotel/Senior Care/Housing',

                'Medical Office': 'Health',

                'Other - Services': 'Office',
                'Bank Branch': 'Office',
                'Financial Office': 'Office',
                'Other - Public Services': 'Office',

                'K-12 School': 'Education',
                'Other - Education': 'Education',
                'Vocational School': 'Education',
                'Adult Education': 'Education',
                'Pre-school/Daycare': 'Education',
                'University': 'Education',
                'College/University': 'Education',
                'Library': 'Education'
                }


def mapping_property_use_type(data_frame, property_use_types_columns, usetype_dict):
    """

    :param data_frame:
    :return:
    """
    data = data_frame.copy()
    print("Before")
    print(data[property_use_types_columns].nunique().sort_values())

    print("After")
    for column in property_use_types_columns:
        data[column] = data[column].replace(usetype_dict)

    print(data[property_use_types_columns].nunique().sort_values())
    return data


if __name__ == '__main__':
    # Starting time
    t0 = time()
    cleaning_pipeline()
    # End of pipeline time
    t1 = time()
    print("computing time : {:8.6f} sec".format(t1 - t0))
    print("computing time : " + strftime('%H:%M:%S', gmtime(t1 - t0)))
