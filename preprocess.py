import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Set file name
raw_dataset_file = "./dataset/2016_Building_Energy_Benchmarking.csv"

def load_data():
    """

    :return:
    """

    # Load raw data
    raw_dataset = pd.read_csv(raw_dataset_file)
    return raw_dataset

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


def preprocess_features(raw_data):
    """

    :param raw_data:
    :return:
    """
    columns_to_drop = ["PropertyName", "ListOfAllPropertyUseTypes", "Address", "City", "State", "TaxParcelIdentificationNumber", "Neighborhood", "YearsENERGYSTARCertified", "DefaultData", "Comments", "Latitude", "Longitude"]
    all_data_v1 = drop_selected_features(raw_data, columns_to_drop)

    all_data_v2 = fill_property_use_type_GFA(all_data_v1)

    all_data_v3 = drop_outliers_based_on_dataset(all_data_v2)

    features_with_nan = ["SiteEUI(kBtu/sf)", "SiteEUIWN(kBtu/sf)"]
    all_data_v4 = drop_buildings_subset_nan(all_data_v3, features_with_nan)
    all_data_v4 = fill_nan_column_by_value(all_data_v4, "ENERGYSTARScore", -1)

    all_data_v5 = removing_energy_outliers(all_data_v4, "SourceEUIWN(kBtu/sf)", 0)



    return


def drop_selected_features(data_frame, list_features_to_drop):
    """

    :param data_frame:
    :param list_features_to_drop:
    :return:
    """
    df = data_frame.drop(columns=list_features_to_drop)
    return df


def fill_property_use_type_GFA(data_frame):
    """

    :param data_frame:
    :return:
    """
    # 1) dropna for LargestPropertyUseTypeGFA
    rows_to_drop = data_frame[data_frame["LargestPropertyUseTypeGFA"].isna()]
    index_to_drop = rows_to_drop.index

    # 2) we drop all the rows for which we have a NaN for LargestPropertyUseTypeGFA
    df = data_frame.copy()
    df = df.drop(index=index_to_drop)

    # 3) We fill the NaN for SecondLargestPropertyUseTypeGFA and ThirdLargestPropertyUseTypeGFA
    df["SecondLargestPropertyUseTypeGFA"] = df["SecondLargestPropertyUseTypeGFA"].fillna(0)
    df["ThirdLargestPropertyUseTypeGFA"] = df["ThirdLargestPropertyUseTypeGFA"].fillna(0)

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


def removing_energy_outliers(data_frame, energy_feature, ceiling):
    """

    :param data_frame:
    :param energy_feature:
    :return:
    """
    df = data_frame[data_frame[energy_feature] >= ceiling]
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


if __name__ == '__main__':
    pass