import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Set file name
raw_dataset_file = "./dataset/2016_Building_Energy_Benchmarking.csv"

def load_data(raw_dataset_file):
    """

    :return:
    """

    # Load raw data
    raw_dataset = pd.read_csv(raw_dataset_file)
    return raw_dataset

# Verification functions :

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

def verify_min_value(data_frame, all_features, ceiling):
    """

    :param data_frame:
    :param all_features:
    :param ceiling:
    :return:
    """

    features_with_negative_values = []
    for energy_feature in all_features:
        energy_min = data_frame[energy_feature].min()
        if energy_min < ceiling:
            print("For this feature :", energy_feature, "we have this negative value :", energy_min)
            features_with_negative_values.append(energy_feature)
    return features_with_negative_values

# Cleaning pipeline
def preprocess_features(raw_data):
    """

    :param raw_data:
    :return:
    """
    print("Starting cleaning pipeline.")
    print("Initial shape :", raw_data.shape)

    columns_to_drop = ["OSEBuildingID", "PropertyName", "ListOfAllPropertyUseTypes", "Address", "City", "State", "TaxParcelIdentificationNumber", "Neighborhood", "YearsENERGYSTARCertified", "DefaultData", "Comments", "Latitude", "Longitude"]
    all_data_v1 = drop_selected_features(raw_data, columns_to_drop)
    print("v1:", all_data_v1.shape)

    all_data_v2 = fill_property_use_type_GFA(all_data_v1)
    print("v2:", all_data_v2.shape)

    all_data_v3 = drop_outliers_based_on_dataset(all_data_v2)
    print("v3:", all_data_v3.shape)

    features_with_nan = ["SiteEUI(kBtu/sf)", "SiteEUIWN(kBtu/sf)"]
    all_data_v4 = drop_buildings_subset_nan(all_data_v3, features_with_nan)
    all_data_v4 = fill_nan_column_by_value(all_data_v4, "ENERGYSTARScore", -1)
    print("v4:", all_data_v4.shape)

    all_data_v5 = removing_outliers(all_data_v4, "SourceEUIWN(kBtu/sf)", 0, less_than_or_equal=False)
    print("v5:", all_data_v5.shape)

    print("The computed remaining energy is the absolute value of the percentage of total energy that remaining energy has.")
    print("The computation of the remaining energy is based on a hypothesis that the site energy is the sum of electricity, steam and natural gas.")
    all_data_v6 = compute_TotalEnergy(all_data_v5).sort_values(by="RemainingEnergy(%)", ascending=False)
    print("v6:", all_data_v6.shape)

    all_data_v7 = removing_outliers(all_data_v6, "RemainingEnergy(%)", 0.01, less_than_or_equal=True)
    print("v7:", all_data_v7.shape)

    # We reset the index
    all_data_vf = all_data_v7.reset_index(drop=True)
    print("Final shape of our dataset : ", all_data_vf.shape)

    print("End of pipeline.")
    return all_data_vf


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
    df["RemainingEnergy(kBtu)"] = df["SiteEnergyUse(kBtu)"] - (df["Electricity(kBtu)"] + df["SteamUse(kBtu)"] + df["NaturalGas(kBtu)"])
    # 2) we take the absolute vaue of the difference and round up to the superior unit.
    df["RemainingEnergy(%)"] = round(abs(df["RemainingEnergy(kBtu)"] / df["SiteEnergyUse(kBtu)"] * 100), 1) # abs adds 5 buildings

    return df

def save_cleaned_dataset(df, path):
    # Save
    df.to_csv(path, index=False)


if __name__ == '__main__':
    raw_df = load_data(raw_dataset_file)
    final_df = preprocess_features(raw_df)
    save_cleaned_dataset(final_df, "dataset/2016_Building_Energy_Cleaned.csv")