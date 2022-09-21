# from cleaning import *

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings(action="ignore")


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


# Description du dataframe
def extrems(df):
    """
    Input : dataframe
    output : description du dataframe
    """
    return df.describe().loc[['min', '25%', '50%', '75%', 'max'], :]


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


def remove_percentile_outliers(data_frame, upper_percentile, lower_percentile, features):
    """

    :param data_frame:
    :param upper_percentile:
    :param lower_percentile:
    :param features: (list) of two elements
    :return:
    """
    assert len(features) == 2, print("Features must contains two features only.")

    df = data_frame.copy()
    upper_outliers = df[(df[features[0]] > df[features[0]].quantile(upper_percentile)) & (
            df[features[1]] > df[features[1]].quantile(upper_percentile))]
    lower_outliers = df[(df[features[0]] < df[features[0]].quantile(lower_percentile)) & (
            df[features[1]] < df[features[1]].quantile(lower_percentile))]

    # for each building outlier, we save their index in the list called index_to_drop
    l_upper = upper_outliers.index.tolist()
    l_lower = lower_outliers.index.tolist()
    index_to_drop = list()
    index_to_drop.extend(l_upper)
    index_to_drop.extend(l_lower)

    print("We check that we have all the indexes to drop :", len(index_to_drop))
    print("We wanted to drop :", upper_outliers.shape[0] + lower_outliers.shape[0], "buildings.")
    df = df.drop(index=index_to_drop)
    return df


def drop_selected_features(data_frame, list_features_to_drop):
    """

    :param data_frame:
    :param list_features_to_drop:
    :return:
    """
    df = data_frame.drop(columns=list_features_to_drop)
    return df


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


def input_propertyGFATotal(df):
    """
    6.2)
    """
    print("___Inputting correct propertyGFATotal after removing negative values above___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)  ####

    # On imputele vrai total à la surface totale
    data_frame['PropertyGFATotal'] = data_frame['PropertyGFAParking'] + data_frame['PropertyGFABuilding(s)']

    print("After :", data_frame.shape)  ####
    return data_frame


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


def fill_nan_column_by_value(data_frame, column, value):
    """

    :param data_frame:
    :return:
    """
    df = data_frame.copy()
    df[column] = df[column].fillna(value)
    return df


def dropping_missing_values(df):
    """
    Step 3)
    :param df: (DataFrame)

    """
    print("___Dropping buildings with missing values___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    # Deleting rows containing NaN
    data_frame = data_frame.dropna()  # data_frame.dropna(inplace=True)

    print("After cleaning missing values, the file contains {} rows et {} columns.".format(data_frame.shape[0],
                                                                                           data_frame.shape[1]))
    print("Remaining missing values : " + str(data_frame.isnull().sum().sum()))

    print("After :", data_frame.shape)
    return data_frame


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
