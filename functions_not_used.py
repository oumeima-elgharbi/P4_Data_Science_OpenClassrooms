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


def transforming_building_type(df):
    """
    8
    """
    print("___Transforming building type and primary property type___")
    data_frame = df.copy()
    print("Before :", data_frame.shape)

    print(data_frame["BuildingType"].value_counts())
    print(data_frame["PrimaryPropertyType"].value_counts())

    data_frame.loc[data_frame.BuildingType == "Nonresidential wa", "BuildingType"] = "Nonresidential"
    data_frame.loc[data_frame.PrimaryPropertyType == "Restaurant\n", "PrimaryPropertyType"] = "Restaurant"
    data_frame.loc[data_frame.PrimaryPropertyType == "Non-refrigerated warehouse", "PrimaryPropertyType"] = "Warehouse"

    print("After :", data_frame.shape)
    return data_frame