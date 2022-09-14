# Chargement des librairies
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Divers
from time import time, strftime, gmtime

# Affichage max des colonnes
pd.set_option('display.max_columns', 50)

import warnings
warnings.filterwarnings(action="ignore")


def pipeline_1():
    """

    """
    print("First part")
    # Heure démarrage
    t0 = time()

    # Chargement des données
    input_path = r'C:\Users\oumei\Documents\OC_projets\P4\P4_Data_Science_OpenClassrooms'
    df_2016 = pd.read_csv(input_path + "/dataset/source/2016_Building_Energy_Benchmarking.csv")

    df = df_2016.copy()
    # Réinitialisation de l'index
    df.reset_index(drop=True, inplace=True)

    # Imputation des valeurs manquantes
    df_clean = df.copy()

    # Colonnes à ne pas garder, trop de données manquantes et indépendantes de la problématique
    to_drop = ['Comments', 'Outlier', 'YearsENERGYSTARCertified', 'ENERGYSTARScore']
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

    # Suppression des lignes contenant des NA
    df_clean.dropna(inplace=True)

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


    replace_value_for_a_feature(df_clean, "NumberofBuildings", 0, 1)


    # Valeurs manquantes restantes
    print(
        "Après nettoyage des valeurs manquantes, le fichier comporte {} lignes et {} colonnes".format(df_clean.shape[0],
                                                                                                      df_clean.shape[
                                                                                                          1]))
    print("Valeurs manquantes restantes : " + str(df_clean.isnull().sum().sum()))

    # Variables qualitatives => type objet
    col_object = ['OSEBuildingID', 'CouncilDistrictCode']

    # Contrôle du type des variables
    df_types = df_clean.join(df['ENERGYSTARScore'])
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

    # Description des variables
    df_correct = df_types.copy()
    print(df_correct.describe().transpose())

    # Liste des colonnes numériques
    cols_numeric = df_correct.select_dtypes(include='number').columns

    # On ne garde que les observations sans valeurs négatives
    df_positive = df_correct.loc[:, [x for x in cols_numeric if x not in ['Longitude', 'Latitude', 'ENERGYSTARScore']]]
    df_correct = df_correct[(df_positive.select_dtypes(include='number') >= 0).all(axis=1)].copy()

    # Valeurs négatives
    print("Après nettoyage des valeurs négatives, le fichier comporte {} lignes".format(df_correct.shape[0]))

    # On imputele vrai total à la surface totale
    df_correct['PropertyGFATotal'] = df_correct['PropertyGFAParking'] + df_correct['PropertyGFABuilding(s)']

    # On ne garde que les observations avec le status "Compliant" (conforme)
    df_correct = df_correct[df_correct['ComplianceStatus'] == 'Compliant']

    # Création d'une variable pour les autres sources d'énérgie
    df_correct['Others(kBtu)'] = df_correct['SiteEnergyUse(kBtu)'] - df_correct['SteamUse(kBtu)'] - df_correct[
        'Electricity(kBtu)'] - df_correct['NaturalGas(kBtu)']

    # 2) we take the absolute value of the difference and round up to the superior unit.
    df_correct["RemainingEnergy(%)"] = round(
        abs(df_correct["Others(kBtu)"] / df_correct["SiteEnergyUse(kBtu)"] * 100),
        1)  # abs adds 5 buildings

    print("Shape : 3219, 44", df_correct.shape)

    # Suppression des observations avec total énergie qui est significativement inférieur à la somme des composantes
    # ce qui correspond à "Others(kbtu)" inférieur à 0,
    # car elle vient d'être calculée par différence entre le total énérgie et ses 3 premières composantes

    # On supprime les observations avec des consommations d'autres énergie inférieures à 0 ==> 823 lignes
    to_drop1 = df_correct['Others(kBtu)'] < 0

    # On ne supprime que si le montant inférieur à 0 est significatif par rapport à la consommation totale ==> 36 lignes
    to_drop2 = np.abs(df_correct['Others(kBtu)']) > (0.001 * df_correct['SiteEnergyUse(kBtu)'])

    to_drop3 = np.abs(df_correct["Others(kBtu)"]) > (0.01 * df_correct['SiteEnergyUse(kBtu)'])

    # suppression des consommations d'autres énérgie négatives lorsque significatives
    to_drop = to_drop1 & to_drop2
    df_correct = df_correct[~to_drop3] ####

    # Imputation à 0 des consommations d'autres énérgie négatives lorsque non significatives
    df_correct.loc[df_correct['Others(kBtu)'] < 0, 'Others(kBtu)'] = 0

    # Renomme la variable de consommation totale d'énergie
    df_correct = df_correct.rename(columns={'SiteEnergyUse(kBtu)': 'Total_energy'})

    print("Shape : 3214, 44", df_correct.shape)

    print(df_correct["BuildingType"].value_counts())
    print(df_correct["PrimaryPropertyType"].value_counts())

    df_correct.loc[df_correct.BuildingType == "Nonresidential wa", "BuildingType"] = "Nonresidential"
    df_correct.loc[df_correct.PrimaryPropertyType == "Restaurant\n", "PrimaryPropertyType"] = "Restaurant"
    df_correct.loc[df_correct.PrimaryPropertyType == "Non-refrigerated warehouse", "PrimaryPropertyType"] = "Warehouse"

    df_correct = mapping_property_use_type(df_correct, property_use_types_columns, usetype_dict)

    print("Second part")
    df_explo = df_correct.copy()

    # Nombre de modalités des varaibles qualitatives
    df_explo.describe(exclude='number').loc['unique', :]

    def box_categorical(df, col_categorical="PrimaryPropertyType", col_numeric="TotalGHGEmissions"):
        """
        Input : dataframe, colonne d'une variable qualitative, colonne d'une variable quantitative
        output : boxplot de la variable quantitative en fonction de la variable qualitative
        """
        # Largeur du graphe en fonction du nombre de modalités de col_categorical
        xsize = min(round(df[col_categorical].drop_duplicates().shape[0] / 2), 20)

        # liste triée des zones selon médiane des valeurs
        order = df.groupby(col_categorical)[col_numeric].median().sort_values(ascending=False).index

        # Graphique
        fig, ax = plt.subplots(figsize=(xsize, 4))
        bp = sns.boxplot(x=col_categorical, y=col_numeric, data=df, order=order, showfliers=False)
        bp.set_xticklabels(bp.get_xticklabels(), rotation=45)
        bp.set_title(col_categorical)
        plt.xticks(rotation=90)
        plt.show()

    # Variable BuildingType / émission CO2
    box_categorical(df_explo, "BuildingType")

    # Variable Neighborhood / émission CO2
    box_categorical(df_explo, "Neighborhood")

    # Variable PrimaryPropertyType" / émission CO2
    box_categorical(df_explo, "PrimaryPropertyType")

    # Changement de valeurs pour les valeurs avec faibles occurences
    df_explo.loc[df_explo['PrimaryPropertyType'] == 'Restaurant\n', 'PrimaryPropertyType'] = 'Restaurant'
    df_explo.loc[df_explo['PrimaryPropertyType'] == 'Non-refrigerated warehouse', 'PrimaryPropertyType'] = 'Warehouse'

    # Liste des variables qualitatives pour la modélisation
    select_non_numeric = ['BuildingType', 'PrimaryPropertyType', 'Neighborhood']

    # Sélection des variables numériques
    df_numeric = df_correct.select_dtypes(include='number').copy()

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

    # Heatmap des corrélations
    def correlations(df, largeur=15, hauteur=15):
        """
        Input : dataframe, largeur du graphique, hauteur du graphique
        output : heatmap de la table des corrélations
        """
        # Table des corrélations
        corr = df.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Add the mask to the heatmap
        plt.figure(figsize=(largeur, hauteur))

        sns.heatmap(corr, center=0, cmap=sns.color_palette("RdBu_r", 7), linewidths=1,
                    annot=True, annot_kws={"size": 9}, fmt=".02f")

        plt.title('Tableau des corrélations', fontsize=18)
        plt.xticks(fontsize=12, rotation=90)
        plt.yticks(fontsize=12)
        plt.show()

    # Courbes de densité des variables
    densite(df_numeric.loc[:, df_numeric.columns != 'ENERGYSTARScore'])

    # transformation log pour les variables avec un skewness élevé
    df_numeric_log = pd.DataFrame()

    for i, val in enumerate(df_numeric.columns.tolist()):
        if (df_numeric[val].skew() > 2):
            df_numeric_log[val] = np.log(df_numeric[val] + 1)
        else:
            df_numeric_log[val] = df_numeric[val]

    # Courbes de densité des variables après transformation
    densite(df_numeric_log.loc[:, df_numeric.columns != 'ENERGYSTARScore'])

    # Table des corrélations
    correlations(df_numeric_log)

    # Suppression des variables
    low_correlated = ['Latitude', 'Longitude', 'DataYear']
    high_correlated = ['Electricity(kWh)', 'NaturalGas(therms)', 'PropertyGFABuilding(s)', 'LargestPropertyUseTypeGFA']
    target_linked = ['GHGEmissionsIntensity', 'SiteEnergyUseWN(kBtu)',
                     'SiteEUI(kBtu/sf)', 'SiteEUIWN(kBtu/sf)', 'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)']

    to_drop = list(set(low_correlated + high_correlated + target_linked))
    df_select = df_numeric_log.drop(to_drop, axis=1).copy()

    # Table des corrélations des varaibles numériques retenues pour les modélisations
    correlations(df_select, 8, 8)

    # Jointure des colonnes pour modélisation
    df_export = df_select.join(df_explo[select_non_numeric])

    # type float64 pour toutes les variables numériques
    numeric_col = df_export.select_dtypes(include='number').columns
    df_export[numeric_col] = df_export[numeric_col].astype(np.float64)

    # Sauvegarde du dataframe en csv pour le notebook d'exploration
    df_export.to_csv(
        r'C:\Users\oumei\Documents\OC_projets\P4\P4_Data_Science_OpenClassrooms\dataset\output/data_cleaned.csv',
        encoding='utf-8', index=False)
    df_correct.to_csv(
        r'C:\Users\oumei\Documents\OC_projets\P4\P4_Data_Science_OpenClassrooms\dataset\output/data_fixed.csv',
        encoding='utf-8', index=False)

    t1 = time()
    print("computing time : {:8.6f} sec".format(t1 - t0))
    print("computing time : " + strftime('%H:%M:%S', gmtime(t1 - t0)))


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
    print("Notebook 1")
    pipeline_1()

#%%
