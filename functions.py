import matplotlib.pyplot as plt
import seaborn as sns


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

def log_transformation(data_frame, features_to_predict):
    """

    :param data_frame:
    :param features_to_predict:
    :return:
    """
    df = data_frame.copy()
    print('Log2-transformation of the variables to predict.')
    for feature in features_to_predict:
        # we name the new variable
        log_feature = "Log2-{}".format(feature)
        # we add the transformed variable to our dataframe
        df[log_feature] = np.log2(1 + df[feature])
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


if __name__ == '__main__':
    densite()
