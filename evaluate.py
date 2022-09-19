import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time, strftime, gmtime

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, median_absolute_error

pd.set_option('display.float_format', lambda x: '%.5f' % x)

global results
results = pd.DataFrame({})

global results_cv
results_cv = pd.DataFrame({})

global prediction_time
prediction_time = pd.DataFrame({})


def evaluate_regression(model_name, result, y_test, y_pred):
    """
    :param model_name:
    :param result:
    :param y_test:
    :param y_pred:

    :UC: y_test must be a Pandas Series with a label

    """
    print("Prediction for : ", y_test.name)  # name Pandas Series
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    median_ae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    result = pd.concat([result, pd.DataFrame({"Model": [model_name],
                                              "RMSE": [rmse],
                                              "MSE": [mse],
                                              "MAE": [mae],
                                              "Median Absolute Error": [median_ae],
                                              "R² = 1 - RSE": [r2]})])
    # we sort the datafraeme of results by best : by=["R² = 1 - RSE", "RMSE", "MAE"]
    result = result.sort_values(by=["R² = 1 - RSE"], ascending=False)
    display(result)

    # 2) graph
    plt.title("Scatter plot of the predicted values as a function of the true values.")
    plt.legend("If the prediction was good, we would see a line.")
    plt.figure(0)
    plt.scatter(y_test, y_pred, color='coral')

    plt.title("Distribution of the prediction errors")
    err_hist = np.abs(y_test - y_pred)
    plt.figure(1)
    plt.hist(err_hist, bins=50, color='steelblue')

    return result


def summary_results_cv(model_name, mean_cv_score, result_cv):
    """
    Mean cross-validated score of the best_estimator
    """
    print("Results Cross-Validated")
    result_cv = pd.concat([result_cv, pd.DataFrame({"Model": [model_name],
                                                    "Mean CV R²": [mean_cv_score]})])

    result_cv = result_cv.sort_values(by=["Mean CV R²"], ascending=False)
    display(result_cv)

    return result_cv


def compute_prediction_time(model, model_name, X, df_prediction_time):
    """

    """
    # Starting time
    t0 = time()
    model.predict(X)
    t1 = time()
    pred_time = t1 - t0
    print("Prediction time : {:8.6f} sec".format(pred_time))

    df_prediction_time = pd.concat([df_prediction_time, pd.DataFrame({"Model name": [model_name],
                                                                      "Prediction time": [pred_time]
                                                                      })])
    df_prediction_time = df_prediction_time.sort_values(by=["Prediction time"], ascending=True)

    # display(df_prediction_time)
    return df_prediction_time


def display_barplot_errors(results, baseline_model, title, metric):
    """

    """
    sns.set(rc={'figure.figsize': (14, 10)})
    plt.title(title)
    graph = sns.barplot(x=results["Model"],
                        y=results[metric])

    baseline_rmse = results[metric][results["Model"] == baseline_model][0]  # we take the value inside
    # Drawing a horizontal line at point 1.25
    graph.axhline(baseline_rmse, color="b")


def display_barplot_prediction_time(df_prediction_time):
    title = "Barplot of prediction time per model"
    sns.set(rc={'figure.figsize': (14, 10)})
    plt.title(title)
    name_model = df_prediction_time["Model name"]
    graph = sns.barplot(x=name_model,
                        y=df_prediction_time["Prediction time"])

    min_time = df_prediction_time["Prediction time"].min()  # we take the value inside
    # Drawing a horizontal line at point 1.25
    graph.axhline(min_time, color="b")


# NOT USED FOR NOW
def display_prediction(y_test, y_pred):
    # plt.scatter(y_test, y_pred, color='coral')

    sizes = {}  # clé : coordonnées ; valeur : nombre de points à ces coordonnées
    for (yt, yp) in zip(list(y_test), list(y_pred)):
        if (yt, yp) in sizes:
            sizes[(yt, yp)] += 1
        else:
            sizes[(yt, yp)] = 1

    keys = sizes.keys()
    plt.scatter(
        [k[0] for k in keys],  # vraie valeur (abscisse)
        [k[1] for k in keys],  # valeur predite (ordonnee)
        s=[sizes[k] for k in keys],  # taille du marqueur
        color='coral', alpha=0.8)


if __name__ == '__main__':
    pass
