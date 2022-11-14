import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

GM_COMPONENTS = 200
PERC_THRESHOLD = 1
NUM_FEATURES = 200
GBR_ESTIMATORS = 500


def load_data(use_imp_data=False, save_imp_data=True, gm_components=10):
    if use_imp_data:
        x_train_imp = pd.read_csv("data/x_train_imp.csv")
        test_imp = pd.read_csv("data/test_imp.csv")
        y_train = pd.read_csv("data/y_train.csv").drop("id", axis=1)

        id = test_imp["id"].to_numpy()
        test_imp = test_imp.drop("id", axis=1).to_numpy()

        return x_train_imp.to_numpy(), test_imp, id, y_train.to_numpy()

    else:
        x_train = pd.read_csv("data/X_train.csv").drop("id", axis=1)
        y_train = pd.read_csv("data/y_train.csv").drop("id", axis=1).to_numpy()
        test = pd.read_csv("data/X_test.csv")

        id = test["id"].to_numpy()
        test = test.drop("id", axis=1)

        imp = get_imputer(x_train, gm_components)
        x_train_imp = imp.transform(x_train)
        test_imp = imp.transform(test)

        if save_imp_data:
            pd.DataFrame(x_train_imp).to_csv("data/x_train_imp.csv", index=False)
            test_df = pd.DataFrame(test_imp)
            test_df["id"] = id
            test_df.to_csv("data/test_imp.csv", index=False)
        return x_train_imp, test_imp, id, y_train


def get_imputer(x_train, k_neighbors, max_iter=10):
    imp = SimpleImputer(strategy="median")
    imp.fit(x_train)
    return imp


def filter_outliers(x_train, y_train, threshold, n_components):
    gm = train_outlier_detection_model(x_train, n_components=n_components, random=False)
    mask = get_outlier_mask(x_train, gm, threshold)
    return x_train[np.nonzero(mask)[0]], y_train[np.nonzero(mask)[0]]


def train_outlier_detection_model(x_train, n_components, random=True):
    if random:
        gm = BayesianGaussianMixture(n_components=n_components)
        # gm = GaussianMixture(n_components=n_components)
    else:
        gm = BayesianGaussianMixture(n_components=n_components, random_state=0)
        # gm = GaussianMixture(n_components=n_components, random_state=0)
    gm.fit(x_train)
    return gm


def get_outlier_mask(x_train, gm, threshold, print_stats=False):

    scores = gm.score_samples(x_train)
    perc = np.percentile(scores, threshold)
    if print_stats:
        print(f"The threshold of the score is {perc:.2f}")
    return scores > perc


def select_features(x_train, y_train, x_test, k, score_func=None, use_mutual_info=False):

    if use_mutual_info and score_func is None:
        score_func = mutual_info_regression
    elif score_func is None:
        score_func = f_regression

    f_selector = get_f_selector(x_train, y_train, k, score_func)
    return f_selector.transform(x_train), f_selector.transform(x_test)


def get_regressor(x_train, y_train, regressor_steps):
    clf = GradientBoostingRegressor(loss="squared_error", n_estimators=regressor_steps, max_depth=5)
    clf.fit(x_train, y_train)
    return clf


def get_f_selector(x_train, y_train, k, score_func):
    f_selector = SelectKBest(score_func=score_func, k=k)
    f_selector.fit(x_train, y_train)
    return f_selector


def save_submission(id, y_pred, name="submission.csv"):
    submission = pd.DataFrame()
    submission["id"] = id
    submission["y"] = y_pred
    submission.to_csv(f"{name}", index=False)


if __name__ == "__main__":
    # Load data
    x_train, x_test, test_id, y_train = load_data(use_imp_data=False)
    print("Data loaded")

    # Normalize data
    scaler = QuantileTransformer(output_distribution="normal", random_state=0).fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train, y_train = filter_outliers(
        x_train,
        y_train,
        threshold=PERC_THRESHOLD,
        n_components=GM_COMPONENTS,
        local_outlier_factor=False,
    )

    y_train, x_train = filter_outliers(
        y_train,
        x_train,
        n_components=1,
        threshold=1,
        local_outlier_factor=False,
    )

    x_train, x_test = select_features(x_train, y_train, x_test, 200, use_mutual_info=False)

    regressor = get_regressor(x_train, y_train, GBR_ESTIMATORS)
    y_pred = regressor.predict(x_test)
    save_submission(test_id, y_pred)
