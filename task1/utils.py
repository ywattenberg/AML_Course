import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression


def dump_scores_to_file(scores, path):
    if os.path.exists(path):
        with open(path, "r+") as f:
            data = json.load(f)
            runs = data["runs"]
            runs[len(runs)] = scores
            data["optimal"].append(scores[max(scores.keys())])
            f.seek(0)
            json.dump(data, f, indent=4)
    else:
        with open(path, "w") as f:
            data = {"optimal": [scores[max(scores.keys())]], "runs": {0: scores}}
            json.dump(data, f, indent=4)


def safe_cross_scores(score_map):
    for key, value in score_map.items():
        dump_scores_to_file(value, f"results/{key}.json")


def get_imputer(x_train, k_neighbors, max_iter=10):
    imp = IterativeImputer(
        max_iter=max_iter, n_nearest_features=k_neighbors, random_state=0
    )
    imp.fit(x_train)
    return imp


def load_data(use_imp_data=False, safe_imp_data=True):
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

        imp = get_imputer(x_train)
        x_train_imp = imp.transform(x_train)
        test_imp = imp.transform(test)

        if safe_imp_data:
            pd.DataFrame(x_train_imp).to_csv("data/x_train_imp.csv", index=False)
            test_df = pd.DataFrame(test_imp)
            test_df["id"] = id
            test_df.to_csv("data/test_imp.csv", index=False)
        return x_train_imp, test_imp, id, y_train


def train_outlier_detection_model(x_train, n_components, random=True):
    if random:
        gm = GaussianMixture(n_components=n_components)
    else:
        gm = GaussianMixture(n_components=n_components, random_state=0)
    gm.fit(x_train)
    return gm


def get_outlier_mask(x_train, gm, threshold, print_stats=False):

    scores = gm.score_samples(x_train)
    perc = np.percentile(scores, threshold)
    if print_stats:
        print(f"The threshold of the score is {perc:.2f}")
    return scores > perc


def get_r2_score(x_train_cv, x_test_cv, y_train_cv, y_test_cv, regressor_steps):
    clf = GradientBoostingRegressor(loss="squared_error", n_estimators=regressor_steps)
    clf.fit(x_train_cv, y_train_cv)
    y_pred = clf.predict(x_test_cv)
    curr_score = r2_score(y_test_cv, y_pred)
    return curr_score


def get_split_from_index(x_train, y_train, train_index, test_index):
    x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]
    y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
    return x_train_cv, x_test_cv, y_train_cv, y_test_cv


def get_split_from_index(x_train, y_train, train_index, test_index):
    x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]
    y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
    return x_train_cv, x_test_cv, y_train_cv, y_test_cv


def filter_outliers(x_train, y_train, threshold, n_components):
    gm = train_outlier_detection_model(x_train, n_components=n_components)
    mask = get_outlier_mask(x_train, gm, threshold)
    return x_train[np.nonzero(mask)[0]], y_train[np.nonzero(mask)[0]]


def get_f_selector(x_train, y_train, k, score_func):
    f_selector = SelectKBest(score_func=score_func, k=k)
    f_selector.fit(x_train, y_train)
    return f_selector


def select_features(
    x_train, y_train, x_test, k, score_func=None, use_mutual_info=False
):

    if use_mutual_info and score_func is None:
        score_func = mutual_info_regression
    elif score_func is None:
        score_func = f_regression

    f_selector = get_f_selector(x_train, y_train, k, score_func)
    return f_selector.transform(x_train), f_selector.transform(x_test)


def train_outlier_detection_model(x_train, n_components):
    gm = GaussianMixture(n_components=n_components)
    gm.fit(x_train)
    return gm
