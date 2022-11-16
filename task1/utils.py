import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import LocalOutlierFactor

enable_iterative_imputer


def train_pca_model(x_train, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(x_train)
    return pca


def pca_transform(x_train, x_test, n_components):
    pca = train_pca_model(x_train, n_components)
    return pca.transform(x_train), pca.transform(x_test)


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
    imp = SimpleImputer(strategy="median")
    imp.fit(x_train)
    return imp


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


def get_regressor(x_train, y_train, regressor_steps):
    clf = GradientBoostingRegressor(loss="squared_error", n_estimators=regressor_steps)
    clf.fit(x_train, y_train)
    return clf


def get_r2_score(x_train_cv, x_test_cv, y_train_cv, y_test_cv, regressor_steps):
    clf = get_regressor(x_train_cv, y_train_cv, regressor_steps)
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

#added an optinal parameter loacl_outlier_factor, if set to true, the local outlier factor is used instead of gmm or bgm
def filter_outliers(x_train, y_train, threshold, n_components, local_outlier_factor=False):
    if not local_outlier_factor:
        gm = train_outlier_detection_model(x_train, n_components=n_components)
        mask = get_outlier_mask(x_train, gm, threshold)
    else:
        mask = train_local_outlier_factor(x_train=x_train, n_neighbors=n_components)
    return x_train[np.nonzero(mask)[0]], y_train[np.nonzero(mask)[0]]

# Function to train the local outlier factor model
def train_local_outlier_factor(x_train, n_neighbors):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    mask = lof.fit_predict(x_train)
    return mask == 1


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
    gm = BayesianGaussianMixture(n_components=n_components)
    #gm = GaussianMixture(n_components=n_components)
    gm.fit(x_train)
    return gm


def save_submission(id, y_pred, name="submission.csv"):
    submission = pd.DataFrame()
    submission["id"] = id
    submission["y"] = y_pred
    submission.to_csv(f"{name}", index=False)