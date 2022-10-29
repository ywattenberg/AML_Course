import os
import warnings
import json


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

IMP_NN = 32
LOAD_IMP_DATA = True
GM_COMPONENTS = 140
PERC_THRESHOLD = 10
LASSO_ALPHA = 5
GBR_ESTIMATORS = 200
D_COLS_ROUND = 16


def get_imputer(x_train, max_iter=10, k_neighbors=IMP_NN):
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


def train_outlier_detection_model(x_train, n_components=GM_COMPONENTS, random=True):
    if random:
        gm = GaussianMixture(n_components=n_components)
    else:
        gm = GaussianMixture(n_components=n_components, random_state=0)
    gm.fit(x_train)
    return gm


def get_outlier_mask(x_train, gm, threshold=PERC_THRESHOLD, print_stats=False):
    scores = gm.score_samples(x_train)
    perc = np.percentile(scores, threshold)
    if print_stats:
        print(f"The threshold of the score is {perc:.2f}")
    return scores > perc


def cross_validation_gmm_components(
    x_train,
    y_train,
    num_of_splits=5,
    regressor_steps=GBR_ESTIMATORS,
    threshold=PERC_THRESHOLD,
    begin=10,
    end=200,
    step=10,
):
    scores = {}
    kf = KFold(n_splits=num_of_splits, shuffle=True)
    for components in range(begin, end, step):
        score = 0
        print(f"Components: {components}")
        for train_index, test_index in kf.split(x_train):
            x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]
            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
            gm = train_outlier_detection_model(x_train_cv, n_components=components)

            mask = get_outlier_mask(x_train_cv, gm, threshold)
            x_train_cv = x_train_cv[np.nonzero(mask)[0]]
            y_train_cv = y_train_cv[np.nonzero(mask)[0]]

            clf = GradientBoostingRegressor(
                loss="squared_error", n_estimators=regressor_steps
            )
            clf.fit(x_train_cv, y_train_cv)
            y_pred = clf.predict(x_test_cv)
            curr_score = r2_score(y_test_cv, y_pred)
            score += curr_score
            print(f"Current score {components} components: {curr_score:.4f}")
        scores[score / num_of_splits] = components
        print("-" * 50)
        print(f"Final score for {components} components: {score / num_of_splits:.4f}")
        print("-" * 50)
    return scores


def cross_validation_gmm_threshold(
    x_train,
    y_train,
    num_of_splits=5,
    regressor_steps=GBR_ESTIMATORS,
    threshold=PERC_THRESHOLD,
    n_components=GM_COMPONENTS,
    begin=0,
    end=80,
    step=5,
):
    scores = {}
    kf = KFold(n_splits=num_of_splits, shuffle=True)
    models = []
    for train_index, test_index in kf.split(x_train):
        x_train_cv = x_train[train_index]
        models.append(
            train_outlier_detection_model(x_train_cv, n_components=n_components)
        )

    for threshold in range(begin, end, step):
        score = 0
        print(f"Threshold: {threshold}%")
        for idx, (train_index, test_index) in enumerate(kf.split(x_train)):
            x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]
            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

            mask = get_outlier_mask(x_train_cv, models[idx], threshold)
            x_train_cv = x_train_cv[np.nonzero(mask)[0]]
            y_train_cv = y_train_cv[np.nonzero(mask)[0]]

            clf = GradientBoostingRegressor(
                loss="squared_error", n_estimators=regressor_steps
            )
            clf.fit(x_train_cv, y_train_cv)
            y_pred = clf.predict(x_test_cv)
            curr_score = r2_score(y_test_cv, y_pred)
            score += curr_score
            print(f"Current score with threshold {threshold}%: {curr_score:.4f}")
        scores[score / num_of_splits] = threshold
        print("-" * 50)
        print(f"Final score with threshold {threshold}%: {score / num_of_splits:.4f}")
        print("-" * 50)
    return scores


def cross_validation_regression_num_regressors(
    x_train,
    y_train,
    num_of_splits=5,
    regressor_steps=GBR_ESTIMATORS,
    begin=100,
    end=1000,
    step=50,
):
    scores = {}
    kf = KFold(n_splits=num_of_splits, shuffle=True)
    for regressors in range(begin, end, step):
        score = 0
        print(f"Regressor steps: {regressors}")
        for train_index, test_index in kf.split(x_train):
            x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]
            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

            clf = GradientBoostingRegressor(
                loss="squared_error", n_estimators=regressors
            )
            clf.fit(x_train_cv, y_train_cv)
            y_pred = clf.predict(x_test_cv)
            curr_score = r2_score(y_test_cv, y_pred)
            score += curr_score
            print(f"Current score {regressors} regressors: {curr_score:.4f}")
        scores[score / num_of_splits] = regressors
        print("-" * 50)
        print(f"Final score for {regressors} regressors: {score / num_of_splits:.4f}")
        print("-" * 50)
    return scores


def filter_outliers(
    x_train, y_train, threshold=PERC_THRESHOLD, n_components=GM_COMPONENTS
):
    gm = train_outlier_detection_model(x_train, n_components=n_components)
    mask = get_outlier_mask(x_train, gm, threshold)
    return x_train[np.nonzero(mask)[0]], y_train[np.nonzero(mask)[0]]


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


def safe_cross_scores(
    component_scores=None, threshold_scores=None, regressor_scores=None
):
    if component_scores is not None:
        dump_scores_to_file(component_scores, "gmm_components.json")
    if threshold_scores is not None:
        dump_scores_to_file(threshold_scores, "gmm_threshold.json")
    if regressor_scores is not None:
        dump_scores_to_file(regressor_scores, "regressor_num.json")


if __name__ == "__main__":

    # Load data
    x_train, test, test_id, y_train = load_data(use_imp_data=True)
    print("Data loaded")

    # Normalize data
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    test = scaler.transform(test)

    # Select outlier detection model
    component_scores = cross_validation_gmm_components(x_train, y_train)
    print("Finished cross validation for components")
    best_num_of_components = component_scores[max(component_scores.keys())]
    print(f"Best model has {best_num_of_components} components")

    threshold_scores = cross_validation_gmm_threshold(
        x_train, y_train, n_components=best_num_of_components
    )
    print("Finished cross validation for threshold")
    best_threshold = threshold_scores[max(threshold_scores.keys())]
    print(f"Best model has {best_threshold} threshold")

    print(
        f"Filtering outliers with {best_num_of_components} components and threshold {best_threshold}%"
    )

    x_train, y_train = filter_outliers(
        x_train, y_train, threshold=best_threshold, n_components=best_num_of_components
    )

    num_regressor_score = cross_validation_regression_num_regressors(x_train, y_train)
    print("Finished cross validation for number of regressors")
    best_num_of_regressors = num_regressor_score[max(num_regressor_score.keys())]
    print(f"Best model has {best_num_of_regressors} regressors")

    safe_cross_scores(component_scores, threshold_scores, num_regressor_score)
    # Train model

    quit()

    k = x_train.shape[1] - 1

    best_score = -1
    best_columns = 0
    columns_to_drop = []
    component_scores = []
    features = []

    for i in range(0, k, D_COLS_ROUND):
        model = Lasso(alpha=LASSO_ALPHA)
        model.fit(x_train, y_train)
        coef = model.coef_

        print(f"Dropping columns: ")

        for j in range(D_COLS_ROUND):
            if i + j >= k:
                continue

            idx = np.argmin(np.abs(coef))
            coef = np.delete(coef, idx)

            columns_to_drop.append(idx)
            dropped_column = x_train[:, [idx]]

            x_train = np.delete(x_train, idx, 1)
            x_test = np.delete(x_test, idx, 1)

        clf = GradientBoostingRegressor(
            loss="squared_error", n_estimators=GBR_ESTIMATORS
        )
        clf.fit(x_train, y_train)

        print(f"New column count: {len(coef)}")
        score = r2_score(y_test, clf.predict(x_test))
        if best_score < score:
            best_score = score
            best_dropped_columns = i + 1
            pd.DataFrame(x_train).to_csv("best_train.csv", index=False)
            pd.DataFrame(x_test).to_csv("best_test.csv", index=False)

        component_scores.append(score)
        features.append(len(coef))

        print("Dropped columns: {} R2 score: {}".format(i + 1, score))
        pd.DataFrame(columns_to_drop).to_csv("columns_to_drop.csv", index=False)

    plt.plot(features, component_scores)
    plt.savefig(
        f"runs/{IMP_NN}_{LOAD_IMP_DATA}_{GM_COMPONENTS}_{PERC_THRESHOLD}_{LASSO_ALPHA}_{GBR_ESTIMATORS}_{D_COLS_ROUND}_{best_score}.png"
    )

    if not os.path.exists("runs/saved_runs.csv"):
        pd.DataFrame(
            [
                [
                    IMP_NN,
                    LOAD_IMP_DATA,
                    GM_COMPONENTS,
                    PERC_THRESHOLD,
                    LASSO_ALPHA,
                    GBR_ESTIMATORS,
                    D_COLS_ROUND,
                    best_score,
                ]
            ],
            columns=[
                "IMP_NN",
                "LOAD_IMP_DATA",
                "GM_COMPONENTS",
                "PERC_THRESHOLD",
                "LASSO_ALPHA",
                "GBR_ESTIMATORS",
                "D_COLS_ROUND",
                "best_score",
            ],
        ).to_csv(
            "runs/saved_runs.csv",
            index=False,
            mode="a",
        )
    else:
        df = pd.read_csv("runs/saved_runs.csv")
        pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        IMP_NN,
                        LOAD_IMP_DATA,
                        GM_COMPONENTS,
                        PERC_THRESHOLD,
                        LASSO_ALPHA,
                        GBR_ESTIMATORS,
                        D_COLS_ROUND,
                        best_score,
                    ]
                ),
            ],
            axis=1,
        ).to_csv("runs/saved_runs.csv", index=False)

    print("Best score: {}".format(best_score))
    print("Best dropped columns: {}".format(best_dropped_columns))

    # sel = VarianceThreshold(threshold=(0.95))
    # x_train = sel.fit_transform(x_train, y_train)
    # x_test = sel.transform(x_test)

    # clf.fit(x_train, y_train)

    # create pediction file
    # y_pred = clf.predict(imp.transform(test))
    # pred["y"] = pd.DataFrame(y_pred, columns=["y"])
    # pred.to_csv("data/y_pred.csv", index=False)

    # print("R2 score: {}".format(r2_score(y_test, clf.predict(x_test))))
