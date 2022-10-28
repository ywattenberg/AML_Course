import os
import warnings

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
GM_COMPONENTS = 100
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


def train_outlier_detection_model(x_train, n_components=GM_COMPONENTS, random=False):
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
    x_train, y_train, num_of_splits=5, regressor_steps=GBR_ESTIMATORS
):
    scores = []
    kf = KFold(n_splits=num_of_splits, shuffle=True)
    for components in range(10, 200, 10):
        score = 0
        for train_index, test_index in kf.split(x_train):
            x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]
            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
            gm = train_outlier_detection_model(
                x_train_cv, threshold, n_components=componenets
            )

            mask = get_outlier_mask(x_train_cv, gm, threshold)
            x_train_cv = x_train_cv[np.nonzero(mask)[0]]
            y_train_cv = y_train_cv[np.nonzero(mask)[0]]

            clf = GradientBoostingRegressor(
                loss="squared_error", n_estimators=regressor_steps
            )
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test_cv)
            score += r2_score(y_test_cv, y_pred)
        scores.append(score / 4)
        print("Threshold: ", threshold, "Score: ", score / 4)
    return scores


if __name__ == "__main__":

    # Load data
    x_train, test, test_id, y_train = load_data(use_imp_data=False)
    print("Data loaded")

    # normalize data
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    test = scaler.transform(test)

    scores = cross_validation_gmm(x_train, y_train)
    print("Cross validation scores: ", scores)
    quit()
    # make train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    gm = GaussianMixture(n_components=GM_COMPONENTS, random_state=0).fit(x_train)
    score = gm.score_samples(x_train)

    # Get the score threshold for anomaly
    pct_threshold = np.percentile(score, PERC_THRESHOLD)  # Print the score threshold
    print(f"The threshold of the score is {pct_threshold:.2f}")  # Label the anomalies
    outliers = score > pct_threshold

    x_train = x_train[np.nonzero(outliers)[0]]
    y_train = y_train[np.nonzero(outliers)[0]]

    k = x_train.shape[1] - 1

    best_score = -1
    best_columns = 0
    columns_to_drop = []
    scores = []
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

        scores.append(score)
        features.append(len(coef))

        print("Dropped columns: {} R2 score: {}".format(i + 1, score))
        pd.DataFrame(columns_to_drop).to_csv("columns_to_drop.csv", index=False)

    plt.plot(features, scores)
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
