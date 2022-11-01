import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import utils

warnings.filterwarnings("ignore")

GM_COMPONENTS = 120
PERC_THRESHOLD = 10
GBR_ESTIMATORS = 200


# def cross_validation_regression_num_dim(
#     x_train,
#     y_train,
#     num_of_splits=5,
#     regressor_steps=GBR_ESTIMATORS,
#     begin=10,
#     end=200,
#     step=10,
# ):


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
            x_train_cv, x_test_cv, y_train_cv, y_test_cv = utils.get_split_from_index(
                x_train, y_train, train_index, test_index
            )
            gm = utils.train_outlier_detection_model(
                x_train_cv, n_components=components
            )

            mask = utils.get_outlier_mask(x_train_cv, gm, threshold)
            x_train_cv = x_train_cv[np.nonzero(mask)[0]]
            y_train_cv = y_train_cv[np.nonzero(mask)[0]]

            curr_score = utils.get_r2_score(
                x_train_cv, x_test_cv, y_train_cv, y_test_cv, regressor_steps
            )
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
            utils.train_outlier_detection_model(x_train_cv, n_components=n_components)
        )

    for threshold in range(begin, end, step):
        score = 0
        print(f"Threshold: {threshold}%")
        for idx, (train_index, test_index) in enumerate(kf.split(x_train)):
            x_train_cv, x_test_cv, y_train_cv, y_test_cv = utils.get_split_from_index(
                x_train, y_train, train_index, test_index
            )

            mask = utils.get_outlier_mask(x_train_cv, models[idx], threshold)
            x_train_cv = x_train_cv[np.nonzero(mask)[0]]
            y_train_cv = y_train_cv[np.nonzero(mask)[0]]

            curr_score = utils.get_r2_score(
                x_train_cv, x_test_cv, y_train_cv, y_test_cv, regressor_steps
            )
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
    threshold=PERC_THRESHOLD,
    n_components=GM_COMPONENTS,
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
            x_train_cv, x_test_cv, y_train_cv, y_test_cv = utils.get_split_from_index(
                x_train, y_train, train_index, test_index
            )
            x_train_cv, y_train_cv = utils.filter_outliers(
                x_train_cv, y_train_cv, threshold=threshold, n_components=n_components
            )

            curr_score = utils.get_r2_score(
                x_train_cv, x_test_cv, y_train_cv, y_test_cv, regressors
            )
            score += curr_score
            print(f"Current score {regressors} regressors: {curr_score:.4f}")
        scores[score / num_of_splits] = regressors
        print("-" * 50)
        print(f"Final score for {regressors} regressors: {score / num_of_splits:.4f}")
        print("-" * 50)
    return scores


def cross_validation_feature_selection(
    x_train,
    y_train,
    num_of_splits=5,
    regressor_steps=GBR_ESTIMATORS,
    threshold=PERC_THRESHOLD,
    n_components=GM_COMPONENTS,
    begin=10,
    end=500,
    step=10,
):
    scores = {}
    kf = KFold(n_splits=num_of_splits, shuffle=True)
    for features in range(begin, end, step):
        score = 0
        print(f"Number of features: {features}")
        for train_index, test_index in kf.split(x_train):
            x_train_cv, x_test_cv, y_train_cv, y_test_cv = utils.get_split_from_index(
                x_train, y_train, train_index, test_index
            )
            x_train_cv, y_train_cv = utils.filter_outliers(
                x_train_cv, y_train_cv, threshold=threshold, n_components=n_components
            )

            x_train_cv, y_train_cv = utils.select_features(
                x_train_cv, y_train_cv, x_test_cv, features, use_mutual_info=False
            )
            curr_score = utils.get_r2_score(
                x_train_cv, x_test_cv, y_train_cv, y_test_cv, regressor_steps
            )
            score += curr_score
            print(f"Current score {features} features: {curr_score:.4f}")
        scores[score / num_of_splits] = features
        print("-" * 50)
        print(f"Final score for {features} features: {score / num_of_splits:.4f}")
        print("-" * 50)
    return scores


if __name__ == "__main__":

    # Load data
    x_train, x_test, test_id, y_train = utils.load_data(use_imp_data=True)
    print("Data loaded")

    # Normalize data
    # QuantileTransformer(output_distribution='normal', random_state=0).fit_transform(x_train)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Select outlier detection model
    # component_scores = cross_validation_gmm_components(x_train, y_train)
    # print("Finished cross validation for components")
    # best_num_of_components = component_scores[max(component_scores.keys())]
    # print(f"Best model has {best_num_of_components} components")

    # threshold_scores = cross_validation_gmm_threshold(
    #     x_train, y_train, n_components=best_num_of_components
    # )
    # print("Finished cross validation for threshold")
    # best_threshold = threshold_scores[max(threshold_scores.keys())]
    # print(f"Best model has {best_threshold} threshold")

    # print(
    #     f"Filtering outliers with {best_num_of_components} components and threshold {best_threshold}%"
    # )

    # x_train, y_train, gm = filter_outliers(
    #     x_train, y_train, threshold=best_threshold, n_components=best_num_of_components
    # )

    # num_regressor_score = cross_validation_regression_num_regressors(x_train, y_train)
    # print("Finished cross validation for number of regressors")
    # best_num_of_regressors = num_regressor_score[max(num_regressor_score.keys())]
    # print(f"Best model has {best_num_of_regressors} regressors")
    # scores_map = {"component_scores":component_scores, "threshold_scores":threshold_scores, "num_regressor_score":num_regressor_score}
    # utils.safe_cross_scores(scores_map)
    ## Train model

    # feature_selection_scores = cross_validation_feature_selection(x_train, y_train)
    # print("Finished cross validation for feature selection")
    # best_num_of_features = feature_selection_scores[
    #     max(feature_selection_scores.keys())
    # ]
    # print(f"Best model has {best_num_of_features} features")
    x_train, y_train = utils.filter_outliers(
        x_train, y_train, threshold=PERC_THRESHOLD, n_components=GM_COMPONENTS
    )
    x_train, x_test = utils.select_features(
        x_train, y_train, x_test, 150, use_mutual_info=False
    )
    regressor = utils.get_regressor(x_train, y_train, GBR_ESTIMATORS)
    y_pred = regressor.predict(x_test)
    utils.save_submission(test_id, y_pred)

    # sel = VarianceThreshold(threshold=(0.95))
    # x_train = sel.fit_transform(x_train, y_train)
    # x_test = sel.transform(x_test)

    # clf.fit(x_train, y_train)

    # create pediction file
    # y_pred = clf.predict(imp.transform(test))
    # pred["y"] = pd.DataFrame(y_pred, columns=["y"])
    # pred.to_csv("data/y_pred.csv", index=False)

    # print("R2 score: {}".format(r2_score(y_test, clf.predict(x_test))))
