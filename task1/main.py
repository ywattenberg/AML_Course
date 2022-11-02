import os
from re import T
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

GM_COMPONENTS = 10
PERC_THRESHOLD = 10
GBR_ESTIMATORS = 200
PCA_COMPONENTS = 30
NUM_FEATURES = "all"


# def cross_validation_regression_num_dim(
#     x_train,
#     y_train,
#     num_of_splits=5,
#     regressor_steps=GBR_ESTIMATORS,
#     begin=10,
#     end=200,
#     step=10,
# ):


def cross_validation_pca_num_dimensions(
    x_train,
    y_train,
    num_of_splits=5,
    regressor_steps=GBR_ESTIMATORS,
    threshold=PERC_THRESHOLD,
    n_components=GM_COMPONENTS,
    begin=10,
    end=200,
    step=10,
):
    scores = {}
    kf = KFold(n_splits=num_of_splits, shuffle=True)

    for components in range(begin, end, step):
        score = 0
        print(f"Components: {components}")
        for idx, (train_index, test_index) in enumerate(kf.split(x_train)):

            x_train_cv, x_test_cv, y_train_cv, y_test_cv = utils.get_split_from_index(
                x_train, y_train, train_index, test_index
            )

            pca = utils.train_pca_model(x_train_cv, n_components=components)
            x_train_cv = pca.transform(x_train_cv)
            x_test_cv = pca.transform(x_test_cv)

            x_train_cv, y_train_cv = utils.filter_outliers(
                x_train_cv, y_train_cv, n_components=n_components, threshold=threshold
            )

            # x_train_cv, x_test_cv = utils.select_features(
            #     x_train_cv, y_train_cv, x_test_cv, 'all'
            # )

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


def cross_validation_gmm_components(
    x_train,
    y_train,
    num_of_splits=5,
    regressor_steps=GBR_ESTIMATORS,
    threshold=PERC_THRESHOLD,
    n_components=GM_COMPONENTS,
    pca_components=PCA_COMPONENTS,
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
            x_train_cv, y_train_cv = utils.pca_transform(
                x_train_cv, x_test_cv, pca_components
            )

            x_train_cv, y_train_cv = utils.filter_outliers(
                x_train_cv, y_train_cv, n_components=components, threshold=threshold
            )

            # x_train_cv, x_test_cv = utils.select_features(
            #     x_train_cv, y_train_cv, x_test_cv, 120
            # )

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


def cross_validation(x_train, y_train, num_of_splits=5):
    columns = [
        "num_of_regressors",
        "num_of_pca_dims",
        "num_of_gmm_comps",
        "threshold",
        "num_of_features",
        "r2_score",
    ]
    scores = pd.DataFrame(columns=columns)
    best_run = pd.DataFrame(columns=columns)
    kf = KFold(n_splits=num_of_splits, shuffle=True)
    for num_of_regressors in range(100, 400, 100):
        for num_of_pca_dims in [20, 30, 50, 100, 150]:
            for num_of_gmm_comps in [1,5,10,50,100]:
                for threshold in [1,5,10,15]:
                    for num_of_features in range(10, num_of_pca_dims+1, int(num_of_pca_dims / 5)):
                        score = 0
                        for train_index, test_index in kf.split(x_train):
                            (
                                x_train_cv,
                                x_test_cv,
                                y_train_cv,
                                y_test_cv,
                            ) = utils.get_split_from_index(
                                x_train, y_train, train_index, test_index
                            )

                            x_train_cv, x_test_cv = utils.pca_transform(
                                x_train_cv, x_test_cv, num_of_pca_dims
                            )

                            x_train_cv, y_train_cv = utils.filter_outliers(
                                x_train_cv,
                                y_train_cv,
                                n_components=num_of_gmm_comps,
                                threshold=threshold,
                            )

                            x_train_cv, x_test_cv = utils.select_features(
                                x_train_cv, y_train_cv, x_test_cv, num_of_features
                            )

                            curr_score = utils.get_r2_score(
                                x_train_cv,
                                x_test_cv,
                                y_train_cv,
                                y_test_cv,
                                num_of_regressors,
                            )
                            score += curr_score
                            curr_run = pd.DataFrame.from_dict(
                                {
                                    "num_of_regressors": [num_of_regressors],
                                    "num_of_pca_dims": [num_of_pca_dims],
                                    "num_of_gmm_comps": [num_of_gmm_comps],
                                    "threshold": [threshold],
                                    "num_of_features": [num_of_features],
                                    "r2_score": [score / num_of_splits],
                                }
                            )
                        print(pd.concat([best_run, curr_run]))
                        if (
                            best_run.empty
                            or curr_run.iloc[0, -1] > best_run.iloc[0, -1]
                        ):
                            best_run = curr_run
                    scores = pd.concat(
                        [scores, curr_run],
                        ignore_index=True,
                    )
                    scores.to_csv("tmp.csv", index=False)
    return scores


def cross_validation_gmm_threshold(
    x_train,
    y_train,
    num_of_splits=5,
    regressor_steps=GBR_ESTIMATORS,
    n_components=GM_COMPONENTS,
    pca_components=PCA_COMPONENTS,
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
            x_train_cv, y_train_cv = utils.pca_transform(
                x_train_cv, x_test_cv, pca_components
            )

            x_train_cv, y_train_cv = utils.filter_outliers(
                x_train_cv, y_train_cv, n_components=n_components, threshold=threshold
            )

            # x_train_cv, x_test_cv = utils.select_features(x_train_cv, x_test_cv, 120)

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
    pca_components=PCA_COMPONENTS,
    num_features=NUM_FEATURES,
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
            x_train_cv, y_train_cv = utils.pca_transform(
                x_train_cv, x_test_cv, pca_components
            )

            x_train_cv, y_train_cv = utils.filter_outliers(
                x_train_cv, y_train_cv, threshold=threshold, n_components=n_components
            )

            x_train_cv, x_test_cv = utils.select_features(
                x_train_cv, x_test_cv, num_features
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
    pca_components=PCA_COMPONENTS,
    begin=5,
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
            x_train_cv, y_train_cv = utils.pca_transform(
                x_train_cv, x_test_cv, pca_components
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
    x_train, x_test, test_id, y_train = utils.load_data(use_imp_data=False)
    print("Data loaded")

    # Normalize data
    # QuantileTransformer(output_distribution='normal', random_state=0).fit_transform(x_train)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    scores = cross_validation(x_train, y_train)
    scores.to_csv("scores.csv", index=False)
    print("Done with cross validation")
    print(print(scores[scores.score == scores.score.max()]))

    quit()
    ## ------------------ Cross validation ------------------ ##
    pca_scores = cross_validation_pca_num_dimensions(x_train, y_train)
    print("PCA cross validation done")
    best_pca_components = pca_scores[max(pca_scores.keys())]
    print(f"Best PCA components: {best_pca_components}")

    # Select outlier detection model
    component_scores = cross_validation_gmm_components(
        x_train, y_train, pca_components=best_pca_components
    )
    print("Finished cross validation for components")
    best_num_of_components = component_scores[max(component_scores.keys())]
    print(f"Best model has {best_num_of_components} components")

    threshold_scores = cross_validation_gmm_threshold(
        x_train,
        y_train,
        n_components=best_num_of_components,
        pca_components=best_pca_components,
    )
    print("Finished cross validation for threshold")
    best_threshold = threshold_scores[max(threshold_scores.keys())]
    print(f"Best model has {best_threshold} threshold")

    feature_scores = cross_validation_feature_selection(
        x_train,
        y_train,
        pca_components=best_pca_components,
        n_components=best_num_of_components,
        threshold=best_threshold,
        begin=5,
        end=best_pca_components,
        step=5,
    )
    print("Finished cross validation for number of features")
    best_num_of_features = feature_scores[max(feature_scores.keys())]
    print(f"Best model has {best_num_of_features} features")

    num_regressor_score = cross_validation_regression_num_regressors(
        x_train,
        y_train,
        pca_components=best_pca_components,
        n_components=best_num_of_components,
        threshold=best_threshold,
        num_features=best_num_of_features,
        begin=100,
        end=500,
        step=50,
    )
    print("Finished cross validation for number of regressors")
    best_num_of_regressors = num_regressor_score[max(num_regressor_score.keys())]
    print(f"Best model has {best_num_of_regressors} regressors")

    ## ------------------ Save cross validation runs ------------------ ##

    scores_map = {
        "component_scores": component_scores,
        "threshold_scores": threshold_scores,
        "num_regressor_score": num_regressor_score,
        "pca_scores": pca_scores,
        "feature_scores": feature_scores,
    }
    # scores_map = {"pca_scores": pca_scores}
    utils.safe_cross_scores(scores_map)

    ## ------------------ Final model ------------------ ##

    # x_train, y_train = utils.filter_outliers(
    #     x_train, y_train, threshold=PERC_THRESHOLD, n_components=GM_COMPONENTS
    # )
    # x_train, x_test = utils.select_features(
    #     x_train, y_train, x_test, 150, use_mutual_info=False
    # )
    # regressor = utils.get_regressor(x_train, y_train, GBR_ESTIMATORS)
    # y_pred = regressor.predict(x_test)
    # utils.save_submission(test_id, y_pred)
