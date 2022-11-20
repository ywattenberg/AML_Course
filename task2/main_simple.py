import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import utils


def gridsearch(model, param_grid, X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Test set score: {:.2f}".format(grid.score(X_val, y_val)))
    return grid


def main():
    y_train = pd.read_csv("data/y_train.csv")
    # X_train = pd.read_csv("data/X_train.csv")

    # X_test = pd.read_csv("data/X_test.csv")

    # X_train_1 = utils.process_data(X_train)
    # X_test_1 = utils.process_data(X_test)

    # X_train_2 = utils.process_data2(X_train)
    # X_test_2 = utils.process_data2(X_test)

    # imputer = SimpleImputer(strategy="median", fill_value=0).fit(X_train_2)
    # X_train_2 = imputer.transform(X_train_2)
    # X_test_2 = imputer.transform(X_test_2)
    # scaler = StandardScaler().fit(X_train_2)
    # X_train_2 = scaler.transform(X_train_2)
    # X_test_2 = scaler.transform(X_test_2)

    # X_train_con = np.concatenate((X_train_1, X_train_2), axis=1)
    # X_test_con = np.concatenate((X_test_1, X_test_2), axis=1)

    y_train = y_train["y"].to_numpy().flatten()

    # pd.DataFrame(X_train_1).to_csv("data/X_train_1.csv", index=False)
    # pd.DataFrame(X_test_1).to_csv("data/X_test_1.csv", index=False)
    # pd.DataFrame(X_train_2).to_csv("data/X_train_2.csv", index=False)
    # pd.DataFrame(X_test_2).to_csv("data/X_test_2.csv", index=False)
    # pd.DataFrame(X_train_con).to_csv("data/X_train_con.csv", index=False)
    # pd.DataFrame(X_test_con).to_csv("data/X_test_con.csv", index=False)

    X_train_1 = pd.read_csv("data/X_train_1.csv")
    X_test_1 = pd.read_csv("data/X_test_1.csv")
    X_train_2 = pd.read_csv("data/X_train_2.csv")
    X_test_2 = pd.read_csv("data/X_test_2.csv")
    X_train_con = pd.read_csv("data/X_train_con.csv")
    X_test_con = pd.read_csv("data/X_test_con.csv")

    param_grid = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
    }
    gridsearch(
        model=XGBClassifier(objective="multi:softprob"),
        param_grid=param_grid,
        X_train=X_train_1,
        y_train=y_train,
    )
    gridsearch(
        model=XGBClassifier(objective="multi:softprob"),
        param_grid=param_grid,
        X_train=X_train_2,
        y_train=y_train,
    )
    gridsearch(
        model=XGBClassifier(objective="multi:softprob"),
        param_grid=param_grid,
        X_train=X_train_con,
        y_train=y_train,
    )

    # XGBClassifier_con = XGBClassifier(max_depth=5, n_estimators=500, learning_rate=0.05).fit(X_train_con, y_train)
    # XGBClassifier_1 = XGBClassifier(max_depth=5, n_estimators=250, learning_rate=0.05).fit(X_train_1, y_train)
    # XGBClassifier_2 = XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05).fit(X_train_2, y_train)
    # GBClassifier_con = GradientBoostingClassifier(max_depth=5, n_estimators=500, learning_rate=0.1, max_features=60).fit(X_train_con, y_train)
    # GBClassifier_1 = GradientBoostingClassifier(max_depth=5, n_estimators=250, learning_rate=0.1, max_features=60).fit(X_train_1, y_train)
    # GBClassifier_2 = GradientBoostingClassifier(max_depth=5, n_estimators=300, learning_rate=0.1, max_features=60).fit(X_train_2, y_train)
    # AdaClassifier_con = AdaBoostClassifier(n_estimators=500, learning_rate=0.1).fit(X_train_con, y_train)
    # AdaClassifier_1 = AdaBoostClassifier(n_estimators=250, learning_rate=0.1).fit(X_train_1, y_train)
    # AdaClassifier_2 = AdaBoostClassifier(n_estimators=300, learning_rate=0.1).fit(X_train_2, y_train)
    # RFClassifier_con = RandomForestClassifier(max_depth=5, n_estimators=500, max_features=60).fit(X_train_con, y_train)
    # RFClassifier_1 = RandomForestClassifier(max_depth=5, n_estimators=250, max_features=60).fit(X_train_1, y_train)
    # RFClassifier_2 = RandomForestClassifier(max_depth=5, n_estimators=300, max_features=60).fit(X_train_2, y_train)

    # pred_XGBClassifier_con = XGBClassifier_con.predict(X_test_con)
    # pred_XGBClassifier_1 = XGBClassifier_1.predict(X_test_1)
    # pred_XGBClassifier_2 = XGBClassifier_2.predict(X_test_2)
    # pred_GBClassifier_con = GBClassifier_con.predict(X_test_con)
    # pred_GBClassifier_1 = GBClassifier_1.predict(X_test_1)
    # pred_GBClassifier_2 = GBClassifier_2.predict(X_test_2)
    # pred_AdaClassifier_con = AdaClassifier_con.predict(X_test_con)
    # pred_AdaClassifier_1 = AdaClassifier_1.predict(X_test_1)
    # pred_AdaClassifier_2 = AdaClassifier_2.predict(X_test_2)
    # pred_RFClassifier_con = RFClassifier_con.predict(X_test_con)
    # pred_RFClassifier_1 = RFClassifier_1.predict(X_test_1)
    # pred_RFClassifier_2 = RFClassifier_2.predict(X_test_2)

    # pred_df = pd.DataFrame()
    # pred_df["XGBClassifier_con"] = pred_XGBClassifier_con
    # pred_df["XGBClassifier_1"] = pred_XGBClassifier_1
    # pred_df["XGBClassifier_2"] = pred_XGBClassifier_2
    # pred_df["GBClassifier_con"] = pred_GBClassifier_con
    # pred_df["GBClassifier_1"] = pred_GBClassifier_1
    # pred_df["GBClassifier_2"] = pred_GBClassifier_2
    # pred_df["AdaClassifier_con"] = pred_AdaClassifier_con
    # pred_df["AdaClassifier_1"] = pred_AdaClassifier_1
    # pred_df["AdaClassifier_2"] = pred_AdaClassifier_2
    # pred_df["RFClassifier_con"] = pred_RFClassifier_con
    # pred_df["RFClassifier_1"] = pred_RFClassifier_1
    # pred_df["RFClassifier_2"] = pred_RFClassifier_2
    # pred_df.to_csv("pred_df.csv", index=False)

    # out = pd.DataFrame()
    # out["y"] = pred_df.mode(axis=1)[0]
    # out["id"] = X_test["id"]
    # out.to_csv("out.csv", index=False)


if __name__ == "__main__":
    main()
