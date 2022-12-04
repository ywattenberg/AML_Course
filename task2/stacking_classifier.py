import pandas as pd
import torch
import dataset
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

X_train = pd.read_csv("data/X_train_2.csv")
X_test = pd.read_csv("data/X_test_2.csv")
y_train = pd.read_csv("data/y_train.csv")
id_col = y_train["id"]
y_train.drop("id", axis=1, inplace=True)

estimators = [
    ("rf", RandomForestClassifier(max_depth=5, n_estimators=300, max_features=60)),
    ("svr", make_pipeline(StandardScaler(), LinearSVC(random_state=42))),
    ("xgb", XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05)),
    (
        "gbc",
        GradientBoostingClassifier(
            max_depth=5, n_estimators=300, learning_rate=0.1, max_features=60
        ),
    ),
    ("hgbc", GradientBoostingClassifier(max_depth=5, n_estimators=500, learning_rate=0.1)),
    ("ada", AdaBoostClassifier(n_estimators=300, learning_rate=0.1)),
]

clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


out = pd.DataFrame()
out["y"] = y_pred
out["id"] = id_col
out.to_csv("out_stacking.csv", index=False)
