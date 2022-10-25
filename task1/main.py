import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

if __name__ == "__main__":

    # Load data
    x_train = pd.read_csv("data/X_train.csv")
    test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv")
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    clf = Ridge(alpha=1.0)

    x_train.drop("id", axis=1, inplace=True)
    y_train.drop("id", axis=1, inplace=True)
    test.drop("id", axis=1, inplace=True)

    clf.fit(imp.fit_transform(x_train), y_train)
    y_pred = clf.predict(imp.transform(test))

    pd.DataFrame(y_pred).to_csv("data/y_pred.csv", index=False)

    print("R2 score: {}".format(r2_score(y_train, clf.predict(imp.transform(x_train)))))
