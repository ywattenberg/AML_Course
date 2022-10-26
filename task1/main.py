import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

if __name__ == "__main__":

    # Load data
    x_train = pd.read_csv("data/X_train.csv")

    test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv")

    pred = pd.DataFrame(test["id"])

    # Remove nan values
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")

    clf = Ridge(alpha=1000.0)
    # clf = Lasso(alpha=0)
    # clf = GaussianMixture(n_components=2, random_state=0)

    # normalize data
    x_train = preprocessing.StandardScaler().fit_transform(x_train)
    # y_train = preprocessing.StandardScaler().fit_transform(y_train)

    # make train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    x_train = np.delete(x_train, axis=1, obj=0)
    x_test = np.delete(x_test, axis=1, obj=0)
    # test = np.delete(test, axis=1, obj=0)

    print(x_train)
    print(y_train)

    x_train = imp.fit_transform(x_train)
    x_test = imp.transform(x_test)

    gm = GaussianMixture(n_components=100, random_state=0).fit(x_train)
    score = gm.score_samples(x_train)

    # Get the score threshold for anomaly
    pct_threshold = np.percentile(score, 3)  # Print the score threshold
    print(f"The threshold of the score is {pct_threshold:.2f}")  # Label the anomalies
    # outliers = score.apply(lambda x: 1 if x < pct_threshold else 0)
    outliers = score > pct_threshold
    print(len(np.nonzero(outliers)[0]))

    x_train = x_train[np.nonzero(outliers)[0]]
    # y_train = y_train[np.nonzero(outliers)[0]]

    print(x_train)
    # print(outliers.shape)
    # print(outliers)

    sel = VarianceThreshold(threshold=(0.95))
    print(x_train.shape)
    x_train = sel.fit_transform(x_train, y_train)
    print(x_train.shape)
    x_test = sel.transform(x_test)

    clf.fit(x_train, y_train)

    # create pediction file
    # y_pred = clf.predict(imp.transform(test))
    # pred["y"] = pd.DataFrame(y_pred, columns=["y"])
    # pred.to_csv("data/y_pred.csv", index=False)

    print("R2 score: {}".format(r2_score(y_test, clf.predict(x_test))))
