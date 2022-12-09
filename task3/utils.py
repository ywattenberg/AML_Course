import pickle
import gzip
import numpy as np
import os


def load_zipped_pickle(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


def save_zipped_pickle(obj, filename):
    with gzip.open(filename, "wb") as f:
        pickle.dump(obj, f, 2)


def test_pred():
    # load data
    train_data = load_zipped_pickle("train.pkl")
    test_data = load_zipped_pickle("test.pkl")
    samples = load_zipped_pickle("sample.pkl")
    # make prediction for test
    predictions = []
    for d in test_data:
        prediction = np.array(np.zeros_like(d["video"]), dtype=np.bool)
        height = prediction.shape[0]
        width = prediction.shape[1]
        prediction[
            int(height / 2) - 50 : int(height / 2 + 50), int(width / 2) - 50 : int(width / 2 + 50)
        ] = True

        # DATA Strucure
        predictions.append({"name": d["name"], "prediction": prediction})
        # save in correct format

    save_zipped_pickle(predictions, "my_predictions.pkl")
