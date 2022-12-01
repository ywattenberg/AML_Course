import csv

import pandas as pd
import torch
from classifier_model import ClassifierModel


def main():
    X_test = pd.read_csv("data/X_test_2.csv").iloc[:, 1:]

    pred_file = open('predictions.csv', 'w', newline='')
    writer = csv.writer(pred_file, delimiter=',')
    fields = ['id','y']
    writer.writerow(fields)

    sm = torch.nn.Softmax(dim=1)
    print(len(X_test.columns))
    model = ClassifierModel(935)
    model.load_state_dict(torch.load("model_2.pth"))
    model.eval()

    for i in range(len(X_test)):
        x = X_test.iloc[i].to_numpy().flatten()
        x = torch.tensor(x).type(torch.float)
        # print(x.squeeze(0).squeeze(0))
        y_pred = model(x.unsqueeze(0))
        # print(sm(y_pred))
        y_pred = torch.argmax(sm(y_pred))
        y_pred = y_pred.detach().numpy()
        # print(y_pred)
        writer.writerow([i, y_pred])

if __name__ == "__main__":
    main()