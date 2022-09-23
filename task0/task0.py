import pandas as pd 
import numpy as np
from sklearn.linear_model import Ridge

def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    k = test.iloc[:, 1:].to_numpy()

    y = train.y.to_numpy()
    X = train.iloc[:,2:].to_numpy()

    model = Ridge(alpha=0.0)
    model.fit(X,y)
    
    pred = model.predict(k)

    prediction = pd.Dataframe(test.Id)
    prediction['y'] = pred
    prediction.to_csv("prediction.csv", index=False)


if __name__ == '__main__':
    main()
    
