import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/iris.csv")

X = data.drop("species", axis=1)
y = data["species"]

model = joblib.load("model/model.pkl")

preds = model.predict(X)

acc = accuracy_score(y, preds)

print("Accuracy:", acc)

if acc < 0.8:
    raise Exception("Model accuracy too low")