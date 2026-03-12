from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model/model.pkl")


class Features(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(data: Features):

    sample = pd.DataFrame(
        [data.features],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ],
    )

    prediction = model.predict(sample)

    return {"prediction": int(prediction[0])}