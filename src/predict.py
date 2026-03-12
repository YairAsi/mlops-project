import joblib
import pandas as pd

model = joblib.load("model/model.pkl")

sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)

prediction = model.predict(sample)

print("Prediction:", prediction)