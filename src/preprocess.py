from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

df.to_csv("data/iris.csv", index=False)

print("Dataset saved to data/iris.csv")