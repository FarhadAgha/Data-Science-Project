import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("IRIS DATASET")
print("=" * 50)
print(df.head())
print("\nShape:", df.shape)

print("\n SUMMARY STATISTICS")
print("=" * 50)
print(df.describe())

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.pairplot(df, hue="species", diag_kind="hist")
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.savefig("iris_pairplot.png")  # saves in current folder
plt.show()

X = df[iris.feature_names]   
y = df['target']             

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n MODEL EVALUATION")
print("=" * 50)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # example: setosa
predicted_class = model.predict(sample)
print("\n Sample prediction (sepal length=5.1, width=3.5, petal length=1.4, width=0.2):")
print(f"Predicted species: {iris.target_names[predicted_class][0]}")