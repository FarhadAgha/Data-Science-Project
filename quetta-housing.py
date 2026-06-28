import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create fake Quetta housing data
np.random.seed(42)
n = 1000  # 1000 houses

data = {
    'MedInc': np.random.uniform(2, 8, n),      # median income
    'HouseAge': np.random.randint(10, 60, n),  # house age
    'AveRooms': np.random.uniform(3, 8, n),    # average rooms
    'AveOccup': np.random.uniform(1, 5, n),    # average occupancy
    'target': np.random.uniform(50000, 200000, n)  # price in PKR
}

df = pd.DataFrame(data)

print("🏠 QUETTA HOUSING DATASET (DUMMY DATA)")
print("=" * 50)
print(df.head())
print("\nShape:", df.shape)

# Summary statistics
print("\n📈 SUMMARY STATISTICS")
print("=" * 50)
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Feature Correlations - Quetta Housing")
plt.tight_layout()
plt.savefig("quetta_housing_corr.png")
plt.show()

# Train model
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n🤖 MODEL EVALUATION")
print("=" * 50)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f}")

# Plot predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price (PKR)")
plt.ylabel("Predicted Price (PKR)")
plt.title("Actual vs Predicted - Quetta Housing")
plt.savefig("quetta_predictions.png")
plt.show()