import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gradient_descent import LinearRegressionGD
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load cleaned data
df = pd.read_csv("data/processed/flight_delay_cleaned.csv")

# Select features and target
X = df[["DEP_DELAY", "DISTANCE", "CRS_ELAPSED_TIME"]].values
y = df["ARR_DELAY"].values

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
print("NaNs in X_train:", np.isnan(X_train).any())
print("NaNs in y_train:", np.isnan(y_train).any())
print("NaNs in X_test:", np.isnan(X_test).any())
print("NaNs in y_test:", np.isnan(y_test).any())
'''
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LinearRegressionGD(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mse = np.mean((y_test - predictions) ** 2)
print(f"Test MSE: {mse:.2f}")

# Plot learning curve
plt.plot(model.loss_history)
plt.title("Loss over Iterations")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.show()
