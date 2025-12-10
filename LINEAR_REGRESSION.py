# IMPLEMENTING LINEAR REGRESSION FROM SCRATCH USING PYTHON
# STEP 1: Create dataset manually
# STEP 2: Visualize the data
# STEP 3: Define loss function (MSE)
# STEP 4: Define gradient descent
# STEP 5: Train the model
# STEP 6: Evaluate with MAE, RMSE, R²
# STEP 7: Visualize model fit & loss curve
# STEP 8: Compare with sklearn
# STEP 9: Compare R² scores
# step 10:compare loss curves

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# STEP 1 — Create Data
X_data = np.array([1.5,1.8,2.0,2.5,3.3,3.8,4.0,4.9,5.5,5.9,6.0,6.3,7.0,7.2,8.9,9.0,9.5])
Y_data = np.array([50,55,54,58,60,65,69,72,75,79,80,82,84,81,86,89,92])

df = pd.DataFrame({
    "Hours_Studied": X_data,
    "Exam_Score": Y_data
})

df.to_csv("Student_data.csv", index=False)
print(df)


# STEP 2 — Visualize Data
plt.figure(figsize=(8,6))
plt.scatter(df["Hours_Studied"], df["Exam_Score"], color="blue")
plt.title("Hours Studied VS Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()


# STEP 3 — Loss Function (MSE)
def mean_squared_error(X, Y, m, b):
    predicted_Y = m * X + b
    return np.mean((Y - predicted_Y)**2)


# STEP 4 — Gradient Descent
def gradient_descent_step(X, Y, m, b, learning_rate):
    N = len(Y)
    predicted_Y = m * X + b
    error = predicted_Y - Y   # Standard error = y_hat - y

    # gradients
    gradient_m = (2/N) * np.sum(error * X)
    gradient_b = (2/N) * np.sum(error)

    # update parameters
    m -= learning_rate * gradient_m
    b -= learning_rate * gradient_b

    return m, b


# STEP 5 — Training Loop
m = 0.0
b = 0.0
learning_rate = 0.01
epochs = 500

m_history = []
b_history = []
loss_history = []

for epoch in range(epochs):
    m, b = gradient_descent_step(X_data, Y_data, m, b, learning_rate)
    loss = mean_squared_error(X_data, Y_data, m, b)

    m_history.append(m)
    b_history.append(b)
    loss_history.append(loss)
print(f"m_history:{m_history}")
print(f"b_history:{b_history}")
print(f"loss_history:{loss_history}")


print(f"\nFinal Parameters : m={m:.4f}, b={b:.4f}")


# STEP 6 — Evaluation Metrics
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)


# Calculate Metrics
final_predicted_Y = m * X_data + b
mae = mean_absolute_error(Y_data, final_predicted_Y)
rmse = root_mean_squared_error(Y_data, final_predicted_Y)
r2_custom = r2_score(Y_data, final_predicted_Y)

print("\nModel Evaluation Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2_custom:.4f}")


# STEP 7 — Visualizations
plt.figure(figsize=(12,5))

# Regression Fit
plt.subplot(1,2,1)
plt.scatter(X_data, Y_data, color="blue")
plt.plot(X_data, final_predicted_Y, color="red", linewidth=2)
plt.title("Linear Regression Fit (Scratch)")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.grid(True)

# Loss Curve
plt.subplot(1,2,2)
plt.plot(loss_history, color="green")
plt.title("MSE Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.show()


# STEP 8 — Compare with sklearn
from sklearn.linear_model import LinearRegression

sk_model = LinearRegression()
sk_model.fit(X_data.reshape(-1,1), Y_data)
y_pred_sklearn = sk_model.predict(X_data.reshape(-1,1))

print("\n--- Comparison ---")
print(f"Scratch Model → m: {m:.4f}, b: {b:.4f}")
print(f"sklearn Model → m: {sk_model.coef_[0]:.4f}, b: {sk_model.intercept_:.4f}")


# STEP 9 — R² Comparison
r2_sklearn = r2_score(Y_data, y_pred_sklearn)

print(f"\nR² (Scratch): {r2_custom:.4f}")
print(f"R² (sklearn): {r2_sklearn:.4f}")
print(f"R² Score (Scratch Model): {r2_custom*100:.2f}%")

# STEP 10 — Compare Loss Curve: Scratch vs Sklearn

# Compute constant MSE for sklearn model
sklearn_mse = mean_squared_error(X_data, Y_data, sk_model.coef_[0], sk_model.intercept_)
sklearn_loss_line = [sklearn_mse] * epochs  # same constant loss for all epochs

plt.figure(figsize=(10,6))
plt.plot(loss_history, label="Scratch Model Loss (MSE)", linewidth=2)
plt.plot(sklearn_loss_line, label="Sklearn Model Loss (MSE)", linestyle='--', linewidth=2)

plt.title("Loss Curve Comparison: Scratch vs Sklearn")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.legend()
plt.show()

