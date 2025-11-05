import numpy as np

# -----------------------------
# Dataset: 3-input AND gate
# -----------------------------
# Input (x1, x2, x3)
X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

# Labels (target values) for 3-input AND
# Only [1,1,1] should output 1, all others 0
y = np.array([0, 0, 0, 0, 0, 0, 0, 1])

# -----------------------------
# Initialization of parameters
# -----------------------------
w = np.array([0.0, 0.0, 0.0])  # weights for 3 features
b = 0.0                        # bias
eta = 0.1                      # learning rate
epochs = 20                    # number of training epochs

# -----------------------------
# Activation function
# -----------------------------
def step_function(z):
    return 1 if z >= 0 else 0

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(epochs):
    errors = 0
    for x, target in zip(X, y):
        # Linear combination
        z = np.dot(x, w) + b
        print(x, w, np.dot(x, w))
        # Prediction
        y_pred = step_function(z)
        # Error
        update = eta * (target - y_pred)
        # Update rule
        w += update * x
        b += update
        # Count errors
        errors += int(update != 0.0)

    if errors == 0:
        print(f"Training complete at epoch {epoch+1}")
        break

# -----------------------------
# Final results
# -----------------------------
print("Final weights:", w)
print("Final bias:", b)

# Test predictions
#for sample in X:
#    print(f"Input {sample} -> Prediction: {step_function(np.dot(sample, w) + b)}")
