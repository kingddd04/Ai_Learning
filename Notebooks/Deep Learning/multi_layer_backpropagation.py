import numpy as np   # numerical library for arrays and matrix operations

# ----- Funzioni di attivazione -----
def relu(x):
    return np.maximum(0, x)   # ReLU activation: passes positive values, zeroes negatives

def relu_derivative(x):
    return (x > 0).astype(float)   # Derivative of ReLU: 1 where x>0, else 0

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)   # Mean Squared Error loss function

# ----- Dataset XOR -----
X_train = np.array([[0.0, 0.0],   # Input samples (XOR truth table)
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0]])
y_train = np.array([[0.0],        # Target outputs (XOR labels)
                    [1.0],
                    [1.0],
                    [0.0]])

# ----- Hyperparametri -----
learning_rate = 0.1   # step size for gradient descent
epochs = 20           # number of training iterations
n_samples, n_features = X_train.shape   # 4 samples, 2 features
n_hidden1 = 4         # neurons in first hidden layer
n_hidden2 = 4         # neurons in second hidden layer
n_outputs = y_train.shape[1]   # 1 output neuron

# ----- Inizializzazione pesi -----
weights_input_hidden1 = np.random.randn(n_hidden1, n_features)   # W1: (4x2) random init
bias_hidden1 = np.zeros((1, n_hidden1))                          # b1: (1x4) zeros

weights_hidden1_hidden2 = np.random.randn(n_hidden2, n_hidden1)  # W2: (4x4) random init
bias_hidden2 = np.zeros((1, n_hidden2))                          # b2: (1x4) zeros

weights_hidden2_output = np.random.randn(n_outputs, n_hidden2)   # W3: (1x4) random init
bias_output = np.zeros((1, n_outputs))                           # b3: (1x1) zero

# ----- Training loop -----
for epoch in range(epochs):

    # ----- Forward pass -----
    z_hidden1 = X_train @ weights_input_hidden1.T + bias_hidden1   # linear transform for hidden1
    a_hidden1 = relu(z_hidden1)                                    # apply ReLU activation

    z_hidden2 = a_hidden1 @ weights_hidden1_hidden2.T + bias_hidden2  # linear transform for hidden2
    a_hidden2 = relu(z_hidden2)                                      # apply ReLU activation

    z_output = a_hidden2 @ weights_hidden2_output.T + bias_output    # linear transform for output
    predictions = z_output   # here no activation (linear regression style)

    # ----- Loss -----
    loss = mse_loss(predictions, y_train)   # compute MSE between predictions and true labels

    # ----- Backward pass -----
    # Output layer
    delta_output = (2.0 / n_samples) * (predictions - y_train)   # local error at output (dL/dZ3) # error of layer
    grad_weights_hidden2_output = delta_output.T @ a_hidden2     # dL/dW3 = δ3^T * A2 # vector of error of each layer
    grad_bias_output = delta_output.sum(axis=0, keepdims=True)   # dL/db3 = sum of δ3 over batch

    # Hidden layer 2
    delta_hidden2 = (delta_output @ weights_hidden2_output) * relu_derivative(z_hidden2) # δ2 = (δ3 * W3) ⊙ ReLU'(Z2)
    grad_weights_hidden1_hidden2 = delta_hidden2.T @ a_hidden1   # dL/dW2 = δ2^T * A1
    grad_bias_hidden2 = delta_hidden2.sum(axis=0, keepdims=True) # dL/db2 = sum of δ2

    # Hidden layer 1
    delta_hidden1 = (delta_hidden2 @ weights_hidden1_hidden2) * relu_derivative(z_hidden1)  # δ1 = (δ2 * W2) ⊙ ReLU'(Z1)
    grad_weights_input_hidden1 = delta_hidden1.T @ X_train       # dL/dW1 = δ1^T * X
    grad_bias_hidden1 = delta_hidden1.sum(axis=0, keepdims=True) # dL/db1 = sum of δ1

    # ----- Gradient descent update -----
    weights_hidden2_output -= learning_rate * grad_weights_hidden2_output   # update W3
    bias_output -= learning_rate * grad_bias_output                         # update b3

    weights_hidden1_hidden2 -= learning_rate * grad_weights_hidden1_hidden2 # update W2
    bias_hidden2 -= learning_rate * grad_bias_hidden2                       # update b2

    weights_input_hidden1 -= learning_rate * grad_weights_input_hidden1     # update W1
    bias_hidden1 -= learning_rate * grad_bias_hidden1                       # update b1

    print(f"Epoch {epoch+1} | Loss: {loss:.4f}")   # monitor training progress
