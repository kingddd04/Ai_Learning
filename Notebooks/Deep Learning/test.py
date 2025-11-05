import numpy as np

# -----------------------------
# Utility functions
# -----------------------------

def leaky_relu(x, negative_slope=0.01):
    return np.where(x > 0, x, negative_slope * x)

def leaky_relu_derivative(x, negative_slope=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = negative_slope
    return dx

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability trick
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    # y_true is one-hot encoded
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
    return np.sum(log_likelihood) / m

# -----------------------------
# Neural Network Class
# -----------------------------

class SimpleFFNN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01, negative_slope=0.01):
        self.lr = lr
        self.negative_slope = negative_slope
        
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))
        
        self.W3 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b3 = np.zeros((1, output_dim))
    
    def forward(self, X):
        # Layer 1
        self.z1 = X @ self.W1 + self.b1
        self.a1 = leaky_relu(self.z1, self.negative_slope)
        
        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = leaky_relu(self.z2, self.negative_slope)
        
        # Output layer
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = softmax(self.z3)
        
        return self.a3
    
    def backward(self, X, y_true):
        m = X.shape[0]
        
        # Output layer error
        dz3 = self.a3 - y_true
        dW3 = (self.a2.T @ dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Hidden layer 2
        da2 = dz3 @ self.W3.T
        dz2 = da2 * leaky_relu_derivative(self.z2, self.negative_slope)
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer 1
        da1 = dz2 @ self.W2.T
        dz1 = da1 * leaky_relu_derivative(self.z1, self.negative_slope)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Gradient descent update
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = cross_entropy_loss(y_pred, y)
            self.backward(X, y)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    np.random.seed(42)
    
    # Dummy dataset: 4 samples, 2 features, 2 classes
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[1,0],[0,1],[0,1],[1,0]])  # one-hot
    
    model = SimpleFFNN(input_dim=2, hidden_dim=4, output_dim=2, lr=0.1)
    model.train(X, y, epochs=1000)
    
