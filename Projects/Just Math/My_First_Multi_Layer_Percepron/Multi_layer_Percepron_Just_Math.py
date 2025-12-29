import numpy as np


# --- Activation functions ---
def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def binary_cross_entropy(y_pred, y_true, eps=1e-15):
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Compute BCE
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


# --- Hyperparameters ---

mlp_architecture = [4, 6, 8, 5, 1]  # Input layer: 4, Hidden layers: 4 neurons each, Output layer: 1
epochs = 200
batch_size = 4
lr = 0.1

# --- Synthetic data ---
X_train = np.array([[0.2042004490654904, -0.981643337710476, 1.1668681601409305, -0.16473182061373173],
                    [0.3464193932125635, 0.6346366688888685, 0.3445628142342713, 0.8461254884365395],
                    [-0.5156282344121331, -0.9090594797458548, 2.262211517785421, -0.39046867551992726],
                    [0.2216842728597084, -1.1040425196698564, 0.3098869825878104, -0.6524148138957611],
                    [-0.3732548752786023, -0.7517203256622179, -1.8577774119602624, 0.7766059444189485],
                    [0.4480094116904102, 0.8590583902667492, -1.4245952456290907, 0.5238287571746911],
                    [1.1767489262063602, -1.5405675091091204, -1.1653344977807831, -0.4970465416316781],
                    [1.0839898883407273, 0.18500609595561965, -0.3675551566263139, -2.4036212863624487],
                    [0.945599517902921, -0.08712828049201037, 0.025329931567315975, -1.2151272792870456],
                    [0.8740954299984632, -0.9367257397000532, 0.11829572551490666, 2.348459753830983],
                    [1.0437799967436923, -0.7249833976034493, 0.04942126636187027, -1.0083117602851286],
                    [1.1459400766468815, 0.2013541776499479, -0.3576440625551027, -0.7999637672250738],
                    [0.0590885363964252, 0.28562836704470135, 1.3831081110649832, 0.5191410964746214],
                    [1.154845774178582, 0.8563761879564187, 0.63542932597061, -0.16029467836488684],
                    [1.0394374906829318, -1.2538640490970918, 0.2591224499096588, 0.581847860724566],
                    [-1.2258685337789483, -0.8518965654879387, -0.4089839254733032, -0.7474956476844435],
                    [-0.003216825452349415, -0.20485089192122277, -0.0017734324998246972, 0.4368549120801247],
                    [-1.541099617761669, 1.9993419139119637, 1.212083584347714, 0.8425575417960754],
                    [1.1731537436689818, -0.5350981692714062, 0.8570692883616162, 0.006031983494241065],
                    [-0.47335774115890417, 0.4717407360443464, -0.6111651093865168, -0.9796470895680595]])
Y_train = np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1])

n_samples = X_train.shape[0]

#               weight_size(no_of_inputs), no_of_neurons
W1 = np.random.randn(mlp_architecture[0], mlp_architecture[1]) * 0.1
b1 = np.zeros(mlp_architecture[1])

W2 = np.random.randn(mlp_architecture[1], mlp_architecture[2]) * 0.1
b2 = np.zeros(mlp_architecture[2])

W3 = np.random.randn(mlp_architecture[2], mlp_architecture[3]) * 0.1
b3 = np.zeros(mlp_architecture[3])

W4 = np.random.randn(mlp_architecture[3], mlp_architecture[4]) * 0.1
b4 = np.zeros(mlp_architecture[4])

for epoch in range(epochs):
    losses = []
    for index in range(0, n_samples, batch_size):
        end = index + batch_size
        Xb = X_train[index:end]
        Yb = Y_train[index:end].reshape(-1, 1)

        # === Forward pass ===
        Z1 = Xb @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        A2 = relu(Z2)
        Z3 = A2 @ W3 + b3
        A3 = relu(Z3)
        Z4 = A3 @ W4 + b4
        A4 = sigmoid(Z4)

        loss = binary_cross_entropy(A4, Yb)
        losses.append(loss)

        # === Backward pass ===
        d4 = (A4 - Yb)
        dW4 = A3.T @ d4
        db4 = d4.sum(axis=0)
        dA4 = d4 @ W4.T  # d4: (batch, 1), W(5, 1) -> W.T(1, 5)

        d3 = dA4 * relu_deriv(Z3)  # errore locale deel layer: (batch, 5) # 1, 5
        dW3 = A2.T @ d3  # A(batch, 8) -> A2.T(8, batch) @ (batch, 5): (8, 5)
        db3 = d3.sum(axis=0)
        dA3 = d3 @ W3.T

        d2 = dA3 * relu_deriv(Z2)
        dW2 = A1.T @ d2
        db2 = d2.sum(axis=0)
        dA2 = d2 @ W2.T

        d1 = dA2 * relu_deriv(Z1)
        dW1 = Xb.T @ d1
        db1 = d1.sum(axis=0)

        # === Update weights and biases ===
        W4 -= lr * dW4
        b4 -= lr * db4
        W3 -= lr * dW3
        b3 -= lr * db3
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        #print(f"mini batch [{index}][{end}] loss: {loss}")

    if epoch % 10 == 0:
        print(f"Epoca {epoch+1}/{epochs} loss: {np.mean(losses):.4f}")







