import numpy as np
import pandas as pd
from Neuron_Layer import Neuron_Layer

class Multi_Layer_Neural_Perceptron:
    def __init__(self,mlp_architecture, learning_rate):
        self.mlp_architecture = mlp_architecture
        self.layers = []
        self.learning_rate = learning_rate
        for n in range(1,len(self.mlp_architecture)): 
            layer_weight_size = self.mlp_architecture[n-1]
            layer_neurons_number = self.mlp_architecture[n] 
            layer = Neuron_Layer(layer_weight_size, layer_neurons_number, "hidden",n)
            self.layers.append(layer) 
        self.layers.append(Neuron_Layer(self.mlp_architecture[-1],1, "output", len(self.mlp_architecture)))

    def process(self, inputs):
        current_inputs = inputs
        for layer in self.layers[:-1]:
            layer.process(current_inputs)
            current_inputs = layer.get_outputs()
        self.layers[-1].process(current_inputs)
        prediction = self.layers[-1].get_outputs()[0]
        return prediction
    

    def train(self, training_data, labels, epochs):
        def binary_cross_entropy(y_true, y_pred):
            eps = 1e-8  # avoid log(0)
            y_pred = np.clip(y_pred, eps, 1 - eps)
            loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            return loss

        
        for epoch in range(epochs):
            epoch_losses = []
            for index in range(len(training_data)):
                x, y = training_data[index], labels[index]
                prediction = self.process(x)
                loss = binary_cross_entropy(np.array([y]), np.array([prediction]))
                epoch_losses.append(loss)
                mlp_state = self.get_perceptron_state()
                self.backpropagation(x,prediction,y, mlp_state)
                print_perceptron_state(mlp_state)
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")
            
                

    def backpropagation(self, x, prediction, true_value, mlp_state):
        def sigmoid(x):
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x):
            s = sigmoid(x)
            return s * (1 - s)

        def relu_derivative(x):
            return (x > 0).astype(float)

        backprop_info = {}
        for layer in reversed(self.layers):
            layer_id = layer.get_layer_id()
            if layer.get_layer_type() == "output":
                # Error at output
                delta_layer = np.array([prediction - true_value])
                # Inputs to output layer = outputs of last hidden layer
                inputs_to_layer = mlp_state[layer_id - 1]["outputs"]
                delta_weights = np.outer(delta_layer, inputs_to_layer)
                delta_biases = delta_layer
                backprop_info[layer_id] = {
                    "layer_delta": delta_layer,
                    "delta_weights": delta_weights,
                    "delta_biases": delta_biases
                }

            elif layer.get_layer_type() == "hidden":
                # Propagate error backwards
                
                delta_hidden_temp = mlp_state[layer_id + 1]["weights"].T @ backprop_info[layer_id + 1]["layer_delta"]
                delta_hidden = delta_hidden_temp * relu_derivative(mlp_state[layer_id]["preactivations"])
                print("delta hidden", delta_hidden_temp, delta_hidden)

                # Inputs to this layer
                if layer_id == 1:
                    inputs_to_layer = x
                else:
                    inputs_to_layer = mlp_state[layer_id - 1]["outputs"]

                delta_weights = np.outer(delta_hidden, inputs_to_layer)

                delta_biases = delta_hidden
                backprop_info[layer_id] = {
                    "layer_delta": delta_hidden,
                    "delta_weights": delta_weights,
                    "delta_biases": delta_biases
                }

        self.gradient_descent(backprop_info)


    def gradient_descent(self,backprop_info):
        for layer in self.layers:
            layer_id = layer.get_layer_id()
            layer_weights = layer.get_weights()
            layer_biases = layer.get_biases()
            delta_weights = backprop_info[layer_id]["delta_weights"]
            delta_biases = backprop_info[layer_id]["delta_biases"]
            layer_weights -= self.learning_rate * delta_weights
            layer_biases -= self.learning_rate * delta_biases
            layer.set_weights(layer_weights)
            layer.set_biases(layer_biases)

    def get_perceptron_state(self):
        mlp_full_state = {}
        for index, layer in enumerate(self.layers):

            mlp_full_state[layer.get_layer_id()] = {
                "preactivations": np.array(layer.get_preactivations()),
                "outputs": np.array(layer.get_outputs()),
                "weights": np.array(layer.get_weights()),
                "biases": np.array(layer.get_biases())
            }
        return mlp_full_state

def print_perceptron_state(mlp_state):
    for layer_id, layer_state in mlp_state.items():
        print(f"Layer {layer_id}:")
        print(f"  Preactivations: {layer_state['preactivations']}")
        print(f"  Outputs: {layer_state['outputs']}")
        print(f"  Weights: {layer_state['weights']}")
        print(f"  Biases: {layer_state['biases']}")
                


def main():
    # Architecture: 2 inputs -> 4 hidden -> 1 output
    mlp = Multi_Layer_Neural_Perceptron([2, 2, 2], learning_rate=0.01)

    # XOR dataset
    training_data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    labels = np.array([0, 1, 1, 0])

    # Train
    mlp.train(training_data, labels, epochs=2)
    mlp

    

    # Test predictions
    for x, y in zip(training_data, labels):
        pred = mlp.process(x)
        print(f"Input: {x}, True: {y}, Predicted: {pred:.4f}")

if __name__ == "__main__":
    main()