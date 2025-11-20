import numpy as np
from Neuron import Neuron

class Neuron_Layer:
    def __init__(self, layer_weights_size, layer_number_neurons,  layer_type, layer_id):
        self.neurons = [Neuron(layer_weights_size, layer_type) for _ in range(layer_number_neurons)]
        print(self.neurons)
        self.layer_type = layer_type
        self.layer_id = layer_id

    def process(self, inputs):
        for neuron in self.neurons:
            neuron.process(inputs)

    def get_outputs(self):
        return np.array([neuron.get_output() for neuron in self.neurons])
    
    def get_preactivations(self):
        return np.array([neuron.get_preactivations() for neuron in self.neurons])
    
    def get_weights(self):
        return np.array([neuron.weights for neuron in self.neurons])
    
    def set_weights(self, weights_list):
        for neuron, weights in zip(self.neurons, weights_list):
            neuron.set_weights(weights)

    def get_biases(self):
        return np.array([neuron.bias for neuron in self.neurons])

    def set_biases(self, biases):
        for neuron, bias in zip(self.neurons, biases):
            neuron.set_bias(bias)

    def get_layer_type(self):
        return self.layer_type
    
    def get_layer_id(self):
        return self.layer_id
    
if __name__ == "__main__":
    # Example: create a layer with 3 neurons, each expecting 4 inputs
    input_size = 4
    num_neurons = 3
    layer = Neuron_Layer(layer_weights_size=input_size,
                         layer_number_neurons=num_neurons,
                         layer_type="hidden",
                         layer_id=1)

    # Generate a random input vector of size 4
    sample_input = np.array([0.5, -1.2, 0.3, 2.0])
    print("\nFeeding input:", sample_input)

    # Forward pass through the layer
    layer.process(sample_input)

    # Collect outputs
    outputs = layer.get_outputs()
    preacts = layer.get_preactivations()
    weights = layer.get_weights()
    biases = layer.get_biases()

    print("\n--- Layer Results ---")
    print("Outputs:", outputs)
    print("Pre-activations:", preacts)
    print("Weights:", weights)
    print("Biases:", biases)