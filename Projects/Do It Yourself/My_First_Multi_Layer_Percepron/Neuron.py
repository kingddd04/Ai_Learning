import numpy as np

class Neuron:
    def __init__(self, weights_size , neuron_type="hidden"):
        self.weights = np.random.randn(weights_size)
        self.preactivation = None
        self.bias = np.random.randn() 
        self.output = None
        self.neuron_type = neuron_type

    def process(self, inputs):
        def relu(x):
            return np.maximum(0.0, x)
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        inputs = np.array(inputs, dtype=float)

        self.preactivation = np.dot(inputs, np.array(self.weights)) + self.bias

        # Apply activation depending on neuron type
        if self.neuron_type == "hidden":
            self.output = relu(self.preactivation)
        elif self.neuron_type == "output":
            self.output = sigmoid(self.preactivation)

    def get_output(self):
        return self.output
    
    def set_weights(self, weights):
        self.weights = list(weights)
    
    def set_bias(self, bias):
        self.bias = float(bias)

    def get_preactivations(self):
        return self.preactivation



if __name__ == "__main__":
    # Example usage
    neuron = Neuron(neuron_type="output")
    neuron.set_weights([0.5, -0.6, 0.2])
    neuron.set_bias(0.1)
    
    # Single input
    inputs_single = [1.0, 0.0, 1.0]
    neuron.process(inputs_single)
    print("Single input output:", neuron.get_output())

    # Batch of inputs (3 samples, each with 3 features)
    inputs_batch = [
        [1.0, 0.0, 1.0],
        [0.5, 1.0, -0.5],
        [2.0, -1.0, 0.0]
    ]
    neuron.process(inputs_batch)
    print("Batch output:", neuron.get_output())
