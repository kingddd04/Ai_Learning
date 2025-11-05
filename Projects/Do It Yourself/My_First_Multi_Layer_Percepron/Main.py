from Dataset_loader import Dataset_loader
from Multi_Layer_Neural_Perceptron import Multi_Layer_Neural_Perceptron
import numpy as np

if __name__ == "__main__":

    dataset_loader = Dataset_loader("Cardiovascular_Disease_Dataset.csv")
    data, labels = dataset_loader.get_x_y()

    mlp = Multi_Layer_Neural_Perceptron([12, 24, 34,13 ,6], 0.001)
    mlp.train(data, labels, epochs=1000)

    # Test prediction after training
    test_sample = data[0]
    prediction = mlp.process(test_sample)
    print(f"Test sample prediction: {prediction}, True label: {labels[0]}")
    test_sample2 = data[1]
    prediction2 = mlp.process(test_sample2)
    print(f"Test sample prediction: {prediction2}, True label: {labels[1]}")