import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset (60,000 training images, 10,000 test images)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build a simple feedforward neural network
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),          # Flatten 28x28 images into 784 vector
    layers.Dense(128, activation='relu'),          # Hidden layer
    layers.Dropout(0.2),                           # Dropout for regularization
    layers.Dense(10, activation='softmax')         # Output layer (10 classes)d
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("\nTest accuracy:", test_acc)
