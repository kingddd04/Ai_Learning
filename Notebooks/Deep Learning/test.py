import numpy as np

# Create array of zeros
arr = np.zeros((60000, 255, 255, 255), dtype=np.float32)
arr = np.expand_dims(arr, axis=-1)  # (60000, 28, 28, 1)
arr = np.repeat(arr, repeats=3, axis=-1)  # (60000, 28, 28, 3)
print(arr.shape)   # (60000, 244, 244)
print(arr.nbytes / (1024**3), "GB")  # memory size in GB