import numpy as np
import pandas as pd

# Generate random input features (e.g., just random values between 0 and 1)
# These could represent something like pixel intensities or image attributes
num_samples = 1000  # Number of samples in the dataset
input_features = np.random.rand(num_samples, 3)  # 3 features for simplicity

# Generate random RGB outputs (also values between 0 and 255 for raw RGB values)
rgb_outputs = np.random.randint(0, 256, size=(num_samples, 3))

# Create a DataFrame for easy visualization or saving
dataset = pd.DataFrame(input_features, columns=['Feature1', 'Feature2', 'Feature3'])
dataset['R'] = rgb_outputs[:, 0]
dataset['G'] = rgb_outputs[:, 1]
dataset['B'] = rgb_outputs[:, 2]

# Save to CSV
dataset.to_csv('synthetic_rgb_dataset.csv', index=False)

print(dataset.head())