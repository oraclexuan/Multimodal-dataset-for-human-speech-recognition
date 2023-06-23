import numpy as np

# Load the first file
data1 = np.load(r'F:\Scientific_data\radar_data\word\UWB1\result\Python\0.01_100_32_resnet18_10_0.65_seed2_last_layer_outputs.npy')

# Load the second file
data2 = np.load(r'F:\Scientific_data\radar_data\word\UWB1\result\Python\0.01_100_32_resnet18_10_0.65_seed2_last_layer_outputs.npy')

# Add the two arrays together
sum_data = data1 + data2

# Specify the path where you want to save the file
save_path = r'F:\Scientific_data\radar_data\word\UWB1+2\result\seed2_sum_data.npy'

# Save the sum into a new file at the specified path
np.save(save_path, sum_data)
