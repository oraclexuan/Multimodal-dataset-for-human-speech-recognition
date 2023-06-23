import numpy as np

# # 指定 Numpy 文件的路径
# output_file = r'F:\Scientific_data\radar_data\word\UWB2\result\Python\0.01_100_32_resnet18_10_0.65_seed2_last_layer_outputs.npy'

# # 读取 Numpy 文件
# loaded_outputs = np.load(output_file)

# # 在这里可以使用 loaded_outputs 进行后续操作
# # 例如打印数组的形状和内容
# print("Shape of loaded_outputs:", loaded_outputs.shape)
# print("Content of loaded_outputs:")
# print(loaded_outputs)

# Load your array
array = np.load(r'F:\Scientific_data\radar_data\word\UWB1\result\Python\0.01_100_32_resnet18_10_0.65_seed0_last_layer_outputs.npy')

# Print the shape of the array
print(array.shape)
