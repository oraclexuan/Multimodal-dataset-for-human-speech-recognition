import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据的函数
def load_data(data_dir):
    data = []
    labels = []
    files = os.listdir(data_dir)
    max_length = 0  # 最大序列长度
    for f in files:
        if f.endswith('.npy'):
            d = np.load(os.path.join(data_dir, f))
            label = int(f.split('_')[2])  # 假设标签在文件名中的位置
            max_length = max(max_length, len(d))  # 更新最大序列长度
            data.append(d)
            labels.append(label)
    return data, labels, max_length


# 根据索引获取数据的函数
def get_data_by_index(data, index):
    return [data[i] for i in index if i < len(data)]

# 加载数据
train_data_radar1, train_labels_radar1, max_length_radar1 = load_data('D:/data/uwb1_train_12people')
train_data_radar2, train_labels_radar2, max_length_radar2 = load_data('D:/data/uwb2_train_12people')

# 计算两个模态数据的最大序列长度
max_length = max(max_length_radar1, max_length_radar2)

# 加载模型
device = torch.device('cuda')  # 指定设备为GPU
models_radar1 = [torch.load(os.path.join('F:/Scientific_data/radar_data/word/UWB1/result/Python/12_people/pth', f), map_location=device) for f in os.listdir('F:/Scientific_data/radar_data/word/UWB1/result/Python/12_people') if f.endswith('.pth')]
models_radar2 = [torch.load(os.path.join('F:/Scientific_data/radar_data/word/UWB2/result/Python/12_people/pth', f), map_location=device) for f in os.listdir('F:/Scientific_data/radar_data/word/UWB2/result/Python/12_people') if f.endswith('.pth')]


# 初始化准确率列表
train_acc_list = []
val_acc_list = []

# 使用分层K折来生成训练集和验证集的索引
skf = StratifiedKFold(n_splits=5)
for fold, (train_index, val_index) in tqdm(enumerate(skf.split(train_data_radar1, train_labels_radar1)), total=5, desc='Training...'):
    # 生成训练集和验证集
    X_train_radar1, X_val_radar1 = get_data_by_index(train_data_radar1, train_index), get_data_by_index(train_data_radar1, val_index)
    X_train_radar2, X_val_radar2 = get_data_by_index(train_data_radar2, train_index), get_data_by_index(train_data_radar2, val_index)
    y_train, y_val = get_data_by_index(train_labels_radar1, train_index), get_data_by_index(train_labels_radar1, val_index)

    # 跳过迭代，如果任何数据集为空
    if not X_train_radar1 or not X_val_radar1 or not X_train_radar2 or not X_val_radar2 or not y_train or not y_val:
        print(f'Skipping Fold {fold+1} due to empty dataset.')
        continue

    # 填充序列至最大长度
    X_train_radar1 = [np.pad(d, (0, max_length - len(d)), mode='constant', constant_values=0) for d in X_train_radar1]
    X_val_radar1 = [np.pad(d, (0, max_length - len(d)), mode='constant', constant_values=0) for d in X_val_radar1]
    X_train_radar2 = [np.pad(d, (0, max_length - len(d)), mode='constant', constant_values=0) for d in X_train_radar2]
    X_val_radar2 = [np.pad(d, (0, max_length - len(d)), mode='constant', constant_values=0) for d in X_val_radar2]
    
    

    # 模型对训练集和验证集进行预测
    train_preds_radar1 = [model(torch.Tensor([x]).to(device)) for model in models_radar1 for x in X_train_radar1]
    val_preds_radar1 = [model(torch.Tensor([x]).to(device)) for model in models_radar1 for x in X_val_radar1]
    train_preds_radar2 = [model(torch.Tensor([x]).to(device)) for model in models_radar2 for x in X_train_radar2]
    val_preds_radar2 = [model(torch.Tensor([x]).to(device)) for model in models_radar2 for x in X_val_radar2]

    # 对所有模型的预测结果进行平均，得到最终的预测结果
    train_preds = np.mean(train_preds_radar1 + train_preds_radar2, axis=0)
    val_preds = np.mean(val_preds_radar1 + val_preds_radar2, axis=0)
    
    print(train_preds.shape)
    print(val_preds.shape)

#     # 评估预测结果
#     train_acc = accuracy_score(y_train, np.argmax(train_preds, axis=1))
#     val_acc = accuracy_score(y_val, np.argmax(val_preds, axis=1))
#     train_acc_list.append(train_acc)
#     val_acc_list.append(val_acc)
#     print(f'Fold {fold+1} - Train accuracy: {train_acc}, Val accuracy: {val_acc}')

#     # 生成混淆矩阵并显示
#     cm = confusion_matrix(y_val, np.argmax(val_preds, axis=1))
#     plt.figure(figsize=(10, 10))
#     sns.heatmap(cm, annot=True, fmt='d')
#     plt.title(f'Confusion Matrix for Fold {fold+1}')
#     plt.savefig(f'confusion_matrix_fold_{fold+1}.png')
#     plt.close()

#     # 输出精确度，召回率，F1分数
#     report = classification_report(y_val, np.argmax(val_preds, axis=1), target_names=[str(i) for i in range(12)])
#     print(report)

# # 生成训练进度图片
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(range(1, len(train_acc_list)+1), train_acc_list, 'b', label='Train Accuracy')
# ax.plot(range(1, len(val_acc_list)+1), val_acc_list, 'r', label='Val Accuracy')
# plt.legend()
# plt.xlabel('Fold')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.savefig('training_progress.png')
# plt.close()