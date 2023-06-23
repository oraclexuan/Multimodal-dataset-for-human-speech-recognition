import os
import sys
import argparse
import csv
import pandas as pd
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
from random import shuffle

import torch
import torch.nn as nn
from torch.nn import functional
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.utils.data import DataLoader

import loaddataset
import conf

def train(epoch, x, train_loss_list, train_acc_list, val_loss_list, val_acc_list, last_layer_outputs):

    start = time.time()
    x.append(epoch)
    train_loss = 0.0
    val_loss = 0.0
    correct_train = 0.0
    correct_val = 0.0
    net.train()
    for batch_index, (data, labels) in enumerate(train_loader):
        labels = labels.cuda()
        data = data.cuda()
        optimizer.zero_grad()
        outputs = net(data)
        loss = loss_function(outputs, labels)
        train_loss += loss.item()
        _, preds = outputs.max(1)
        correct_train += preds.eq(labels).sum()
        loss.backward()
        optimizer.step()
    train_loss_list.append(train_loss / len(train_loader.dataset))
    train_acc_list.append(correct_train.cpu() / len(train_loader.dataset))
    scheduler.step()

    torch.cuda.empty_cache()
    net.eval()
    for (data, labels) in val_loader:
        data = data.cuda()
        labels = labels.cuda()

        outputs = net(data)
        loss = loss_function(outputs, labels)
        val_loss += loss.item()
        _, preds = outputs.max(1)
        correct_val += preds.eq(labels).sum()
    print('Training Epoch: {epoch} \tLoss: {:0.4f}\tLR: {:0.9f}'.format(
        loss.item(),
        optimizer.param_groups[0]['lr'],
        epoch=epoch
    ))
    val_loss_list.append(val_loss / len(val_loader.dataset))
    val_acc_list.append(correct_val.cpu() / len(val_loader.dataset))
    # update training loss for each iteration
    finish = time.time()
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    accuracy = correct_val / len(val_loader.dataset)
    print('Average train loss: {:.4f}'.format(train_loss))
    print('Validation set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        epoch,
        val_loss,
        accuracy,
    ))
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

    torch.cuda.empty_cache()
    # Save last layer outputs
    last_layer_outputs.extend(outputs.detach().cpu())

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training the THAT model with timedata dataset")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--nclass', type=int, default=20)
    parser.add_argument('--env', type=int, default=0)
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('--data_dir', default=r'D:\data\uwb1_train_12people_compared')
    parser.add_argument('--save_dir', default=r'F:\Scientific_data\radar_data\word\UWB1\result\Python')
    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(args.device)

    net = conf.get_network(args)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),weight_decay=0.0001)
    milestones = list(range(10, 500, 10))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.65)
    step = milestones[1] - milestones[0]
    milestones = '_'.join(map(str, scheduler.milestones))
    gamma = scheduler.gamma
    filename = f"{args.lr}_{args.epoch}_{args.batch}_{args.net}_{step}_{gamma}_seed{args.seed}"
    
    datalist = os.listdir(args.data_dir)
    train_list_new = []
    test_list_new = []


    random.seed(args.seed)
    all_files = os.listdir(args.data_dir)
    shuffle(all_files)

    def get_label_for_file(filename):
        return int(filename.split("_")[2])

    all_labels = [get_label_for_file(f) for f in all_files]

    train_files, test_files, train_labels, test_labels = train_test_split(all_files, all_labels, test_size=0.1, stratify=all_labels, random_state=0)
    train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=0.2, stratify=train_labels, random_state=0)

    train_data = loaddataset.uwbDataset(args.data_dir, train_files, num_classes=args.nclass, norm=True, abs=True)
    val_data = loaddataset.uwbDataset(args.data_dir, val_files, num_classes=args.nclass, norm=True, abs=True)
    test_data = loaddataset.uwbDataset(args.data_dir, test_files, num_classes=args.nclass, norm=True, abs=True)

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch)
    test_loader = DataLoader(test_data,batch_size=args.batch)


    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    j = 0
    x = []
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    last_layer_outputs = []

    for i in range(0, args.epoch):
        train(i, x, train_loss_list, train_acc_list, val_loss_list, val_acc_list, last_layer_outputs)

        plt.close()
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax1.plot(x, train_loss_list, 'r')
        ax1.plot(x, val_loss_list, 'b')
        plt.legend(['train_loss', 'val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        ax2 = fig.add_subplot(222)
        ax2.plot(x, train_acc_list, 'g')
        ax2.plot(x, val_acc_list, 'y')
        plt.legend(['train_acc', 'val_acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.draw()
        plt.pause(1)
        plt.savefig(os.path.join(args.save_dir, f"{filename}_training_progress.png"))

        
     # Start generating confusion matrix after training...
    net.eval()

    all_labels = []
    all_predictions = []

    for (data, labels) in test_loader:
        data = data.cuda()
        labels = labels.cuda()

        outputs = net(data)
        _, preds = outputs.max(1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(preds.tolist())

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion matrix shape:", cm.shape)
    print('Training dataset size:', len(train_data))
    print('Validation dataset size:', len(val_data))
    print('Test dataset size:', len(test_data))

    row_sums = cm.sum(axis=1)
    row_sums[row_sums == 0] = 1e-10

    # Normalization
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(20,20))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    unique_classes = sorted(set(all_labels))

    class_names = [str(i) for i in unique_classes]
    average_diagonal_accuracy = np.trace(cm) / np.sum(cm)
    ax.set(xticks=np.arange(len(unique_classes)),
           yticks=np.arange(len(unique_classes)),
           xticklabels=class_names, 
           yticklabels=class_names,
           title='Normalized Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_title('Normalized Confusion matrix\nAverage Diagonal Accuracy: {:.2f}%'.format(average_diagonal_accuracy * 100))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.savefig(os.path.join(args.save_dir, f"{filename}_normalized_confusion_matrix.png"))
    torch.save(net.state_dict(), os.path.join(args.save_dir, f"{filename}_model.pth"))

    # Save last layer outputs
    last_layer_outputs = torch.cat(last_layer_outputs)
    output_file = os.path.join(args.save_dir, f"{filename}_last_layer_outputs.npy")
    np.save(output_file, last_layer_outputs.cpu().numpy())

    torch.cuda.empty_cache()
    
