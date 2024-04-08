# -*- coding: utf-8 -*-
"""timbre_model_6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kZDN7Z9b2B6xCah-p2Gbz-2Vj8FQBeKR
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import IPython.display as ipd
import csv
import random
from sklearn.manifold import TSNE
from info_nce import InfoNCE, info_nce
from dataloader import DataLoader

train_input_data_path = 'your_path'
train_mel_spectrogram_path = 'your_path'
test_input_data_path = 'your_path'
test_mel_spectrogram_path = 'your_path'
model_path = 'your_path'
optimizer_path = 'your_path'

"""# 모델 선언"""

class Extract_Model(nn.Module):
    def __init__(self):
        super(Extract_Model, self).__init__()#input.shape == batch,128,157
        self.conv0 = nn.Conv2d(1, 256, kernel_size=5, padding=2, padding_mode='zeros', stride=1)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=5, padding=2, padding_mode='zeros', stride=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=5, padding=2, padding_mode='zeros', stride=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='zeros', stride=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='zeros', stride=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='zeros', stride=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='zeros', stride=1)
        self.pool = nn.MaxPool2d(2)
        self.fc0 = nn.Linear(8*9*256, 512)
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 1, 128, 157)
        # x = (x/80)+1
        x = self.pool(self.relu(self.conv0(x)))#output.shape == 64,78
        y = x
        x = self.relu(self.conv1(x))#output.shape == 64,78
        x = self.relu(self.conv2(x))#output.shape == 64,78
        x = self.relu(self.conv3(x)+y)#output.shape == 64,78
        x = self.pool(self.relu(self.conv4(x)))#output.shape == 32,39
        x = self.pool(self.relu(self.conv5(x)))#output.shape == 16,19
        x = self.pool(self.relu(self.conv5(x)))#output.shape == 8,9
        x = torch.flatten(x,1)
        x = self.fc0(x)
        x = self.fc1(x)
        return x #output.shape == batch, 256

device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
# extract_model = Extract_Model().to(device)

extract_model = Extract_Model()
extract_model.load_state_dict(torch.load(model_path))
extract_model.to(device)

optimizer = optim.Adam(extract_model.parameters(),lr=1e-6)

criterion = InfoNCE(negative_mode='paired')

"""# Test_show()"""

test_data = pd.read_csv('test_input_data 복사본.csv')
final_test_dataset = test_data.sort_values('name').reset_index().iloc[:,1:]
final_test_dataset= final_test_dataset.iloc[[0,10,20,30,40,50,60,70,80,90]].reset_index().iloc[:,1:]
test_artist_list = final_test_dataset['name'].unique().tolist()

# final_test_dataset= final_test_dataset[:100].reset_index().iloc[:,1:]
# test_artist_list = final_test_dataset['name'].unique().tolist()

# for i in range(11,100):
#         print(test_mel_spectrogram_path+ final_test_dataset['file'][i])
#         print(final_test_dataset['name'][i])

import platform
from unicodedata import normalize
from dataloader import trim_mel

from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus']= False

if platform.system() == 'Darwin': # 맥os 사용자의 경우에
    rc('font', family = 'AppleGothic')

elif platform.system() == 'Windows': # 윈도우 사용자의 경우에
    path = 'c:/Windows/Fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)

def test_show():   # 최종 테스트 셋 220개에 대해서 보여줌
    labels = []

    x = torch.tensor(trim_mel(np.load(test_mel_spectrogram_path+ final_test_dataset['file'][0])).reshape(1,128,157)).to(device)
    x = extract_model(x).cpu().detach().numpy()
    labels.append(final_test_dataset['name'][0])

    for j in range(9):
        song = np.load(test_mel_spectrogram_path+ final_test_dataset['file'][0])
        rd = random.randint(0,song.shape[1]-200)
        mel = torch.tensor(np.array(song[:,rd:rd+157]).reshape(1,128,157)).to(device)
        mel = extract_model(mel).cpu().detach().numpy()
        x = np.concatenate((x,mel),axis=0)
        labels.append(final_test_dataset['name'][0])


    for i in range(1,10):
        for j in range(10):
            song = np.load(test_mel_spectrogram_path+ final_test_dataset['file'][i])
            rd = random.randint(0,song.shape[1]-200)
            mel = torch.tensor(np.array(song[:,rd:rd+157]).reshape(1,128,157)).to(device)
            mel = extract_model(mel).cpu().detach().numpy()
            x = np.concatenate((x,mel),axis=0)
            labels.append(final_test_dataset['name'][0])


    tsne = TSNE(n_components=2, random_state=1,perplexity=40)
    X_tsne = tsne.fit_transform(x)

    sample_sizes = [10] * 10
    labels_uni = []
    for i in test_artist_list:
        labels_uni.append(normalize('NFC', i))

    y = np.array(labels_uni)
    singer_names = labels_uni

    plt.figure(figsize=(10, 8))
    for i in range(len(sample_sizes)):
        plt.scatter(X_tsne[i * sample_sizes[i]:(i + 1) * sample_sizes[i], 0],
                    X_tsne[i * sample_sizes[i]:(i + 1) * sample_sizes[i], 1],label=singer_names[i])

    print('how many different voice :', np.unique(y).shape[0])
    plt.legend(labels=singer_names, loc='upper right')
    plt.show()

"""# 학습"""

# dataloader = DataLoader()
# for batch in dataloader.generate_batch()[:1]:
#     for sample in batch:
#         anc, pos, neg = sample[0], sample[1], sample[2:7]
#         print("Anchor:", anc)
#         print("Positive:", pos)
#         print("Negatives:", neg)
#         print(len(sample))
#     print(len(batch))

# np.array(batch)[:,0].shape # curr

# np.array(batch)[:,1].shape # pos

# np.array(batch)[:,2:].shape # neg

def train(batch):#배치로 넣음 (batch,7,128,157) anchor1 +1 -5
    output = [extract_model(batch[:,x]) for x in range(7)]

    query = output[0]
    positive_key = output[1]
    negative_keys = torch.stack(output[2:],dim=1)
    loss = criterion(query, positive_key, negative_keys)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

# def test(batch):#배치로 넣음 (batch,7,128,157) anchor1 +1 -5
#     output = [extract_model(batch[:,x]) for x in range(7)]
#
#     query = output[0]
#     positive_key = output[1]
#     negative_keys = torch.stack(output[2:],dim=1)
#     loss = criterion(query, positive_key, negative_keys)
#     optimizer.zero_grad()
#
#     return loss

def evaluate(test_batch):
    extract_model.eval()

    with torch.no_grad():
        output = [extract_model(test_batch[:,x]) for x in range(7)]
        query = output[0]
        positive_key = output[1]
        negative_keys = torch.stack(output[2:],dim=1)
        loss = criterion(query, positive_key, negative_keys)

    return loss

dataloader = DataLoader()
train_loss_visual = [2, 1.4, 1.3075682901786172, 1.2109897223224149]
test_loss_visual = [2, 1.75, 1.5586630759439766, 1.4235615837040232]
lowest_loss = 1.4235615837040232

epoch = 3

for i in range(epoch):
    mean_count = 0
    mini = 0

    loss_mean = 0
    loss_epoch_mean = 0

    test_loss_mean = 0
    test_loss_epoch_mean = 0

    batches = dataloader.generate_batch()

    for batch in tqdm(batches):
        mean_count = mean_count + 1

        train_loss = train(torch.tensor(np.array(batch)).to(device)).cpu().detach().numpy()    # 학습 loss

        test_batch = dataloader.generate_test_batch()
        # test_loss = test(torch.tensor(np.array(test_batch)).to(device)).cpu().detach().numpy() # 테스트 loss
        test_loss  = evaluate(torch.tensor(np.array(test_batch)).to(device)).cpu().detach().numpy()

        loss_mean += train_loss
        loss_epoch_mean += train_loss

        test_loss_mean += test_loss
        test_loss_epoch_mean += test_loss

        mini = mini + 1
        if(mini% 100 == 0):
            print('loss :',loss_mean/100)
            print('test_loss :',test_loss_mean/100)
            print(i+mini*32/3409,'epoch : ', i+1)
            print('--------------------')
            loss_mean = 0
            test_loss_mean = 0

    test_show()

    train_loss_visual.append(loss_epoch_mean/mean_count)
    test_loss_visual.append(test_loss_epoch_mean/mean_count)
    plt.plot(train_loss_visual,label='train loss')
    plt.plot(test_loss_visual,label='test loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    print('epoch_mean :',loss_epoch_mean/mean_count)
    print('test_epoch_mean :',test_loss_epoch_mean/mean_count)
    print(mean_count)

    if test_loss_epoch_mean/mean_count < lowest_loss:
        lowest_loss = test_loss_epoch_mean/mean_count
        torch.save(extract_model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)

"""# Test SHow 불러오기"""

torch.save(extract_model.state_dict(), model_path)
torch.save(optimizer.state_dict(), optimizer_path)

extract_model = Extract_Model()
extract_model.load_state_dict(torch.load(model_path))
extract_model.to(device)

optimizer.load_state_dict(torch.load(optimizer_path))

test_show()