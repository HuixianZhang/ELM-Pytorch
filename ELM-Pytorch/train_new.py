import argparse
import torch
from models import ELM
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 6
hidden_size = 20
num_classes = 1

torch.set_default_dtype(torch.float64)

data = pd.read_csv('/home/alissa/Downloads/ELM-Pytorch/AAPL.csv')

#Split Data
train_data = data[:3020]
test_data = data[3021:]

#First calculate the weight of each indicator
open_data = train_data['Open']
high_data = train_data['High']
low_data = train_data['Low']
volume_data = train_data['Volume']
target_data = train_data['Close']

open_data_test = test_data['Open']
high_data_test = test_data['High']
low_data_test = test_data['Low']
volume_data_test = test_data['Volume']
target_data_test = test_data['Close']

open_df = abs(open_data - target_data)
high_df = abs(high_data - target_data)
low_df = abs(low_data - target_data)
volume_df = abs(volume_data - target_data)

weight_open = sum((min(open_df) + 0.5*max(open_df))/(0.5*max(open_df) + open_df))/len(open_df)
weight_high = sum((min(high_df) + 0.5*max(high_df))/(0.5*max(high_df) + high_df))/len(high_df)
weight_low = sum((min(low_df) + 0.5*max(low_df))/(0.5*max(low_df) + high_df))/len(low_df)
weight_volume = sum((min(volume_df) + 0.5*max(volume_df))/(0.5*max(volume_df) + volume_df))/len(volume_df)

#Normalize indicators
normalize_open = -1 + 2*(open_data - min(open_data))/(max(open_data)- min(open_data))
normalize_high = -1 + 2*(high_data - min(high_data))/(max(high_data)- min(high_data))
normalize_low = -1 + 2*(low_data - min(low_data))/(max(low_data)- min(low_data))
normalize_volume = -1 + 2*(volume_data - min(volume_data))/(max(volume_data)- min(volume_data))
normalize_target = -1 + 2*(target_data - min(target_data))/(max(target_data)- min(target_data))

normalize_open_test = -1 + 2*(open_data_test - min(open_data_test))/(max(open_data_test)- min(open_data_test))
normalize_high_test = -1 + 2*(high_data_test - min(high_data_test))/(max(high_data_test)- min(high_data_test))
normalize_low_test = -1 + 2*(low_data_test - min(low_data_test))/(max(low_data_test)- min(low_data_test))
normalize_volume_test = -1 + 2*(volume_data_test - min(volume_data_test))/(max(volume_data_test)- min(volume_data_test))
normalize_target_test = -1 + 2*(target_data_test - min(target_data_test))/(max(target_data_test)- min(target_data_test))
#print(normalize_open_test)
open_test = torch.Tensor(normalize_open_test.values)
high_test = torch.Tensor(normalize_high_test.values)
low_test = torch.Tensor(normalize_low_test.values)
volume_test = torch.Tensor(normalize_volume_test.values)
target_test = torch.Tensor(normalize_target_test.values)
cat_test = torch.stack((open_test,high_test,low_test,volume_test,target_test),0)

open = torch.Tensor(normalize_open)
high = torch.Tensor(normalize_high)
low = torch.Tensor(normalize_low)
volume = torch.Tensor(normalize_volume)
target = torch.Tensor(normalize_target)
cat = torch.stack((open,high,low,volume,target),0)

n = 6
L = len(cat[0])
for i in range(L-n):
	cat_seq = cat[:,i:i+n]
	up = max(target[i:i+n])
	low = min(target[i:i+n])
	Up = torch.full((1, n), up)
	Low = torch.full((1, n), low)
	train_seq = torch.cat((cat_seq, Up, Low),0)
	train_label = target[i+n:i+2*n]
	train_label_new = train_label.repeat(7, 1)
	train_seq = train_seq.cuda()
	train_label_new = train_label_new.cuda()
#print(train_label)
	elm = ELM(input_size=image_size, h_size=hidden_size, num_classes=num_classes,device = device)
	#elm = ELM(input_size=image_size, h_size=hidden_size, num_classes=num_classes)
	elm.fit(train_seq, train_label_new)

acc = []
Length = len(cat_test[0])
for i in range(Length - n):
	cat_seq_test = cat_test[:,i:i+n]
	up_test = max(target_test[i:i+n])
	low_test = min(target_test[i:i+n])
	Up_test = torch.full((1, n), up_test)
	Low_test = torch.full((1, n), low_test)
	test_seq = torch.cat((cat_seq_test, Up_test, Low_test),0)
	test_label = target_test[i+n:i+2*n]
	test_label_new = test_label.repeat(7, 1)
	test_seq = test_seq.cuda()
	test_label_new = test_label_new.cuda()

	accuracy,loss = elm.evaluate(test_seq, test_label_new)

	print('Accuracy: {}'.format(accuracy))
	print('Loss: {}'.format(loss))
	acc.append(accuracy)
sum = 0
for i in range(len(acc)):
	sum = sum + acc[i]
acc_f = sum / len(acc)
print(acc_f)

