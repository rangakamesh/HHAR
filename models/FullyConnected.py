#Importing python packages
import numpy as np
import pandas as pd
import os
import argparse
import time

#Importing Pytorch packages
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self,l_col):
        super(Net, self).__init__()
        hidden_out = 400
        self.fc1 = nn.Linear(l_col-1, hidden_out)
        self.fc2 = nn.Linear(hidden_out, hidden_out)
        self.fc3 = nn.Linear(hidden_out, hidden_out)
        self.fc4 = nn.Linear(hidden_out, hidden_out)
        self.fc5 = nn.Linear(hidden_out, hidden_out)
        self.fc6 = nn.Linear(hidden_out, hidden_out)
        self.fc7 = nn.Linear(hidden_out, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        x = F.relu(self.fc4(x)) 
        x = F.relu(self.fc5(x)) 
        x = F.relu(self.fc6(x)) 
        x = self.fc7(x)           
        return F.log_softmax(x,1)
        # return x

class FullyConnected():
    def __init__(self,l_col):

       self.net = Net(l_col)
       self.l_col = l_col
        

    def train(self,train_loader,test_loader,lr,epochs,n_feat):
        
        net = self.net
        # batch_size=batch_sizex
        learning_rate=lr
        epochs=epochs
        log_interval=1000
        l_col = self.l_col
        total = 0
        correct = 0
        train_accuracy = list()
        epoch_times = list()
        epoch_loss = list()
        test_acc = list()
      
        # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9,nesterov=True)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate,amsgrad=True)
        
        criterion = nn.CrossEntropyLoss()
    
        for epoch in range(epochs):
          epoch_strt_time = time.time()
          for batch_idx, sub_data in enumerate(train_loader):
            data =sub_data[:,0:l_col-1].float()
            targ = sub_data[:,l_col-1].float()
            data, targ = Variable(data), Variable(targ)
            # data = data.view(-1, 3*32*32)
            optimizer.zero_grad()
            net_out = net(data.float())
            # print(net_out.shape,targ.shape)
            targ = targ.type(torch.LongTensor)
            loss = criterion(net_out, targ)
            _, predicted = torch.max(net_out.data, 1)
            total += targ.size(0)
            correct += (predicted == targ).sum().item()
                        
            if ((batch_idx % log_interval) == 0):
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data))
            loss.backward()
            optimizer.step()

          epoch_stop_time = time.time()
          tot_epoch_time = epoch_stop_time - epoch_strt_time
          epoch_acc = (correct/total)*100
          print('\n Epoch '+str(epoch)+' time take :'+str(tot_epoch_time)+' Accuracy: '+str(epoch_acc))
          print('<--------------------------------------------------------------------------------------->')
          train_accuracy.append(epoch_acc)
          epoch_times.append(tot_epoch_time)
          epoch_loss.append(loss.data)
          tacc,_,_ = self.predict(test_loader)
          test_acc.append(tacc)
        return train_accuracy,test_acc,epoch_loss,epoch_times

    def predict(self,test_loader):
        
        correct = 0
        total = 0
        net=self.net
        l_col = self.l_col
        
        
        for batch_idx, sub_data in enumerate(test_loader):
          data =sub_data[:,0:l_col-1].float()
          targ = sub_data[:,l_col-1].float()
          net_out = net(data.float())
          
          # sum up batch loss
          _, predicted = torch.max(net_out.data, 1)
          total += targ.size(0)
          correct += (predicted == targ).sum().item()
    
        #print('\nTest set: \033[1m \033[4m Accuracy:\033[0m {}/{} ({:.0f}%)\n'.format(correct, total,100. * correct / total))
        return (100. * correct / total),targ,predicted
