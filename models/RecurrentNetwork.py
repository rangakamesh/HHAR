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

class RNN(nn.Module):
    def __init__(self, batch_size, n_inputs, n_neurons):
        super(RNN, self).__init__()
        
        self.batch_size = batch_size
        self.n_neurons = n_neurons

        self.rnn = nn.RNNCell(n_inputs, n_neurons)
        self.hx = torch.randn(batch_size, n_neurons) # initialize hidden state
        self.FC = nn.Linear(n_neurons, 6)

    def init_hidden(self):
        self.hx = torch.randn(self.batch_size, self.n_neurons)
        
    def forward(self, X):
        # for each time step
        for i in range(4):
          self.hx = F.relu(self.rnn(X, self.hx))
          self.out = self.FC(self.hx)
        return F.log_softmax(self.out,1)

class RecurrentNetwork():
  def __init__(self,batch_size,n_inputs,n_neurons,lr,epochs):
    
    self.batch_size = batch_size
    self.n_neurons = n_neurons
    self.n_inputs = n_inputs
    self.lr = lr
    self.epochs = epochs

    self.net = RNN(batch_size,n_inputs,n_neurons)
  
  def train(self,train_loader,validation_loader):
    BATCH_SIZE = self.batch_size
    N_INPUT = self.n_inputs
    N_NEURONS = self.n_neurons
    lr = self.lr

    acc_list = list()
    test_list = list()
    time_list = list()
    loss_list = list()

    batch_size=BATCH_SIZE

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model instance
    model = self.net
    # checkpoint = torch.load('./checkpoint/rnn_net.pth')
    # model.load_state_dict(checkpoint['net'])
    # net=self.net

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(self.epochs):  # loop over the dataset multiple times
        epoch_strt_time = time.time()
        train_running_loss = 0.0
        train_acc = 0.0
        model.train()
        print("In Epoch:",epoch)
        
        # TRAINING ROUND
        for i, data in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # reset hidden states
            model.hidden = model.init_hidden() 

            sub_data= data
            inputs = sub_data[:,0:9].float()
            labels =  sub_data[:,9].float()

            # forward + backward + optimize
            outputs = model(inputs)
            labels = labels.type(torch.LongTensor)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            train_running_loss += loss.detach().item()
            mba = self.get_accuracy(outputs, labels, BATCH_SIZE)
            train_acc += mba
            
        model.eval()
        print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' 
              %(epoch, train_running_loss / i, train_acc/i))

        epoch_stop_time = time.time()
        test_acc,_,_ = self.predict(validation_loader)
        test_list.append(test_acc)
        tot_epoch_time = epoch_stop_time - epoch_strt_time
        time_list.append(tot_epoch_time)
        acc_list.append(train_acc/i)
        loss_list.append(train_running_loss/i)
        print('Epoch '+str(epoch)+' time take :'+str(tot_epoch_time))
    return acc_list,test_list,loss_list,time_list

  def get_accuracy(self,logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    # print(torch.max(logit, 1)[1].view(target.size()))
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data) .sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

  def predict(self,test_loader):
    correct = 0
    total = 0
    model = self.net

    for batch_idx, sub_data in enumerate(test_loader):
      data =sub_data[:,0:9].float()
      targ = sub_data[:,9].float()
      net_out = model(data.float())

      # sum up batch loss
      _, predicted = torch.max(net_out.data, 1)
      total += targ.size(0)
      correct += (predicted == targ).sum().item()

    # print('\nTest set: \033[1m \033[4m Accuracy:\033[0m {}/{} ({:.0f}%)\n'.format(correct, total,100. * correct / total))
    return (100. * correct / total),targ,predicted