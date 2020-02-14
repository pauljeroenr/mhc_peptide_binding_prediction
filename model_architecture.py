#!/usr/bin/env python
# coding: utf-8

# In[3]:


from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


# In[4]:


# add dynamic size for linear layer
class convnet(nn.Module):
    def __init__(self, inputchannel, L, dropout):
        
        super(convnet, self).__init__()    
        self.L = L
        self.inputchannel = inputchannel
        self.dropout = dropout
        self.conv1 = nn.Conv2d(self.inputchannel * 2, 40 , 3, stride=(1,1), padding=(1,1), dilation=(1,1))
        self.conv1_bn = nn.BatchNorm2d(40)
        self.conv2 = nn.Conv2d(40, 50, 3, stride=(1,1), padding=(1,1), dilation=(1,1))
        self.conv2_bn = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 30, 5, stride=(1,1), padding=(2,2), dilation=(1,1))
        self.conv3_bn = nn.BatchNorm2d(30)
        self.conv4 = nn.Conv2d(30, 1, 3, stride=(1,1), padding=(1,1), dilation=(1,1))
        self.conv4_bn = nn.BatchNorm2d(1)

        self.L1 = nn.Linear(self.L * self.L, 1)
        self.dropout = nn.Dropout2d(self.dropout)
        self.dropoutearly = nn.Dropout2d(0.0)
        #self.dropout = nn.Dropout2d(0.1) #2d use lower dropout because of spatial 
        
    def forward(self, inp): 
        pep = inp[:, 0, :]
        mhc = inp[:, 1, :]
        #print(mhc.size(), pep.size())
        sa = pep.permute(0,2,1).unsqueeze(3)
        sb = mhc.permute(0,2,1).unsqueeze(2)
        ones = torch.ones_like(sa)
        ones_t = torch.ones_like(sb)
        s = torch.mul(sa, ones_t)
        s_t = torch.mul(sb, ones)
        x_comp = torch.cat((s, s_t), dim=1)
        #print(x_comp.size())
        x = self.conv1(x_comp)
        x = self.dropoutearly(F.relu(self.conv1_bn(x)))
        x = self.conv2(x)
        x = self.dropoutearly(F.relu(self.conv2_bn(x)))
        x = self.conv3(x)
        x = self.dropout(F.relu(self.conv3_bn(x)))  
        x = self.conv4(x)        
        x = self.dropout(F.relu(self.conv4_bn(x)))  
        x = x.view(-1, self.num_flat_features(x))
        x = self.L1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[5]:


# add embedding maybe ?
# add dynamic hidden dim and other parameter
class lstm(nn.Module):

    def __init__(self, input_dim=600, hidden_dim=15, batch_size=32, output_dim=1, num_layers=10):
        super(lstm, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, x):
        out = x.view(len(x), 1, -1)
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(out)

        y_pred = self.linear(lstm_out.view(len(lstm_out), -1))
        return y_pred


# In[ ]:


class conv3x3(nn.Module):
    def __init__(self, inputchannel, L, dropout, dropoutearly):

        super(conv3x3, self).__init__()    
        self.L = L
        self.inputchannel = inputchannel
        self.dropout = dropout
        self.dropoutearly = dropoutearly

        self.conv1 = nn.Conv2d(self.inputchannel * 2, self.inputchannel * 2, 3, stride=(1,1), padding=(1,1), dilation=(1,1), bias=False)
        self.conv1_bn = nn.BatchNorm2d(self.inputchannel * 2)
        self.conv2 = nn.Conv2d(self.inputchannel * 2, self.inputchannel * 2, 3, stride=(1,1), padding=(1,1), dilation=(1,1), bias=False)
        self.conv2_bn = nn.BatchNorm2d(self.inputchannel * 2)
        self.conv3 = nn.Conv2d(self.inputchannel * 2, self.inputchannel * 2, 3, stride=(1,1), padding=(1,1), dilation=(1,1), bias=False)
        self.conv3_bn = nn.BatchNorm2d(self.inputchannel * 2)
        self.conv4 = nn.Conv2d(self.inputchannel * 2, self.inputchannel * 2, 3, stride=(1,1), padding=(1,1), dilation=(1,1), bias=False)
        self.conv4_bn = nn.BatchNorm2d(self.inputchannel * 2)
        self.conv5 = nn.Conv2d(self.inputchannel * 2, self.inputchannel * 2, 3, stride=(1,1), padding=(1,1), dilation=(1,1), bias=False)
        self.conv5_bn = nn.BatchNorm2d(self.inputchannel * 2)
        self.conv6 = nn.Conv2d(self.inputchannel * 2, self.inputchannel * 2, 3, stride=(1,1), padding=(1,1), dilation=(1,1), bias=False)
        self.conv6_bn = nn.BatchNorm2d(self.inputchannel * 2)
        self.conv7 = nn.Conv2d(self.inputchannel * 2, 1, 3, stride=(1,1), padding=(1,1), dilation=(1,1), bias=False)
        self.conv7_bn = nn.BatchNorm2d(1)

        self.L1 = nn.Linear(self.L * self.L, 1)
        self.dropout = nn.Dropout(self.dropout)
        self.dropoutearly = nn.Dropout(self.dropoutearly)
        #self.dropout = nn.Dropout2d(0.1) #2d use lower dropout because of spatial 
        
    def forward(self, inp): 
        pep = inp[:, 0, :]
        mhc = inp[:, 1, :]
        #print(mhc.size(), pep.size())
        sa = pep.permute(0,2,1).unsqueeze(3)
        sb = mhc.permute(0,2,1).unsqueeze(2)
        ones = torch.ones_like(sa)
        ones_t = torch.ones_like(sb)
        s = torch.mul(sa, ones_t)
        s_t = torch.mul(sb, ones)
        x_comp = torch.cat((s, s_t), dim=1)
        #print(x_comp.size())
        x = self.conv1(x_comp)
        x = self.dropoutearly(F.relu(self.conv1_bn(x)))
        x = self.conv2(x)
        x = self.dropoutearly(F.relu(self.conv2_bn(x)))
        x = self.conv3(x)
        x = self.dropoutearly(F.relu(self.conv3_bn(x)))  
        x = self.conv4(x)        
        x = self.dropout(F.relu(self.conv4_bn(x))) 
        x = self.conv5(x)        
        x = self.dropout(F.relu(self.conv5_bn(x))) 
        x = self.conv6(x)        
        x = self.dropout(F.relu(self.conv6_bn(x))) 
        x = self.conv7(x)   
        x = self.dropout(F.relu(self.conv7_bn(x))) 
        x = x.view(-1, self.num_flat_features(x))
        x = self.L1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[6]:


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_value = 0):
        super(BasicBlock, self).__init__()
        self.dropout_value = dropout_value
        self.dropout = nn.Dropout2d(self.dropout_value)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.dropout(F.relu(out))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dropout_value = 0):
        super(Bottleneck, self).__init__()
        self.dropout_value = dropout_value
        self.dropout = nn.Dropout2d(self.dropout_value)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, inputchannel, block, num_blocks, num_classes=1, dropout = 0):
        super(ResNet, self).__init__()
        self.dropout = dropout
        self.in_planes = 64
        self.inputchannel = inputchannel

        self.conv1 = nn.Conv2d(self.inputchannel * 2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout = self.dropout)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout = self.dropout)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout = self.dropout)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout = self.dropout)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_value = self.dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, inp):
        pep = inp[:, 0, :]
        mhc = inp[:, 1, :]
        #print(mhc.size(), pep.size())
        sa = pep.permute(0,2,1).unsqueeze(3)
        sb = mhc.permute(0,2,1).unsqueeze(2)
        ones = torch.ones_like(sa)
        ones_t = torch.ones_like(sb)
        s = torch.mul(sa, ones_t)
        s_t = torch.mul(sb, ones)
        x_comp = torch.cat((s, s_t), dim=1)
        #print(x_comp.size())
        out = F.relu(self.bn1(self.conv1(x_comp)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

