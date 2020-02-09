#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import time 
import math
import torch
import numpy as np
import sys
import re

from torchvision import transforms, utils
from torch import nn 
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm as tqdm

import metrics


# In[3]:


def eval_epoch(model, validation_data,  optimizer,device, criterion):
    ''' Epoch operation in evaluation phase '''
    model.eval()
    pred_list = []
    true_list = []
    accuracy = 0
    with torch.no_grad():
        count = 0
        losssum = 0
        for batch in tqdm(validation_data, mininterval=2, desc='  - (Validation)   ', leave=False, position=0):
            inp, target = map(lambda x: x.to(device), batch) 
            optimizer.zero_grad()
            pred = model(inp)
            loss = criterion(pred, target.float())
            count += 1
            losssum += loss.item()
            batch_accuracy = metrics.pearsonr_torch(pred.detach().double(), target.detach().double())
            accuracy += batch_accuracy
            pred_list.append(pred.detach().cpu().numpy())
            true_list.append(target.detach().cpu().numpy())   
        losssum = losssum / count
        accuracy = accuracy / count
        return losssum, pred_list, true_list, accuracy


# In[4]:


def train_epoch(model, training_data, optimizer, device, criterion):
    ''' Epoch operation in training phase'''
    model.train()
    count = 0 
    losssum = 0
    pred_list = []
    true_list = []
    accuracy = 0
    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False, position=0):
        inp, target = map(lambda x: x.to(device), batch) 
        optimizer.zero_grad()
        pred = model(inp)
        loss = criterion(pred, target.float())
        loss.backward()
        optimizer.step()  
        count += 1
        losssum += loss.item()
        batch_accuracy = metrics.pearsonr_torch(pred.detach().double(), target.detach().double())
        accuracy += batch_accuracy
        pred_list.append(pred.detach().cpu().numpy())
        true_list.append(target.detach().cpu().numpy())
    losssum = losssum / count
    accuracy = accuracy / count
    return losssum, pred_list, true_list, accuracy


# In[5]:


def start_training(path_checkpoint, epoch_num, model, train_dataloader, eval_dataloader, optimizer, device, criterion):
    """Start Training and save checkpoint each epoch and return loss and accuracy of the train/valid data"""
    
    valid_accus = []
    valid_losss = []
    train_accus = []     
    train_losses = []
    for epoch_i in range(epoch_num):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, pred_train, true_train, train_accu = train_epoch(model, train_dataloader, optimizer, device, criterion)
        print('  - (Training)   ppl: {0}, Accuracy: {1}, elapse: {2} min'.format(
                    train_loss,
                    train_accu,
                    (time.time()-start)/60))




        start = time.time()
        valid_loss, pred_valid, true_valid, valid_accu = eval_epoch(model, eval_dataloader, optimizer,  device, criterion)
        print('  - (Validation) ppl: {0}, Accuracy: {1}, elapse: {2} min'.format(
                    valid_loss,
                    valid_accu,
                    (time.time()-start)/60))

        valid_losss += [valid_loss]        
        train_losses += [train_loss]
        valid_accus += [valid_accu]
        train_accus += [train_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
                'model': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch_i,
                'train_losss_chk': train_losses,
                'valid_losss_chk': valid_losss,
                'train_accu_chk': train_accus,
                'valid_accu_chk': valid_accus}

        torch.save(checkpoint, path_checkpoint)
    return(valid_losss, train_losses, valid_accus, train_accus, true_valid, pred_valid)


# In[6]:


def continue_training(path_checkpoint, epoch_num, model, train_dataloader, eval_dataloader, optimizer, device, criterion):     
    """Continue Training from checkpoint and save checkpoint each epoch and return loss and accuracy of the train/valid data"""
    
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs_yet = checkpoint['epoch']+1
    print("Checkpoint found and loaded - Resuming training")

    valid_losss = checkpoint['valid_losss_chk']     
    train_losses = checkpoint['train_losss_chk']
    valid_accus = checkpoint['valid_accu_chk']
    train_accus = checkpoint['train_accu_chk']

    for epoch_i in range(epochs_yet, epoch_num):
        print('[ Epoch', epoch_i + 1, ']')

        start = time.time()
        train_loss, pred_train, true_train, train_accu = train_epoch(model, train_dataloader, optimizer, device, criterion)
        print('  - (Training)   ppl: {0}, Accuracy: {1}, elapse: {2} min'.format(
                    train_loss,
                    train_accu,
                    (time.time()-start)/60))




        start = time.time()
        valid_loss, pred_valid, true_valid, valid_accu = eval_epoch(model, eval_dataloader, optimizer,  device, criterion)
        print('  - (Validation) ppl: {0}, Accuracy: {1}, elapse: {2} min'.format(
                    valid_loss,
                    valid_accu,
                    (time.time()-start)/60))

        valid_losss += [valid_loss]        
        train_losses += [train_loss]
        valid_accus += [valid_accu]
        train_accus += [train_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
                'model': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch_i,
                'train_losss_chk': train_losses,
                'valid_losss_chk': valid_losss,
                'train_accu_chk': train_accus,
                'valid_accu_chk': valid_accus}

        torch.save(checkpoint, path_checkpoint)
    return(valid_losss, train_losses, valid_accus, train_accus, true_valid, pred_valid)


# In[7]:


def prediction_only(path_checkpoint, model, test_dataloader, optimizer, device, criterion):     
    """Prediction for new values"""
    if str(device) == "cpu":
        checkpoint = torch.load(path_checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(path_checkpoint)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Checkpoint found and loaded - Predict values")

    for epoch_i in range(1):
        print('[ Epoch', epoch_i + 1, ']')

        start = time.time()
        _, pred_valid, true_valid, _ = eval_epoch(model, test_dataloader, optimizer,  device, criterion)
        print('  - (Test) Elapse: {0} min'.format(
                    (time.time()-start)/60))

    return(pred_valid, true_valid)

