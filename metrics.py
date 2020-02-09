#!/usr/bin/env python
# coding: utf-8

# In[1]:


import unittest
import torch
from torch import nn
from scipy.stats import spearmanr, pearsonr


# In[12]:


def pearsonr_torch(prediction,target):
    nominator = torch.sum((prediction - prediction.mean()) * (target - target.mean()))
    dominator = torch.sqrt(torch.sum(torch.pow(prediction-prediction.mean(),2)))*torch.sqrt(torch.sum(torch.pow(target-target.mean(),2)))
    return torch.div(nominator, dominator).item()

def select_criterion(method='MSE'):
    if method == 'MSE':
        return nn.MSELoss()
    elif method == 'CROSS_ENTROPY':
        return nn.CrossEntropyLoss()
    elif method == 'L1':
        return nn.L1Loss()
    elif method == 'SmoothL1':
        return nn.SmoothL1()
    elif method == 'KLDiv':
        return nn.KLDivLoss()
    elif method == 'BCELoss':
        return nn.BCELoss()
    else:
        raise NotImplementedError


# In[13]:


class metrics_test(unittest.TestCase):

    def test_pearsonr_torch(self):
        self.assertAlmostEqual(pearsonr_torch(torch.tensor([0.2, 0.6, 1.0]), torch.tensor([0.5, 0.1, 1])), 0.5544, places=4)
    def test_select_criterion(self):
        self.assertEqual(type(select_criterion()), torch.nn.modules.loss.MSELoss)
        self.assertEqual(type(select_criterion(method = "L1")), torch.nn.modules.loss.L1Loss)
        self.assertRaises(NotImplementedError, select_criterion, "nonimplemented")
        
        self.mseloss = select_criterion()
        self.assertEqual(self.mseloss(torch.tensor(1.0), torch.tensor(3.0)), torch.tensor(4.))

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

