#!/usr/bin/env python
# coding: utf-8

# In[5]:


from mhcflurry import Class1AffinityPredictor
import pandas as pd
import numpy as np


# In[34]:


def reverse_log_transformation(y):
    return(np.exp(-1*(y * np.log(50000) - 1)))

def root_mean_squared(data, allele_name, modelname):
    '''input data as dataframe containing column pred and true
       input allele_name - name of the allele as string'''
    error = data.pred - data.true
    error2 = np.square(error)
    rmse = np.sqrt(np.mean(error2))
    return({allele_name + modelname: rmse})

# need to change weight name in mhc flurry library to work on windows
def mhcflurry_test(data, allele_name):
    '''input data as dataframe containing column allele, peptide, measurement_value
       input allele_name - name of the allele as string'''
    predictor = Class1AffinityPredictor.load()
    data = data[data.allele == allele_name]
    mhcflurry_data = predictor.predict(allele=allele_name, peptides=data.peptide.tolist())
    output_data = pd.DataFrame(list(zip(mhcflurry_data,
                                      data.measurement_value.tolist(),
                                      data.allele.tolist(),
                                      data.peptide.tolist())), 
                             columns =['pred', 'true',
                                       'allele', 'peptide'])
    return(output_data)

def mymodel_test():
    
    return()

