#!/usr/bin/env python
# coding: utf-8

# In[1]:


import unittest
import pandas as pd
import os
import numpy as np


# In[2]:


codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


# In[6]:


class data_transformation():
    '''
    Class required path to full dataset and mhc dataset
    - aligned_mhc_dataset for smaller mhc sequence - otherwise mhc_sequence_complete
    allele_name as string to only return this allele
    quant_data True/False if True only quantitative data returned
    encoding - which encoding - only one-hot implemented
    '''
    def __init__(self, path_data, path_mhc, allele_name = None, quant_data = True, encoding = "one-hot", dbscan = True):
        self.path_data = path_data
        self.path_mhc = path_mhc
        self.quant_data = quant_data
        self.encoding = encoding
        self.allele_name = allele_name
        self.dbscan = dbscan
        
    def read_data(self):
        '''Reads dataset, mhc sequence data and joins them - returns the data as pandas dataframe'''
        os.chdir('/content/drive/My Drive/master_thesis/scripts/fill/framework')
        cwd = os.getcwd()
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')
        current_path = os.getcwd()
        data = pd.read_csv(current_path + self.path_data)
        mhc_data = pd.read_csv(current_path + self.path_mhc).loc[:, ["allele", "mhc_sequence"]]
        joined_data = pd.merge(data, mhc_data, left_on = "original_allele", right_on = "allele", how='inner').drop(["allele_y", "original_allele"], axis = 1)
        os.chdir(cwd) 
        return joined_data
    
    def filter_data(self):
        '''Filter specific allele_name if not None and filter quantitative if True
           and log transform the data'''
        if not self.allele_name:
            data = self.read_data()
        else:
            data = self.read_data()
            data = data[data.allele_x == self.allele_name]
        if self.quant_data:
            data = data[data.measurement_type == "quantitative"].reset_index(drop = True)
        else:
            data = data.reset_index(drop = True)
        if self.dbscan:
            data = data.drop(['Unnamed: 0'], axis = 1).reset_index(drop = True)
        data["measurement_value"] = (1 - np.log(data["measurement_value"])) / np.log(50000)
        #data = data.loc[:, ["peptide", "mhc_sequence", "measurement_value"]]
        data = data.drop_duplicates().reset_index(drop = True)
        unique_data = data.groupby(['peptide', 'mhc_sequence']).mean()
        data = pd.merge(data, unique_data, left_on = ["peptide", 'mhc_sequence'], right_on = ["peptide", 'mhc_sequence'], how='inner').drop(['measurement_value_x', 'measurement_source'], axis = 1)     
        data = data.drop_duplicates().reset_index(drop = True)
        return data
    
    def encode_sequence(self, row, enc):
        '''Only OneHot Implemented right now - Blosum coming
           Encode peptide and mhc_sequence'''
        if self.encoding == "one-hot":
            if enc == "peptide":
                list_peptide = []
                for j in list(row.reset_index(drop = True).at[0, 'peptide']):
                    list_peptide.append([i for i,x in enumerate(sorted(codes)) if x == j])
                list_peptide = np.array(list_peptide)
                results = np.zeros((len(list_peptide), len(sorted(codes))), dtype = np.uint8)
                for i, sequence in enumerate(list_peptide):
                    results[i, sequence] = 1
            elif enc == "mhc":
                list_mhc = []
                for j in list(row.reset_index(drop = True).at[0, 'mhc_sequence']):
                    list_mhc.append([i for i,x in enumerate(sorted(codes)) if x == j])
                list_mhc = np.array(list_mhc)
                results = np.zeros((len(list_mhc), len(sorted(codes))), dtype = np.uint8)
                for i, sequence in enumerate(list_mhc):
                    results[i, sequence] = 1
        else:
            raise NotImplementedError
        return results        
    
    def __getitem__(self):
        '''Return encoded sequences and target as 3 arrays'''
        data = self.filter_data()
        peptide_sequence = []
        mhc_sequence = []
        max_sequence = len(max(max(data.mhc_sequence.values, key=len), max(data.peptide.values, key=len)))
        output_array_mhc = np.zeros((len(data), max_sequence, 20), dtype = np.uint8)
        output_array_peptide = np.zeros((len(data), max_sequence, 20), dtype = np.uint8)
        for i in range(len(data)):
            peptide_sequence.append(self.encode_sequence(data.loc[i:i, :], enc = "peptide"))
            output_array_peptide[i][:peptide_sequence[i].shape[0],:peptide_sequence[i].shape[1]] = peptide_sequence[i]  
            
            mhc_sequence.append(self.encode_sequence(data.loc[i:i, :], enc = "mhc"))
            output_array_mhc[i][:mhc_sequence[i].shape[0],:mhc_sequence[i].shape[1]] = mhc_sequence[i] 
        target = np.array([data.measurement_value_y])
        if self.dbscan:
            dbscan = np.array([data.dbscan_cluster_y])
            return output_array_peptide, output_array_mhc, target, dbscan
        else:
            return output_array_peptide, output_array_mhc, target


# In[7]:


class data_class_test(unittest.TestCase):
    def setUp(self):
        self.class_data = data_transformation(path_data = "/data/curated_training_data_no_mass_spec_dbscan.csv",
                                              path_mhc = "/data/aligned_mhc_dataset.csv",
                                              allele_name = "HLA-A*02:01",
                                              quant_data = True,
                                              encoding = "one-hot",
                                              dbscan = True)
    
    def test_encode_sequence(self):
        self.assertEqual(self.class_data.encode_sequence(self.class_data.filter_data().loc[:0, :], enc = "peptide").shape,
                         (len(self.class_data.filter_data().loc[0:0, ["peptide"]].peptide[0]), 20))
        
    
    def test_getitem(self):
        peptide_seq, mhc_sequence, target, dbscan = self.class_data.__getitem__()
        self.assertEqual(peptide_seq.shape, (11705, 34, 20))
        self.assertEqual(mhc_sequence.shape, (11705, 34, 20))
        
if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

