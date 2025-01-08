import time
import importlib
import sys
import pickle
import pandas as pd
import matplotlib.cm as cm
import pandas as pd
from scipy.stats import norm
from scipy import stats
import torch 
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy
import os
import pandas as pd
from general import normalize,normalize4, divergence, NNPU, list_array, NMF_divergence, index_array_3D, NMF_fixed_a
from AsymLoss import *

import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import preprocessing
from enum import Enum 
import copy
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.autograd import Variable

# Initiazation matrix X 
def init_X(XBkg,position,spec_list):
    """
    position: position of thickness (or alpha for shift) used to create X
    """
    X=np.concatenate([XBkg,spec_list[position,:,:]],axis=1)
    return X



def create_scenario_ML(dictionary, minmax_counting_all , minmax_a_Bkg, min_counting_radio, list_lamb=None, nb_sce=100, max_active=4,file_save=None):
    """
    Creata list of unmixing scenario 
    ----------
    Input:
    dictionary: Dictionary of radionuclides
    minmax_counting_all: minimum, maximum of total counting 
    minmax_a_Bkg: mixing weight of Bkg 
    min_counting_radio: minimal counting pour each radionuclide
    list_lamb: if None: without variability or drift (known X), not None: with variability or drift
    nb_sce: number of scenario
    max_active: maximum number of active radionuclide (4 radio + 1 Bkg)
    file_save: name of saved file, if None: don't save file
    ----------
    Output: list of unmixing scenario, matrix (nbr_scenario * (dictionary+1)) 
    """
    list_sce=[]
    nbr_radio=len(dictionary)+1 # +Bkg
    for i in range(int(nb_sce)):
        condi=1 
        if list_lamb is not None: #  variability or drift (can be alpha)
            lamb=np.random.choice(list_lamb)
        nb_radio_active=np.random.randint(0,max_active+1)
        while condi==1:
            counting=10**np.random.uniform( np.log10(minmax_counting_all[0]),np.log10(minmax_counting_all[1])) # total counting
            a=np.zeros(nbr_radio)
            if nb_radio_active==0:
                a[0]=1  # only Bkg
                condi=0 # end loop
            else:
                list_radio_active=np.random.choice(dictionary, nb_radio_active, replace=False) # list of active radio
                list_radio_active=np.sort(list_radio_active) # sort
                a[0]=np.random.uniform(minmax_a_Bkg[0],minmax_a_Bkg[1]) # a_Bkg
                for j in range(len(list_radio_active)): 
                    a[list_radio_active[j]]=np.random.uniform(0,1) # a of actif radio
                a[1:]=a[1:]/np.sum(a[1:])*(1-a[0]) # sum =1
                if np.all((a[a>0]*counting)>min_counting_radio[a>0]): # > min counting
                    condi=0
            a=np.int_(a*counting)
        if list_lamb is not None: # variablity
            list_sce+=[np.append(a,lamb)]
        else: # without variablity
            list_sce+=[a]
    list_sce=np.array(list_sce)
    np.random.shuffle(list_sce)
    if file_save is None:
        return list_sce
    else:
        with open('../data/'+file_save, "wb") as fp:
            pickle.dump(list_sce, fp)
        return list_sce



class CustomStarDataset(Dataset):
    """
    Create pytorch dataset   
    """
    # This loads the scenarios, create simulated spectra from scenarios 
    def __init__(self,data,XBkg,spec_list,variablity=True,drift=False,classifi=True): # data: list of mixing scenarios
        """
        init  
        ----------
        data: list of mixing scenarios
        XBkg: natural background (Bkg) 
        spec_list: spectral signatures of all radionuclides with different thicknesses (or gain shift)
        variablity: True: with variablity , False: without variablity
        drift: clas: with spectral drift, False: without spectral drift
        classifi: True: classification, False: regression
        """
        # load data
        self.scenarios=data
        if variablity==False:
            if drift==False:
                X=init_X(XBkg,0,spec_list) # 0: without deformation, can be modified
        data_X=[]
        for i in range(len(data)):
            if variablity==False: # without variability
                if drift==False: # no drift
                    np.random.seed(i)
                    data_X+=[np.random.poisson(X.dot(data[i,:])) ]
                else: # with drift
                    X=init_X(XBkg[data[i,-1]],0,spec_list[data[i,-1],:,:,:])
                    np.random.seed(i)
                    data_X+=[np.random.poisson(X.dot(data[i,:-1]))]
            else : # with variability
                X=init_X(XBkg,data[i,-1],spec_list)
                np.random.seed(i)
                data_X+=[np.random.poisson(X.dot(data[i,:-1]))]
        data_X=np.array(data_X)
        ## normalisation
        data_X=data_X/np.sum(data_X,1).reshape(-1,1)
        data_X=data_X[:,None,:]
        # conver to torch dtypes
        self.dataset=torch.tensor(data_X).float()
        # counting of all radionuclides
        if (variablity==False) and (drift==False): # X known
            data_tmp=data
        else:
            data_tmp=data[:,:-1] # last column is thickness or alpha of spectral drift
        if classifi==True: # classification
            self.labels=torch.tensor((data_tmp[:,1:]>0)*1,dtype=torch.float32) # Bkg is alway present
        else: # regression
            self.labels=torch.tensor(data_tmp/np.sum(data_tmp,1).reshape(-1,1),dtype=torch.float32)

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.dataset[idx],self.labels[idx]


# the same thing but only returns X, not the label
# Use only for tests with the GPU 
class CustomStarDataset_test(Dataset):
    """
    The same thing but only returns X, not the label
    Use only for tests with the GPU 
    """
    # This loads the data and converts it, make data ready
    def __init__(self,data_cs):# data_cs: CustomStarDataset objet
        self.dataset=data_cs.dataset
    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.dataset[idx]



def get_data_loader_multi(data,XBkg,spec_list,variability=True,drift=False,batch_size=512, num_workers=0,classifi=True):
    """
    Prepare the data for training ML model
    ----------
    data: list of mixing scenarios
    XBkg: natural background (Bkg) 
    spec_list: spectral signatures of all radionuclides with different thicknesses
    variablity: True: with variablity , False: without variablity
    drift: True: with spectral drift, False: without spectral drift   
    batch_size: batch size
    num_workers: number of workers for Dataloader
    """
    # get train, validation, test set
    data_train,data_val,data_test=data
    # create CustomStarDataset for each dataset
    train_set=CustomStarDataset(data_train,XBkg,spec_list,variability,drift,classifi=classifi)
    val_set=CustomStarDataset(data_val,XBkg,spec_list,variability,drift,classifi=classifi)
    test_set=CustomStarDataset(data_test,XBkg,spec_list,variability,drift,classifi=classifi)
    ## create dataloader
    train_dataloader = DataLoader(train_set, batch_size=batch_size,shuffle=0, num_workers=num_workers)
    val_dataloader = DataLoader(val_set, batch_size=batch_size,shuffle=0, num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=batch_size,shuffle=0, num_workers=num_workers)
    # get validation and test Dataset (use with gpu when prediction)
    test_set2=CustomStarDataset_test(test_set)
    test_dataloader2 = DataLoader(test_set2, batch_size=batch_size,shuffle=0, num_workers=num_workers)
    val_set2=CustomStarDataset_test(val_set)
    val_dataloader2 = DataLoader(val_set2, batch_size=batch_size,shuffle=0, num_workers=num_workers)
    ## get numpy test data
    test_sce=test_set.scenarios
    test_y=test_set.labels.detach().numpy()
    test_X=test_set.dataset.detach().numpy()
    data_load=[train_dataloader,val_dataloader,test_dataloader] # dataloader
    data_set=[train_set,val_set,test_set] # CustomStarDataset
    data_test=[test_sce,test_y,test_X] # data for test
    data_loader_test=[val_dataloader2,test_dataloader2] # dataloader for test with gpu
    return data_load,data_set,data_test,data_loader_test


class CustomStarDatasetMono(Dataset):
    """
    Binary classification (mono-label) (binary relevant)
    Same as CustomStarDataset but return the label of each radionuclide
    """
    # This loads the data and converts it, make data rdy
    def __init__(self,data_cs,radio):
        # load data
        self.dataset=data_cs.dataset
        # conver to torch dtypes
        self.labels=data_cs.labels[:,radio:radio+1]
        #self.labels=torch.tensor((data[:,1:]>0)*1,dtype=torch.int)

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.dataset[idx],self.labels[idx]


def get_data_loader_mono(data_set_multi,radio=0,batch_size=512,num_workers=0):
    """
    Prepare the data for training ML model for binary classification
    """
    train_set,val_set,test_set=data_set_multi
    
    train_set_mono=CustomStarDatasetMono(train_set,radio)
    val_set_mono=CustomStarDatasetMono(val_set,radio)
    test_set_mono=CustomStarDatasetMono(test_set,radio)

    train_dataloader_mono = DataLoader(train_set_mono, batch_size=batch_size,shuffle=0, num_workers=num_workers)
    val_dataloader_mono = DataLoader(val_set_mono, batch_size=batch_size,shuffle=0, num_workers=num_workers)
    test_dataloader_mono = DataLoader(test_set_mono, batch_size=batch_size,shuffle=0, num_workers=num_workers)
    return train_dataloader_mono, val_dataloader_mono,test_dataloader_mono




