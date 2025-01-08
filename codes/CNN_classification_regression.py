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

#from read_data import get_X_drift
from general import normalize,normalize4, divergence, NNPU, list_array
from AsymLoss import *

#initialize the matrix X
def init_X(XMVP,position,spec_list):
    X=np.concatenate([XMVP,spec_list[position,:,:]],axis=1)
    return X

# class of CNN model for classification
class CNN_Classifier(pl.LightningModule):
    def __init__(self,config,NUM_CLASSES=1,input_shape=(512, 1, 1024)):
        """
        Initialize a class
        Parameters
        ----------
        config: list of hyper-parameters
        NUM_CLASSES: number of labels: 1: mono-label, binary classification, N>1: multi-label
        input_shape: shape of input: batchsize * 1* number of channels
        ---------
        """
        super(CNN_Classifier, self).__init__()
        self.save_hyperparameters()
        self.NUM_CLASSES=NUM_CLASSES
        
        if NUM_CLASSES==1: # binary classification
            self.train_accuracy = Accuracy(task="BINARY")
            self.val_accuracy = Accuracy(task="BINARY")
            # loss function : binary cross entropy 
            # logit (sigmoid) applied in loss function (more stable than in the activation function)
            self.loss_fn= torch.nn.BCEWithLogitsLoss() 
        else: # multi label
            self.train_accuracy = Accuracy(task="multilabel", num_labels=NUM_CLASSES)
            self.val_accuracy = Accuracy(task="multilabel", num_labels=NUM_CLASSES)
            # Asymmetric loss
            if "gamma_neg" in config.keys():
            # Asymmetric loss: default: BCE + Logit
                if 'dis_loss' in config.keys():
                    self.loss_fn = AsymmetricLoss(gamma_neg=config["gamma_neg"], gamma_pos=config["gamma_pos"], clip=config["loss_clip"]
                                             ,disable_torch_grad_focal_loss=True)
                else:
                    self.loss_fn = AsymmetricLoss(gamma_neg=config["gamma_neg"], gamma_pos=config["gamma_pos"], clip=config["loss_clip"])
            else:
                self.loss_fn= torch.nn.BCEWithLogitsLoss()
        self.activation=F.relu #activation
        self.lr=config["lr"] # learning rate
        self.nbr_layers=config["nbr_layer_cnn"] # number of CNN layer
        self.nbr_layers_lin=config["nbr_layer_lin"] # number of hidden full connected layers 

        for i in range(config["nbr_layer_cnn"]):
            list_keys=["layer_size_","kernel_","stride_","max_pool_"]
            if i==0: # input layer
                dim=1
            else:
                dim=list_param[0]
            list_param=[]
            for j in list_keys:
                if (j+str(i+1)) in config.keys():
                    list_param+=[config[j+str(i+1)]]
                else:
                    list_param+=[config[j+"default"]]
            
            setattr(self,'conv'+str(i+1),torch.nn.Conv1d(dim, list_param[0], list_param[1],stride=list_param[2]))
            setattr(self,'pool'+str(i+1),torch.nn.MaxPool1d( list_param[3]))
            setattr(self,'batch'+str(i+1),torch.nn.BatchNorm1d( list_param[0]))
            #setattr(self,'drop'+str(i+1),torch.nn.Dropout( config["dropout"]))
       
        self.flatten = nn.Flatten(1)
        # Lazylinear works in training but does not work for fine-tune
        #self.fc1 = nn.LazyLinear(config["linear_neurons"])
        ####
        ## for fine tune, need to calculate size after Conv layer
        n_size = self._get_conv_output(input_shape)
        
        # hidden fully connected layers
        for i in range(config["nbr_layer_lin"]):
            list_keys=["layer_fn_size_","dropout_"]
            if i==0: # input layer
                dim=n_size
            else:
                dim=list_param[0]
            list_param=[]
            for j in list_keys:
                if (j+str(i+1)) in config.keys():
                    list_param+=[config[j+str(i+1)]]
                else:
                    list_param+=[config[j+"default"]]
            setattr(self,'fc'+str(i+1),torch.nn.Linear(dim, list_param[0]))
            setattr(self,'dropfc'+str(i+1),torch.nn.Dropout( list_param[1]))
        # output layer
        self.fco = nn.Linear(list_param[0], NUM_CLASSES)
    # calulate dimension after CNN layer (before flatten)    
    def _get_conv_output(self, shape):
        inp = Variable(torch.rand( *shape))
        output_feat = self._forward_features(inp)
        n_size = output_feat.size(1)
        return n_size
    # forward function for calculate dimensions
    def _forward_features(self, x):
        list_fct=['conv','pool','batch']
        for i in range(self.nbr_layers):
            for j in list_fct:
                x=getattr(self,j+str(i+1))(x)
            ## activation
            x=self.activation(x)
            ## dropout
            #x=getattr(self,'drop'+str(i+1))(x)
        x = self.flatten(x)  # flatten all dimensions except batch
        return x
    # main forward
    def forward(self, x):
        list_fct=['conv','pool','batch']
        for i in range(self.nbr_layers):
            for j in list_fct:
                x=getattr(self,j+str(i+1))(x)
            ## activation
            x=self.activation(x)
            ## dropout
            #x=getattr(self,'drop'+str(i+1))(x)
        x = self.flatten(x)  # flatten all dimensions except batch
        
        for i in range(self.nbr_layers_lin):
            x=getattr(self,'fc'+str(i+1))(x)
            ## activation
            x=self.activation(x)
            ## dropout
            x=getattr(self,'dropfc'+str(i+1))(x)
         
        ## since sigmoid including in loss function, no need sigmoid here
        #x = torch.sigmoid(self.fc2(x))
        x = self.fco(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        acc = self.train_accuracy(logits, y)
        self.log("train_acc", acc,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        #self.log("val_loss", loss,prog_bar=True,on_step=False,on_epoch=True)
        acc = self.val_accuracy(logits, y)
        self.log("val_acc", acc,prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss)
        acc = self.val_accuracy(logits, y)
        self.log("test_acc", acc,on_epoch=True)
        #return {'val_loss': loss}        
    def predict(self, batch, batch_idx: int , dataloader_idx: int = None):
        return self(batch)
    def configure_optimizers(self):
        # can be modified
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        return optimizer
    

    
# class of CNN model for regression

class CNN_Regressfier(pl.LightningModule):
    def __init__(self,config,NUM_CLASSES=1,input_shape=(512, 1, 1024)):
        super(CNN_Regressfier, self).__init__()
        self.save_hyperparameters()
        self.loss_type=config["loss_type"]
        
        self.NUM_CLASSES=NUM_CLASSES
        if config["loss_type"]==0: #MSE:
            self.loss_fn= torch.nn.MSELoss() 
        else: # cross entropy
            self.loss_fn= torch.nn.CrossEntropyLoss()
                    
        self.train_accuracy = self.loss_fn
        self.val_accuracy = self.loss_fn
        
        self.activation=F.relu #activation
        self.lr=config["lr"] # learning rate
        self.nbr_layers=config["nbr_layer_cnn"] # number of CNN layer
        self.nbr_layers_lin=config["nbr_layer_lin"] # number of hidden full connected layers 
        
        
        for i in range(config["nbr_layer_cnn"]):
            list_keys=["layer_size_","kernel_","stride_","max_pool_"]
            if i==0: # input layer
                dim=1
            else:
                dim=list_param[0]
            list_param=[]
            for j in list_keys:
                if (j+str(i+1)) in config.keys():
                    list_param+=[config[j+str(i+1)]]
                else:
                    list_param+=[config[j+"default"]]
            
            setattr(self,'conv'+str(i+1),torch.nn.Conv1d(dim, list_param[0], list_param[1],stride=list_param[2]))
            setattr(self,'pool'+str(i+1),torch.nn.MaxPool1d( list_param[3]))
            setattr(self,'batch'+str(i+1),torch.nn.BatchNorm1d( list_param[0]))
            #setattr(self,'drop'+str(i+1),torch.nn.Dropout( config["dropout"]))
       
        self.flatten = nn.Flatten(1)
        # Lazylinear works in training but does not work for fine-tune
        #self.fc1 = nn.LazyLinear(config["linear_neurons"])
        ####
        ## for fine tune, need to calculate size after Conv layer
        n_size = self._get_conv_output(input_shape)
        
        # hidden fully connected layers
        for i in range(config["nbr_layer_lin"]):
            list_keys=["layer_fn_size_","dropout_"]
            if i==0: # input layer
                dim=n_size
            else:
                dim=list_param[0]
            list_param=[]
            for j in list_keys:
                if (j+str(i+1)) in config.keys():
                    list_param+=[config[j+str(i+1)]]
                else:
                    list_param+=[config[j+"default"]]
            setattr(self,'fc'+str(i+1),torch.nn.Linear(dim, list_param[0]))
            setattr(self,'dropfc'+str(i+1),torch.nn.Dropout( list_param[1]))
        # output layer
        self.fco = nn.Linear(list_param[0], NUM_CLASSES)
    # calulate dimension after CNN layer (before flatten)    
    def _get_conv_output(self, shape):
        inp = Variable(torch.rand( *shape))
        output_feat = self._forward_features(inp)
        n_size = output_feat.size(1)
        return n_size
    # forward function for calculate dimensions
    def _forward_features(self, x):
        list_fct=['conv','pool','batch']
        for i in range(self.nbr_layers):
            for j in list_fct:
                x=getattr(self,j+str(i+1))(x)
            ## activation
            x=self.activation(x)
            ## dropout
            #x=getattr(self,'drop'+str(i+1))(x)
        x = self.flatten(x)  # flatten all dimensions except batch
        return x
    # main forward
    def forward(self, x):
        list_fct=['conv','pool','batch']
        for i in range(self.nbr_layers):
            for j in list_fct:
                x=getattr(self,j+str(i+1))(x)
            ## activation
            x=self.activation(x)
            ## dropout
            #x=getattr(self,'drop'+str(i+1))(x)
        x = self.flatten(x)  # flatten all dimensions except batch
        
        for i in range(self.nbr_layers_lin):
            x=getattr(self,'fc'+str(i+1))(x)
            ## activation
            x=self.activation(x)
            ## dropout
            x=getattr(self,'dropfc'+str(i+1))(x)
         
        ## since sigmoid including in loss function, no need sigmoid here
        #x = torch.sigmoid(self.fc2(x))
        x = self.fco(x)
        ## BCElogit includes softmax
        if (self.NUM_CLASSES!=1) and (self.loss_type==0):
            x=F.softmax(x,dim=1)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss,prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss,on_epoch=True)

        #return {'val_loss': loss}        
    def predict(self, batch, batch_idx: int , dataloader_idx: int = None):
        return self(batch)
    def configure_optimizers(self):
        # can be modified
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    








