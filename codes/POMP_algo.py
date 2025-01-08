from IPython.display import HTML
from IPython.display import clear_output
import matplotlib.animation as animation
import time
from scipy.stats.distributions import chi2


import numpy as np
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import scipy
import os
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

import IAE_CNN_TORCH_Oct2023 as cnn 
import torch
from general import divergence, NNPU, NMF_fixed_a

def POMP(y,X,alpha=5/100,max_ite=500,tol=10**-10,turn=1,I0=None):
    """
    Identify a!=0 and Estimate X (or lambda), a 
    Parameters
    ----------
    y: mesured spectrum
    X: spectral singatures
    alpha: expected false positive rate
    max_ite: maximum iterations
    tol: tolerance
    turn: 1: no additional post-processing test, 2: with additional test, 3: additional test with few modifications (for small dictionary size and large FPR)
    I0: list of active radionuclides, None by default: I0=[0] (Bkg)
    --------------
    """
    M,N=np.shape(X)
    # Init
    if I0 is None:
        I0=[0] # only bkg
        I=list(np.arange(1,N)) # list of tested radionuclide 
        X0=X[:,0:1]  # X0 = Bkg
    else:
        I_tmp=list(np.arange(1,N)) # list of tested radionuclides I
        I=[item for item in I_tmp if item not in I0]  # remove radionuclide from I which is in I0 
        X0=X[:,I0]
        
    weight_esti=NNPU(y,X0) # estimated a
    L0=divergence(X0,y,weight_esti) # loss
    list_weight=[]
    std_final=np.zeros(N)
    list_loss=[L0]
    # turn 1: forward
    ####################
    I0,I,L0,weight_esti,list_loss= select_forward_POMP(y,X,I0,I,L0,alpha,max_ite,tol,list_loss,weight_esti)
    ###################  
    list_act=I0  # list activities of procedure
    weight_esti_final=np.zeros(N) # save estimated a
    weight_esti_final[I0]=weight_esti
    list_weight+=[weight_esti_final]
    
    # turn 2
    ############################
    if (turn>1) & (len(I0)>2): # at least 2 radionuclides, use backward
        I0,I,L0,weight_esti, list_act=select_backward_POMP(y,X,I0,I,L0,alpha,max_ite,tol,turn,list_act,weight_esti)
    ############################
    # save estimated a for turn 2
    weight_esti_final=np.zeros(N)
    weight_esti_final[I0]=weight_esti
    list_weight+=[weight_esti_final]
    
    # turn 3
    ###############################
    if (turn==3) and (len(list_act)!=len(I0)): # turn 3, if turn 2 is do st
        list_loss=[L0]
        I0,I,L0,weight_esti,list_loss= select_forward_POMP(y,X,I0,I,L0,alpha,max_ite,tol,list_loss,weight_esti)
    ##############################################################
    weight_esti_final=np.zeros(N)
    weight_esti_final[I0]=weight_esti
    list_weight+=[weight_esti_final]
    ############
    # uncertainty std
    ##########
    std=std_fisher(X[:,I0],weight_esti) 
    std_final[I0]=std
    #print(I)
    return {'a':weight_esti_final,'Iden':I0,'Std':std_final,'Procedure':list_act,'Alist':list_weight,'LossList':list_loss}



def select_forward_POMP(y,X,I0,I,L0,alpha,max_ite,tol,list_loss,weight_esti):
    flag=1
    while (flag==1) & (len(I)!=0):
        weight_esti_list=[]
        L_test=np.zeros(len(I))
        DT=chi2.ppf(1-2*alpha/(len(I)), df=1) # chisquare 1 
        for i in range(len(I)):
            I_test=I0+[I[i]]  #add radio i in a tested dictionary
            X_test=X[:,I_test].copy()
            weight_esti_test=NNPU(y,X_test,niter_max=max_ite,tol=tol) # estimated weight
            weight_esti_list+=[weight_esti_test]
            L_test[i]=divergence(X_test,y,weight_esti_test)# loss
        j=np.argmin(L_test) # min of loss
        list_loss+=[L_test[j]]
        if (-2*(L_test[j]-L0)>DT):
            I0=I0+[I[j]] # update list of active radionuclides
            I.pop(j) # remove j from a tested dictionary
            L0=L_test[j] # update loss
            weight_esti=weight_esti_list[j] # update a
        else:
            flag=0     
    return I0,I,L0,weight_esti,list_loss

def select_backward_POMP(y,X,I0,I,L0,alpha,max_ite,tol,turn,list_act,weight_esti):
    flag=1
    while (flag==1) & (len(I0)!=1):
        L_test=np.zeros(len(I0)-1)
        if turn==2:
            DT=chi2.ppf(1-2*alpha/((len(I)+1)), df=1) # turn 2, same alpha
        else:
            DT=chi2.ppf(1-2*1/1000/((len(I)+1)), df=1) # turn 3, very small alpha in backward, same in new forward
        for i in range(1,len(I0)):
            I_test=I0.copy()
            del I_test[i] # tested dictionary: list of active radionuclides, remove i from a tested dictionary
            X_test=X[:,I_test]
            weight_esti_test=NNPU(y,X_test,niter_max=max_ite,tol=tol) # estiamted a
            L_test[i-1]=divergence(X_test,y,weight_esti_test) # loss
        j=np.argmin(L_test)+1 # min of loss
        I_test=I0.copy()
        del I_test[j]    
        if ((2*(L_test[j-1]-L0))<DT): # deviance > threshold
            I+=[I0[j]] # update list of inactive radionulcide
            list_act+=[I0[j]] # update unmixing procedure
            I0=I_test.copy()  # update list of active radionuclides
            L0=L_test[j-1] # update loss
            weight_esti=NNPU(y,X[:,I0]) # update a

        else:
            flag=0
    return I0,I,L0,weight_esti, list_act


# loss function, neg log likelihood =divergence
def cost_function(weight_esti,spectrum,X):
    tmp=X.dot(weight_esti)
    cost=np.sum(tmp-spectrum*np.log(tmp)) 
    return cost

# calculate std of a using Fisher when X is known
def std_fisher(X,a):
    """
    Calculate std using the Fisher information matrix
    Parameters
    ----------
    a: estimated a
    X: spectral signatures
    --------------
    """
    std=np.zeros(len(a))
    X_reduced=X[:,a>0] # only active radionuclides
    weight_reduced=a[a>0]
    M,N=np.shape(X_reduced)
    fisher=np.zeros((N,N))
    tmp=X_reduced.dot(weight_reduced)
    for i in range(N):
        for j in range(N):
            fisher[i,j]=np.dot(X_reduced[:,i]*X_reduced[:,j],1/tmp)
    var=np.linalg.inv(fisher)
    std[a>0]=np.sqrt(np.diag((var)))
    return std
