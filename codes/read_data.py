import time
import importlib
import sys
import pickle
import pandas as pd
import matplotlib.cm as cm
import pandas as pd
from scipy.stats import norm
from scipy import stats
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import scipy
import os
import time

def read_spectrum(fname):
    spec = []
    for line in open(fname, 'r'):
        spec.append(float(line.rstrip()))
    return np.array(spec)


def GetListData(name,RN_NAME,PATH=None):    
    list_thickness=[]
    list_data=[]
    if PATH is None:
        PATH_DATA=os.path.join('../data/Simulation_steel_sphere/', RN_NAME[name])
    else:
        PATH_DATA=os.path.join(PATH, RN_NAME[name])
    ### search dat file
    for file in os.listdir(PATH_DATA):
        if file.endswith(".dat"):
            tmp_file=os.path.join(PATH_DATA, file)
            list_data+=[tmp_file]
    ## list thickness
    ## replace __ = . , (2__0=2.0mm)
    for tmp_file in list_data:
        pos=tmp_file.find("STEEL")
        tmp_thick=tmp_file.replace(tmp_file[:pos+5],"")
        tmp=tmp_thick.replace("mm.dat","")
        if "__" in tmp:
            tmp=tmp.replace("__",".")
        list_thickness+=[float(tmp)]
    ##sort data 
    ind=np.argsort(list_thickness)
    list_thickness=np.sort(list_thickness)
    list_data = [list_data[i] for i in ind]
    return list_data,list_thickness

def GetSpectra(name,RN_NAME,PATH=None,max_channel_list=None):
    """
    Read the spectral singatures
    ----------
    name: position of radionuclide
    RN_NAME: list of radionuclides
    """
    list_data,list_thickness=GetListData(name,RN_NAME,PATH)
    spec=[]
    for i in range(len(list_data)):
        spec += [read_spectrum(list_data[i])]
    spec=np.array(spec).T
    # cut 20kev 
    # 1024 channels, 1 channel=2kev
    spec = spec[20:2048+20,:]
    spec=spec[0::2,:]+spec[1::2,:]
    if max_channel_list is not None:
        spec[max_channel_list[name]:,:]=0
    ## Am241 with high thicknesses, spectral signatures are almost absored by attenuation 
    ## X of Am241 for thickness >4 mm = X of 4 mm  
    if name==5: 
        spec[:,30:]=spec[:,29:30].dot(np.ones((1,spec.shape[1]-30)))
    return spec.T,list_thickness
    
def GetSpectraDrift(name,RN_NAME,PATH=None,max_channel_list=None,alpha=0):
    """
    For the drift of spectral singatures
    ----------
    name: position of radionuclide
    RN_NAME: list of radionuclides
    alpha: per mille( 1/1000) of shift  
    """
    #######
    if PATH is None:
        PATH_DATA=os.path.join('../data/drift_data/', RN_NAME[name])
    else:
        PATH_DATA=os.path.join(PATH, RN_NAME[name])
    list_thickness=[]
    list_data=[]
    spec=[]
    for file in os.listdir(PATH_DATA):
        if alpha==0:
            end_file=".dat"
        elif alpha<0:
            end_file=".neg_"+str(np.abs(int(alpha*100)))+"dat"
        else:
            end_file="."+str(int(alpha*100))+"dat" # file name: 100*alpha
        if file.endswith(end_file):
            tmp_file=os.path.join(PATH_DATA, file)
            list_data+=[file]
            spec += [read_spectrum(tmp_file)]
    # get list of thickness
    for tmp_file in list_data:
        pos=tmp_file.find("STEEL")
        tmp_thick=tmp_file.replace(tmp_file[:pos+5],"")
        pos=tmp_thick.find("mm")
        tmp_thick=tmp_thick.replace(tmp_thick[pos:],"")
        if "__" in tmp_thick:
            tmp_thick=tmp_thick.replace("__",".")
        list_thickness+=[float(tmp_thick)]
    ##sort list of thickness
    ind=np.argsort(list_thickness)
    list_thickness=np.sort(list_thickness)
    list_data = [list_data[i] for i in ind]
    spec=[spec[i] for i in ind]
    ################      
    spec=np.array(spec)
    spec = spec[:,20:2048+20]
    spec=spec[:,0::2]+spec[:,1::2]
    if max_channel_list is not None:
        spec[:,max_channel_list[name]:]=0    
#     if name==5:
#         spec[30:,:]=np.dot(np.ones((spec.shape[0]-30,1)),spec[29:30,:])

    return spec,list_thickness



# function modifies energy 
def func(x,alpha):
    """
    Function modifies energy in fonction of alpha
    """
    return x*(1-alpha[0])+alpha[1]



# Initiazation matrix X 
def init_X(XBkg,position,spec_list):
    """
    position: position of thickness (or alpha for shift) used to create X
    """
    X=np.concatenate([XBkg,spec_list[position,:,:]],axis=1)
    return X




def get_drift_spectrum(spec,alpha,func=func, nbr_particles=1e6):
    """
    Get the drift spectrum fom the observed spectrum
    ----------
    spec: observed spectrum, array 1D (histogram)
    alpha: parameter of drift function
    func: function creates spectral drift
    nbr_particles: number of particles
    ----------
    Idea: from a observed spectrum, vector of size n, tranform it into list of particles (here 1e6 particles) by sampling
    Then, apply the func to modifies the paricle energy and build a new spectrum
    """
    shape=spec.shape
    counting=np.sum(spec)
    ## tranform the normalized observed spectrum into high stat, e.g, * 1e6 (can be modified)
    n_spec_drift=np.int_(np.round(spec/np.sum(spec)*nbr_particles,0)).squeeze()
    ener_drift=2*np.arange(len(spec))+20 # channel -> energy
    spectre_drift_ener=[]
    for i in range(len(ener_drift)):
        spectre_drift_ener+=list(np.linspace(ener_drift[i],ener_drift[i]+2,n_spec_drift[i])) # from histogram, get the particle energies 
    spectre_drift_ener=np.array(spectre_drift_ener) # nbr_particles of particle enrgies
    ### apply the func
    spectre_drift_ener=func(spectre_drift_ener,alpha)
    #######
    # from array of particles, get the histogram (spectrum)
    hist=np.histogram(spectre_drift_ener,bins=np.arange(2*len(spec)+20+1))
    spectre_drift_est=hist[0][20:2*len(spec)+20]
    spectre_drift_est=spectre_drift_est[0::2]+spectre_drift_est[1::2]
    spectre_drift_est=spectre_drift_est/np.sum(spectre_drift_est)
    return (spectre_drift_est*counting).reshape(shape)

# get drift X from histograms 
def get_X_drift_est(X,alpha=0,func=func):
    """
    Like get_X_drift but use the estimated drift spectra (from the initial spectra)
    ----------
    X: the initial spectral singatures (without drift)
    alpha: parameter of drift function
    func: function creates spectral drift
    """
    X_drift=[]
    for i in range(X.shape[1]):
        tmp=get_drift_spectrum(X[:,i],alpha,func)
        if i==0:# Bkg, avoid 0
            tmp[tmp==0]=np.min(tmp[tmp>0])/100
        X_drift+=[tmp]
    X_drift=np.array(X_drift).T
    X_drift=X_drift/np.sum(X_drift,0)
    return X_drift



