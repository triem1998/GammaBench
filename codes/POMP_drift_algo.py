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
import scipy
import os
import time
from general import normalize,normalize4, divergence, NNPU 
from scipy.stats.distributions import chi2
from POMP_algo import POMP,select_backward_POMP,select_forward_POMP,std_fisher
from read_data import get_X_drift_est
# function modifies energy 
def func(x,alpha):
    return x*(1-alpha[0])+alpha[1]


def find_drift_spectrum(spec_drift,spec_no_drift,func=func,bounds=None,loss_type=1,niter=100,step_size=1e-3,tole=1e-4,alpha0=None,optim=0,nbr_particles=1e6):
    """
    Transform  the spectrum with no drift into the drift spectrum
    Estimate the parameter of drift function
    ----------
    spec_drift: drift spectrum, vector 1D
    spec_no_drift: initial spectrum, vector 1D
    bounds: bounds of parameters (alpha) in funcs 
    func: function creates spectral drift
    loss_type: 0: l2 norm, 1: divergence
    niter: max number of iteration
    step_size: step_size used in optimizer
    tole: stopping criterion
    nbr_particles: number of particles
    alpha0: intial value of alpha
    optim: 'Nelder-Mead','Powell','L-BFGS-B','TNC'
    """
    from scipy.optimize import minimize
    # spectrum -> array of particles energies
    def get_drift_energy(spec):
        n_spec_drift=np.int_(spec/np.sum(spec)*nbr_particles).squeeze() # vector 1D
        ener_drift=2*np.arange(len(spec))+20
        spectre_drift_ener=[]
        for i in range(len(ener_drift)):
            spectre_drift_ener+=list(np.linspace(ener_drift[i],ener_drift[i]+2,n_spec_drift[i]))
        spectre_drift_ener=np.array(spectre_drift_ener)
        return spectre_drift_ener
    
    counting=np.sum(spec_no_drift)
    n=len(spec_no_drift)
    shape=spec_no_drift.shape
    spectre_drift_ener=get_drift_energy(spec_no_drift)
    
    # function from array of particles energies -> spectrum
    def get_drift_spectrum_v2(spectre_drift_ener,alpha):
        ### can be modified
        spectre_drift_ener=func(spectre_drift_ener,alpha)
        ###
        hist=np.histogram(spectre_drift_ener,bins=np.arange(2*n+20+1))
        spectre_drift_est=hist[0][20:2*n+20]
        spectre_drift_est=spectre_drift_est[0::2]+spectre_drift_est[1::2]
        spectre_drift_est=spectre_drift_est/np.sum(spectre_drift_est)
        return (spectre_drift_est*counting).reshape(shape)
    
    # cost function
    def cost_drift(param,*args):
        s1,typ,spectre_drift_ener=args
        s2=get_drift_spectrum_v2(spectre_drift_ener, param)
        # s1: observed drif spectrum, s2: estimated drif spectrum
        if typ==0: # l2 norm
            return np.linalg.norm(s1-s2)
        else: # divergence
            return np.sum(s2-s1*np.log(s2+1e-16)-s1+s1*np.log(s1+1e-16))
    if alpha0 is None:
        x0_list=np.array([[-0.14,0],[-0.07,0],[0,0],[0.07,0],[0.14,0]])
        loss_list=np.array([cost_drift(i,spec_drift,loss_type,spectre_drift_ener) for i in x0_list])
        x0=x0_list[np.argmin(loss_list)]
    else:
        x0=alpha0
    optim_list=['Nelder-Mead','Powell','L-BFGS-B','TNC']
    if bounds is None:
        bounds=((-0.15,0.15),(-1,1)) # max 15% drift, offset [-1,1] keV
    sol = minimize(cost_drift,x0=x0,args=(spec_drift,loss_type,spectre_drift_ener),bounds=bounds,method=optim_list[optim],tol=tole,options={'maxiter':niter}) 
    param=sol.x    
    return get_drift_spectrum_v2(spectre_drift_ener, param), param


def NNPU_drift(y,X,a0=None,alpha0=None,func=func,estimed_aMVP=1,niter_max_in=1000,niter_max_out=10,tol=10**-4,optim=0):
    """
    Estimate a from the drift measured spectrum (mixing) y and the initial spectral singatures (without drift) X
    ----------
    y: measured spectrum (with drift)
    X : spectral signatures (without drift)
    a0 : initial mixing coefficient or counting
    func: function creates spectral drift
    estimed_aMVP: 1 - estimate the natural Bkg counting or 0 - do not 
    niter_max_in: maximum iteration of inner loop
    niter_max_out: maximum iteration of outer loop
    tol: stopping criterion
    optim: 'Nelder-Mead','Powell','L-BFGS-B','TNC'

    """
    M,N=np.shape(X)
    if a0 is None:
        a0=np.ones(N)/N*np.sum(y)
    ak=a0.copy()
    niter_out=0
    err=1 # initial error, > tol
    Xk=X.copy()
    while (niter_out<niter_max_out) & (err>tol) :
        niter_in=0
        err_in=1
        while (niter_in<niter_max_in) & (err_in>tol) :
            ak1=ak*(np.dot(np.transpose(Xk),(y/(Xk.dot(ak)))))
            niter_in+=1
            if estimed_aMVP==0:
                ak1[0]=ak[0]
            err_in=np.linalg.norm(ak1-ak)/np.linalg.norm(ak)# relative error
            ak=ak1.copy()
        ## drift
        y_est_k=Xk.dot(ak)
        if niter_out==0: # estimate alpha initially 
            y_est_k1,alpha_est=find_drift_spectrum(y,X.dot(ak),loss_type=1,alpha0=alpha0,func=func,optim=optim)
        else: # use same alpha for init
            y_est_k1,alpha_est=find_drift_spectrum(y,X.dot(ak),loss_type=1,alpha0=alpha_est,func=func,optim=optim)
        err_drift=np.linalg.norm(y_est_k1-y_est_k)/np.linalg.norm(y_est_k)
        err=np.maximum(err_in,err_drift)
        Xk=get_X_drift_est(X,alpha_est,func)
        niter_out+=1
    return ak,alpha_est,divergence(Xk,y,ak)




def POMP_drift(y,X_init,fpr=1/100,niter_max_in=500,niter_max_out=10,tol=10**-3,turn=1,I0=None,option=0,func=func,optim=0,alpha0=None):
    """
    Identify a!=0 and Estimate X (or lambda), a 
    Parameters
    ----------
    y: mesured spectrum (drift)
    X_init: initial spectral singatures (without drift)
    fpr: expected false positive rate
    max_ite: maximum iterations
    tol: tolerance
    turn: 1: no additional post-processing test, 2: with additional test, 3: additional test with few modifications (small dictionary size and large FPR)
    I0: list of active radionuclides, None by default: I0=[0] (Bkg)
    option: 0 : estimate alpha at begining, save and use it later; 1: estimate for each iteration in model selection
    func: function creates spectral drift
    optim: 'Nelder-Mead','Powell','L-BFGS-B','TNC'
    alpha0: intial value of alpha
    --------------
    """
    M,N=np.shape(X_init)
    # Init
    if option==0: # estimate alpha at the begining and then apply POMP for it
        _,alpha_est,_=NNPU_drift(y,X_init,func=func,niter_max_in=niter_max_in,niter_max_out=niter_max_out,tol=tol,optim=optim,alpha0=alpha0)
        X=get_X_drift_est(X_init,alpha_est,func=func)
        dic=POMP(y,X,fpr,niter_max_in,tol=tol,turn=turn,I0=I0)
        dic['Alpha']=np.array([alpha_est])
        return dic
    else:
        X=X_init
        if I0 is None:
            I0=[0] # only bkg
            I=list(np.arange(1,N)) # list of tested radionuclide 
            X0=X[:,0:1]  # X0 = Bkg
        else:
            I_tmp=list(np.arange(1,N)) # list of tested radionuclides I
            I=[item for item in I_tmp if item not in I0]  # remove radionuclide from I which is in I0 
            X0=X[:,I0]

        weight_esti,alpha_est,L0= NNPU_drift(y, X0,func=func, niter_max_in=niter_max_in,niter_max_out=niter_max_out, tol=tol,alpha0=alpha0,optim=optim)
        list_weight=[]
        list_alpha_est=[alpha_est]
        list_loss=[L0]
        
        # turn 1
        ##########
        param_drift=[func,niter_max_in, niter_max_out,tol, optim]
        I0,I,L0,weight_esti,list_loss,alpha_est,list_alpha_est=forward_POMP_drift(y,X,I0,I,L0,fpr,list_loss, weight_esti, alpha_est,list_alpha_est, param_drift)
        ##########
        list_act=I0  # list activities of procedure
        weight_esti_final=np.zeros(N) # save estimated a
        weight_esti_final[I0]=weight_esti
        list_weight+=[weight_esti_final]
        X_est=get_X_drift_est(X,alpha_est,func)
        
        # turn 2
        ########
        if (turn>1) & (len(I0)>2):## turn 2
            ### POMP backward
            #I0,I,L0,weight_esti, list_act=select_backward_POMP(y,X_est,I0,I,L0,fpr,niter_max_in,tol,turn,list_act,weight_esti)
            ### POMP drift backward
            I0,I,L0,weight_esti,list_loss,alpha_est,list_alpha_est, list_act= backward_POMP_drift(y,X,I0,I,L0,fpr,list_loss,weight_esti,alpha_est,list_alpha_est,turn,list_act,param_drift)
        ########      
        # save estimated a for turn 2
        weight_esti_final=np.zeros(N)
        weight_esti_final[I0]=weight_esti
        list_weight+=[weight_esti_final]
        # turn 3
        #########
        if (turn==3) and (len(list_act)!=len(I0)): # turn 3, if turn 2 does st
            list_loss=[L0]
            I0,I,L0,weight_esti,list_loss= select_forward_POMP(y,X_est,I0,I,L0,fpr,niter_max_in,tol,list_loss,weight_esti)
        #########
            
        weight_esti_final=np.zeros(N)
        weight_esti_final[I0]=weight_esti
        list_weight+=[weight_esti_final]
        # uncertainty std
        std_final=np.zeros(N)
        std_final[I0]=std_fisher(X[:,I0],weight_esti) 
        #print(I)
        return {'a':weight_esti_final,'Iden':I0,'Std':std_final,'Procedure':list_act,'Alist':list_weight, 'LossList':list_loss, 'Alpha':list_alpha_est}
        #return weight_esti_final,I0,std_final,list_act,list_weight,list_loss,list_alpha_est


def forward_POMP_drift(y,X,I0,I,L0,fpr,list_loss,weight_esti,alpha_est,list_alpha_est,param_drift):
    func,niter_max_in, niter_max_out,tol, optim = param_drift
    flag=1
    while (flag==1) & (len(I)!=0):
        weight_esti_list=[]
        L_test=np.zeros(len(I))
        alpha_test=np.zeros((len(I),len(alpha_est)))
        DT=chi2.ppf(1-2*fpr/(len(I)), df=1) # chisquare 1 
        if len(I0)>=2:
            alpha0=alpha_est
        else:
            alpha0=None
        for i in range(len(I)):
            I_test=I0+[I[i]]  #add radio i in a tested dictionary
            X_test=X[:,I_test].copy()
            weight_esti_test,alpha_test[i],L_test[i]= NNPU_drift(y,X_test,func=func, niter_max_in=niter_max_in, niter_max_out=niter_max_out, tol=tol,optim=optim,alpha0=alpha0) # estimated weight
            weight_esti_list+=[weight_esti_test]

        j=np.argmin(L_test) # min of loss
        list_loss+=[L_test[j]]
        if (-2*(L_test[j]-L0)>DT):
            I0=I0+[I[j]] # update list of active radionuclides
            I.pop(j) # remove j from a tested dictionary
            L0=L_test[j] # update loss
            weight_esti=weight_esti_list[j] # update a
            alpha_est=alpha_test[j]
            list_alpha_est+=[alpha_est]
        else:
            flag=0     
    return I0,I,L0,weight_esti,list_loss,alpha_est,list_alpha_est

def backward_POMP_drift(y,X,I0,I,L0,fpr,list_loss,weight_esti,alpha_est,list_alpha_est,turn,list_act,param_drift):
    func,niter_max_in, niter_max_out,tol, optim = param_drift
    flag=1
    while (flag==1) & (len(I0)!=2): # I0 contains at least 2 radio (Bkg+ 1 radionuclide)
        weight_esti_list=[]
        L_test=np.zeros(len(I0)-1)
        alpha_test=np.zeros((len(I0)-1,len(alpha_est)))
        L_test=np.zeros(len(I0)-1)
        if turn==2:
            DT=chi2.ppf(1-2*fpr/((len(I)+1)), df=1)
        else:
            DT=chi2.ppf(1-2*fpr/100/((len(I)+1)), df=1) # turn 3, use very small alpha in backward and then same alpha in forward
        res=[]
        for i in range(1,len(I0)): # i>=1 since bkg is always present
            I_test=I0.copy()
            del I_test[i]
            X_test=X[:,I_test].copy()
            weight_esti_test,alpha_test[i-1],L_test[i-1]= NNPU_drift(y,X_test,func=func, niter_max_in=niter_max_in, niter_max_out=niter_max_out, tol=tol,optim=optim,alpha0=alpha_est) # estimated weight
            weight_esti_list+=[weight_esti_test]
        j=np.argmin(L_test)+1 # min loss
        I_test=I0.copy()
        del I_test[j] # remove j from list of active radionuclide
        if (np.abs(-2*(L_test[j-1]-L0))<DT): # Deviance < threshold -> radionuclide is not present
            I+=[I0[j]] # Add radio j in the tested dictionary
            list_act+=[I0[j]] # update unmixing procedure
            I0=I_test.copy()# update list of active radio
            print(I0)
            L0=L_test[j-1]
            weight_esti=weight_esti_list[j-1]
            alpha_est=alpha_test[j-1]
            list_alpha_est+=[alpha_est]
        else:
            flag=0
    return I0,I,L0,weight_esti,list_loss,alpha_est,list_alpha_est,list_act
