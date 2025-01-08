
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

# SemSun
def BCD(y,X0,a0,list_model,X=None,max_channel_list=None,radio=None,estimed_aMVP=1,Lambda0=None,
         step_size_BSP=1e-3,tol=1e-4,niter_max_BSP=1000,niter_max_out=100,tol_BSP=1e-8,norm='1',optim=0):
    """
    Estimate X, a from y 
    Parameters
    ----------
    y: mesured spectrum
    X0 : initial spectral signature
    X : input spectral signature, for NMSE computation purposes
    a0 : initial mixing weight
    max_channel_list: max channel for each radionuclide, channel > max : value = 0
    radio: mask for radionuclides, used for unmixing procedure ; 1: test, 0: No , None: test all radionuclides 
    list_model: list of pre-trained IAE models
    estimed_aMVP : estimation of MVP, 1 : yes, 0: no
    step_size_BSP: learning rate in BSP
    tol: tolerance in the outer loop
    niter_max_BSP : maximum number of iterations for inner loop when estimating X
    niter_max_out : maximum number of external iterations
    tol_in: tolerance in the inner loop
    norm: norm to normalize data
    optim: solver, 0: BFGS, 1: SLSQP
    ---------
    Output: X, a, loss, NMSE, lambda
    """
    ####  initial outer loop
    err=1
    ite=0
    itp=0
    ak=a0.copy()
    Xk=X0.copy()

    ####
    loss=[divergence(X0,y,a0)] #loss
    if X is not None:
        NMSE_list=[-10*np.log10((np.sum((X0[:,1:]-X[:,1:])**2,axis=0)/np.sum(X[:,1:]**2,axis=0)))] # NMSE
    else:
        NMSE_list = None
        
    if radio is None:
        radio=(np.ones(len(a0))==1)
     ## initial value of lambda
    Lambda_list = [0 for r in range(len(a0)-1)]
    OldErrIn = 1e32
    OldLoss = 1e32
    I = np.linspace(0,len(a0)-1,len(a0)).astype('int32')
    ## function to estimate X,a
    fct=barycentric_span_projection

    #### while loop
    while (ite<niter_max_out) & (err>tol) :
        ### Estimate the mixing weights a
        ak_p=np.zeros(len(a0))
        ak_p[radio] = NNPU(y,Xk[:,radio],ak[radio],estimed_aMVP)
        ### initial inner loop
        Xp = Xk.copy()

        ###joint IAE model
        if True:   
            d=np.shape(list_model[0].anchorpoints)[1]
            Bg = Xp[:,0].dot(ak_p[0]) # Bkg* its counting
            if ite<1:
                # iteration 0: no lambda value, estimate X by NMF then use Fast Interpolation to give a initial value of lambda
                if Lambda0 is None:
                    tmp=NMF_fixed_a(y,X0,ak_p)
                    Lambda0=list_model[0].fast_interpolation((tmp)[np.newaxis,:,1:])["Lambda"][0][0].detach().numpy()
                ## Use barycentric_span_projection to estimate lambda (or X)
                rec = fct(y[np.newaxis,:d,np.newaxis] , tole=tol_BSP,Bg = Bg[np.newaxis,:d,np.newaxis] , model=list_model[0], Lambda0=Lambda0,  a0=ak_p[1:], niter=niter_max_BSP, optim=optim, step_size=step_size_BSP,norm=norm,max_channel_list=max_channel_list)
                # save the estimated lambda in the lambda list and use it as the initial value for the next iteration
                Lambda_list[0] = rec['Lambda']
            else:
                # iteration > 0
                rec = fct(y[np.newaxis,:d,np.newaxis] , tole=tol_BSP,Bg = Bg[np.newaxis,:d,np.newaxis] , model=list_model[0], Lambda0=Lambda_list[0], a0=ak_p[1:], niter=niter_max_BSP, optim=optim, step_size=step_size_BSP,norm=norm,max_channel_list=max_channel_list)
                Lambda_list[0] = rec['Lambda'].squeeze()
            # update X
            summ = rec["XRec"].squeeze()
            Xp[:d,1:]=summ # normalize X
            Xp[:,1:]/=np.sum(Xp[:,1:],axis=0)     
        # stopping criterion                  
        errA= np.mean(np.linalg.norm(ak_p-ak)/np.linalg.norm(ak))
        errX= np.mean(np.linalg.norm(Xp-Xk)/np.linalg.norm(Xk))
        err=np.maximum(errA,errX)
        if X is not None:
            NMSE_list.append(-10*np.log10((np.sum((Xp[:,1:]-X[:,1:])**2,axis=0)/np.sum(X[:,1:]**2,axis=0))))

        # Update variables
        Xk=Xp.copy()
        ak=ak_p.copy()

        cost=divergence(Xk,y,ak)
        loss+=[cost]

        ite+=1
        itp+=1
        
        OldErrIn = err
        OldLoss = cost


        cmd = 'loss: '+str(cost)+' / ErrX: '+str(errX)+' / ErrA: '+str(errA)
        
    #print('iteration outer: ',ite,' / ',cmd)        
    return Xk,ak,np.array(loss),np.array(NMSE_list),np.array(Lambda_list[0]).squeeze()

###############################################################################################################################################

def _get_barycenter(Lambda,amplitude=None,model=None,fname=None,max_channel_list=None):
    """
    Reconstruct a barycenter from Lambda
    Parameters
    ----------
    Lambda: lambda used to reconstruct the barycenter
    amplitude: amplitude of X, if: None -> the vector 1
    model: IAE model
    fname: name of IAE model if model is not provided
    max_channel_list: max channel for each radionuclide
    -------
    Output: X
    
    """
    #from scipy.optimize import minimize

    if model is None:
        model = load_model(fname)
    model.nneg_output=True

    PhiE,_ = model.encode(model.anchorpoints)
    
    
    B = []
    for r in range(model.NLayers):
        B.append(torch.einsum('ik,kjl->ijl',torch.as_tensor(Lambda.astype("float32")), PhiE[model.NLayers-r-1]))
    tmp=model.decode(B)
    if max_channel_list is not None:
        for i in range(len(max_channel_list)):
            tmp[:,max_channel_list[i]:,i]=0

    if  amplitude is not None:
        tmp=torch.einsum('ijk,i -> ijk',tmp,torch.as_tensor(amplitude.astype("float32")))
    return tmp.detach().numpy()


def barycentric_span_projection(y, Bg = None, model=None, Lambda0=None,a0=None, tole=1e-4,
                                niter=100, optim=None, norm='1',constr=True,step_size=1e-3,bound=0,max_channel_list=None):
    from scipy.optimize import minimize
    """
    Estimate lambda using numerical solver (a is fixed)
    Parameters
    ----------
    y: mesured spectrum
    Bg : terms fixed
    model: pre-trained IAE model
    optim: solver, 0: BFGS, 1: SLSQP
    Lambda0: initial value of lambda
    a0 : mixing weight (fixed)
    tol: tolerance in the outer loop
    niter : maximum number of iterations
    tole: tolerance
    norm: norm to normalize data
    step_size: Step size used for numerical approximation of the Jacobian
    bound: 0: lambda in [0,1], otherwise: lambda = min, max obtained when training IAE. 
    """
    if model is None:
        model = load_model(fname)
    r=0
    PhiE,_ = model.encode(model.anchorpoints) # encode anchor points
    ## these calculs for simplex constraint of lambda, used in later
    PhiE2 = torch.einsum('ijk,ljk -> il',PhiE[model.NLayers-r-1], PhiE[model.NLayers-r-1])
    iPhiE = torch.linalg.inv(PhiE2 + model.reg_inv*torch.linalg.norm(PhiE2,ord=2)* torch.eye(model.anchorpoints.shape[0],device=model.device))
    # shape and init values
    d = model.anchorpoints.shape[0]
    b,tx,ty = y.shape
    ty=len(a0)
    loss_val = []
    a = a0
    # init lambda
    if Lambda0 is None:
        if len(a0)==1:# the counting of X has dimension one -> individual IAE model
            #initial lambda is calculated by fast_interpolation
            Lambda = model.fast_interpolation(x-Bg,Amplitude=a0[:,np.newaxis])["Lambda"][0].detach().numpy()
            Lambda=Lambda.squeeze()
        else: # joint IAE model, set lambda = lambda of first anchor point
            Lambda=np.zeros(d)
            Lambda[0]=1
    else:
        Lambda = Lambda0
    # define simplex constraint 
    def simplex_constraint(param):
        return np.sum(param)-1
    ## limit value of lambda
    if bound==0:
        bnds=((0,1),(0,1)) # lambda in [0,1]
    else:
        bnds=(model.bounds) # min, max of lambda obtained in training
    constraints=[{'type': 'eq','fun':simplex_constraint}]
        
    # function to get barycenter : provide X with given lambda
    def Func(P):
        P=torch.tensor(P.astype('float32'))
        Lambda= P.reshape(b,-1)
        if optim==0: # BFGS, consider simplex constraint
            ones = torch.ones_like(Lambda)
            mu = (1 - torch.sum(torch.abs(Lambda),dim=1))/torch.sum(iPhiE)
            Lambda = Lambda +  torch.einsum('ij,i -> ij',torch.einsum('ij,jk -> ik', ones, iPhiE),mu)
   
        B = []
        for r in range(model.NLayers):
            B.append(torch.einsum('ik,kjl->ijl',Lambda , PhiE[model.NLayers-r-1]))
        # decode barycenter 
        XRec=model.decode(B)
        ## 0 for channel > max channel
        if max_channel_list is not None:
            for i in range(len(max_channel_list)):
                XRec[:,max_channel_list[i]:,i]=0
        #### normalisation 
        XRec=torch.einsum('ijk,ik-> ijk',XRec,1/torch.sum(XRec,axis=(1)))
        return XRec.detach().numpy()
     # cost function
    def get_cost(param,*args):       
        y,Bg,a0=args# recover the parameters in arguments
        XRec = Func(param) # X estimated
        XRec=np.einsum('ijk,ilk->ijl', XRec, a0[np.newaxis, np.newaxis, :]) # X*a
        Tot = XRec+Bg
        Tot = Tot*(Tot > 0)
        return np.sum(Tot-y*np.log(Tot+1e-10)) # negative log likelihood
    Lambda=Lambda.squeeze()
    if optim!=0: # use SLSQP,consider automatically simplex constraints by simplex_constraint defined above
        sol = minimize(get_cost,x0=Lambda,args=(y,Bg,a0),constraints=constraints, 
                       bounds=bnds,method='SLSQP',tol=tole,options={'maxiter':niter,'eps':step_size}) 
    else: # use BFGS, simplex constraint is taken into account in the Func(P)
        sol = minimize(get_cost,x0=Lambda,args=(y,Bg,a0),
                       bounds=bnds,method='L-BFGS-B',tol=tole,options={'maxiter':niter,'eps':step_size}) 
        
    param =sol.x # solution obtained by solver
    Lambda=param
    if len(Lambda.shape)==1:
        Lambda=Lambda[np.newaxis,:]
            
    if optim==0: # apply simplex constraint
        ones = np.ones(Lambda.shape)
        iPhiE=iPhiE.detach().numpy()
        mu = (1 - np.sum(np.abs(Lambda),1))/np.sum(iPhiE)
        Lambda = Lambda +  np.einsum('ij,i -> ij',np.einsum('ij,jk -> ik', ones, iPhiE),mu)

    Params={}
    Params["Lambda"] = Lambda
    Params['XRec']=_get_barycenter(Lambda,model=model,max_channel_list=max_channel_list)
    return Params



def MoSeVa(y,X0,a0, list_model, X=None, estimed_aMVP=1,optim=0,alpha=1/100,turn=1,I0=None,
         step_size_BSP=1e-4,tol=1e-5,niter_max_BSP=100,niter_max_out=30,tol_BSP=1e-8,max_channel_list=None):
    from scipy.stats.distributions import chi2
    """
    Identify a!=0 and Estimate X (or lambda), a 
    Parameters
    ----------
    y: measured spectrum
    alpha: expected false positive rate
    a0 : initial mixing weight; a0_i = 0 -> dictionary does not containt radionuclide i
    turn: 1 no additional post-processing test, 2: with additional test, 3: additional test with few modifications (small dictionary size and large FPR)
    I0: list of active radionuclides, None by default: I0=[0] (Bkg)
    X0 : initial spectral signature
    X : input spectral signature, for NMSE computation purposes
    max_channel_list: max channel for each radionuclide, channel > max :value =0
    list_model: list of pre-trained IAE models
    estimed_aMVP : estimation of Bkg, 1 : yes, 0: no
    step_size_BSP: learning rate in BSP
    tol: tolerance in the outer loop
    niter_max_BSP : maximum number of iterations for inner loop when estimating X
    niter_max_out : maximum number of external iterations
    tol_in: tolerance in the inner loop
    optim: solver used in BSP,  0: BFGS, 1: SLSQP
    --------------
    Output
    Identification :list of active radionuclides
    Quantification: Estimated X and a
    Standard deviation of a (and  lambda)
    
    """
   
    M,N=np.shape(X0)
    ### Init
    if I0 is None:  
        I0=[0] # only bkg
        I=list(np.arange(1,N)[a0[1:]>0]) # list of tested radionuclide (a0_i==0, this radionuclide is not in dictionary)
        a_init=np.zeros(len(a0)) # init a
        a_init[0]=NNPU(y,X0[:,0:1]) #calculate  a_bkg 
        Lambda0=None
        L0=divergence(X0,y,a_init)
        res_final=[X0,a_init,L0,None,None] # final result of quantification (X,a,loss, NMSE, lambda)
        lambda_list=[] # list of estimated lambda in unmixing procedure

    else:
        I_tmp=list(np.arange(1,N)[a0[1:]>0]) # list of tested radionuclides I
        I=[item for item in I_tmp if item not in I0] # remove radionuclide from I which is in I0 
        ## create a mask for interessted radionuclides I0
        radio=np.zeros(N)
        radio[I0]=1
        radio=(radio==1)
        ## estimate X and a for this I0
        res_final=BCD(y,X0,a0,X=X,list_model=list_model,estimed_aMVP=estimed_aMVP,tol_BSP=tol_BSP,
                      tol=tol,niter_max_out=niter_max_out,niter_max_BSP=niter_max_BSP,step_size_BSP=step_size_BSP,
                      optim=optim,radio=radio, max_channel_list=max_channel_list)
        L0=res_final[2][-1] # loss 
        Lambda0=res_final[4] # estimated lambda
        lambda_list=[Lambda0]
    
    list_a=[] # list of a for different turns (with/without additional tests)
    list_loss=[L0] # list of loss
    
    # turn 1
    ############
    param_SemSun=[y,X0,a0,X,list_model,estimed_aMVP,tol_BSP,max_channel_list,tol,niter_max_out, niter_max_BSP, step_size_BSP,optim]
    list_loss, lambda_list,I0,I,L0,res_final,Lambda0= forward_MoSeVa(Lambda0,I0,I,L0,res_final,list_loss,lambda_list,alpha,N,param_SemSun )
    ########################################################

    # save output    
    res_final=list(res_final)
    res_final[2]=np.array(res_final[2]).reshape(-1)
    L0=res_final[2][-1]
    list_act=I0 # unmxing procedure
    list_a+=[res_final[1]]
    # turn 2
    ### additional test
    ##############
    if (turn>1) & (len(I0)>2):## turn 2
        I,list_act,I0,L0,res_final,Lambda0,lambda_list= backward_MoSeVa(Lambda0,turn,I0,I,L0,res_final,list_loss,lambda_list,alpha,N, list_act,param_SemSun)
    #####################################################
    list_a+=[res_final[1]]    
    
    #turn 3
    ###########
    if (turn==3) and (len(list_act)!=len(I0)):
        list_loss+=[L0] # loss 
        list_loss, lambda_list,I0,I,L0,res_final,Lambda0= forward_MoSeVa(Lambda0,I0,I,L0,res_final,list_loss,lambda_list,alpha, N, param_SemSun )
    ############

    # standrad deviation using Fisher matrix
    std_tmp=get_std_autograd(res_final[1],res_final[4],y,list_model[0],X0[:,0:1],max_channel_list)    
    # list of estimated a        
    list_a+=[res_final[1]]
    list_a=np.array(list_a)
    return {'Iden':I0,'LambdaList':lambda_list,'Quan':res_final,'Std':std_tmp,'Procedure':list_act,'AList':list_a,'LossList':list_loss}


def forward_MoSeVa(Lambda0,I0,I,L0,res_final,list_loss,lambda_list,alpha,N,param_SemSun ):
    flag=1
    y,X0,a0,X,list_model,estimed_aMVP,tol_BSP,max_channel_list,tol,niter_max_out, niter_max_BSP, step_size_BSP,optim=param_SemSun
    while (flag==1) & (len(I)!=0):
        L_test=np.zeros(len(I))
        if len(I0)==1: # y only contains Bkg, chisquare 2, 
            DT=chi2.ppf(1-2*alpha/(len(I)), df=2) # can be modified depends on the application
        else: # at least one radionuclide
            DT=chi2.ppf(1-2*alpha/(len(I)), df=1) # chisquare 1
        res=[]
        for i in range(len(I)):
            I_test=I0+[I[i]] # add radio i in tested dictionary
            ## create a mask for this dictionary
            radio=np.zeros(N)
            radio[I_test]=1
            radio=(radio==1)
            ### estimate X and a
            result_BCD_optim=BCD(y,X0,a0,X=X,list_model=list_model,estimed_aMVP=estimed_aMVP,tol_BSP=tol_BSP,
                              max_channel_list=max_channel_list,tol=tol,niter_max_out=niter_max_out,niter_max_BSP=niter_max_BSP,
                              step_size_BSP=step_size_BSP,optim=optim,radio=radio,Lambda0=Lambda0)
            L_test[i]=result_BCD_optim[2][-1] # add loss into list of loss 
            res+=[result_BCD_optim] # list of res
        j=np.argmin(L_test) # min loss
        list_loss+=[L_test[j]] # update list of loss in selection procedure
        lambda_list+=[res[j][4]] # update list of loss in selection procedure
        
        if (-2*(L_test[j]-L0)>DT): # deviance test, deviance > threshold
            I0=I0+[I[j]] # update list of active radionuclides
            I.pop(j)  # remove radionuclide j from the tested dictionary
            print(I0)
            L0=L_test[j] # update loss
            res_final=res[j] # final result
            Lambda0=res_final[4] # update lambda
        else:
            flag=0 # stop
    return list_loss, lambda_list,I0,I, L0, res_final, Lambda0

def backward_MoSeVa(Lambda0,turn,I0,I,L0,res_final,list_loss,lambda_list,alpha,N,list_act,param_SemSun ):
    flag=1
    y,X0,a0,X,list_model,estimed_aMVP,tol_BSP,max_channel_list,tol,niter_max_out, niter_max_BSP, step_size_BSP,optim=param_SemSun

    while (flag==1) & (len(I0)!=2): # I0 contains at least 2 radio (Bkg+ 1 radionuclide)
        L_test=np.zeros(len(I0)-1)
        if turn==2:
            DT=chi2.ppf(1-2*alpha/((len(I)+1)), df=1)
        else:
            DT=chi2.ppf(1-2*1/1000/((len(I)+1)), df=1) # turn 3, use very small alpha in backward and then same alpha in forward
        res=[]
        for i in range(1,len(I0)): # i>=1 since bkg is always present
            I_test=I0.copy()
            del I_test[i]
            radio=np.zeros(N)
            radio[I_test]=1
            radio=(radio==1) # update mask
            result_BCD_optim=BCD(y,X0,a0,X=X,list_model=list_model,estimed_aMVP=estimed_aMVP,tol_BSP=tol_BSP,
                                 tol=tol,niter_max_out=niter_max_out,niter_max_BSP=niter_max_BSP,step_size_BSP=step_size_BSP,
                                 optim=optim,radio=radio,Lambda0=Lambda0)

            L_test[i-1]=result_BCD_optim[2][-1]
            res+=[result_BCD_optim]

        j=np.argmin(L_test)+1 # min loss
        I_test=I0.copy()
        del I_test[j] # remove j from list of active radionuclide
        if (np.abs(-2*(L_test[j-1]-L0))<DT): # Deviance < threshold -> radionuclide is not present
            I+=[I0[j]] # Add radio j in the tested dictionary
            list_act+=[I0[j]] # update unmixing procedure
            I0=I_test.copy()# update list of active radio
            print(I0)
            L0=L_test[j-1]
            res_final=res[j-1]
            Lambda0=res_final[4]
            lambda_list+=[Lambda0]
        else:
            flag=0
    return I,list_act,I0,L0,res_final,Lambda0,lambda_list
def loss_fct(inp_tensor,y_tensor,model,MVP_tensor,max_channel_list,radio):
    """
    Calculate loss function
    Parameters
    ----------
    inp_tensor: input tensor (a,lambda)
    y_tensor: observed spectrum
    model : IAE pre-trained model
    MVP_tensor: Background tensor
    max_channel_list:  max channel for each radionuclide
    --------------
    """
    a=inp_tensor[:-1]
    Lambda=inp_tensor[-1][None]
    X_ten=_get_barycenter_ver2(Lambda,model=model,max_channel_list=max_channel_list)[:,radio[1:]] # estimated X
    X_ten=torch.cat((MVP_tensor,X_ten),1) # concatenante with Bkg
    return torch.sum(torch.matmul(X_ten,a)-y_tensor*torch.log(torch.matmul(X_ten,a))) # cost function
# function return X from lambda
def _get_barycenter_ver2(Lambda,model,max_channel_list=None):
    Lambda=torch.cat((Lambda,1-Lambda),0)
    model.nneg_output=True
    PhiE,_ = model.encode(model.anchorpoints)
    B = []
    #Lambda=torch.as_tensor( Lambda[np.newaxis,:].astype("float32"))
    for r in range(model.NLayers):
        B.append(torch.einsum('ik,kjl->ijl',Lambda[None, :], PhiE[model.NLayers-r-1]))
    tmp=model.decode(B)
    if max_channel_list is not None:
        for i in range(len(max_channel_list)):
            tmp[:,max_channel_list[i]:,i]=0
    tmp=torch.einsum('ijk,ik-> ijk',tmp,1/torch.sum(tmp,axis=(1)))
    return tmp.squeeze()
# autograd approximate hessian matrix and calculate std
def get_std_autograd(a_est,lambda_est,y,model,MVP,max_channel_list=None):
    """
    Approximate hessian matrix by autograd and calculate std
    Parameters
    ----------
    a_est: estimated a
    y: mesured spectrum
    lambda_est: estimated lambda
    model: pre-trained IAE models
    MVP: normalized Bkg
    --------------
    """
    #tranform into tensor
    y_tensor=torch.tensor( y.astype("float32"),requires_grad=True)
    a_tensor=torch.tensor( a_est.astype("float32"),requires_grad=True)
    MVP_tensor=torch.tensor( MVP.astype("float32"))
    lamb_tensor=torch.tensor(lambda_est[0:1].astype("float32"),requires_grad=True)
    radio=(a_est>0)
    #input tensor (a,lambda)
    inp_tensor=torch.cat((a_tensor[radio],lamb_tensor),0)
    hess=torch.autograd.functional.hessian(lambda t: loss_fct(t,y_tensor,model,MVP_tensor,max_channel_list,radio), inp_tensor).detach().numpy() 
    var=np.linalg.inv(hess)
    tmp=np.sqrt(np.diag((var))*(np.diag((var))>0))
    # return vector std of a and lambda
    std_tmp=np.zeros(len(a_est)+1)
    if len(tmp)-1==len(a_est[a_est>0]):
        std_tmp[:-1][a_est>0]=np.int_(tmp[:-1]) # std a, only active radios
        std_tmp[-1]=np.round(tmp[-1],4)#
    else: #  estimated a of some radios are too small -> std=0 -> reshape output 
        amin=(np.sort(a_est[a_est>0])[len(a_est[a_est>0])+1-len(tmp)-1]+np.sort(a_est[a_est>0])[len(a_est[a_est>0])+1-len(tmp)])/2 
        # amin= 1/2*(min+ near min) # a>amin -> fit dimension   
        std_tmp[:-1][a_est[1]>amin]=np.int_(tmp[:-1])
        std_tmp[-1]=np.round(tmp[-1],4)   
    return std_tmp


