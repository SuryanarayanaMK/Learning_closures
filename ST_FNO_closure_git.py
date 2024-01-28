import numpy as np
import torch
from torch.autograd import grad
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from pinnutils_git import *
from scipy.interpolate import griddata
from itertools import product, combinations
from scipy.io import savemat, loadmat
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from tqdm import tqdm_notebook as tqdm 
import torch.nn as nn
import torch.nn.functional as F

import scipy.io as sio
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def return_agg(X_test):
    ## input shape: # channels, [NxN], time snaps
    imgs  = torch.stack([img_t for img_t in X_test], dim=3)
    print(" imgs shape ", imgs.shape, " X_test shape ", X_test.shape,  imgs.view(X_test.shape[1],-1).shape)
    means = imgs.view(X_test.shape[1],-1).mean(dim=1)
    stds  = imgs.view(X_test.shape[1],-1).std(dim=1)
    return means, stds

L = np.double(sys.argv[1]); 
zeta = np.double(sys.argv[2]);
N = 256;

def train_model(zeta, model, criterion, optimizer, scheduler, n_epochs, batch_s, loader, scheduler_flag=True):
    
    model.train();
    
    vec = np.zeros((batch_s,3,N,N))
    vec[:,0,:,:] = 1; vec[:,1,:,:] = 0; vec[:,2,:,:] = 1;
    torch_vec = torch.tensor(vec, dtype=torch.float32).to(device)
    
    start_time = time.time()
    for epoch in range(0, n_epochs+1):
        # monitor training loss
        train_loss = 0.0

        for i, (x_batch, y_batch) in enumerate(loader):
            optimizer.zero_grad()
            x_batch = x_batch.to(device).requires_grad_(True)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            #print(" outputs shape ", outputs.shape)
            
            beta0 = outputs[:,0,:,:].reshape(batch_s,1,N,N)
            beta1 = outputs[:,1,:,:].reshape(batch_s,1,N,N)
            alpha0 = outputs[:,2,:,:].reshape(batch_s,1,N,N)
            alpha1 = outputs[:,3,:,:].reshape(batch_s,1,N,N)
            alpha2 = outputs[:,4,:,:].reshape(batch_s,1,N,N)
            pred = (alpha0 + 2*zeta*beta0)*torch_vec + (alpha1 + 2*zeta*beta1)*y_batch[:,0:3,:,:] + alpha2*y_batch[:,3:6,:,:]
            
            loss = criterion(pred, y_batch[:,6:9,:,:])
            loss.backward()
            optimizer.step()
            
        if(epoch%100==0):
            print('Epoch: {} \t learning-rate: {} \t loss: {} \t lr: {}'.format(
                epoch, 
                scheduler.get_last_lr(),
                loss,
                optimizer.param_groups[0]['lr']
                ))
        
        if(epoch % 10000 == 0):
            model_name = "Closure_FNO_L_{}_zeta{}_ep{}_sin.pth".format(L, zeta,epoch)
            torch.save({
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "loss": train_loss,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },model_name)
        
        if(scheduler_flag):
            scheduler.step()

    elapsed_time = time.time() - start_time
    print(' total CPU time for training = ',elapsed_time)
    
    return model

def evaluate_STFNO(model, device, L, zeta, load_seed, full=True):
    
    model.to(device='cpu')
    model.eval();
    
    data = loadmat('reference_data_L_' + str(L) + '_zeta_'+ str(zeta)+'_seed'+str(load_seed)+ '_.mat')
    D    = data["D"][:,:,:,:]; DD   = data["DD"][:,:,:,:]; 
    SD   = data["SD"][:,:,:,:]; SE   = data["SE"][:,:,:,:]; 
    EE   = data["EE"][:,:,:,:]; E   = data["E"][:,:,:,:]; 
    n_snaps = D.shape[3];
        ## reshaping for CNN feed
    D  = np.transpose(D, [3,2,0,1]);    D  = D.reshape(n_snaps,3,N,N)
    SD = np.transpose(SD, [3,2,0,1]);   SD = SD.reshape(n_snaps,3,N,N)
    DD = np.transpose(DD, [3,2,0,1]);   DD = DD.reshape(n_snaps,3,N,N)
    E = np.transpose(E, [3,2,0,1]);     E = E.reshape(n_snaps,3,N,N)
    EE = np.transpose(EE, [3,2,0,1]);   EE = EE.reshape(n_snaps,3,N,N)
    SE = np.transpose(SE, [3,2,0,1]);   SE = SE.reshape(n_snaps,3,N,N)
    
    trace_DD = DD[:,0,:,:] + DD[:,2,:,:];
    trace_EE = EE[:,0,:,:] + EE[:,2,:,:];
    trace_DE =  D[:,0,:,:]*E[:,0,:,:] + 2*D[:,1,:,:]*E[:,1,:,:] + D[:,2,:,:]*E[:,2,:,:]
    
    trace_DD = trace_DD.reshape(n_snaps,1,N,N)
    trace_EE = trace_EE.reshape(n_snaps,1,N,N)
    trace_DE = trace_DE.reshape(n_snaps,1,N,N)
    
    # input: D (# snaps x #channels x N x N); trDD (# snaps x N x N); SD (# snaps x # channels x N x N)
    X_eval = torch.tensor(np.hstack([trace_DD, trace_EE, trace_DE]), dtype=torch.float32)
    ST      = SE + 2*zeta*SD
    y_eval = torch.tensor(np.hstack([D, E, ST]), dtype=torch.float32)
    
    output_eval = model(X_eval) #.cpu().data.numpy();
    beta0      = output_eval[:,0,:,:].reshape(n_snaps,N,N);
    beta1      = output_eval[:,1,:,:].reshape(n_snaps,N,N);
    alpha0     = output_eval[:,2,:,:].reshape(n_snaps,N,N);
    alpha1     = output_eval[:,3,:,:].reshape(n_snaps,N,N);
    alpha2     = output_eval[:,4,:,:].reshape(n_snaps,N,N);
    
    print(y_eval[:,0,:,:].shape, alpha0.shape, alpha1.shape)
    
    recon_st11        = ((alpha0 + 2*zeta*beta0)*1 + (alpha1 + 2*zeta*beta1)*y_eval[:,0,:,:] + alpha2*y_eval[:,3,:,:]).cpu().data.numpy();
    recon_st12        = ((alpha0 + 2*zeta*beta0)*0 + (alpha1 + 2*zeta*beta1)*y_eval[:,1,:,:] + alpha2*y_eval[:,4,:,:]).cpu().data.numpy();
    recon_st22        = ((alpha0 + 2*zeta*beta0)*1 + (alpha1 + 2*zeta*beta1)*y_eval[:,2,:,:] + alpha2*y_eval[:,5,:,:]).cpu().data.numpy();
    
    if(full):  
        rmse_st = np.zeros((3,))
        rmse_st[0] = np.mean((ST[:,0,:,:].flatten() - recon_st11.flatten())**2)/np.mean(ST[:,0,:,:].flatten()**2)
        rmse_st[1] = np.mean((ST[:,1,:,:].flatten() - recon_st12.flatten())**2)/np.mean(ST[:,1,:,:].flatten()**2)
        rmse_st[2] = np.mean((ST[:,2,:,:].flatten() - recon_st22.flatten())**2)/np.mean(ST[:,2,:,:].flatten()**2)
        
        return rmse_st
    else:
        rmse_st = np.zeros((n_snaps, 3))
        for k in range(0, n_snaps):
            rmse_st[k,0] = np.mean((ST[k,0,:,:].flatten() - recon_st11[k,:,:].flatten())**2)/np.mean(ST[k,0,:,:].flatten()**2)
            rmse_st[k,1] = np.mean((ST[k,1,:,:].flatten() - recon_st12[k,:,:].flatten())**2)/np.mean(ST[k,1,:,:].flatten()**2)
            rmse_st[k,2] = np.mean((ST[k,2,:,:].flatten() - recon_st22[k,:,:].flatten())**2)/np.mean(ST[k,2,:,:].flatten()**2)
            
        return rmse_st

def add_snaps(snaps, test_):
    count = 0; arr_ = np.argsort(test_)[::-1];
    for i in range(0, test_.shape[0]):
        if( (snaps.tolist().count(arr_[i]) == 0) and count < n_samples):
            snaps = np.hstack([snaps, arr_[i]])
            count = count + 1
        elif(count>=n_samples):
            return snaps;
        else:
            continue
    
def generate_snaps(list_, n_snaps, seed):
    
    print(" length of list_ ", len(list_))
    
    np.random.seed(seed);
    temp  = np.arange(0, len(list_));
    np.random.shuffle(temp);
    ind = temp[0:n_snaps];
    snaps = list_[ind];
    print(" removing... ", snaps)
    list_ = np.asarray([i for i in list_ if i not in list_[ind]])
    
    return snaps, list_
    
def generate_loader(L, zeta, load_seed, snaps, batch_s):
    print(" snaps used for training ", snaps)
                   
    data = loadmat('reference_data_L_' + str(L) + '_zeta_'+ str(zeta)+'_seed'+str(load_seed)+ '_.mat')
    D    = data["D"][:,:,:,snaps]; DD   = data["DD"][:,:,:,snaps]; 
    SD   = data["SD"][:,:,:,snaps]; EE   = data["EE"][:,:,:,snaps]; 
    E   = data["E"][:,:,:,snaps]; SE = data["SE"][:,:,:,snaps];
    
    ## reshaping for CNN feed
    D  = np.transpose(D, [3,2,0,1]);    D  = D.reshape(len(snaps),3,N,N)
    SD = np.transpose(SD, [3,2,0,1]);   SD = SD.reshape(len(snaps),3,N,N)
    DD = np.transpose(DD, [3,2,0,1]);   DD = DD.reshape(len(snaps),3,N,N)
    E = np.transpose(E, [3,2,0,1]);     E = E.reshape(len(snaps),3,N,N)
    EE = np.transpose(EE, [3,2,0,1]);   EE = EE.reshape(len(snaps),3,N,N)
    SE = np.transpose(SE, [3,2,0,1]);   SE = SE.reshape(len(snaps),3,N,N)
    
    trace_DD = DD[:,0,:,:] + DD[:,2,:,:];
    trace_EE = EE[:,0,:,:] + EE[:,2,:,:];
    trace_DE =  D[:,0,:,:]*E[:,0,:,:] + 2*D[:,1,:,:]*E[:,1,:,:] + D[:,2,:,:]*E[:,2,:,:]
    
    trace_DD = trace_DD.reshape(len(snaps),1,N,N)
    trace_EE = trace_EE.reshape(len(snaps),1,N,N)
    trace_DE = trace_DE.reshape(len(snaps),1,N,N)
    
    X_train = torch.tensor(np.hstack([trace_DD, trace_EE, trace_DE]), dtype=torch.float32)
    ST      = SE + 2*zeta*SD
    y_train = torch.tensor(np.hstack([D, E, ST]), dtype=torch.float32)
    print('X_train', X_train.shape, y_train.shape) 

    loader  = FastTensorDataLoader(X_train, y_train, batch_size=batch_s, shuffle=True)
    print(" number of batches ", len(loader))
    
    return loader

load_seed = int(sys.argv[3])
batch_s = 10; n_samples = 10;

#####################################################
snaps = np.arange(0,n_samples)
loader  = generate_loader(L, zeta, load_seed, snaps, batch_s)
print(" snaps ", snaps)
####################################################

Ninp = 3; Nout = 5; width = 2; modes = 16; # 4, 18
model = FNO2d(Ninp, Nout, modes, width).to(device) # width, modes
print("#parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
# model = model.double();
print(model)
#net = net.double()
print(model)

# specify loss function
criterion = nn.MSELoss()
# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# https://arxiv.org/abs/1711.00489 - Don't decay the LR, increase the batch-size
milestones = [[2500,5000,7500]] 
#milestones = [[5000]]
scheduler = MultiStepLR(optimizer, milestones[0], gamma=0.1)

n_epochs = 10_001;
model_ft  = train_model(zeta, model, criterion, optimizer, scheduler, n_epochs, batch_s, loader, scheduler_flag=True)  
rmse_st   = evaluate_STFNO(model_ft, device, L, zeta, load_seed, full=False);
test_ = np.mean(rmse_st,axis=1)
print(" test_ ", test_, " aggregated ", np.mean(test_))
print(" max ind ", np.argsort(test_)[::-1][0:5])

