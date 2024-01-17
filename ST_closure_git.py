import numpy as np
import torch
from torch.autograd import grad
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from pinnutils import *
from scipy.interpolate import griddata
from itertools import product, combinations
from scipy.io import savemat, loadmat
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from tqdm import tqdm_notebook as tqdm 

import scipy.io as sio
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

L = np.double(sys.argv[1]); 
zeta = np.double(sys.argv[2]);
N = 256;

class DNN(nn.Module):
    
    def __init__(self, sizes, mean=0, std=1, seed=0, activation=nn.Tanh()):
        super(DNN, self).__init__()
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.bn = BatchNorm(mean, std)
        
        layer = []
        for i in range(len(sizes)-2):
            linear = LayerNoWN(sizes[i], sizes[i+1], seed, activation)
            layer += [linear, activation]
            
        layer += [LayerNoWN(sizes[-2], sizes[-1], seed, activation)]
        
        self.net = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.net(self.bn(x))


def train_model(zeta, model, criterion, optimizer, scheduler, n_epochs, batch_s, loader, scheduler_flag=True):
    
    model.train();
    
    vec = np.zeros((batch_s,3))
    vec[:,0] = 1; vec[:,1] = 0; vec[:,2] = 1;
    torch_vec = torch.tensor(vec, dtype=torch.float32).to(device)
    
    start_time = time.time()
    for epoch in range(n_epochs):
        train_loss = 0.0
        
        for i, (X_batch, Y_batch) in enumerate(loader):
            optimizer.zero_grad()
            X_batch = X_batch.to(device).requires_grad_(True)
            Y_batch = Y_batch.to(device)
            outputs = model(X_batch)
            
            # print(" outputs shape ", outputs.shape)
            # print(" Y_batch shape ", Y_batch[:,0:3].shape, Y_batch[:,3:6].shape)
            # print(" torch_vec shape ", torch_vec.shape)
            
            pred = (outputs[:,2:3] + 2*zeta*outputs[:,0:1])*torch_vec + \
                   (outputs[:,3:4] + 2*zeta*outputs[:,1:2])*Y_batch[:,0:3] + \
                    outputs[:,4:5]*Y_batch[:,3:6]

            loss = criterion(pred, Y_batch[:,6:9])
        
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
            model_name = "/mnt/ceph/users/smaddu/closure_modeling/sn_git_DNN_closure_L_{}_zeta{}_ep{}_sin.pth".format(L, zeta, epoch)
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

def evaluate_STCNN(model, device, L, zeta, load_seed, full=True):
    
    model.to(device='cpu')
    model.eval();
    
    data = loadmat('/mnt/ceph/users/smaddu/closure_modeling/test_matlab_matrix_L_'+str(L)+'_zeta_'+str(zeta)+\
                   '_seed'+str(load_seed)+'_.mat')
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
    
    d11 = D[:,0,:,:].ravel();    d12 = D[:,1,:,:].ravel();   d22 = D[:,2,:,:].ravel();
    e11 = E[:,0,:,:].ravel();    e12 = E[:,1,:,:].ravel();   e22 = E[:,2,:,:].ravel();
    
    ST  = SE + 2*zeta*SD
    se11 = SE[:,0,:,:].ravel(); se12 = SE[:,1,:,:].ravel(); se22 = SE[:,2,:,:].ravel();
    st11 = ST[:,0,:,:].ravel(); st12 = ST[:,1,:,:].ravel(); st22 = ST[:,2,:,:].ravel();

    X = np.vstack([trace_DD.ravel(), trace_EE.ravel(), trace_DE.ravel()]).T
    Y = np.vstack([d11, d12, d22, e11, e12, e22, st11, st12, st22]).T 
    y_test = Y; 
    
    X_test = torch.tensor(X, dtype=torch.float32)
    # y_test = torch.tensor(Y, dtype=torch.float32)
    print(" X_test shape ", X_test.shape, " y_test shape ", y_test.shape)
    
    if(full):  
        rmse_st = np.zeros((3,))
        rmse_st[0] = np.mean((ST[:,0,:,:].flatten() - recon_st11.flatten())**2)/np.mean(ST[:,0,:,:].flatten()**2)
        rmse_st[1] = np.mean((ST[:,1,:,:].flatten() - recon_st12.flatten())**2)/np.mean(ST[:,1,:,:].flatten()**2)
        rmse_st[2] = np.mean((ST[:,2,:,:].flatten() - recon_st22.flatten())**2)/np.mean(ST[:,2,:,:].flatten()**2)
        
        return rmse_st
    else:
        rmse_st = np.zeros((n_snaps, 3))
        for k in range(0, n_snaps):
            out = model(X_test[k*(N*N):(k+1)*(N*N),:]).cpu().data.numpy();
            
            pred = (out[:,2] + 2*zeta*out[:,0])*1 + \
                   (out[:,3] + 2*zeta*out[:,1])*y_test[k*(N*N):(k+1)*(N*N),0] + \
                   out[:,4]*y_test[k*(N*N):(k+1)*(N*N),3]
            rmse_st[k,0] = np.mean((ST[k,0,:,:].flatten() - pred.flatten())**2)/np.mean(ST[k,0,:,:].flatten()**2)
            
            pred = (out[:,2] + 2*zeta*out[:,0])*0 + \
                   (out[:,3] + 2*zeta*out[:,1])*y_test[k*(N*N):(k+1)*(N*N),1] + \
                   out[:,4]*y_test[k*(N*N):(k+1)*(N*N),4]
            rmse_st[k,1] = np.mean((ST[k,1,:,:].flatten() - pred.flatten())**2)/np.mean(ST[k,1,:,:].flatten()**2)
            
            pred = (out[:,2] + 2*zeta*out[:,0])*1 + \
                   (out[:,3] + 2*zeta*out[:,1])*y_test[k*(N*N):(k+1)*(N*N),2] + \
                   out[:,4]*y_test[k*(N*N):(k+1)*(N*N),5]
            rmse_st[k,2] = np.mean((ST[k,2,:,:].flatten() - pred.flatten())**2)/np.mean(ST[k,2,:,:].flatten()**2)
            
        return rmse_st

def generate_loader(L, zeta, load_seed, snaps, batch_s, num_batch):
    print(" snaps used for training ", snaps)
    
    data = loadmat('/mnt/ceph/users/smaddu/closure_modeling/test_matlab_matrix_L_'+str(L)+'_zeta_'+str(zeta)+\
                   '_seed'+str(load_seed)+'_.mat')
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
    
    d11 = D[:,0,:,:].ravel();    d12 = D[:,1,:,:].ravel();   d22 = D[:,2,:,:].ravel();
    e11 = E[:,0,:,:].ravel();    e12 = E[:,1,:,:].ravel();   e22 = E[:,2,:,:].ravel();
    
    ST  = SE + 2*zeta*SD
    se11 = SE[:,0,:,:].ravel(); se12 = SE[:,1,:,:].ravel(); se22 = SE[:,2,:,:].ravel();
    st11 = ST[:,0,:,:].ravel(); st12 = ST[:,1,:,:].ravel(); st22 = ST[:,2,:,:].ravel();

    X = np.vstack([trace_DD.ravel(), trace_EE.ravel(), trace_DE.ravel()]).T
    Y = np.vstack([d11, d12, d22, e11, e12, e22, st11, st12, st22]).T 
    
    print(" X shape ", X.shape, " Y shape ", Y.shape, " X[:,0] shape ", X[:,0].shape)

    np.random.seed(load_seed)
    torch.manual_seed(load_seed)

    idxs = np.random.choice(X[:,0].size, num_batch*batch_s, replace=False)
    X_train = torch.tensor(X[idxs], dtype=torch.float32, device=device)
    y_train = torch.tensor(Y[idxs], dtype=torch.float32, device=device)
    
    print(" X_train shape ", X_train.shape, " y_train ", y_train.shape)

    # setup data loaders
    loader       = FastTensorDataLoader(X_train, y_train, batch_size=batch_s, shuffle=True)
    print(" number of batches ", len(loader))

    return loader

load_seed = int(sys.argv[3])
batch_s   = 16384
num_batch = 40;
n_samples = 10;

#####################################################
# sind = 0; eind = 81;
# np.random.seed(0);
# list_ = np.arange(sind,eind); #np.random.shuffle(arange_)
# snaps, list_ = generate_snaps(list_, n_samples, 10);
# loader  = generate_loader(L, zeta, load_seed, snaps, batch_s, num_batch)
# print(" snaps ", snaps)
# print(" list_ ", list_, len(list_))
#snaps = [0,1,2,3,4,5,6,7,8,9] 
snaps = np.arange(0,10)
loader  = generate_loader(L, zeta, load_seed, snaps, batch_s, num_batch)
print(" snaps ", snaps)
####################################################

model = DNN(sizes=[3,50,50,50,50,50,5],seed=0, activation=Sin()).to(device)
print("#parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
#model = model.double()
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
rmse_st   = evaluate_STCNN(model_ft, device, L, zeta, load_seed, full=False);
test_ = np.mean(rmse_st,axis=1)
print(" test_ ", test_, " aggregated ", np.mean(test_))
print(" max ind ", np.argsort(test_)[::-1][0:5])




