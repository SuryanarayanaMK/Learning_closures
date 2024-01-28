import os
import time
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
from torch.autograd import grad
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

import sys

class BatchNorm(object):
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
        
    def __call__(self, x):
        return (x-self.mean)/self.std
    
class BatchNormCNN(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
        
    def __call__(self,x):
        norm  = transforms.Normalize(self.mean,self.std)
        imgs_norm = torch.stack([norm(img_) for img_ in x], dim=3)
        imgs_norm = imgs_norm.permute(3,0,1,2)
        return imgs_norm

class LayerNoWN(nn.Module):
    def __init__(self, in_features, out_features, seed, activation):
        super(LayerNoWN, self).__init__()
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
            
        gain = 5/3 if isinstance(activation, nn.Tanh) else 1
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.zeros_(self.linear.bias)
        
        self.linear = self.linear
        
    def forward(self, x):
        return self.linear(x)

class PINNNoWN(nn.Module):
    
    def __init__(self, sizes, mean=0, std=1, seed=0, activation=nn.Tanh()):
        super(PINNNoWN, self).__init__()
        
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

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class Sin(nn.Module):
    
    def forward(self, x):
        return torch.sin(x)

######################################################################    
class reConvNetST(nn.Module):
    def __init__(self,inp_dim=1, out_dim=1,num_nodes=8,fs=3,seed=0):
        super(reConvNetST, self).__init__()
        
        torch.manual_seed(seed)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(inp_dim, num_nodes,  kernel_size=fs, stride=1, padding=(1,1), padding_mode='circular'), #channel 1, channel out, kernel_size ...
            Sin())
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_nodes, num_nodes, kernel_size=fs, stride=1, padding=(1,1), padding_mode='circular'),
            Sin())
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_nodes, num_nodes, kernel_size=fs, stride=1, padding=(1,1), padding_mode='circular'),
            Sin())
        self.layer4 = nn.Sequential(
            nn.Conv2d(num_nodes, num_nodes, kernel_size=fs, stride=1, padding=(1,1), padding_mode='circular'),
            Sin())
        self.layer5 = nn.Sequential(
            nn.Conv2d(num_nodes, num_nodes, kernel_size=fs, stride=1, padding=(1,1), padding_mode='circular'),
            Sin())
        self.layer6 = nn.Sequential(
            nn.Conv2d(num_nodes, out_dim, kernel_size=fs, stride=1, padding=(1,1), padding_mode='circular'))

    def forward(self, x): # shows how the data flows through layers
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out
    
######################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super(SpectralConv2d, self).__init__()
        
        torch.manual_seed(0) # setting seed
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1./(in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, self.modes, dtype=torch.complex64))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, self.modes, dtype=torch.complex64))

    def compl_mul1d(self, input, weights):   
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.tensor:
        
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.complex64)
        out_ft[:,:,:self.modes,:self.modes] = self.compl_mul1d(x_ft[:,:,:self.modes,:self.modes], self.weights1)
        out_ft[:,:,-self.modes:,:self.modes] = self.compl_mul1d(x_ft[:,:,-self.modes:,:self.modes], self.weights2)
        x = torch.fft.irfft2(out_ft,s=(x.size(-2), x.size(-1)))
        return x
    
class FNO2d(nn.Module):
    def __init__(self, Ninp, Nout, modes, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: a driving function observed at T timesteps + 1 locations (u(1, x), ..., u(T, x),  x).
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.modes = modes
        self.width = width

        self.conv0 = SpectralConv2d(Ninp,self.width,self.modes)
        self.conv1 = SpectralConv2d(self.width,self.width,self.modes)
        # self.conv2 = SpectralConv2d(self.width,self.width,self.modes)
        # self.conv3 = SpectralConv2d(self.width,self.width,self.modes)
        self.conv4 = SpectralConv2d(self.width,Nout,self.modes)
        
        # self.w0 = nn.Conv2d(Ninp, self.width, kernel_size=1)
        # self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1)
        # self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1)
        # # self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1)
        # self.w4 = nn.Conv2d(self.width, Nout, 1)

    def forward(self, x):
        x1 = self.conv0(x); #x2 = self.w0(x); 
        x = x1 #+ x2; 
        x = Sin()(x)
        
        x1 = self.conv1(x); #x2 = self.w1(x); 
        x = x1 #+ x2; 
        x = Sin()(x)
        
        x1 = self.conv4(x); #x2 = self.w4(x); 
        x = x1 #+ x2;
        return x    
