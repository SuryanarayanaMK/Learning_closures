import sys
sys.executable
sys.path.insert(0,'./anaconda3/lib/python3.9/site-packages')

import dataclasses
import numpy as np
from timeit import default_timer as timer
import math
#### pytorch related dependencies
import torch
import scipy.io as sio
from scipy.io import savemat, loadmat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" device ", device)
print(" done with imports... ")

####################################################################################################
# This code generates solution of Kinetic theory detailed in @https://arxiv.org/abs/2308.06675
################################################################################################### 

def con(arr):
    return arr.cpu().data.numpy()

def to_ZNN(func_):
    return np.transpose(func_, [2,0,1])

def to_NNZ(func_):
    return np.transpose(func_, [1,2,0])

def torch_NNZ(func_):
    return torch.transpose(func_,1,2).transpose(0,2)

def torch_ZNN(func_):
    return torch.transpose(func_,2,0).transpose(1,2)

def initialize2_pytorch(seed_):
    psi = torch.ones((N,N,Nth),dtype=torch.float64,device=device)
    Nk  = 8; pert = 0.1;
    torch.manual_seed(seed_)
    np.random.seed(seed_)
    print(L)
    for i in range(0,Nk):
        k1 = np.random.randint(1,Nk+1); k2 = np.random.randint(1,Nk+1)
        rand = np.random.uniform(0,1,4)
        psi = psi + pert*(rand[0]-0.5)\
                        *torch.cos(2.*torch.pi*k1*xx_/L + rand[1])\
                        *torch.cos(2.*torch.pi*k2*yy_/L + rand[2])\
                        *torch.cos(2.*th_ + rand[3])**(i+1);
    
    psi = (L**2)*psi/(torch.sum(psi*dx_*dx_*dth_))
    psi_hat = torch.fft.fftn(psi)
    return psi_hat

def sigma_h(psi_h, u_h, v_h):
    
    e11 = torch.real(torch.fft.ifft2(kx_op*u_h)); ## ux
    e12 = 0.5*torch.real(torch.fft.ifft2(ky_op*u_h + kx_op*v_h)); ## 0.5(uy + vx)
    e22 = -e11; ## vy
        
    # Moments of the distribution functions
    psi_h0 = torch.fft.ifft2(psi_h[:,:,0])*dth_;
    psi_h2 = torch.fft.ifft2(psi_h[:,:,-2])*dth_;
    psi_h4 = torch.fft.ifft2(psi_h[:,:,-4])*dth_;
    
    # Concentration
    c = torch.real(psi_h0);
    #print(" c mean ", np.mean(con(c).flatten()), np.std(con(c).flatten()))
    #print(" max c ", jnp.max(c), " min c ", jnp.min(c))
    
    # D tensor 
    d11 = 0.5*(torch.real(psi_h0) + torch.real(psi_h2)); 
    d12 = 0.5*torch.imag(psi_h2);
    d22 = 0.5*(torch.real(psi_h0) - torch.real(psi_h2)); 
    
    # S tensor
    s1111 = 0.125*( 3.0*torch.real(psi_h0) + 4.0*torch.real(psi_h2) + torch.real(psi_h4) ); # verified in mathematica
    s1112 = 0.125*( 2.0*torch.imag(psi_h2) + torch.imag(psi_h4) ); # verified in mathematica
    s1122 = d11 - s1111;
    s1222 = d12 - s1112;
    s2222 = d22 - s1122;
    
    # D*D
    dd11  = d11*d11 + d12*d12; 
    dd12  = d11*d12 + d12*d22; 
    dd22  = d12*d12 + d22*d22;
    
    # S:E
    se11  = s1111*e11 + 2.0*s1112*e12 + s1122*e22; 
    se12  = s1112*e11 + 2.0*s1122*e12 + s1222*e22; 
    se22  = s1122*e11 + 2.0*s1222*e12 + s2222*e22;
    
    # S:D
    sd11  = s1111*d11 + 2.0*s1112*d12 + s1122*d22; 
    sd12  = s1112*d11 + 2.0*s1122*d12 + s1222*d22; 
    sd22  = s1122*d11 + 2.0*s1222*d12 + s2222*d22;
    
    #print(" alpha ", alpha, " beta ", beta,  " zeta ", zeta)
    
    # Tensor components
    s11_h = torch.fft.fft2( alpha*(d11 - c/2.) + beta*se11 - 2*zeta*beta*(dd11 - sd11) ); 
    s12_h = torch.fft.fft2( alpha* d12         + beta*se12 - 2*zeta*beta*(dd12 - sd12) );
    s22_h = torch.fft.fft2( alpha*(d22 - c/2.) + beta*se22 - 2*zeta*beta*(dd22 - sd22) );
    s21_h = s12_h;
    
    return s11_h, s12_h, s21_h, s22_h;    

def Stokes(s11_h, s12_h, s21_h, s22_h):
    
    u_hat =   L11_h*(kx_op*s11_h + ky_op*s21_h) + L12_h*(kx_op*s12_h + ky_op*s22_h);
    v_hat =   L21_h*(kx_op*s11_h + ky_op*s21_h) + L22_h*(kx_op*s12_h + ky_op*s22_h);
    
    return u_hat, v_hat

def flux(psi_h, u_h, v_h, p1, p2, filter_):
    
    # velocity and distribution function in real space
    u = torch.real(torch.fft.ifft2(u_h)); 
    v = torch.real(torch.fft.ifft2(v_h));
    psi = torch.real(torch.fft.ifftn(psi_h)); # ifft2 was done instead of ifftn
    
    # compute derivatives
    ux = torch.real(torch.fft.ifft2(kx_op * u_h ));
    uy = torch.real(torch.fft.ifft2(ky_op * u_h ));
    vx = torch.real(torch.fft.ifft2(kx_op * v_h ));
    vy = -ux;
    
    # get integral values from Fourier transform
    psi_h0 = torch.fft.ifft2(psi_h[:,:, 0]) * dth_;   # can we store this value and call maybe ?
    psi_h2 = torch.fft.ifft2(psi_h[:,:,-2]) * dth_;  # can we store this value and call maybe ?
    
    # D-tensor
    d11 = 0.5*(torch.real(psi_h0) + torch.real(psi_h2));
    d12 = 0.5*torch.imag(psi_h2); d21 = d12;
    d22 = 0.5*(torch.real(psi_h0) - torch.real(psi_h2));
    
    # T = grad(u) + 2*zeta*D
    t11 = ux + 2.0*zeta*d11; t12 = uy + 2.0*zeta*d12;
    t21 = vx + 2.0*zeta*d21; t22 = vy + 2.0*zeta*d22;
    
    # Conformational fluxes
    xdot_psi_h = torch.fft.fftn( torch_NNZ(u + V0*torch_ZNN(p1)) * psi); 
    ydot_psi_h = torch.fft.fftn( torch_NNZ(v + V0*torch_ZNN(p2)) * psi);
    
    pdot     = -p2*torch_NNZ( t11*torch_ZNN(p1) + t12*torch_ZNN(p2) ) + p1*torch_NNZ( t21*torch_ZNN(p1) + t22*torch_ZNN(p2) ); 
    #print(" type ", pdot.dtype, psi.dtype, pdot.shape, psi.shape)
    divp_psi = kth_op*(torch.fft.fftn(pdot*psi));
    divx_psi = torch_NNZ( kx_op*torch_ZNN( xdot_psi_h ) + ky_op*torch_ZNN( ydot_psi_h) );
    
    B_h = -(divx_psi + divp_psi)*filter_ # design this filter 
    #print(" B_h device ", B_h.get_device())
    
    return B_h

def sbdf2(psi_h, psim1_h, B_h, Bm1_h):
    return K_h*(4.0*psi_h - psim1_h + 2.0*dt*(2.0*B_h - Bm1_h))

def write_grad(u_h, v_h):
    ux_ = torch.real(torch.fft.ifft2(kx_op*u_h));
    uy_ = torch.real(torch.fft.ifft2(ky_op*u_h));
    vx_ = torch.real(torch.fft.ifft2(kx_op*v_h));
    vy_ = torch.real(torch.fft.ifft2(ky_op*v_h));
    
    return ux_, uy_, vx_, vy_;

def write_out(psi_h, u_h, v_h):
    
    e11 = torch.real(torch.fft.ifft2(kx_op*u_h)); ## ux
    e12 = 0.5*torch.real(torch.fft.ifft2(ky_op*u_h + kx_op*v_h)); ## 0.5(uy + vx)
    e21 = e12;
    e22 = -e11; ## vy
    
    w12 = torch.real(torch.fft.ifft2(ky_op*u_h - kx_op*v_h)); ## 0.5(uy - vx)
    
    psi_h0 = torch.fft.ifft2(psi_h[:,:,0])*dth_;
    psi_h2 = torch.fft.ifft2(psi_h[:,:,-2])*dth_;
    psi_h4 = torch.fft.ifft2(psi_h[:,:,-4])*dth_;
    
    c = torch.real(psi_h0);
    # D tensor 
    d11 = 0.5*(torch.real(psi_h0) + torch.real(psi_h2)); 
    d12 = 0.5*torch.imag(psi_h2); d21 = d12;
    d22 = 0.5*(torch.real(psi_h0) - torch.real(psi_h2)); 
    
    dd11 = d11*d11 + d12*d21;
    dd12 = d11*d12 + d12*d22;
    dd21 = dd12;
    dd22 = d21*d12 + d22*d22;
    
    ee11 = e11*e11 + e12*e21;
    ee12 = e11*e12 + e12*e22;
    ee21 = ee12;
    ee22 = e21*e12 + e22*e22;
    
    # S tensor
    s1111 = 0.125*( 3.0*torch.real(psi_h0) + 4.0*torch.real(psi_h2) + torch.real(psi_h4) ); # verified in mathematica
    s1112 = 0.125*( 2.0*torch.imag(psi_h2) + torch.imag(psi_h4) ); # verified in mathematica
    s1122 = d11 - s1111;
    s1222 = d12 - s1112;
    s2222 = d22 - s1122;
    
    # S:D
    sd11  = s1111*d11 + 2.0*s1112*d12 + s1122*d22; 
    sd12  = s1112*d11 + 2.0*s1122*d12 + s1222*d22; 
    sd22  = s1122*d11 + 2.0*s1222*d12 + s2222*d22;
    
    # S:E
    se11  = s1111*e11 + 2.0*s1112*e12 + s1122*e22; 
    se12  = s1112*e11 + 2.0*s1122*e12 + s1222*e22; 
    se22  = s1122*e11 + 2.0*s1222*e12 + s2222*e22;
    
    return c, d11, d12, d22, e11, e12, e22, sd11, sd12, sd22, dd11, dd12, dd22, ee11, ee12, ee22, w12, se11, se12, se22

### Design operators and global parameters

# Global parameters
######################################
L = np.double(sys.argv[1]);     # domain size
dT = 0.05;  # translational diffusion
dR = 0.05;  # rotational diffusion
alpha = -1; # dipole strength
beta = 0.8; # particle concentration
zeta = np.double(sys.argv[2]);   # alignment strength 
V0 = 0;     # swimming speed

# Time stepping
#######################################
tf = 200; # final time
dt = 0.0004;#0.00625/16; # time step # I changed here for 2 to 4
Nt = np.floor(tf/dt+0.1);

# Discretization
#######################################
N = 256; # number of spatial modes
Nth = 256; # number oef orientational modes
Lth = 2.0*np.pi;
dx_ = L/N; dth_ = Lth/Nth;
print(" dx_ ", dx_, " dth_ ", dth_, " dt ", dt)

x_ = np.linspace(0, L-dx_, N);
y_ = np.linspace(0, L-dx_, N);
th_ = np.linspace(0, Lth-dth_, Nth);
xx_, yy_, th_ = np.meshgrid(x_, y_, th_); # note that I've flipped xx_jax, yy_jax and vice-versa

print(" xx_ ", xx_.shape, " yy_ ", yy_.shape, " yy_ ", th_.shape)

xx_ = np.transpose(xx_,[1,0,2]);
yy_ = np.transpose(yy_,[1,0,2]);

ik_x  = (1./L)*np.hstack((np.arange(0,int(N/2)+1), np.arange(-int(N/2)+1,0)));
ik_y  = (1./L)*np.hstack((np.arange(0,int(N/2)+1), np.arange(-int(N/2)+1,0)));
ik_th = (1./Lth)*np.hstack((np.arange(0,int(Nth/2)+1), np.arange(-int(Nth/2)+1,0)));
kx_, ky_, kth_ = np.meshgrid(ik_x, ik_y, ik_th); # this is done just to get kth_jax

kx2d_, ky2d_ = np.meshgrid(ik_x, ik_y);
kx2d_ = kx2d_.T; ky2d_ = ky2d_.T; # transpose important

kx_op  = 2.0*np.pi*1j*kx2d_; 
ky_op  = 2.0*np.pi*1j*ky2d_; 
kth_op = 2.0*np.pi*1j*kth_;
 
# Laplacian operators
Lx_h = kx_op**2 + ky_op**2; # spatial laplacian
#Lx_h = Lx_h.at[0, 0].set(1)
Lth_h = kth_op**2;  # orientational laplacian

ksq = np.abs(Lx_h); ksq[0,0] = 1 #ksq.at[0,0].set(1); # magnitude of Fourier modes
kx_n = np.imag(kx_op)/np.sqrt(ksq);
ky_n = np.imag(ky_op)/np.sqrt(ksq); # remember this doesn't have i in the front

linear_ = to_NNZ(dT*Lx_h + dR*to_ZNN(Lth_h));
#linear_ = linear_.at[0,0,:].set(1) # check this.
K_h = 1./(3. - 2.*dt*linear_);

# Building stokes operator
L11_h = (1./ksq)*(1.0 - kx_n*kx_n); # the sign is important, as kx_n is normalized - check overleaf for derivation
L12_h = (1./ksq)*(0.0 - kx_n*ky_n);
L21_h = (1./ksq)*(0.0 - ky_n*kx_n);
L22_h = (1./ksq)*(1.0 - ky_n*ky_n);

## filter for 2/3-aliasing
max_k = np.max(np.abs(ky_[:,:,:]))
max_th = np.max(np.abs(kth_[:,:,:]))
print(" max_k ", max_k, " max_th ", max_th)
filter_ = (np.abs(kx_) < (2/3.)*max_k) & (np.abs(ky_) < (2/3.)*max_k) & (np.abs(kth_) < (2/3.)*max_th);
print(" Lx_h ", Lx_h.shape, " Lth_h ", Lth_h.shape)

ik_x  = (1./L)*np.hstack((np.arange(0,int(N/2)+1), np.arange(-int(N/2)+1,0)));
ik_y  = (1./L)*np.hstack((np.arange(0,int(N/2)+1), np.arange(-int(N/2)+1,0)));
ik_x[int(N/2)] = 0; ik_y[int(N/2)] = 0;

kx2d_, ky2d_ = np.meshgrid(ik_x, ik_y);
kx2d_ = kx2d_.T; ky2d_ = ky2d_.T; # transpose important

kx_op  = 2.0*np.pi*1j*kx2d_; 
ky_op  = 2.0*np.pi*1j*ky2d_; 

############################# conversion to pytorch #############################
p1 = torch.tensor(np.cos(th_),dtype=torch.float64,device=device); 
p2 = torch.tensor(np.sin(th_),dtype=torch.float64,device=device); 

xx_ = torch.tensor(xx_, dtype=torch.float64,device=device);
yy_ = torch.tensor(yy_, dtype=torch.float64,device=device);
th_ = torch.tensor(th_, dtype=torch.float64,device=device);
K_h = torch.tensor(K_h, dtype=torch.complex128,device=device);
kx_op = torch.tensor(kx_op, dtype=torch.complex128,device=device);
ky_op = torch.tensor(ky_op, dtype=torch.complex128,device=device);
kth_op = torch.tensor(kth_op, dtype=torch.complex128,device=device);
L11_h = torch.tensor(L11_h, dtype=torch.float64,device=device);
L12_h = torch.tensor(L12_h, dtype=torch.float64,device=device);
L21_h = torch.tensor(L21_h, dtype=torch.float64,device=device);
L22_h = torch.tensor(L22_h, dtype=torch.float64,device=device);
filter_ = torch.tensor(filter_,dtype=torch.bool,device=device);
############################# conversion to pytorch #############################

# # initialize psi_h
inp_seed = np.int(sys.argv[3]); 
psim1_h = initialize2_pytorch(inp_seed);
psi_h = psim1_h + 0;

################
psi_h0 = torch.fft.ifft2(psi_h[:,:,0])*dth_;
psi_h2 = torch.fft.ifft2(psi_h[:,:,-2])*dth_;
psi_h4 = torch.fft.ifft2(psi_h[:,:,-4])*dth_;
# Concentration
c = torch.real(psi_h0);
print(" c ", np.mean(con(c).flatten()), np.std(con(c).flatten()))
d11 = 0.5*(torch.real(psi_h0) + torch.real(psi_h2)); 
d12 = 0.5*torch.imag(psi_h2);
d22 = 0.5*(torch.real(psi_h0) - torch.real(psi_h2));

print(" done with initialization ")    
print(" max  c ", torch.max((c)), " min c ", torch.min(c))
print(" max  d11 ", torch.max(d11), " min d11 ", torch.min(d11))
print(" max  d12 ", torch.max(d12), " min d12 ", torch.min(d12), " sum ", torch.max(d12)+torch.min(d12))
print(" max  d22 ", torch.max(d22), " min d22 ", torch.min(d22))

# # initialize u_h, v_h
u = torch.zeros((N,N),dtype=torch.float64,device=device);
v = torch.zeros((N,N),dtype=torch.float64,device=device);
u_h = torch.fft.fft2(u); v_h = torch.fft.fft2(v); 
s11_h, s12_h, s21_h, s22_h = sigma_h(psim1_h, u_h, v_h)
u_h, v_h = Stokes(s11_h, s12_h, s21_h, s22_h);
Bm1_h = flux(psim1_h, u_h, v_h, p1, p2, filter_)

def kinetic(array_):
    psi_h, psim1_h, u_h, v_h, Bm1_h = array_
    
    s11_h, s12_h, s21_h, s22_h = sigma_h(psi_h, u_h, v_h)
    u_hat, v_hat = Stokes(s11_h, s12_h, s21_h, s22_h)
    B_h      = flux(psi_h, u_hat, v_hat, p1, p2, filter_)
    psip1_h  = sbdf2(psi_h, psim1_h, B_h, Bm1_h)
    
    psim1_h  = psi_h + 0;
    psi_h    = psip1_h + 0;
    Bm1_h    = B_h + 0;
    return (psi_h, psim1_h, u_hat, v_hat, Bm1_h)

#outer_steps = 250; inner_steps = 25; # changed outer-iterations to 
outer_steps = 80; inner_steps = 640;
# ## data generated with outer_steps = 80; inner_steps = 640;

def loop_kinetic(psi_h, psim1_h, u_h, v_h, Bm1_h):
    arr_ = (psi_h, psim1_h, u_h, v_h, Bm1_h)
    for i in range(0, inner_steps):
        arr_ = kinetic(arr_)
    return arr_

#store_psi   = np.zeros((N,N,Nth,outer_steps))
store_U     = np.zeros((N,N,2,outer_steps+1))
tensor_c    = np.zeros((N,N,outer_steps+1))
tensor_D    = np.zeros((N,N,3,outer_steps+1))
tensor_E    = np.zeros((N,N,3,outer_steps+1))
tensor_DD   = np.zeros((N,N,3,outer_steps+1))
tensor_SD   = np.zeros((N,N,3,outer_steps+1))
tensor_SE   = np.zeros((N,N,3,outer_steps+1))
tensor_EE   = np.zeros((N,N,3,outer_steps+1))
tensor_W    = np.zeros((N,N,outer_steps+1))
tensor_dU   = np.zeros((N,N,4,outer_steps+1))

c_, d11_, d12_, d22_,  e11_, e12_, e22_, sd11_, sd12_, sd22_, dd11_, dd12_, dd22_, \
                                         ee11_, ee12_, ee22_, w12_, se11_, se12_, se22_ = write_out(psi_h, u_h, v_h)

### writing initial conditions
store_U[:,:,0,0] = con(torch.real(torch.fft.ifft2(u_h)));
store_U[:,:,1,0] = con(torch.real(torch.fft.ifft2(v_h)));
tensor_D[:,:,0,0]  = con(d11_);   tensor_D[:,:,1,0]  = con(d12_);  tensor_D[:,:,2,0]  = con(d22_);
tensor_SD[:,:,0,0] = con(sd11_);  tensor_SD[:,:,1,0] = con(sd12_); tensor_SD[:,:,2,0] = con(sd22_);
tensor_DD[:,:,0,0] = con(dd11_);  tensor_DD[:,:,1,0] = con(dd12_); tensor_DD[:,:,2,0] = con(dd22_);
tensor_SE[:,:,0,0] = con(se11_);  tensor_SE[:,:,1,0] = con(se12_); tensor_SE[:,:,2,0] = con(se22_);
tensor_E[:,:,0,0]  = con(e11_);   tensor_E[:,:,1,0]  = con(e12_);  tensor_E[:,:,2,0]  = con(e22_);
tensor_EE[:,:,0,0] = con(ee11_);  tensor_EE[:,:,1,0] = con(ee12_); tensor_EE[:,:,2,0] = con(ee22_);
tensor_c[:,:,0] = con(c_);
tensor_W[:,:,0] = con(w12_);
ux_, uy_, vx_, vy_ = write_grad(u_h, v_h);
tensor_dU[:,:,0,0] = con(ux_);  tensor_dU[:,:,1,0] = con(uy_);
tensor_dU[:,:,2,0] = con(vx_);  tensor_dU[:,:,3,0] = con(vy_);
### done with initial conditions

print(" successful ")

total_time = 0;
for write_ in range(0, outer_steps):

    start = timer();
    psi_h, psim1_h, u_h, v_h, Bm1_h = loop_kinetic(psi_h, psim1_h, u_h, v_h, Bm1_h)
    end = timer();
    
    total_time = total_time + end-start;
    print(" time taken ", end-start, " at outer ", write_, " per-step ", (end-start)/inner_steps, " sim-    time", write_*inner_steps*dt)

    c_, d11_, d12_, d22_,  e11_, e12_, e22_, sd11_, sd12_, sd22_, dd11_, dd12_, dd22_, \
                                         ee11_, ee12_, ee22_, w12_, se11_, se12_, se22_ = write_out(psi_h, u_h, v_h)

    tensor_c[:,:,write_+1] = con(c_)
    tensor_D[:,:,0,write_+1]  = con(d11_);  tensor_D[:,:,1,write_+1]  = con(d12_);  tensor_D[:,:,2,write_+1]  = con(d22_);
    tensor_SD[:,:,0,write_+1] = con(sd11_); tensor_SD[:,:,1,write_+1] = con(sd12_); tensor_SD[:,:,2,write_+1] = con(sd22_);
    tensor_DD[:,:,0,write_+1] = con(dd11_); tensor_DD[:,:,1,write_+1] = con(dd12_); tensor_DD[:,:,2,write_+1] = con(dd22_);
    tensor_SE[:,:,0,write_+1] = con(se11_); tensor_SE[:,:,1,write_+1] = con(se12_); tensor_SE[:,:,2,write_+1] = con(se22_);
    tensor_E[:,:,0,write_+1]  = con(e11_);  tensor_E[:,:,1,write_+1]  = con(e12_);  tensor_E[:,:,2,write_+1]  = con(e22_);
    tensor_EE[:,:,0,write_+1] = con(ee11_); tensor_EE[:,:,1,write_+1] = con(ee12_); tensor_EE[:,:,2,write_+1] = con(ee22_);
    tensor_W[:,:,write_+1]    = con(w12_) 
    ux_, uy_, vx_, vy_ = write_grad(u_h, v_h);
    tensor_dU[:,:,0,write_+1] = con(ux_);  tensor_dU[:,:,1,write_+1] = con(uy_);
    tensor_dU[:,:,2,write_+1] = con(vx_);  tensor_dU[:,:,3,write_+1] = con(vy_);
     
    if(np.isnan(tensor_W[:,:,write_]).any()):
        print(" nan valued encountered \n ")
        print(" Terminating program !! ")
        break;
        
    #store_psi[:,:,:,write_] = np.real(np.fft.ifftn(psi_h));
    store_U[:,:,0,write_+1]     = con(torch.real(torch.fft.ifft2(u_h)));
    store_U[:,:,1,write_+1]     = con(torch.real(torch.fft.ifft2(v_h)));

print(" total-time taken by the simulation ", total_time)

mdic = {"c": tensor_c, "D": tensor_D, "DD": tensor_DD, "SD": tensor_SD, "SE": tensor_SE, "E": tensor_E, "EE": tensor_EE, "W":tensor_W, "label": "experiment","tensor_dU": tensor_dU, "store_U":store_U}
savemat("reference_data_L_" + str(L) + "_zeta_"+ str(zeta)+"_seed"+str(inp_seed)+ "_.mat", mdic)

