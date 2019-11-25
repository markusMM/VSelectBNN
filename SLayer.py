# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:07:39 2019

@author: Markus Meister
"""
#%% -- imports --
import math
import torch
from torch import nn
from pytorch_complex_tensor import ComplexTensor

#%% -- constants --

torch.pi = torch.Tensor([math.pi])
torch.j  = ComplexTensor([[0],[1]])

#%% -- Discrete s-plane --

class S_Layer(nn.Module):
    """
        Windowed-S-Transform-Layer
        
    The window funktion follows the following equation:
        w[n] = a-(1-a) cos( (n2pi)/(N-1) ); n = 0...N-1
    with parameter a.
    
    Thus: x'[n] = w[n]x[n]; n = 0...N-1
    
    The S-Plane is a simple S-Transform of the batch/window:
        X'[k] = sum( x'[n]exp( sigma - jnk/N ) )
    
    with k      ... number of Fourier coefficients 
    and  sigma  ... real part of the impulse response
    
    """
    
    def __init__(
            self, 
            input_dim, 
            window_length, 
            filter_length, 
            a_init = .5, 
            device = 'cuda', 
            abs_flag = False, 
            log_flag = False, 
        ):
        
        super().__init__()
        
        # window parameter
        self.a = nn.Parameter(torch.ones(input_dim)*a_init).to(device)
        
        # real part parameter
        self.sigma = nn.Parameter(torch.rand(input_dim)).to(device)
        
        self.D = input_dim
        self.L = window_length
        self.K = filter_length
        
        self.abs_flag = abs_flag
        self.log_flag = log_flag
        
        self.device = device
        self.to(self.device)
        
    def forward(self, input, abs_flag=None, log_flag=None):
        
        if type(abs_flag) == type(None):
            abs_flag = self.abs_flag
        if type(log_flag) == type(None):
            log_flag = self.log_flag
        
        if input.shape[-1] != self.D:
            input = MoveLastToFirst(input)
        
        L = self.L
        K = self.K
        D = self.D
        N = input.shape[0]
        
        
        
        x = input * ( self.a-(1-self.a) * torch.cos( (torch.arange(L)*2*torch.pi)/(L-1) ) )
        lnX = torch.j*2*torch.pi*(torch.arange(L)[:,None] @ torch.arange(K)[None,:])
        expjOmega = ComplexTensor(torch.exp(lnX / (L-1)))
        expSigma  = torch.exp(self.sigma / (L-1))
        X = ( x * expSigma[:,None,None] * torch.ones(D,L,K) * expjOmega ).sum(dim=-2)
        
        X = ComplexTensor(torch.fft(-torch,1))
        
        if abs_flag:
            X = X.abs()
        if log_flag:
            X = torch.log(X)
            X[X!=X] = 0
        
        return X
        
#%%        
        
        