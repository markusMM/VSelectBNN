# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:53:17 2019

@author: Markus.Meister1
"""

import torch
import numpy as np

#%%

def t_mask(t_shape, index, missing=None):
    
#    assert len(t_shape) > len(index.shape)
#    if len(t_shape) > len(index.shape):
#        missing = np.where(t_shape not in index.shape)
#    
#    if missing:
#        i_idx = np.ones(len(t_shape)-1).astype(str) * ':'
#        i_idx[missing] = 'None'
#        index = torch.ones(*t_shape[1:],index.shape[-1]) * eval('index[%s]' %(','.join(i_idx)))
    
    sum_tsh = sum(t_shape) - t_shape[-1]    
    mask = torch.zeros(t_shape).reshape(-1) 
    
    summand = torch.arange(sum_tsh)*t_shape[-1]
    #summand = eval('summand[:,%s]' %(','.join(['None']*(len(t_shape)-1))))
    
    index = index.reshape(sum_tsh,index.shape[-1]) + summand[:,None]
    
    mask[index.reshape(-1)] = 1
    
    return mask.reshape(t_shape).bool()



