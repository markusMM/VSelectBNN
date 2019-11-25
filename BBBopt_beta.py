# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 05:16:31 2019

@author: Master
"""
#%% -- imports --

import torch
import torch.nn.functional as F
from torch.distributions import Normal, Exponential, Cauchy, HalfNormal, HalfCauchy

###############################################################################
#%% -- sampling algorithm --

# recursive state drawing
def K_recurs(state,K):
    if type(state) == type(list()):
        return [K_recurs(s,K) for s in state]
    else:
        return state[K]

# stack params recursive
def p_recurs(param1,param2):
    if type(param1) == type(list()):
        return [p_recurs(param1[p],param2[p]) for p in range(len(param1))]
    else:
        return torch.cat([param1,param2])

def sample_elbo(
        model, 
        input, target, 
        S = None, K = None, 
        sample_var = 1.0, loglike_method = None, 
        data_noise = 1.0e-07, 
        verbose = False,
    ):
    # we calculate the negative elbo, which will be our loss function
    
    if type(loglike_method) == type(None):
        loglike_method = model.loglike_method
    
    if type(K) == type(None):
        K = model.K
    
    if type(S) == type(None):
        S = max(model.S, model.K_)
    
    # casting them to integer
    # in case we get a floating point from any scheme
    S = int(S)
    K = int(K)
    
    # target for log likelihood
    if      len(target.shape) < 3 and type(model).__name__ == 'MLP_BBB':
        target = target[None,:]
    elif    len(target.shape) < 4:
        target = target[:,None]
    
    w_epsilon = model.variational_states['w_epsilon']
    b_epsilon = model.variational_states['b_epsilon']
    z_epsilon = model.variational_states['z_epsilon']
    
    spill_over = 0
    if hasattr(model, 'full_conv'):
        if model.full_conv:
            spill_over += model.filter_length -1
    if hasattr(model, 'zero_padd'):
        spill_over += model.zero_padd
    
    # new prediction from old K best
    if type(w_epsilon) != type(None):
        
        out_K = model(
                input, 
                w_epsilon , b_epsilon, z_epsilon, 
                sample_var = sample_var, 
                n_sample = model.K_
        )
        if spill_over > 0:
            ret_K = out_K[-spill_over:]
            out_K = out_K[:-spill_over] + torch.randn(*target.shape).to(model.device) * data_noise
        
        # get log probabilities
        log_priors_K   = model.log_prior()
        log_posts_K    = model.log_post()
        # calculate the log likelihood
        if      loglike_method == 'softmax':
            log_likes_K = target*F.log_softmax(out_K, dim=0)
        elif    loglike_method == 'last_weight':
            log_likes_K = model.log_like(out_K, target, model.noise_tol, backtrace=False)
        elif    loglike_method == 'backtrace':
            log_likes_K = model.log_like(out_K, target, model.noise_tol, backtrace=True)
        else:
            log_likes_K = Normal(out_K, model.noise_tol).log_prob(target)
        
        if type(model).__name__ == 'MLP_BBB':
            log_likes_K = (log_likes_K - log_likes_K.max()).mean(dim=[-2,-1]) + log_likes_K.max()
        else:
            log_likes_K = (log_likes_K - log_likes_K.max()).mean(dim=[0,-2,-1]) + log_likes_K.max()
    
    # make predictions 
    # and calculate prior, posterior, and likelihood 
    # for a given number of samples
    out_S = model(
                input, 
                sample_var = sample_var, 
                n_sample = S
    ) + torch.randn(*target.shape).to(model.device) * data_noise
    
    # get new variational samples
    new_ws = [f.w_epsilon for f in model.children()]
    new_bs = [f.b_epsilon for f in model.children()]
    new_zs = [f.z_epsilon for f in model.children()]
    
    # get log probabilities
    log_priors_S   = model.log_prior()
    log_posts_S    = model.log_post()
    
    # calculate the log likelihood
    if      loglike_method == 'softmax':
        log_likes_S = target*F.log_softmax(out_S, dim=0)
    elif    loglike_method == 'last_weight':
        log_likes_S = model.log_like(out_S, target, model.noise_tol, backtrace=False)
    elif    loglike_method == 'backtrace':
        log_likes_S = model.log_like(out_S, target, model.noise_tol, backtrace=True)
    else:
        log_likes_S = Normal(out_S, model.noise_tol).log_prob(target)
    
    if type(model).__name__ == 'MLP_BBB':
        log_likes_S = (log_likes_S - log_likes_S.max()).mean(dim=[-2,-1]) + log_likes_S.max()
    else:
        log_likes_S = (log_likes_S - log_likes_S.max()).mean(dim=[0,-2,-1]) + log_likes_S.max()
        
    
    # new prediction from old K best
    if type(w_epsilon) != type(None):
        
        # add new old stats
        log_posts = torch.cat([log_posts_S, log_posts_K]).squeeze()
        log_priors = torch.cat([log_priors_S[:,None], log_priors_K[:,None]]).squeeze()
        log_likes = torch.cat([log_likes_S, log_likes_K]).squeeze()
        
        # store all variational samples
        new_ws = [p_recurs(new_ws[l], w_epsilon[l]) for l in range(len(w_epsilon))]
        new_bs = [p_recurs(new_bs[l], b_epsilon[l]) for l in range(len(b_epsilon))]
        new_zs = [p_recurs(new_zs[l], z_epsilon[l]) for l in range(len(z_epsilon))]
        out_S = torch.cat([out_S,out_K], dim=0+1*(type(model).__name__ != 'MLP_BBB'))
    else:
        
        log_posts = log_posts_S
        log_priors = log_priors_S
        log_likes = log_likes_S
    
    # removing NaNs
    log_posts[log_posts!=log_posts]     = 0
    log_priors[log_priors!=log_priors]  = 0
    log_likes[log_likes!=log_likes]     = 0
    
    # -- variational selection --
    # sorting for the best posterior w.r.t the current batch
    
    Kopt = K * int(1-model.random_select_ratio)
    Krnd = K - Kopt
    
    idx_k = log_posts.argsort(dim=0)
    if Kopt < 1:
        idx_k_opt = idx_k[:0]
        idx_k_rnd = idx_k[:K]
    else:
        idx_k_opt = idx_k[-Kopt:]
        idx_k_rnd = idx_k[:-Kopt]
        idx_k_rnd = idx_k_rnd[torch.randperm(idx_k_rnd.size(0))][:Krnd]
    
    idx_K = torch.cat([idx_k_opt,idx_k_rnd]).long()
    
    model.log_priors = log_priors[idx_K]
    model.log_posts  = log_posts[idx_K]
    model.log_like  = log_likes[idx_K]
    model.variational_states['w_epsilon'] = [K_recurs(new_ws[l],idx_K) for l in range(len(new_ws))]
    model.variational_states['b_epsilon'] = [K_recurs(new_bs[l],idx_K) for l in range(len(new_bs))]
    model.variational_states['z_epsilon'] = [K_recurs(new_zs[l],idx_K) for l in range(len(new_zs))]
    
    # calculate monte carlo estimate of prior posterior and likelihood
    # the -max / +max part is to avoid memory overflows in the reduce operation
    # it is max, because it is the Evdence Lower Bound
    log_prior   = (model.log_priors - model.log_priors.max()).mean() + model.log_priors.max()
    log_post    = (model.log_posts - model.log_posts.max()).mean() + model.log_posts.max()
    log_like    = (model.log_like - model.log_like.max()).mean() + model.log_like.max()
    
    # calculate the negative elbo (which is our loss function)
    loss = log_post.to(model.device) - log_prior.to(model.device) - log_like.to(model.device)
    
    # calculate the posterior expected activation / output
    if (type(model).__name__ == 'MLP_BBB'):
        exp_act = (out_S[idx_K] * log_posts[idx_K,None,None].to(model.device).exp()).sum(dim=0) 
    else:
        exp_act = (out_S[:,idx_K] * log_posts[idx_K][None,:,None,None].to(model.device).exp()).sum(dim=1)
    # normalization
    exp_act /= log_posts[idx_K].to(model.device).exp().sum() * K
    
    model.K_ = K #idx_K.size(0)
    
    # return loss and output expectation values over the variational samples K
    return  loss, exp_act

#%% ####################################################################### %%#
