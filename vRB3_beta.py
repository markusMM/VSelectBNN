"""
    Bayes By Backprop - variational version
    
This module contains two deep nets (with the possibility to change their activation 
function(s)): A Multi-Layer Perceptron, two (one-sided) Recurrent Neural Network

Here they are named: 
    Elang, Jordan or IIR Recurrent Bayes by Backprop
    Multi-Layer Bayes by Backprop

The classes are:
    MLP_BBB     -   multi-layer Bayes by Backprop
    RNN_BBB     -   recurrent IIR-like layer 
    SRN_BBB     -   Elang / Jordan Bayes by backprop

The variational part:
    
    In terms of variation we do sample similar than in a normal Bayesian Neural 
    Network (BNN).
    But we also sample many parameter sets for each batch at each iteration and 
    select a smaller portion of those samples for actual inference.
    One part of this selection are the Kopt maximum posterior samples and the other 
    part is drawn randomly from the rest.
    
    We do sample randomly because of keeping a part of the variance into our 
    optimization step and thou avoid local optima.

TODO:
    
    - !!! detaching the sampling / training from the particular networks !!!
        -> more dynamic and less redundant code / classes
        -> modularization: "put things together sequentially"
    - evaluate samples with the single datapoint dimension
        -> choosing the best sample for each data point per sample
    - more optimization methods for sampling

@author(s): Markus Meister, ft. Josh Feldman (Feldman 12/17/2018)

@literature:
    
    (Feldman 12/17/2018): 
        Title   = Weight Uncertainty in Neural Networks Tutorial
        Authors = Josh Feldman
        Blog    = Josh Feldman Blog - ml
        Link    = {https://joshfeldman.net/ml/2018/12/17/WeightUncertainty.html}
    
    (Blundell et al 05/21/2015):
        Tilte   = Weight Uncertainty in Neural Networks
        Authors = Charles Blundell CBLUNDELL@GOOGLE.COM, 
                  Julien Cornebise JUCOR@GOOGLE.COM, 
                  Koray Kavukcuoglu KORAYK@GOOGLE.COM, 
                  Daan Wierstra WIERSTRA@GOOGLE.COM
        Journal = stat.ML (in proceeding)
        Link    = {https://arxiv.org/pdf/1505.05424.pdf}
        
    (Laumann 12/12/2018):
        Title   = Bayesian Convolutional Neural Networks with Bayes by Backprop
        Authors = Felix Laumann
        Blog    = Neural Space
        Link    = {https://medium.com/neuralspace/bayesian-convolutional-neural-networks-with-bayes-by-backprop-c84dcaaf086e}
    
    (Lipton et al 23/03/2019):
        Title   = Bayes by Backprop from scratch (NN, classification)
        Authors =   Zachary C. Lipton
                    Mu Li
                    Alex Smola
                    Sheng Zha
                    Aston Zhang
                    Joshua Z. Zhang
                    Eric Junyuan Xie
                    Kamyar Azizzadenesheli
                    Jean Kossaifi
                    Stephan Rabanser
        Blog    = Gluon MXNet Documentation (ft. Read the Docs)
        Link    = {https://gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop.html}
    
@imporvements:
    
    - introduced variational sampling directly into the network
    - select & sample based on K-max. posterior + random samples

@outlook:
    
    - generative model version(s) (where update rules are defined by pretty math)
    - a convolutional filter with any parametric function e.g. ADBUDGE
    - other networks like this e.g. convolutional one (Laumann 12/12/2018)
    
"""
#%% -- imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions \
    import \
        Normal, \
        Poisson, \
        Cauchy, \
        Uniform, \
        HalfCauchy, \
        Exponential, \
        Bernoulli, \
        Binomial
import numpy as np
import warnings
import math
from SignalTorch import ConvT

cpi = torch.Tensor([math.pi])

#%% -- some function(s) --

def roll(x, n):  
    return torch.cat((x[-n:], x[:-n]))

def roll_mat(x,order,strafe=1):
    Xe = torch.zeros(order,*x.shape)
    for m in range(order):
        Xe[m] = roll(x,m*strafe)
    return Xe

# activation function parser
def act_fun(a):
    if type(a) == type('a'):
        try:
            return eval('F.%s' %a)
        except:
            return None
    else:
        return a

def test_act_fun():
    act_funs = [None, 'softmax', 'relu', '+235r+f2fl', F.relu6, None, 'sigmoid', F.softmin]
    length_old = len(act_funs)
    act_funs = list(map(lambda a: act_fun(a), act_funs))
    length_new = len(act_funs)
    assert length_old == length_new
    assert act_funs[0] == None
    assert act_funs[1] == F.softmax
    assert act_funs[length_new-1] == F.softmin
    assert act_funs[2] == F.relu

def parse_layer(layer):
    if type(layer) == type(''):
        try:
            out = eval('LB3_'+layer.split('_')[-1].split('(')[0])
        except Exception as e:
            print('Error parsing layer:')
            print(e)
            print('Using normal distribution!')
            out = LB3_Normal
        return out
    else:
        return layer

def device_check(device):
    # check if cuda is available iff cuda is the device
    if device == 'cuda' and not torch.cuda.is_available():
        warnings.warn('ERROR: CUDA not available! - Using CPU!!', RuntimeWarning)
        device = 'cpu'
    return device

def eval_prop(prop):
    if 'method' in type(prop).__name__.split('_'):
        return prop()
    else:
        return prop

################################
#%% -- standard linear B3 --

class LB3_Normal(nn.Module):
    """
        Layer of our BNN.
        
        implementation: (Feldman 12/17/2018)
        theory and initial work: (Blundell et al 05/21/2015)
    """
    def __init__(self, 
                 input_features, output_features, 
                 prior_var = 1., 
                 prior_pies = .5, 
                 device='cuda', 
                 use_bias=False,
                 dropout_flag = False, 
        ):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 1.
        """
        # checking the device first to see if it is available
        device = device_check(device)
        
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.randn(input_features, output_features))
        self.w_rho = nn.Parameter(torch.rand(input_features, output_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.randn(output_features))
        self.b_rho = nn.Parameter(torch.rand(output_features))
        
        # initialize Bernoulli Dropout probabilities
        if dropout_flag:
            self.z_pies = nn.Parameter(torch.zeros(input_features))
        
        # in case z willbe given in the forward call
        self.dropout_prior = Bernoulli(torch.tensor(prior_pies).to(device))
        
        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None
        self.z = None
        
        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0,prior_var)
        
        # flags for later calculations
        self.use_bias = use_bias
        self.dropout_flag = dropout_flag
        
        # this has to be the same device type, you sent the network to
        self.device = device
        
        self.to(self.device)
    
    def log_like(self, o, y, noise_tol=0.1):
        return Normal(o,noise_tol).log_prob(y)
    
    def forward(
            self, input, 
            w_epsilon = None, b_epsilon = None, z_epsilon = None, 
            sample_var = 1.0, n_sample = 1, z = None, z_pies = None, 
        ):
        """
          Optimization process
          inputs:
              - input           .. a tensor batch of shape (N_batch x D_dim)
              - sample_var      .. variance to noisify the parameters artificially
              - *_epsilon       .. presamples from the 0-mean 1-var normal for reuse
          outputs:
              - a linear feed forward calculation using the sampled weights and biases
          records:
              - *_epsilon       .. samples from the 0-mean 1-var normal for reuse
              - *_log_prior     .. log prior of the respective parameter
              - *_log_post      .. log posterior of the respective parameter
              - log_prior       .. overall sample log prior
              - log_post        .. overall sample log posterior
        """
        
        if type(w_epsilon) != type(None):
            S = w_epsilon.shape[0]
        else:
            S = 0
        
        # input dimensions
        D = self.input_features                 #   input  /  batch  dimension
        H = self.output_features                #   output / hidden  dimension
        S = max(n_sample, S)                    #   number of samples
        
        # getting the standard deviation for annealing variance
        sample_std = torch.sqrt(torch.tensor([sample_var])).to(self.device)
        
        if self.dropout_flag:
            
            if type(z_epsilon) == type(None):
                self.z_epsilon = Uniform(0,1).sample((S, *self.z_pies.shape)).to(self.device)
            else:
                self.z_epsilon = z_epsilon.to(self.device)
            
            # get Bernoulli for input space
            self.z = (self.z_epsilon >= self.z_pies).float()
            
            # get greedy Bernoulli probs
            z_log_prior = self.dropout_prior.log_prob(self.z)
            self.z_post = Bernoulli(self.z_pies).log_prob(self.z)
        
        elif type(z) != type(None):
            
            # use given z
            self.z = z
            
            # get greedy Bernoulli probs
            z_log_prior = self.dropout_prior.log_prob(self.z)
            self.z_post = Bernoulli(self.z_pies).log_prob(self.z)
            
        else:
            
            self.z = torch.ones((S,D)).to(self.device)
            self.z_epsilon = torch.zeros((S,D)).to(self.device)
            self.z_post = torch.zeros((S,D)).to(self.device)
            z_log_prior = torch.zeros((S,D)).to(self.device)
        
        if self.use_bias:
            if type(b_epsilon) == type(None):
                self.b_epsilon = Normal(0,1).sample((S, *self.b_mu.shape)).to(self.device)
            else:
                self.b_epsilon = b_epsilon.to(self.device)
            # bias sample
            self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * sample_std * self.b_epsilon
            # bias prior
            b_log_prior = self.prior.log_prob(self.b)
            # bias posterior
            self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho))).log_prob(self.b)
            
        else:
            
            self.b = torch.zeros((S,H)).to(self.device)
            self.b_epsilon = torch.zeros((S,H)).to(self.device)
            self.b_post = torch.zeros((S,H)).to(self.device)
            b_log_prior = torch.zeros((S,H)).to(self.device)
        
        # sample weights noise if not given
        if type(w_epsilon) == type(None):
            self.w_epsilon = Normal(0,1).sample((S, *self.w_mu.shape)).to(self.device)
        else:
            self.w_epsilon = w_epsilon.to(self.device)
        
        # calculate weights
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * sample_std * self.w_epsilon
        
        # dropout of weights
        # (take only those weights into account whos input is chosen)
        self.w = self.w * self.z[:,:,None]
        
        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w[self.z > 0])
        
        # w posterior
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho))).log_prob(self.w)
        
        # record log prior by evaluating log pdf of prior
        self.log_prior = (
                    w_log_prior.sum(dim = [-1,-2]) / D / H + 
                    b_log_prior.sum(dim = -1) / H + 
                    z_log_prior.sum(dim = -1) / D
                )
        
        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.log_post = (
                    self.w_post.sum(dim = [-1,-2]) / H / D + 
                    self.b_post.sum(dim = -1) / H + 
                    self.z_post.sum(dim = -1) / D
                )
        
        return input @ self.w + self.b[:,None]

#%% -- cauchy version --
class LB3_Cauchy(nn.Module):
    """
        Layer of our BNN.
        
        implementation: (Feldman 12/17/2018)
        theory and initial work: (Blundell et al 05/21/2015)
    """
    def __init__(self, 
                 input_features, output_features, 
                 prior_var=1., 
                 prior_pies = .5, 
                 device='cuda', 
                 use_bias = False, 
                 dropout_flag = False, 
                 ):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 1.
        """
        # checking the device first to see if it is available
        device = device_check(device)
        
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.randn(input_features, output_features))
        self.w_rho = nn.Parameter(torch.rand(input_features, output_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_rho  = nn.Parameter(torch.rand(output_features))    
        self.b_mu   = nn.Parameter(torch.randn(output_features))
        
        # initialize Bernoulli Dropout probabilities
        if dropout_flag:
            self.z_pies = nn.Parameter(torch.rand(input_features))
        
        # in case z willbe given in the forward call
        self.dropout_prior = Bernoulli(torch.tensor(prior_pies).to(device))
        
        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None
        self.z = None
        
        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Cauchy(0,prior_var)
        
        # flags for later calculations
        self.use_bias = use_bias
        self.dropout_flag = dropout_flag
        
        # this has to be the same device type, you sent the network to
        self.device = device
        
        self.to(self.device)
    
    def log_like(self, o, y, noise_tol=0.1):
        return Cauchy(o,noise_tol).log_prob(y)
    
    def forward(
            self, input, 
            w_epsilon = None, b_epsilon = None, z_epsilon = None, 
            sample_var = 1.0, n_sample = 1, z = None, z_pies = None, 
        ):
        """
          Optimization process
          inputs:
              - input           .. a tensor batch of shape (N_batch x D_dim)
              - sample_var      .. variance to noisify the parameters artificially
              - *_epsilon       .. presamples from the 0-mean 1-var normal for reuse
          outputs:
              - a linear feed forward calculation using the sampled weights and biases
          records:
              - *_epsilon       .. samples from the 0-mean 1-var normal for reuse
              - *_log_prior     .. log prior of the respective parameter
              - *_log_post      .. log posterior of the respective parameter
              - log_prior       .. overall sample log prior
              - log_post        .. overall sample log posterior
        """
        
        if type(w_epsilon) != type(None):
            S = w_epsilon.shape[0]
        else:
            S = 0
        
        # input dimensions
        D = self.input_features                 #   input  /  batch  dimension
        H = self.output_features                #   output / hidden  dimension
        S = max(n_sample, S)   #   number of samples
        
        # getting the standard deviation for annealing variance
        sample_std = torch.sqrt(torch.tensor([sample_var])).to(self.device)
        
        
        if self.dropout_flag:
            
            if type(z_epsilon) == type(None):
                self.z_epsilon = Uniform(0,1).sample((S, *self.z_pies.shape)).to(self.device)
            else:
                self.z_epsilon = z_epsilon.to(self.device)
            
            # get Bernoulli for input space
            self.z = (self.z_epsilon >= self.z_pies).to(self.device).float()
            
            # get greedy Bernoulli probs
            z_log_prior = self.dropout_prior.log_prob(self.z)
            self.z_post = Bernoulli(self.z.pies).log_prob(self.z)
        
        elif type(z) != type(None):
            
            # use given z
            self.z = z
            
            # get greedy Bernoulli probs
            z_log_prior = self.dropout_prior.log_prob(self.z)
            self.z_post = Bernoulli(self.z_pies).log_prob(self.z)
            
        else:
            
            self.z = torch.ones((S,D)).to(self.device)
            self.z_epsilon = torch.zeros((S,D)).to(self.device)
            self.z_post = self.z_epsilon
            z_log_prior = torch.zeros((S,D))
        
        if self.use_bias:
            
            if type(b_epsilon) == type(None):
                self.b_epsilon = Uniform(0,1).sample((S, *self.b_mu.shape)).to(self.device)
            else:
                self.b_epsilon = b_epsilon.to(self.device)
            
            # calculate bias with cauchy reparameterization
            self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) *  sample_std * torch.tan(
                         math.pi * ( .5 + self.b_epsilon )
                    )
            
            # bias prior
            b_log_prior = self.prior.log_prob(self.b)
            
            # bias posterior
            self.b_post = Cauchy(self.b_mu.data, torch.log(1+torch.exp(self.b_rho))).log_prob(self.b)
            
        else:
            
            self.b = torch.zeros((S,H)).to(self.device)
            self.b_epsilon = torch.zeros((S,H)).to(self.device)
            self.b_post = torch.zeros((S,H)).to(self.device)
            b_log_prior = torch.zeros((S,H)).to(self.device)
        
        
        if type(w_epsilon) == type(None):
            self.w_epsilon = Uniform(0,1).sample((S, *self.w_mu.shape)).to(self.device)
        else:
            self.w_epsilon = w_epsilon.to(self.device)
        
        
        # calculate weights with cauchy reparameterization
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) *  sample_std * torch.tan(
                     math.pi * ( .5 + self.w_epsilon )
                )
        
        # dropout of weights
        # (take only those weights into account whos input is chosen)
        self.w = self.w * self.z[:,:,None]
        
        # weights prior
        w_log_prior = self.prior.log_prob(self.w[self.z > 0])
        
        # weights posterior
        self.w_post = Cauchy(self.w_mu.data, torch.log(1+torch.exp(self.w_rho))).log_prob(self.w)
        
        
        # record log prior by evaluating log pdf of prior
        self.log_prior = \
            w_log_prior.sum(dim = [-1,-2]) / self.z.sum(dim=-1) / H + \
            b_log_prior.sum(dim = -1) / H + \
            z_log_prior.sum(dim = -1) / D
        
        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.log_post = self.w_post.sum(dim = [-1,-2]) / H / D + \
                        self.b_post.sum(dim = -1) / H + \
                        self.z_post.sum(dim = -1) / D

        return input @ self.w + self.b[:,None]
        
#%%
class LB3_HalfCauchy(nn.Module):
    """
        Layer of our BNN.
        
        implementation: (Feldman 12/17/2018)
        theory and initial work: (Blundell et al 05/21/2015)
    """
    def __init__(self, 
                 input_features, output_features, 
                 prior_var=1., 
                 prior_pies=.5, 
                 device='cuda', 
                 use_bias = False, 
                 dropout_flag = False, 
                 ):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 1.
        """
        # checking the device first to see if it is available
        device = device_check(device)
        
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_rho = nn.Parameter(torch.rand(input_features, output_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_rho = nn.Parameter(torch.rand(output_features))
        
        # initialize Bernoulli Dropout probabilities
        if dropout_flag:
            self.z_pies = nn.Parameter(torch.rand(input_features))
        
        # in case z willbe given in the forward call
        self.dropout_prior = Bernoulli(torch.tensor(prior_pies).to(device))
        
        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None
        self.z = None
        
        # initialize prior distribution for all of the weights and biases
        self.prior = HalfCauchy(prior_var)
        
        # flags for later calculations
        self.use_bias = use_bias
        self.dropout_flag = dropout_flag
        
        # this has to be the same device type, you sent the network to
        self.device = device
        
        self.to(self.device)
    
    def log_like(self, o, y, noise_tol = 0.1 ):
        return HalfCauchy(noise_tol).log_prob(y).mean()
    
    def forward(
            self, input, 
            w_epsilon = None, b_epsilon = None, z_epsilon = None, 
            sample_var = 1.0, n_sample = 1, z = None, z_pies = None, 
        ):
        """
          Optimization process
          inputs:
              - input           .. a tensor batch of shape (N_batch x D_dim)
              - sample_var      .. variance to noisify the parameters artificially
              - *_epsilon       .. presamples from the 0-mean 1-var normal for reuse
          outputs:
              - a linear feed forward calculation using the sampled weights and biases
          records:
              - *_epsilon       .. samples from the 0-mean 1-var normal for reuse
              - *_log_prior     .. log prior of the respective parameter
              - *_log_post      .. log posterior of the respective parameter
              - log_prior       .. overall sample log prior
              - log_post        .. overall sample log posterior
        """
        
        if type(w_epsilon) != type(None):
            S = w_epsilon.shape[0]
        else:
            S = 0
        
        # input dimensions
        D = self.input_features                 #   input  /  batch  dimension
        H = self.output_features                #   output / hidden  dimension
        S = max(n_sample, S)   #   number of samples
        
        # getting the standard deviation for annealing variance
        sample_std = torch.sqrt(torch.tensor([sample_var])).to(self.device)
        
        if self.dropout_flag:
            
            if type(z_epsilon) == type(None):
                self.z_epsilon = Uniform(0,1).sample((S, *self.z_pies.shape)).to(self.device)
            else:
                self.z_epsilon = z_epsilon.to(self.device)
            
            # get Bernoulli for input space
            self.z = (self.z_epsilon >= self.z_pies).to(self.device).float()
            
            # get greedy Bernoulli probs
            z_log_prior = self.dropout_prior.log_prob(self.z)
            self.z_post = Bernoulli(self.z.pies).log_prob(self.z)
        
        elif type(z) != type(None):
            
            # use given z
            self.z = z
            
            # get greedy Bernoulli probs
            z_log_prior = self.dropout_prior.log_prob(self.z)
            self.z_post = Bernoulli(self.z_pies).log_prob(self.z)
            
        else:
            
            self.z = torch.ones((S,D)).to(self.device)
            self.z_epsilon = torch.zeros((S,D)).to(self.device)
            self.z_post = torch.zeros((S,D)).to(self.device)
            z_log_prior = torch.zeros((S,D)).to(self.device)
        
        if self.use_bias():
            
            # sample bias noise if not given
            if type(b_epsilon) == type(None):
                self.b_epsilon = Uniform(0,1).sample((S, *self.b_rho.shape)).to(self.device)
            else:
                self.b_epsilon = b_epsilon.to(self.device)
            
            # calculate bias with cauchy reparameterization
            self.b = ( torch.log(1+torch.exp(self.b_rho)) * sample_std * torch.tan( math.pi* (self.b_epsilon - .5) ) ).abs()
            
            # record log prior by evaluating log pdf of prior at sampled bias
            b_log_prior = self.prior.log_prob(self.b)
            
            # bias log posterior
            self.b_post = HalfCauchy(torch.log(1+torch.exp(self.b_rho)).data).log_prob(self.b)
            
        else:
            
            self.b = torch.zeros((S,*self.b_rho.shape)).to(self.device)
            self.b_epsilon = self.b
            self.b_post = self.b
            b_log_prior = self.b
        
        # sample weights noise if not given
        if type(w_epsilon) == type(None):
            self.w_epsilon = Uniform(0,1).sample((S, *self.w_rho.shape)).to(self.device)
        else:
            self.w_epsilon = w_epsilon.to(self.device)
        
        # calculate weights with cauchy reparameterization
        self.w = ( torch.log(1+torch.exp(self.w_rho)) * sample_std * torch.tan( math.pi* (self.w_epsilon - .5) ) ).abs()
        
        # record log prior by evaluating log pdf of prior at sampled weights
        w_log_prior = self.prior.log_prob(self.w)
        
        # weights log posterior
        self.w_post = HalfCauchy(torch.log(1+torch.exp(self.w_rho)).data).log_prob(self.w)
        
        
        # record log prior by evaluating log pdf of prior
        self.log_prior = \
            w_log_prior.sum(dim = [-1,-2]) / D / H + \
            b_log_prior.sum(dim = -1) / H + \
            z_log_prior.sum(dim = -1) / D
        
        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.log_post = self.w_post.sum(dim = [-1,-2]) / H / D + \
                        self.b_post.sum(dim = -1) / H + \
                        self.z_post.sum(dim = -1) / D

        return input[:,self.z.long()] @ (self.w[self.z.long()]) + self.b[:,None]

#%%
class LB3_Exponential(nn.Module):
    """
        Layer of our BNN.
        
        implementation: (Feldman 12/17/2018)
        theory and initial work: (Blundell et al 05/21/2015)
    """
    def __init__(self, 
                 input_features, output_features, 
                 prior_var=1., 
                 prior_pies=.5, 
                 device='cuda', 
                 use_bias = False, 
                 dropout_flag = False, 
                 ):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 1.
        """
        # checking the device first to see if it is available
        device = device_check(device)
        
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features
        
        # initialize mu and rho parameters for the weights of the layer
        self.w_rho = nn.Parameter(torch.rand(input_features, output_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_rho = nn.Parameter(torch.rand(output_features))
        
        # initialize Bernoulli Dropout probabilities
        if dropout_flag:
            self.z_pies = nn.Parameter(torch.rand(input_features))
        
        # in case z willbe given in the forward call
        self.dropout_prior = Bernoulli(torch.tensor(prior_pies).to(device))
        
        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None
        self.z = None
        
        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Exponential(prior_var)
        
        # flags for later calculations
        self.use_bias = use_bias
        self.dropout_flag = dropout_flag
        
        # this has to be the same device type, you sent the network to
        self.device = device
        
        self.to(self.device)
    
    def log_like(self, o, y, noise_tol = 0.1 ):
        return ( o.clone() * Exponential(noise_tol).log_prob(y) ).sum()
    
    def forward(
            self, input, 
            w_epsilon = None, b_epsilon = None, z_epsilon = None, 
            sample_var = 1.0, n_sample = 1, z = None, z_pies = None, 
        ):
        """
          Optimization process
          inputs:
              - input           .. a tensor batch of shape (N_batch x D_dim)
              - sample_var      .. variance to noisify the parameters artificially
              - *_epsilon       .. presamples from the 0-mean 1-var normal for reuse
          outputs:
              - a linear feed forward calculation using the sampled weights and biases
          records:
              - *_epsilon       .. samples from the 0-mean 1-var normal for reuse
              - *_log_prior     .. log prior of the respective parameter
              - *_log_post      .. log posterior of the respective parameter
              - log_prior       .. overall sample log prior
              - log_post        .. overall sample log posterior
        """
        
        if type(w_epsilon) != type(None):
            S = w_epsilon.shape[0]
        else:
            S = 0
        
        # input dimensions
        D = self.input_features                 #   input  /  batch  dimension
        H = self.output_features                #   output / hidden  dimension
        S = max(n_sample, S)   #   number of samples
        
        # getting the standard deviation for annealing variance
        sample_std = torch.sqrt(torch.tensor([sample_var])).to(self.device)
        
        if self.dropout_flag:
            
            if type(z_epsilon) == type(None):
                self.z_epsilon = Uniform(0,1).sample((S, *self.z_pies.shape)).to(self.device)
            else:
                self.z_epsilon = z_epsilon.to(self.device)
            
            # get Bernoulli for input space
            self.z = (self.z_epsilon <= self.z_pies).to(self.device).float()
            
            # get greedy Bernoulli probs
            z_log_prior = self.dropout_prior.log_prob(self.z)
            self.z_post = Bernoulli(self.z_pies).log_prob(self.z)
        
        elif type(z) != type(None):
            
            # use given z
            self.z = z
            
            # get greedy Bernoulli probs
            z_log_prior = self.dropout_prior.log_prob(self.z)
            self.z_post = Bernoulli(self.z_pies).log_prob(self.z)
            
        else:
            
            self.z = torch.ones((S,D)).to(self.device)
            self.z_epsilon = torch.zeros((S,D)).to(self.device)
            self.z_post = torch.zeros((S,D)).to(self.device)
            z_log_prior = torch.zeros((S,D)).to(self.device)
        
        if self.use_bias:
            
            # sample bias noise if not given
            if type(b_epsilon) == type(None):
                self.b_epsilon = Uniform(0,1).sample((S, *self.b_rho.shape)).to(self.device)
            else:
                self.b_epsilon = b_epsilon.to(self.device)
            
            
            # calculate bias with the exponential of a unit distribution
            # if x is negative, we don't calculate and probability, what so ever
            self.b = torch.log(1+torch.exp(self.b_rho)) * sample_std * \
                torch.exp( - torch.log(1+torch.exp(self.b_rho)) * \
                self.b_epsilon ) * (self.b_epsilon >= 0).float()
            
            # bias log prior
            b_log_prior = self.prior.log_prob(self.b)
            
            # bias log posterior
            self.b_post = Exponential(torch.log(1+torch.exp(self.b_rho))).log_prob(self.b)
            
        else:
            
            self.b = torch.zeros((S,*self.b_rho.shape)).to(self.device)
            self.b_epsilon = self.b
            self.b_post = self.b
            b_log_prior = self.b
        
        # sample weights noise if not given
        if type(w_epsilon) == type(None):
            self.w_epsilon = Uniform(0,1).sample((S, *self.w_rho.shape)).to(self.device)
        else:
            self.w_epsilon = w_epsilon.to(self.device)
        
        # calculate weights with the exponential of a unit distribution
        # if x is negative, we don't calculate and probability, what so ever
        self.w = torch.log(1+torch.exp(self.w_rho)) * sample_std * \
            torch.exp( - torch.log(1+torch.exp(self.w_rho)) * sample_std * \
            self.w_epsilon ) * (self.w_epsilon >= 0).float()
        
        # dropout of weights
        # (take only those weights into account whos input is chosen)
        self.w = self.w * self.z[:,:,None]
        
        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w[self.z > 0])
        
        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Exponential(torch.log(1+torch.exp(self.w_rho))).log_prob(self.w)
        
        
        # record log prior by evaluating log pdf of prior
        self.log_prior = \
            w_log_prior.sum(dim = [-1,-2]) / self.z.sum(dim=-1) / H + \
            b_log_prior.sum(dim = -1) / H + \
            z_log_prior.sum(dim = -1) / D
        
        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.log_post = self.w_post.sum(dim = [-1,-2]) / H / D + \
                        self.b_post.sum(dim = -1) / H + \
                        self.z_post.sum(dim = -1) / D

        return input @ self.w + self.b[:,None]

###############################################################################
#%% -- standard in-place convolutions and other filters --

class Conv1D_BBB(nn.Module):
    """
        Layer of our BNN.
        
        implementation: Nils-Markus Meister
        theory and initial work: (Blundell et al 05/21/2015)
    """
    def __init__(self, 
                 input_features, filter_length, 
                 prior_var = 1., 
                 prior_pies = .5, 
                 device='cuda', 
                 dropout_flag = False, 
                 full_conv = False, 
                 zero_padd = 0, 
                 activation_function = None, 
        ):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 1.
        """
        # checking the device first to see if it is available
        device = device_check(device)
        
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.filter_length = filter_length
        self.zero_padd = zero_padd
        self.full_conv = full_conv

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.randn(input_features, filter_length))
        self.w_rho = nn.Parameter(torch.rand(input_features, filter_length))
        
        # initialize Bernoulli Dropout probabilities
        if dropout_flag:
            self.z_pies = nn.Parameter(torch.rand(input_features))
        
        # in case z willbe given in the forward call
        self.dropout_prior = Bernoulli(torch.tensor(prior_pies).to(device))
        
        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.z = None
        
        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0,prior_var)
        
        # flags for later calculations
        self.dropout_flag = dropout_flag
        
        # activation function
        self.activation_function = act_fun(activation_function)
        
        # this has to be the same device type, you sent the network to
        self.device = device
        
        self.to(self.device)
    
    def log_like(self, o, y, noise_tol=0.1):
        return Normal(o,noise_tol).log_prob(y)
    
    def forward(
            self, input, 
            w_epsilon = None, b_epsilon = None, z_epsilon = None, 
            sample_var = 1.0, n_sample = 1, z = None, z_pies = None, 
        ):
        """
          Optimization process
          inputs:
              - input           .. a tensor batch of shape (N_batch x D_dim)
              - sample_var      .. variance to noisify the parameters artificially
              - *_epsilon       .. presamples from the 0-mean 1-var normal for reuse
          outputs:
              - a linear feed forward calculation using the sampled weights and biases
          records:
              - *_epsilon       .. samples from the 0-mean 1-var normal for reuse
              - *_log_prior     .. log prior of the respective parameter
              - *_log_post      .. log posterior of the respective parameter
              - log_prior       .. overall sample log prior
              - log_post        .. overall sample log posterior
        """
        
        if type(w_epsilon) != type(None):
            S = w_epsilon.shape[0]
        else:
            S = 0
        
        # input dimensions
        D = self.input_features                 #   input  /  batch  dimension
        H = self.filter_length                  #   filter / hidden  dimension
        S = max(n_sample, S)                    #   number of samples
        
        # getting the standard deviation for annealing variance
        sample_std = torch.sqrt(torch.tensor([sample_var])).to(self.device)
        
        if self.dropout_flag:
            
            if type(z_epsilon) == type(None):
                self.z_epsilon = Uniform(0,1).sample((S, *self.z_pies.shape)).to(self.device)
            else:
                self.z_epsilon = z_epsilon.to(self.device)
            
            # get Bernoulli for input space
            self.z = (self.z_epsilon >= self.z_pies).float()
            
            # get greedy Bernoulli probs
            z_log_prior = self.dropout_prior.log_prob(self.z)
            self.z_post = Bernoulli(self.z_pies).log_prob(self.z)
        
        elif type(z) != type(None):
            
            # use given z
            self.z = z
            
            # get greedy Bernoulli probs
            z_log_prior = self.dropout_prior.log_prob(self.z)
            self.z_post = Bernoulli(self.z_pies).log_prob(self.z)
            
        else:
            
            self.z = torch.ones((S,D)).to(self.device)
            self.z_epsilon = torch.zeros((S,D)).to(self.device)
            self.z_post = torch.zeros((S,D)).to(self.device)
            z_log_prior = torch.zeros((S,D)).to(self.device)
        
        # sample weights noise if not given
        if type(w_epsilon) == type(None):
            self.w_epsilon = Normal(0,1).sample((S, *self.w_mu.shape)).to(self.device)
        else:
            self.w_epsilon = w_epsilon.to(self.device)
        
        # calculate weights
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * sample_std * self.w_epsilon
        
        # dropout of weights
        # (take only those weights into account whos input is chosen)
        self.w = self.w * self.z[:,:,None]
        
        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w[self.z > 0])
        
        # w posterior
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho))).log_prob(self.w)
        
        # record log prior by evaluating log pdf of prior
        self.log_prior = (
                    w_log_prior.sum(dim = [-1,-2]) / D / H + 
                    z_log_prior.sum(dim = -1) / D
                )
        
        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.log_post = (
                    self.w_post.sum(dim = [-1,-2]) / H / D + 
                    self.z_post.sum(dim = -1) / D
                )
        
        if self.full_conv:
            input = torch.cat([
                        input,
                        torch.zeros(H-1,*input.shape[1:]).to(self.device)
                    ])
        if self.zero_padd > 0:
            pad = int(self.zero_padd)
            input = torch.cat([
                        input,
                        torch.zeros(pad,*input.shape[1:]).to(self.device)
                    ])
        
        # convolution
        x = ConvT(input, self.w)
    
        if type(self.activation_function) != type(None):
            x = self.activation_function(x)
        
        
        self.b_epsilon = torch.zeros((S,x.shape[-1]))
        
        return x
    
###############################################################################
#%% -- Drop Out Layers --

class Bernoulli_DB3(nn.Module):
    
    def __init__(
                self,
                n_units, pi_prior = .3, 
                gamma = 4, device='cuda'
            ):
        
        # checking the device first to see if it is available
        device = device_check(device)
        
        # initialize the network like you would with a standard RNN, but using the 
        # BBB layer
        super().__init__()
        
        self.n_units = n_units
        self.gamma = gamma
        
        # prior
        self.prior = torch.distributions.Bernoulli(torch.tensor(pi_prior).to(device))
        
        # coin flips
        self.pies = nn.Parameter(0.45*torch.rand(n_units)).to(device)
        
        self.device = device
        self.to(device)
        
    def forward(
                self, 
                input, 
                w_epsilon = None,
                b_epsilon = None, 
                sample_var = 1.0, 
                n_sample = 1, 
                z = None
            ):
        """
          Optimization process
          inputs:
              - input           .. a tensor batch of shape (N_batch x D_dim)
              - sample_var      .. variance to noisify the parameters artificially
              - *_epsilon       .. presamples from the 0-mean 1-var normal for reuse
          outputs:
              - a linear feed forward calculation using the sampled weights and biases
          records:
              - *_epsilon       .. samples from the 0-mean 1-var normal for reuse
              - *_log_prior     .. log prior of the respective parameter
              - *_log_post      .. log posterior of the respective parameter
              - log_prior       .. overall sample log prior
              - log_post        .. overall sample log posterior
        """
        
        # input dimensions
        H = self.n_units
        #gamma = self.gamma
        S = n_sample
        
        # getting the standard deviation for annealing variance
        sample_std = torch.sqrt(torch.tensor([sample_var])).to(self.device)
        
        # sample weights noise if not given
        if type(w_epsilon) == type(None):
            self.w_epsilon = Uniform(0,1).sample((S, *self.pies.shape)).to(self.device)
        else:
            self.w_epsilon = w_epsilon.to(self.device)
        
        # sample weights noise if not given
        if type(b_epsilon) == type(None):
            self.b_epsilon = None
        else:
            self.b_epsilon = None
        
        # calculate weights with the Bernoulli of a unit distribution
        self.w = ((self.w_epsilon <= sample_std*self.pies).float()).to(self.device)
        
        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        
        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Bernoulli(self.pies).log_prob(self.w)
        
        
        # entire log prior
        self.log_prior = \
            torch.sum(w_log_prior, dim = [-1]) / H
        
        # entire log posterior
        self.log_post = \
            self.w_post.sum(dim = [-1]) / H

        return self.w
        
#%% -- time filter layer --
class UB3(nn.Module):
    
    def __init__(
            self,
            input_size,
            output_size,
            filter_length,
            prior_var = 1.0,
            layer_type = 'Normal',
            device = 'cuda',
            reductor = torch.sum, 
            padding = False, 
            use_bias = False, 
            bd3_flags = [], 
        ):
        
        super().__init__()
        
        # pase Bernoulli dropout flag 
        # (if we want to use it inside a specific layer or not)
        if len(bd3_flags) < filter_length:
            bd3_flags += [False] * abs(len(bd3_flags) - filter_length)
        self.bd3_flags = bd3_flags
        
        # layer dimensions
        self.L = filter_length
        self.D = input_size
        self.H = output_size
        
        for l in range(self.L):
            self.add_module(
                    'h%04i' %(l), 
                    parse_layer(layer_type)(
                            input_size, output_size, 
                            prior_var = prior_var, 
                            use_bias = use_bias, 
                            device = device, 
                            dropout_flag = bd3_flags[l], 
                    )
            )
        
        self.padding = padding      # if we want to add zeros at the beginning (only for missing past values) or not
        self.reductor = reductor    # a method with which we reduce our layer responses
        self.device = device
        self.to(device)
    
    def forward(
            self, x, 
            w_epsilon = None, b_epsilon = None, z_epsilon = None,  
            sample_var = 1.0, n_sample = 1, keep_eps = False
        ):
        
        L = self.L
        H = self.H
        D = self.D
        N = x.shape[-2]
        T = x.shape[0]
        S = n_sample
        
        
        # method type epsilon parsing
        if 'method' in type(w_epsilon).__name__.split('_'):
            w_epsilon = w_epsilon()
        if 'method' in type(b_epsilon).__name__.split('_'):
            b_epsilon = b_epsilon()
        if 'method' in type(z_epsilon).__name__.split('_'):
            z_epsilon = z_epsilon()
        
        # keeps previous epsilons
        if hasattr(self,'w_epsilon') and keep_eps:
            w_epsilon = self.w_epsilon
        if hasattr(self,'b_epsilon') and keep_eps:
            b_epsilon = self.b_epsilon
        if hasattr(self,'z_epsilon') and keep_eps:
            z_epsilon = self.z_epsilon
        
        # none type epsilon parsing
        if type(w_epsilon) == type(None):
            w_epsilon = [None] * L
        if type(b_epsilon) == type(None):
            b_epsilon = [None] * L
        if type(z_epsilon) == type(None):
            z_epsilon = [None] * L
        
        # malloc
        y = torch.zeros((T,S,N,H)).to(self.device)
        z = torch.zeros((S,N,D)).to(self.device)
        
        # "new" sampled epsilons
        new_w_eps = []
        new_b_eps = []
        new_z_eps = []
        
        # for each layer as long as we have data backwards in time
        for l, layer in enumerate(self.children()):
            if l >= T:
                if self.padding:
                    y[T-l-1] =  layer(
                            z, w_epsilon[l], b_epsilon[l], z_epsilon[l], 
                            sample_var, S
                        )
                    new_w_eps.append(layer.w_epsilon)
                    new_b_eps.append(layer.b_epsilon)
                    new_z_eps.append(layer.z_epsilon)
            else:
                y[T-l-1] = layer(
                        x[T-l-1], 
                        w_epsilon[l], b_epsilon[l], z_epsilon[l], 
                        sample_var, S
                    )
                new_w_eps.append(layer.w_epsilon)
                new_b_eps.append(layer.b_epsilon)
                new_z_eps.append(layer.z_epsilon)
        
        # fill incalculated epsilons with None
        if len(new_w_eps) < L:
            new_w_eps += [None]*(L-len(new_w_eps))
        if len(new_b_eps) < L:
            new_b_eps += [None]*(L-len(new_b_eps))
        if len(new_z_eps) < L:
            new_z_eps += [None]*(L-len(new_z_eps))
        
        # store epsilons
        self.w_epsilon = new_w_eps
        self.b_epsilon = new_b_eps
        self.z_epsilon = new_z_eps
        
        # response reduction
        # if the reduction function is valid it will be used
        # WARNING: This function is better differntiable!
        #          e.g. sum, gamma*(sum(x))^(1/gamma), etc..
        try:
            y = self.reductor(y, dim=[0])
        except:
            y = y
        
        return y
        
    def log_prior(self):
        return torch.mean( 
                torch.stack( 
                        list( eval_prop( layer.log_prior ) for layer in self.children() )
                ), dim = [0] 
        ).to(self.device)

    def log_post(self):
        return torch.mean( 
                torch.stack( 
                        list( eval_prop( layer.log_post  ) for layer in self.children() )
                ), dim = [0] 
        ).to(self.device)
    
    def log_like(self, o, y, noise_tol = 0.1 ):
        return torch.mean( torch.stack(list( layer.log_like( o, y, noise_tol ) for layer in self.children() )) ).to(self.device)

#%%
class C1B3(nn.Module):
    
    def __init__(
            self,
            input_size,
            output_size,
            filter_length,
            prior_var = 1.0,
            layer_type = 'Normal',
            device = 'cuda',
            padding = False, 
            use_bias = False, 
            bd3_flags = [], 
            individual_dropout = False
        ):
        
        super().__init__()
        
        if type(bd3_flags) != type([]):
            try:
                bd3_flags = bd3_flags.tolist()
            except:
                bd3_flags = [bd3_flags] * filter_length
        
        # pase Bernoulli dropout flag 
        # (if we want to use it inside a specific layer or not)
        if len(bd3_flags) < filter_length:
            bd3_flags += [bd3_flags] * abs(len(bd3_flags) - filter_length)
        self.bd3_flags = bd3_flags
        
        # flag if we want to take dropouts individually for each convolutional filter.
        self.individual_dropout = individual_dropout
        if not self.individual_dropout:
            self.bd3_flags = [False] * filter_length
            self.z_pies = nn.Parameter(torch.rand((input_size)))
        
        # layer dimensions
        self.L = filter_length
        self.D = input_size
        self.H = output_size
        
        for l in range(self.L):
            self.add_module(
                    'h%04i' %(l), 
                    parse_layer(layer_type)(
                            input_size, output_size, 
                            prior_var = prior_var, 
                            use_bias = use_bias, 
                            device = device, 
                            dropout_flag = bd3_flags[l], 
                    )
            )
        
        
        self.variational_states = {
                'w_epsilon'  : None,      # will be [SxLxHxD]
                'b_epsilon'  : None,      # will be [SxLxH]
                'z_epsilon'  : None,      # ... for Bernoulli states
            }
        
        self.padding = padding      # if we want to add zeros at the beginning (only for missing past values) or not
        self.device = device
        self.to(device)
    
    def forward(
            self, x, 
            w_epsilon = None, b_epsilon = None, z_epsilon = None, 
            sample_var = 1.0, n_sample = 1, keep_eps = False
        ):
        
        L = self.L
        H = self.H
        D = self.D
        N = x.shape[-2]
        T = x.shape[0]
        S = n_sample
        
        
        # method type epsilon parsing
        if 'method' in type(w_epsilon).__name__.split('_'):
            w_epsilon = w_epsilon()
        if 'method' in type(b_epsilon).__name__.split('_'):
            b_epsilon = b_epsilon()
        if 'method' in type(z_epsilon).__name__.split('_'):
            z_epsilon = z_epsilon()
        
        # keeps previous epsilons
        if hasattr(self,'w_epsilon') and keep_eps:
            w_epsilon = self.w_epsilon
        if hasattr(self,'b_epsilon') and keep_eps:
            b_epsilon = self.b_epsilon
        if hasattr(self,'z_epsilon') and keep_eps:
            z_epsilon = self.z_epsilon
        
        # none type epsilon parsing
        if type(w_epsilon) == type(None):
            w_epsilon = [None] * L
        if type(b_epsilon) == type(None):
            b_epsilon = [None] * L
        if type(z_epsilon) == type(None):
            if not self.individual_dropout and self.bd3_flags[0]:
                z_epsilon = [Uniform(0,1).sample((S,self.D))] * L
                self.z = (z_epsilon >= self.z_pies).float()
            else:
                z_epsilon = [None] * L
                self.z = None
        
        # malloc
        y = torch.zeros((T,S,N,H)).to(self.device)
        z = torch.zeros((S,N,D)).to(self.device)
        
        # "new" sampled epsilons
        new_w_eps = []
        new_b_eps = []
        new_z_eps = []
        
        # for each layer as long as we have data backwards in time
        for t in range(T):
            o = torch.zeros(y.shape[1:]).to(self.device)
            for l, layer in enumerate(self.children()):
                if l >= T:
                    if self.padding:
                        o +=  layer(
                                z, w_epsilon[l], b_epsilon[l], z_epsilon[l], 
                                sample_var, S, z = self.z
                            )
                        new_w_eps.append(layer.w_epsilon)
                        new_b_eps.append(layer.b_epsilon)
                        new_z_eps.append(layer.z_epsilon)
                else:
                    o += layer(
                            x[T-t-l-1], w_epsilon[l], b_epsilon[l], z_epsilon[l], 
                            sample_var, S, z = self.z
                        )
                    new_w_eps.append(layer.w_epsilon)
                    new_b_eps.append(layer.b_epsilon)
                    new_z_eps.append(layer.z_epsilon)
            y[T-t-1] = o
        
        # fill incalculated epsilons with None
        if len(new_w_eps) < L:
            new_w_eps += [None]*(L-len(new_w_eps))
        if len(new_b_eps) < L:
            new_b_eps += [None]*(L-len(new_b_eps))
        if len(new_z_eps) < L:
            new_z_eps += [None]*(L-len(new_z_eps))
        
        # store epsilons
        self.w_epsilon = new_w_eps
        self.b_epsilon = new_b_eps
        self.z_epsilon = new_z_eps
        
        return y
        
    def log_prior(self):
        return torch.mean( 
                torch.stack( 
                        list( eval_prop( layer.log_prior ) for layer in self.children() )
                ), dim = [0] 
        ).to(self.device)

    def log_post(self):
        return torch.mean( 
                torch.stack( 
                        list( eval_prop( layer.log_post  ) for layer in self.children() )
                ), dim = [0] 
        ).to(self.device)
    
    def log_like(self, o, y, noise_tol = 0.1 ):
        return torch.mean( torch.stack(list( layer.log_like( o, y, noise_tol ) for layer in self.children() )) ).to(self.device)

#####################
#%% -- MLP BBB --
class MLP_BBB(nn.Module):
    '''
        Multi-Layer Perceptron Bayes by Backprop
        a selective samplning approach
        
        This is a generic version of the variatonal Bayes by Backprop
        (Blundell et al 05/21/2015).
        
        The generic part is:
            - how many hidden layers with which dimensions do we have
            - which activation function we use for each layer (act_fun)
            - which of our samples do we want to keep / use for inference
              (based on K-max posterior + a random selection)
        
        The selective sampling method comes from the origin of sampling in the 
        probabilistic domain and a method called select and sample, where an
        activation fuction, for generative models, is used to infer the most
        prominent priors before hand, to avoid a hard cut by conventional
        Maximum A Posterior (MAP) while still have a skalable amount of samples
        left for computation, here: the Backprop algortithm.
        
        The loss for this algorithm is provided by sampling the negative 
        Evidence Lower Bound Objective (ELBO) from our probabilistic weights
        
        act_funs does always count for the number of layers in our network.
        hidden_dim is an array or list of integers containing only the numbers
        of neurons passed to obtain the hidden transformations.
        
    '''
    def __init__(
                self, 
                in_dim, out_dim, hidden_dim = [], act_funs = [],
                n_samples_full = 1000, n_samples_select = 400,
                random_select_ratio = .4, prior_var = 1.0, 
                noise_tol=0.1,  layers = ['Normal'], 
                device = 'cuda', 
                loglike_method = 'softmax', 
                use_bias = False, 
                bd3_flags = [], 
            ):
        # checking the device first to see if it is available
        device = device_check(device)
        
        # initialize layers
        super().__init__()
        
        # pase Bernoulli dropout flag 
        # (if we want to use it inside a specific layer or not)
        if len(bd3_flags) < len(hidden_dim)+1:
            bd3_flags += [False] * abs(len(bd3_flags) - len(layers))
        self.bd3_flags = bd3_flags
        
        self.D = in_dim
        self.H = out_dim
        self.hidden_dim = hidden_dim
        self.all_dim = [in_dim]+hidden_dim+[out_dim]
        self.S = n_samples_full
        self.K = n_samples_select
        self.L = len(hidden_dim) + 1 + np.sum(bd3_flags)
        self.random_select_ratio = random_select_ratio
        self.noise_tol=noise_tol
        self.device = device
        self.loglike_method = loglike_method
        
        # parsing activation functions
        if len(act_funs) < 1:
            act_funs = [None]
        act_funs = list(map(lambda a: act_fun(a), act_funs))
        self.act_funs = act_funs
        
        self.layer_sizes = []
        
        # parsing layers into a list if not given
        if type(layers) != type([]):
            layers = [layers]
        
        # creating linear BBB layers
        for i, n_units in enumerate(self.all_dim[:-1]):
            
            # getting either the ith layer or the first one
            try:
                layer = layers[i]
            except:
                layer = layers[0]
            
            # getting dropout flag
            try:
                do_flg = bd3_flags[i]
            except:
                do_flg = False
            
            # parse and add the BbB layer
            self.add_module(
                    'h%04i' %((i)), 
                    parse_layer(layer)(
                            int(self.all_dim[i]), int(self.all_dim[i+1]), 
                            prior_var = prior_var, 
                            device = device, 
                            use_bias = use_bias, 
                            dropout_flag = do_flg, 
                        )
                )
        
        self.variational_states = {
                'w_epsilon'  : None,      # will be [SxLxHxD]
                'b_epsilon'  : None,      # will be [SxLxH]
                'z_epsilon'  : None,      # ... for Bernoulli states
            }
        
        self.to(self.device)
        
    def forward(
            self, x, 
            w_epsilon = None, b_epsilon = None, z_epsilon = None, 
            sample_var = 1.0, n_sample = 1
        ):
        # again, this is equivalent to a standard MLP
        
        # getting activation functions
        act_funs = self.act_funs
        
        L = self.L
        
        
        # method type epsilon parsing
        if 'method' in type(w_epsilon).__name__.split('_'):
            w_epsilon = w_epsilon()
        if 'method' in type(b_epsilon).__name__.split('_'):
            b_epsilon = b_epsilon()
        if 'method' in type(z_epsilon).__name__.split('_'):
            z_epsilon = z_epsilon()
        
        # none type epsilon parsing
        if type(w_epsilon) == type(None):
            w_epsilon = [None] * L
        if type(b_epsilon) == type(None):
            b_epsilon = [None] * L
        if type(z_epsilon) == type(None):
            z_epsilon = [None] * L
            
        # recurrent forward iterations 
        for l,layer in enumerate(self.children()):
            
            x = layer(x, w_epsilon[l], b_epsilon[l], z_epsilon[l], sample_var, n_sample)
            
            if len(act_funs) > l:
                if type(act_funs[l]) != type(None):
                    x = act_funs[l](x)
        
        # store epsilons
        self.w_epsilon = [l.w_epsilon for l in self.children()]
        self.b_epsilon = [l.b_epsilon for l in self.children()]
        self.z_epsilon = [l.z_epsilon for l in self.children()]
        
        return x
    
    def log_prior(self):
        return torch.mean( 
                torch.stack(
                        list( eval_prop( layer.log_prior ) for layer in self.children() )
                ), dim=[0] 
        ).to(self.device)
    
    def log_post(self):
        return torch.mean( 
                torch.stack( 
                        list( eval_prop( layer.log_post  ) for layer in self.children() )
                ), dim=[0]
        ).to(self.device)
    
    def log_like(self, o, y, noise_tol = 0.1 ):
        return list( layer.log_like( o, y, noise_tol ) for layer in self.children() )[-1].to(self.device)

#%% -- recurrent Bayes by Backprop --
class RNN_BBB(nn.Module):
    """
        Recurrent IIR select and sample for Bayes by Backprop
    
    We here use multiple linear BbB layers in a consecutive manner such that 
    they gain information from the past layers. This method is called Recurrent 
    Neural Network, because it recurrently uses the results of the past 
    (L-1) layers, with L beeing the window / filter length of this network.
    
    input:
        input* / outpunt* *_size    -   input and output dimension
        window_length               -   length of the filter dimension 
            (This is not necesarry the the input data's window length!)
        samples_full                -   the full no. samples for each epoch
                                        and each batch (It is not implemented 
                                        per data point jet!)
        
    
    """
    def __init__(self, 
                 input_size, 
                 output_size, 
                 window_length, 
                 samples_full = 2000, 
                 samples_select = 1200, 
                 noise_tol=0.1,  
                 prior_var=1.0, 
                 random_select_ratio = .4, 
                 layer_type = ['Normal'], 
                 activation_function = 'tanh', 
                 loglike_method = 'softmax', 
                 device='cuda', 
                 h_sizes = [], 
                 h_activations = [], 
                 use_bias = False,
                 bd3_flags = [], 
        ):
        # checking the device first to see if it is available
        device = device_check(device)
        
        # initialize the network like you would with a standard RNN, but using the 
        # BBB layer
        super().__init__()
        
        # some hyper parameters
        self.D = input_size         # input dimension
        self.H = output_size        # output dimension
        self.L = window_length      # filter length
        self.S = samples_full       # sample size
        self.K = samples_select     # reduced samples size
        self.random_select_ratio = random_select_ratio
        
        # this has to be the same type the network will sent to
        self.device = device
        
        # deep hidden layers
        for l in range(window_length):
            self.add_module(
                    'hl%04i' %l,
                    MLP_BBB(
                            input_size, output_size, h_sizes, 
                            h_activations,
                            n_samples_full = samples_full, 
                            n_samples_select = samples_select,
                            random_select_ratio = .4, prior_var = 1.0, 
                            noise_tol=0.1,  layers = layer_type, 
                            device = device, use_bias = use_bias,
                            bd3_flags = bd3_flags,
                    )
            )
        
        # activation function
        self.activation_function = act_fun(activation_function)
        
        self.loglike_method = loglike_method
        self.noise_tol = noise_tol # we will use the noise tolerance to calculate our likelihood
        self.variational_states = {
                'w_epsilon'  : None,      # will be [SxLxHxD]
                'b_epsilon'  : None,      # will be [SxLxH]
                'z_epsilon'  : None,      # ... for Bernoulli states
                }
        
        self.to(self.device)
    
    def forward(
            self, x, 
            w_epsilon = None, b_epsilon = None, z_epsilon = None, 
            sample_var = 1.0, n_sample = 1, keep_eps = False
        ):
        # again, this is equivalent to a standard recurrent net
        L = self.L
        H = self.H
        T = x.size(0)
        N = x.size(-2)
        S = n_sample
        
        # loading in the activation function
        activation = self.activation_function
        
        # preallocation of the output tensor
        y = torch.zeros(T,S,N,H).to(self.device)
        
        # method type epsilon parsing
        if 'method' in type(w_epsilon).__name__.split('_'):
            w_epsilon = w_epsilon()
        if 'method' in type(b_epsilon).__name__.split('_'):
            b_epsilon = b_epsilon()
        if 'method' in type(z_epsilon).__name__.split('_'):
            z_epsilon = z_epsilon()
        
        # keeps previous epsilons
        if hasattr(self,'w_epsilon') and keep_eps:
            w_epsilon = self.w_epsilon
        if hasattr(self,'b_epsilon') and keep_eps:
            b_epsilon = self.b_epsilon
        if hasattr(self,'z_epsilon') and keep_eps:
            z_epsilon = self.z_epsilon
        
        # none type epsilon parsing
        if type(w_epsilon) == type(None):
            w_epsilon = [None] * L
        if type(b_epsilon) == type(None):
            b_epsilon = [None] * L
        if type(z_epsilon) == type(None):
            z_epsilon = [None] * L
        
        # recurrent forward iterations
        # repeat filter inside input batch window until end
        # repeat window inside filter until end
        t = 0
        while t < T-1:
            for l,layer in enumerate(self.children()):
                
                # resetting time
                if t >= T:
                    t = 0
                
                if l > 0:
                    # response + cumulative previous responses
                    o = y[t] + layer(
                            x[t], w_epsilon[l], b_epsilon[l], z_epsilon[l], 
                            sample_var, n_sample
                        ) + y[t-1]
                else:
                    # layer response
                    o = y[t] + layer(
                            x[t], w_epsilon[l], b_epsilon[l], z_epsilon[l], 
                            sample_var, n_sample
                        )
                
                y[t] = o
                
                t += 1
                    
#        # depricated stacking version
#        y = torch.stack([f(x[l]) for l,f in enumerate(regressor.children())])
#        for j in range(1,y.shape[0]):
#            y[j] += y[j-1]
        
        # activation function
        if type(activation) != type(None):
            y = activation(y)
        
        # store epsilons
        self.w_epsilon = [l.w_epsilon for l in self.children()]
        self.b_epsilon = [l.b_epsilon for l in self.children()]
        self.z_epsilon = [l.z_epsilon for l in self.children()]
        
        return y

    def log_prior(self):
        return torch.mean( 
                torch.stack( 
                        list( eval_prop( layer.log_prior ) for layer in self.children() )
                ), dim = [0] 
        ).to(self.device)

    def log_post(self):
        return torch.mean( 
                torch.stack( 
                        list( eval_prop( layer.log_post  ) for layer in self.children() )
                ), dim = [0] 
        ).to(self.device)
    
    def log_like(self, o, y, noise_tol = 0.1 ):
        return torch.mean( torch.stack(list( layer.log_like( o, y, noise_tol ) for layer in self.children() )) ).to(self.device)

#####################################
#%% -- recurrent Bayes by Backprop --
class SRN_BBB(nn.Module):
    """
        Recurrent select and sample for Bayes by Backprop
    
    We here use multiple linear BbB layers in a consecutive manner such that 
    they gain information from the past layers. This method is called Recurrent 
    Neural Network, because it recurrently uses the results of the past 
    (L-1) layers, with L beeing the window / filter length of this network.
    
    input:
        input* / outpunt* *_size    -   input and output dimension
        window_length               -   length of the filter dimension 
            (This is not necesarry the the input data's window length!)
        samples_full                -   the full no. samples for each epoch
                                        and each batch (It is not implemented 
                                        per data point jet!)
    
    """
    def __init__(self, 
                 input_size, 
                 output_size, 
                 filter_length, 
                 h_sizes = [12], 
                 samples_full = 2000, 
                 samples_select = 1200, 
                 noise_tol=0.1,  
                 prior_var=1.0, 
                 random_select_ratio = .4, 
                 layer_types = ['Normal','Exponential'], 
                 activation_functions = ['relu', None], 
                 device='cuda', 
                 rec_type = 'elang', 
                 u_layer = 'Normal', 
                 u_activation = 'relu', 
                 loglike_method = 'softmax', 
                 use_bias = False, 
                 vrb3_flag = False, 
                 conv_flag = False, 
                 ih_bd3_flags = [], 
                 hh_bd3_flag = False, 
                 hy_bd3_flag = False, 
        ):
        
        # checking the device first to see if it is available
        device = device_check(device)
        
        # initialize the network like you would with a standard RNN, but using the 
        # BBB layer
        super().__init__()
        
        # some hyper parameters
        self.D = input_size         # input dimension
        self.H = h_sizes[-1]        # hidden layer sizes
        self.G = output_size        # output dimension
        self.L = filter_length      # filter length
        self.S = samples_full       # sample size
        self.K = samples_select     # reduced samples size
        self.random_select_ratio = random_select_ratio
        self.rec_type = rec_type.lower().strip()    # type of network recording (Elman or Jordan)
        self.loglike_method = loglike_method # method how to take the log-likelihood
        
        # if we, instead of a u-layer take a 1D convolution
        self.conv_flag = conv_flag
        
        # this has to be the same type the network will sent to
        self.device = device
        
        # parse separate u_layer
        if type(u_layer) == type(None):
            if self.rec_type == 'jordan':
                u_layer = layer_types[-1]
            else:
                u_layer = layer_types[-2]
        
        # u activation function
        self.u_activation = act_fun(u_activation)
        
        # parse u layer shape
        U = self.G*( self.rec_type == 'jordan' ) + self.H*( self.rec_type != 'jordan' )
        
        # deep hidden layers
        self.add_module(
                'w_ih',
                MLP_BBB(
                        input_size, h_sizes[-1], h_sizes[:-1], 
                        activation_functions[:-1], 
                        n_samples_full = samples_full, 
                        n_samples_select = samples_select, 
                        random_select_ratio = .4, prior_var = prior_var, 
                        noise_tol=0.1,  layers = layer_types[:-1], 
                        device = device, use_bias = use_bias, 
                        bd3_flags = ih_bd3_flags, 
                )
        )
        
        # time filter layer
        if vrb3_flag:
            self.add_module(
                    'u_hh', 
                    RNN_BBB(
                            U, self.H, self.L-1, 
                            prior_var = prior_var, 
                            layer_type = u_layer, 
                            activation_function = u_activation, 
                            device = device, use_bias = False, 
                            bd3_flags = [hh_bd3_flag] * U, 
                    )
            )
        elif conv_flag:
            self.add_module(
                    'u_hh',
                    C1B3(
                            U, self.H, self.L, 
                            prior_var = prior_var, 
                            layer_type = u_layer, 
                            device = device, use_bias = False,
                            bd3_flags = [hh_bd3_flag] * U, 
                    )
            )
        else:
            self.add_module(
                    'u_hh', 
                    UB3(
                            U, self.H, self.L-1, 
                            prior_var = prior_var, 
                            layer_type = u_layer, 
                            device = device, use_bias = False,
                            bd3_flags = [hh_bd3_flag] * U, 
                    )
            )
        
        self.add_module(
                'w_hy',
                parse_layer(layer_types[-1])(
                        h_sizes[-1], output_size, 
                        prior_var = prior_var, 
                        device = device, 
                        use_bias = use_bias, 
                        dropout_flag = hy_bd3_flag, 
                )
        )
        self.activation_function = act_fun(activation_functions[-1])
        
        self.noise_tol = noise_tol # we will use the noise tolerance to calculate our likelihood
        self.variational_states = {
                'w_epsilon'  : None,      # will be [[KxMx[hixD]],[Lx[HxU]],[DxH]]
                'b_epsilon'  : None,      # will be [KxMx[hi]]
                'z_epsilon'  : None,      # ... for Bernoulli states
            }
        
        self.to(self.device)
    
    def forward(
            self, x, 
            w_epsilon = None, b_epsilon = None, z_epsilon = None, 
            sample_var = 1.0, n_sample = 1
        ):
        # again, this is equivalent to a standard recurrent net
        H = self.H
        G = self.G
        T = x.size(0)
        N = x.size(1)
        S = n_sample
        
        # none type epsilon parsing
        if type(w_epsilon) == type(None):
            w_epsilon = [None]*3
        if type(b_epsilon) == type(None):
            b_epsilon = [None]*3
        if type(z_epsilon) == type(None):
            z_epsilon = [None]*3
        
        # method type epsilon parsing
        if 'method' in type(w_epsilon).__name__.split('_'):
            w_epsilon = w_epsilon()
        if 'method' in type(b_epsilon).__name__.split('_'):
            b_epsilon = b_epsilon()
        if 'method' in type(z_epsilon).__name__.split('_'):
            z_epsilon = z_epsilon()
        
        # preallocation of the output tensor
        h = torch.zeros((T,S,N,H)).to(self.device)
        y = torch.zeros((T,S,N,G)).to(self.device)
        #u_bias = torch.zeros((S,H)) # to make the u layer bias less
    
        # recurrent forward iterations
        # repeat filter inside input window until end
        t = 0
        while t < T:
            if t > 0:
                # unfortunately we have to clone here
                # otherwise it will raise an inplace operation error
                if self.rec_type == 'jordan':
                    if t > 1:
                        u = self.u_hh(
                            y[:t].clone(), 
                            w_epsilon[1], b_epsilon[1], z_epsilon[1], 
                            sample_var, S, keep_eps = True
                        )
                    else:
                        u = self.u_hh(
                            y[:t].clone(), 
                            w_epsilon[1], b_epsilon[1], z_epsilon[1], 
                            sample_var, S, keep_eps = False
                        )
                else:
                    if t > 1:
                        u = self.u_hh(
                            h[:t].clone(), 
                            w_epsilon[1], b_epsilon[1], z_epsilon[1], 
                            sample_var, S, keep_eps = True
                        )
                    else:
                        u = self.u_hh(
                            h[:t].clone(), 
                            w_epsilon[1], b_epsilon[1], z_epsilon[1], 
                            sample_var, S, keep_eps = False
                        )
                hidden = self.w_ih(
                        x[t], w_epsilon[0], b_epsilon[0], z_epsilon[0], sample_var, S
                    ) + u 
            else:
                hidden = self.w_ih(
                        x[t], w_epsilon[0], b_epsilon[0], z_epsilon[0], sample_var, S
                    )
            
            # hidden activation
            if type(self.u_activation) != type(None):
                hidden = self.u_activation(hidden)
            
            # store hidden layer
            h[t] = hidden
                
            # final output
            out = self.w_hy(
                    hidden, w_epsilon[2], b_epsilon[2], z_epsilon[2], sample_var, S
                )
            
            # output activation
            if type(self.activation_function) != type(None):
                out = self.activation_function(out)
            
            # store output layer
            y[t] = out
            
            # time increment
            t += 1
        
        # store epsilons
        self.w_epsilon   = [l.w_epsilon for l in self.children()]
        self.b_epsilon   = [l.b_epsilon for l in self.children()]
        self.z_epsilon   = [l.z_epsilon for l in self.children()]
        
        return y
    
    def log_prior(self):
        return torch.mean( 
                torch.stack(
                        list( eval_prop( layer.log_prior ) for layer in self.children() )
                ), dim=[0] 
        ).to(self.device)
    
    def log_post(self):
        return torch.mean( 
                torch.stack( 
                        list( eval_prop( layer.log_post  ) for layer in self.children() )
                ), dim=[0]
        ).to(self.device)
    
    def log_like(self, o, y, noise_tol = 0.1 ):
        layers = list( layer for layer in self.children() )
        return layers[-1].log_like( o, y, noise_tol ).to(self.device)

#%% -- CNN BbB --

class CNN_BBB(nn.Module):
    '''
        Deep CNN BbB
    
    This layer just combined a time intervall 1D convolutional BbB layer with 
    another MLP
    '''
    def __init__(
            self,
            input_dim, output_dim, filter_dim, hidden_size = [12], 
            zero_padd = 0, full_conv = True, 
            ar_flag = False, ar_length = 3, ar_activation = 'tanh', 
            dropout_flags = [0,1,0], 
            layer_types = ['Normal','Normal','Normal'], 
            act_funs = ['None','None','relu'],
            noise_tol = 0.1, prior_var = 1.0, prior_pies = .5, 
            samples_full = 1000, samples_select = 400, 
            random_select_ratio = .25, 
            loglike_method = 'gauss', 
            device = 'cuda', 
            use_bias = True, 
        ):
        
        
        # checking the device first to see if it is available
        device = device_check(device)
        
        # initialize the network like you would with a standard RNN, but using the 
        # BBB layer
        super().__init__()
        
        try:
            H = hidden_size[-1]
        except:
            try:
                H = int(hidden_size)
            except:
                H = input_dim
        
        # some hyper parameters
        self.D = input_dim          # input dimension
        self.H = H                  # hidden layer size
        self.G = output_dim         # output dimension
        self.L = filter_dim         # filter length
        self.S = samples_full       # sample size
        self.K = samples_select     # reduced samples size
        self.random_select_ratio = random_select_ratio
        self.loglike_method = loglike_method # method how to
        
        # this has to be the same type the network will sent to
        self.device = device
        
        # parse convolutional params
        if type(filter_dim) == type([]):
            self.n_clayer = len(filter_dim)
        else:
            self.n_clayer = 1
            filter_dim = [filter_dim]
        if type(act_funs[0]) != type([]):
            act_funs[0] = [act_funs[0]] * self.n_clayer
        if type(dropout_flags[0]) != type([]):
            dropout_flags[0] = [dropout_flags[0]] * self.n_clayer
        
        # conv1D layer(s)
        for c, f_L in enumerate(filter_dim):
            self.add_module(
                    'c1d_bbb_%03i' %c,
                    Conv1D_BBB(
                            input_dim, f_L, 
                             prior_var = prior_var, 
                             prior_pies = prior_pies, 
                             device=device, 
                             dropout_flag = dropout_flags[0][c], 
                             full_conv = full_conv, 
                             zero_padd = zero_padd, 
                             activation_function = act_funs[0][c], 
                    )
            )
        
        # deep hidden layers
        if not ar_flag or ar_length <= 0:
            self.add_module(
                    'lfc_bbb',
                    MLP_BBB(
                            input_dim, output_dim, hidden_size, 
                            act_funs[1:], 
                            n_samples_full = samples_full, 
                            n_samples_select = samples_select, 
                            random_select_ratio = .4, prior_var = prior_var, 
                            noise_tol=0.1,  layers = layer_types[1:], 
                            device = device, use_bias = use_bias, 
                            bd3_flags = dropout_flags[1:], 
                    )
            )
        else:
            self.add_module(
                    'lfc_bbb',
                    RNN_BBB(
                            input_dim, output_dim, ar_length, 
                            h_sizes = hidden_size, 
                            h_activations = act_funs[1:], 
                            samples_full = samples_full, 
                            samples_select = samples_select, 
                            random_select_ratio = .4, prior_var = prior_var, 
                            noise_tol=0.1,  layer_type = layer_types[1:], 
                            device = device, use_bias = use_bias, 
                            bd3_flags = dropout_flags[1:], 
                            activation_function = ar_activation, 
                    )
            )
        
        self.noise_tol = noise_tol # we will use the noise tolerance to calculate our likelihood
        self.variational_states = {
                'w_epsilon'  : None,      # will be [[KxMx[hixD]],[Lx[HxU]],[DxH]]
                'b_epsilon'  : None,      # will be [KxMx[hi]]
                'z_epsilon'  : None,      # ... for Bernoulli states
            }
        
        self.to(self.device)

    def forward(
            self, x, 
            w_epsilon = None, b_epsilon = None, z_epsilon = None, 
            sample_var = 1.0, n_sample = 1
        ):
        
        # none type epsilon parsing
        if type(w_epsilon) == type(None):
            w_epsilon = [None]*(1+self.n_clayer)
        if type(b_epsilon) == type(None):
            b_epsilon = [None]*(1+self.n_clayer)
        if type(z_epsilon) == type(None):
            z_epsilon = [None]*(1+self.n_clayer)
        
        # method type epsilon parsing
        if 'method' in type(w_epsilon).__name__.split('_'):
            w_epsilon = w_epsilon()
        if 'method' in type(b_epsilon).__name__.split('_'):
            b_epsilon = b_epsilon()
        if 'method' in type(z_epsilon).__name__.split('_'):
            z_epsilon = z_epsilon()
        
        # convolutional layers
        for c in range(self.n_clayer):
            x =    eval('self.c1d_bbb_%03i(\
                            x,\
                            w_epsilon[c], b_epsilon[c], z_epsilon[c], \
                            sample_var, n_sample, \
                        )' %c)
        
        # linear / auto-regression layers
        y     =    self.lfc_bbb(
                            x,
                            w_epsilon[-1], b_epsilon[-1], z_epsilon[-1], 
                            sample_var, n_sample, 
                        )
        
        
        # store epsilons
        self.w_epsilon   = [l.w_epsilon for l in self.children()]
        self.b_epsilon   = [l.b_epsilon for l in self.children()]
        self.z_epsilon   = [l.z_epsilon for l in self.children()]
        
        return y
    
    def log_prior(self):
        return torch.mean( 
                torch.stack(
                        list( eval_prop( layer.log_prior ) for layer in self.children() )
                ), dim=[0] 
        ).to(self.device)
    
    def log_post(self):
        return torch.mean( 
                torch.stack( 
                        list( eval_prop( layer.log_post  ) for layer in self.children() )
                ), dim=[0]
        ).to(self.device)
    
    def log_like(self, o, y, noise_tol = 0.1 ):
        layers = list( layer for layer in self.children() )
        return layers[-1].log_like( o, y, noise_tol ).to(self.device)

#%% ##########################################################################
