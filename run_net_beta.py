# %% Docstring
"""
Run Simple Recurrent Neural Net

Here we use a Kaggle dataset for bike renting (accumulated), including many standard variables.
These networks are estimating a half-Cauchy distributed accumulated renting rate.
Thus, the algorithms uses the Expectation Lower Bound Optimization sampling with Bayes by Backprop
to solve this task.
"""
# %% -- imports --
import numpy as np
import torch
from torch import optim
from vRB3_beta import MLP_BBB, RNN_BBB, SRN_BBB, C1B3, CNN_BBB
from BBBopt_beta import sample_elbo
    
# %% helper functions

def batch_idx(idx,N=1000):
    M = np.ceil(idx.shape[0]/N).astype(int)
    b_idx = []
    for i in range(M):
        if i >= M-1:
            b_idx.append(idx[i*N:])
        else:
            b_idx.append(idx[i*N:(i+1)*N])
    return b_idx

def list_stringification(in_list):
    if type(in_list) != type([]):
        try:
            if type(in_list) == type(''):
                in_list = '"'+in_list+'"'
            else:
                in_list = str(in_list)
            return in_list
        except Exception as e:
            print('ERROR: This function annot stringify:')
            print(in_list)
            print(e)
            return ''
    else:
        str_out = '[' + ', '.join(list(
                map( lambda x: list_stringification(x), in_list )
                )) + ']'
        return str_out

# %% -- main --
if ( __name__ == '__main__' ):
    
    #%% testing evironment
    
    # data parameters
    feature_columns = ['temp','hum_interp','windspeed_interp','workingday'] + ['mnth_%i' %(m+1) for m in range(12)]
    target_columns  = ['cnt']
    
    in_path         = 'test_da.csv'    # dataset with all variables
    nheader         = 1
    
    log_flag        = True             # whether taking the logarithm of the target
    
    # model and training parameters
    model_name      = 'clb3' # 'clb3' # 'c1b3' # 'srnb3' # 'mlrb3' # vrb3 # mlb3  # name of the model to use
    layer_types     = ['Normal','Normal']    # types of the layer distrinutions. here for the RB3
#    layer_types     = ['Normal','Normal','HalfCauchy'] # here for the MLB3
    activation_fcn  = ['None','None','relu']      # activation functions for the differen layers. here: for the rnn
#    activation_fcn  = []
    u_layer         = 'Normal' # filter layer type
    u_activation    = 'relu'        # filter layer activation
    rec_type        = 'elang'       # type of the recurrency of the SRN BbB
    loglike_method  = 'gauss'       # method, how we take the log-likelihood # 'softmax', 'gauss', 'backtrace', 'last_weights'
    lr_start        = 0.10          # initial learning rate
    lr_final        = 0.10e-04      # additional annealing of the learning rate
    window_length   = 14 #2 #'filter_length' #48 #'filter_length' #24  # length: how long is our window | batchsize for MLB3
    window_strafe   = 1             # strafe: how much we shift for the next window
    filter_length   = [7,4] #'h_layer' # 12    # filter length, how many long our RNN is
    hidden_sizes    = [5,5] # [12] #        # hidden layer sizes
    num_inputs      = len(feature_columns)            # input dimension
    num_output      = len(target_columns)             # output dimension
    test_size       =  7*3          # test data // 3 weeks
    batchsize       =  250          # num windows each batch; independent samples
    epochs          =   50          # num interations
    n_burn_in       = int(epochs/100*0) # burn in phase with just one sample each
    S_start         =  800          # initial no. samples per data point each iteration
    S_end           =  800          # final no. samples per data point each iteration
    K_start         =  400          # initial sample selection per data point each epoch
    K_end           =  600          # final sample selection per data point each epoch
    nu_smpl_start   = 2.00          # starting sample noise
    nu_smpl_final   = 1.00          # final sample noise
    nu_smpl_epoch   = 0.75          # fraction of epochs when we want to reach 
                                    # the final sample noise
    
    use_bias        = True          # if using bias, if given
    
    vrb3_flag       = False         # if using SRN BbB whether we use a RNN_BBB (IIR) instead of a U-Layer
    conv_flag       = False         # if using SRN BbB whether we use a C1B3 instead of a U-Layer
    
    # for autoregressive model (only used in clb3 - 1D Convolutional BBB)
    ar_flag         = True          # whether to use AR
    ar_length       = 'h_layer'     # how long is the AR filter
    ar_activation   = 'relu'        # which activation is used at the end
    
    gd_opt          = 'Adam'        # which optimizer we want to use
    
    dropout_flags   = [0,1,0,0]     # list of which layers will be Bernoulli filtered
    uh_dropout_flag = False         # whether to dropout the input(s) of the u-layer
    hy_dropout_flag = False         # whether to dropout the output of the u-layer
    
    # CNN parameters
    full_conv = False   # if doing full convolution
    zero_padd = 0       # if and much we are doing zero padding
    
    # defining test variable from filter length, hidden layer sizes or window length
#    H_param = [[2],[3],[7],[14]]          # hidden layer sizes for daily data
#    H_param = [[10],[15],[10,10],[20,20]]
#    H_param = [[4,4],[5],[5,5],[8,4],[12],[16]]
    H_param = [[2],[3],[4],[5]]         # filter length to be tested  
#    H_param = [[12],[18],[18,12]]         # hidden layer configurations we want to try
    
    # getting an increasing number of samples each epoch
    # 
    # NEW:
    #   - a burn in phase of iterations with just 1 sample each
    # TODO:
    #   - a dynamic mode for 95% error confidence
    # 
    S_scheme                = np.zeros((epochs))
    K_scheme                = np.zeros((epochs))
    S_scheme[:n_burn_in]   +=   1
    K_scheme[:n_burn_in]   +=   1
    S_scheme[n_burn_in:]    = np.linspace(S_start,S_end,epochs-n_burn_in).astype(np.int64)
    K_scheme[n_burn_in:]    = np.linspace(K_start,K_end,epochs-n_burn_in).astype(np.int64)
    
    # annealing of the learning rate
    learning_rate = np.linspace(lr_start, lr_final, epochs)
    
    # getting the annealing scheme for the prior noise 
    s_noise_anneal  = np.linspace(
            nu_smpl_start, 
            nu_smpl_final, 
            np.floor(epochs*nu_smpl_epoch).astype(int)
            )
    s_noise_const   = np.ones((np.ceil(epochs*(1.0-nu_smpl_epoch)).astype(int)))
    s_noise         = np.hstack([s_noise_anneal, s_noise_const])[:epochs].squeeze()
    #s_noise         = torch.from_numpy(s_noise)
    
    #%% parsing params
    import argparse
    
    parser = argparse.ArgumentParser()
    
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
    parser.add_argument(
            '--model_name', dest='model_name', type=str,
            help='''The model name:
# 'srnb3'   -   Simple Recurrent Network BbB 
# 'mlrb3'   -   Multi-Layer Recurrent BbB (IIR)
# 'vrb3'    -   Variational Recurrent BbB (IIR)
# 'mlb3'    -   Multi-Layer Perceptron BbB
            ''', 
            default=model_name
    )
    
    parser.add_argument(
            '--epochs', dest='epochs', type=int,
            help='How many epoch we want to train the model.', 
            default=epochs
    )
    
    parser.add_argument(
            '--window_length', dest='window_length', type=str,
            help='The window / batch length if we use the RNN.', 
            default=window_length
    )
    
    parser.add_argument(
            '--window_strafe', dest='window_strafe', type=int,
            help='The window / batch strafe.', 
            default=window_strafe
    )
    
    parser.add_argument(
            '--filter_length', dest='filter_length', type=str,
            help='The filter length if we use the RNN.', 
            default=filter_length
    )
    
    parser.add_argument(
            '--hidden_sizes', dest='hidden_sizes', nargs='+', type=str,
            help='The hidden_sizes if we use hidden units.', 
            default=hidden_sizes
    )
    
    
    parser.add_argument(
            '--test_params', dest='H_param', nargs='+', type=str,
            help='The set of parameters we want to iterate through. The variable set to h_layer is this value.', 
            default=H_param
    )
    
    parser.add_argument(
            '--batchsize', dest='batchsize', type=int,
            help='The batch size', 
            default=batchsize
    )
    
    parser.add_argument(
            '--rec_type', dest='rec_type', type=str,
            help='SRN BbB: Type of recurrency "elang" or "yordan"', 
            default=rec_type
    )
    
    parser.add_argument(
            '--layers', dest='layer_types', nargs='+', 
            help='The layer types. We have one for the case of a RNN', 
            default=layer_types
    )
    
    parser.add_argument(
            '--actfuns', dest='activation_fcn', nargs='+', 
            help='The layer activations (e.g. relu, tanh, selu,..). We have one for the case of a RNN', 
            default=activation_fcn
    )
    
    
    
    args = parser.parse_args()
    # parsing parser into dict
    params = vars(args)
    
    print(
            '''
            Running Bayesian neural network training script
            
            Parameters:
            '''
    )
    print(params)
    
    # writing to locals
    for p in params:
        locals()[p] = params[p]
    
    #%% data reading
    
    # getting coulumns
    import pandas as pd
    file    = pd.read_csv(in_path, nrows = 1, index_col=0)
    columns = list(file.columns)
    
    # getting number of rows by counting the lines
    #   - insufficient, because a CSV could be very big
    with open(in_path) as f:
        nrows = sum(1 for l in f) - nheader
    f.close()
    
#    # getting number of rows by reading the instant/index of the last line
#    #   - only works for indexed CSVs
#    #   - risky, because the csv might not have an internal index 
#    from file_read_backwards import FileReadBackwards as frb
#    with frb(in_path) as f:
#        lastline = f.readline()
#        nrows = int(lastline.split(',')[0])  + 1
#    f.close()
#    with open(in_path) as f:
#        for h in range(nheader):
#            f.readline()
#        firstline = f.readline()
#        nrows -= int(firstline.split(',')[0]) # subtracting the very first index
#    f.close()
    
    #%% find maximum and subtract it
    
    import dask.dataframe as dd
    target_peak = dd.read_csv(in_path)[target_columns].max().compute().values
    
    #%% data selection and indice transformation
    
    # nrows = nrows//4 # for testing
    
    # selectors for data columns: x, y
    #   x .. independent variables /  input features
    #   y .. dependent variable(s) / target features
    y_cols =  [c in target_columns  for c in columns]
    x_cols =  [c in feature_columns for c in columns]
    
    # make sure we have exactly the right number inputs and outputs
    assert np.sum(x_cols) == num_inputs
    assert np.sum(y_cols) == num_output
    
    # for windowing indices
    from skimage.util.shape import view_as_windows
    
    #%% -- training loop --
    '''
        Here we load the data inside of each iteration in windows.
        
    '''
    from torch.autograd import Variable
    # iterate over all our network architectures
    for h_layer in H_param:
        
        # parse dynamic parameters
        if type(filter_length) == type(''):
            f_L = eval(filter_length)
            while type(f_L) == type(''):
                f_L = eval(f_L)
            while type(f_L) == type([]):
                f_L = f_L[0]
        else:
            f_L = filter_length
        
        if type(ar_length) == type(''):
            ar_L = eval(ar_length)
            while type(ar_L) == type(''):
                ar_L = eval(ar_L)
            while type(ar_L) == type([]):
                ar_L = ar_L[0]
        else:
            f_L = filter_length
            
        if type(window_length) == type(''):
            w_L = eval(window_length)
            while type(w_L) == type(''):
                w_L = eval(w_L)
        else:
            w_L = window_length
        while type(w_L) == type([]):
            w_L = w_L[0]
            
        if type(hidden_sizes) == type(''):
            h_H = eval(hidden_sizes)
            while type(h_H) == type(''):
                h_H = eval(h_H)
        else:
            h_H = hidden_sizes
        
        if w_L // 2 < window_strafe and model_name != 'mlb3':
            print('WARNNG: Window strafe conflicts with the nyquist theorem! - Changing to 1/2 the window length - 1!!', end='\r')
            wdL = w_L//2 - 1
        else:
            wdL = int(window_strafe)
        
        # building windowed index
        # - subject to be optimized, because we just need the first column here
        idx_all = torch.arange(nrows).long()
        idx_trn = view_as_windows(idx_all[:-test_size].numpy(), w_L, window_strafe)
        idx_tst = view_as_windows(idx_all[-test_size:].numpy(), w_L, window_strafe)
        
        # saving the original order (in case its needed)
        idx_trn_, idx_tst_ = idx_trn, idx_tst
        
        # model and optimizer
        if      model_name == 'vrb3':
            regressor       = RNN_BBB(
                        num_inputs, num_output, f_L, 
                        samples_full = S_start, samples_select = K_start, 
                        activation_function = activation_fcn[0], 
                        layer_type = layer_types[-1], 
                        loglike_method = loglike_method, 
                        use_bias = use_bias, 
                        bd3_flags = dropout_flags, 
                    ).cuda()
        elif    model_name.lower() == 'mlb3':
            regressor       = MLP_BBB(
                        num_inputs,num_output, 
                        h_layer, activation_fcn, 
                        S_start, K_start, 
                        layers = layer_types, 
                        loglike_method = loglike_method, 
                        use_bias = use_bias, 
                        bd3_flags = dropout_flags, 
                    ).cuda()
        elif    model_name.lower() == 'mlrb3':
            regressor       = RNN_BBB(
                        num_inputs, num_output, f_L, 
                        samples_full = S_start, samples_select = K_start, 
                        activation_function = activation_fcn, 
                        layer_type = layer_types, 
                        deep_flag = True, h_sizes = h_H, 
                        loglike_method = loglike_method, 
                        use_bias = use_bias, 
                        bd3_flags = dropout_flags, 
                    ).cuda()
        elif    model_name.lower() == 'srnb3':
            regressor       = SRN_BBB(
                        num_inputs, num_output, f_L,
                        samples_full = S_start, samples_select = K_start, 
                        activation_functions = activation_fcn, 
                        u_layer = u_layer, u_activation = u_activation,
                        h_sizes = h_H, layer_types = layer_types, 
                        loglike_method = loglike_method, 
                        use_bias = use_bias, vrb3_flag = vrb3_flag, conv_flag = conv_flag, 
                        hh_bd3_flag = uh_dropout_flag, hy_bd3_flag = uh_dropout_flag, 
                        ih_bd3_flags = dropout_flags, 
                    ).cuda()
        elif    model_name.lower() == 'clb3':
            regressor       = CNN_BBB(
                        num_inputs, num_output, f_L, h_H, 
                        samples_full = S_start, samples_select = K_start, 
                        act_funs = activation_fcn, layer_types = layer_types,
                        dropout_flags = dropout_flags, use_bias = use_bias, 
                        loglike_method = loglike_method, 
                        full_conv = full_conv, 
                        zero_padd = zero_padd, 
                        ar_activation = ar_activation, 
                        ar_flag = ar_flag, 
                        ar_length = ar_L, 
                    ).cuda()
        elif    model_name.lower() == 'c1b3':
            regressor       = C1B3(
                        num_inputs, num_output, f_L,
                        samples_full = S_start, samples_select = K_start, 
                        activation_functions = activation_fcn, 
                        u_layer = u_layer, u_activation = u_activation,
                        h_sizes = h_H, layer_types = layer_types, 
                        loglike_method = loglike_method, 
                        use_bias = use_bias, vrb3_flag = vrb3_flag, conv_flag = conv_flag, 
                        hh_bd3_flag = uh_dropout_flag, hy_bd3_flag = uh_dropout_flag, 
                        ih_bd3_flags = dropout_flags, 
                    ).cuda()
        
        # optimizer - this might define how successfull the algorithm is!
        optimizer       = eval('torch.optim.%s(regressor.parameters(), lr=learning_rate[0])' %gd_opt)
        
        # sampling funtion integrated
        # trying to avoid pickling problems
        regressor.sample_elbo = sample_elbo
        
        #%% pre-loop parameter settings
        n_conv      = 0         # counter for no or bad change in test elbo
        max_conv    = 3         # max no or bad changes in test elbo
        train_elbos = np.zeros(epochs)  # record of average train elbo
        train_mae   = np.zeros(epochs)  # record of average test mae
        train_mad   = np.zeros(epochs)  # record of average test mad
        train_mse   = np.zeros(epochs)  # record of average test mse
        train_rmse  = np.zeros(epochs)  # record of average test rmse
        test_elbos  = np.zeros(epochs)  # record of average test elbo
        test_mae    = np.zeros(epochs)  # record of average test mae
        test_mad    = np.zeros(epochs)  # record of average test mad
        test_mse    = np.zeros(epochs)  # record of average test mse
        test_rmse   = np.zeros(epochs)  # record of average test rmse
        epoch_rec   = np.ones(epochs) * np.nan # record the epochs being done
        tr_indices  = []                # record the starting positions for each training window
        # record the change of the learning rate
        lr_capture  = np.zeros((epochs,len(optimizer.param_groups)))
        iid = np.arange(idx_trn.shape[0])
        breaker = False
        
        windows = idx_trn[:batchsize]
        
        #%% start training loops
        for epoch in range(epochs):  # loop over the dataset multiple times
            
            # breaker if there was any error while training
            if breaker:
                break
            
            print('training epoch %i / %i' %((epoch+1),epochs))
            # shuffling the windows
            np.random.shuffle(iid)
            idx_trn = idx_trn[iid]
            train_losses, train_maes  = [], []
            train_mads, train_mses    = [], []
            train_rmses               = []
            
            train_losses = []
            b_idx = batch_idx(idx_trn, batchsize)
            for n,b in enumerate(b_idx):
                n_str = 'loading batch windows %i / %i -> ' %((n+1)*b.shape[0],idx_trn.shape[0])
                print(n_str, end='\r')
                try:
                    df = pd.read_csv(
                            in_path, 
                            skiprows = b.min(), 
                            nrows = windows.max()+1, 
                            index_col = 0, 
                        )
                    # transforming data into variables
                    x = df.values.copy()[windows[:b.shape[0]]][:, :, x_cols].astype(float)
                    y = df.values.copy()[windows[:b.shape[0]]][:, :, y_cols].astype(float)
                    if log_flag:
                        y = np.log(1 + y)
                    x = Variable(torch.from_numpy(x).transpose(0,1)).float().cuda()
                    y = Variable(torch.from_numpy(y).transpose(0,1)).float().cuda()
                    
                    print(
                            n_str + 
                            'sampling ELBO . . . ', 
                            end='\r'
                    )
                except Exception as e:
                    print('sampling error ! ! !')
                    print(e)
                    print('skipping')
                    breaker = True
                    break
                if model_name == 'mlb3':
                    x = x.squeeze()
                    y = y.squeeze()
                # compute negative elbo loss, with S samples selecting the K best
                loss, _ = regressor.sample_elbo(
                        regressor, 
                        x, y, 
                        S_scheme[epoch], K_scheme[epoch], 
                        s_noise[epoch], loglike_method = loglike_method
                )
                
                # zero gradient (for next optimization step)
                optimizer.zero_grad()
                # back prop
                loss.backward()
                # optimization
                optimizer.step()
                # display loss
                if loss.item() == np.nan:
                    print('Error: No loss caculated!')
                    print('Skipping!! . . . . . . . .',end = '\r')
                else:
                    print(
                            n_str + 
                            'ELBO: %f . . . . . ' %(loss.item())
                    )
                try:
                    train_losses.append(loss.item())
                except Exception as e:
                    train_losses.append(0)
                    print('cannot get ELBO')
                    print(e)
                try:
                    train_maes.append((y - regressor(x)).abs().mean().item())
                except Exception as e:
                    train_maes.append(0)
                    print('cannot get MAE')
                    print(e)
                try:
                    train_mses.append(((y - regressor(x))**2).mean().item())
                except Exception as e:
                    train_mses.append(0)
                    print('cannot get MSE')
                    print(e)
                try:
                    train_mads.append((regressor(x) - regressor(x).mean()).mean().item())
                except Exception as e:
                    train_mads.append(0)
                    print('cannot get MAD')
                    print(e)
                try:
                    train_rmses.append(torch.sqrt((y - regressor(x))**2).mean().item())
                except Exception as e:
                    train_rmses.append(0)
                    print('cannot get RMSE')
                    print(e)
                
                # record loss
                train_losses.append(loss.item())
                
                # empty cache        
                torch.cuda.empty_cache()
                
                del loss, x, y
            
            # display error metrics
            print(
                'ELBO: {}, MAE: {}, MAD: {}, MSE: {}, RMSE: {}'.format(
                        np.mean(train_losses), 
                        np.mean(train_maes), 
                        np.mean(train_mads), 
                        np.mean(train_mses),
                        np.sqrt(np.mean(train_mses)),  
                    )
            )
            
            
            # storing losses and testing values
            train_elbos[epoch]   = np.mean(train_losses)
            train_mae[epoch]     = np.mean(train_maes)
            train_mad[epoch]     = np.mean(train_mads)
            train_mse[epoch]     = np.mean(train_mses)
            train_rmse[epoch]    = np.mean(np.sqrt(train_mses))
            
            # slowly annealing down the learning rate
            for p in range(len(optimizer.param_groups)):
                optimizer.param_groups[p]['lr'] = learning_rate[epoch]
                lr_capture[epoch,p] = optimizer.param_groups[p]['lr']
            
            print('', end='\r')
            print('test epoch %i / %i' %((epoch+1),epochs))
            test_losses, test_maes  = [], []
            test_mads, test_mses    = [], []
            test_rmses              = []
            
            b_idx = batch_idx(idx_tst, batchsize)
            for n,b in enumerate(b_idx):
                n_str = 'loading batch windows %i / %i -> ' %((n+1)*b.shape[0],idx_tst.shape[0])
                print(n_str, end='\r')
                try:
                    df = pd.read_csv(
                            in_path, 
                            skiprows = b.min(), 
                            nrows = windows.max()+1, 
                            index_col = 0, 
                        )
                    
                    # transforming data into variables
                    x = df.values.copy()[windows[:b.shape[0]]][:, :, x_cols].astype(float)
                    y = df.values.copy()[windows[:b.shape[0]]][:, :, y_cols].astype(float)
                    if log_flag:
                        y = np.log(1 + y)
                    x = Variable(torch.from_numpy(x).transpose(0,1)).float().cuda()
                    y = Variable(torch.from_numpy(y).transpose(0,1)).float().cuda()
                    
                    print(
                            n_str + 
                            'sampling ELBO . . . ', 
                            end='\r'
                    )
                    try:
                        
                        if model_name == 'mlb3':
                            x = x.squeeze()
                            y = y.squeeze()
                        optimizer.zero_grad()
                        loss, s_exp = regressor.sample_elbo(
                                regressor, 
                                x, y, 
                                S_scheme[epoch], K_scheme[epoch], 
                                s_noise[epoch], loglike_method = loglike_method
                        )
                        print(
                                n_str + 
                                'ELBO: %f . . . . . ' %(loss.item()) + ' -> taking test metrics',
                                end='\r'
                        )
                        try:
                            test_losses.append(loss.item())
                        except Exception as e:
                            test_losses.append(0)
                            print('cannot get ELBO')
                            print(e)
                        try:
                            test_maes.append((y - regressor(x)).abs().mean().item())
                        except Exception as e:
                            test_maes.append(0)
                            print('cannot get MAE')
                            print(e)
                        try:
                            test_mses.append(((y - regressor(x))**2).mean().item())
                        except Exception as e:
                            test_mses.append(0)
                            print('cannot get MSE')
                            print(e)
                        try:
                            test_mads.append((regressor(x) - regressor(x).mean()).mean().item())
                        except Exception as e:
                            test_mads.append(0)
                            print('cannot get MAD')
                            print(e)
                        try:
                            test_rmses.append(torch.sqrt((y - regressor(x))**2).mean().item())
                        except Exception as e:
                            test_rmses.append(0)
                            print('cannot get RMSE')
                            print(e)
                        
                    except Exception as e:
                        print('sampling error ! ! !')
                        print(e)
                        print('skipping')
                        breaker = True
                        break
                    
                    # empty cache        
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(e)
                    print('skipping')
                    breaker = True
                    break
            
            # display error metrics
            print(
                'ELBO: {}, MAE: {}, MAD: {}, MSE: {}, RMSE: {}'.format(
                        np.mean(test_losses), 
                        np.mean(test_maes), 
                        np.mean(test_mads), 
                        np.mean(test_mses),
                        np.sqrt(np.mean(test_mses)),  
                    )
            )
            
            # storing losses and testing values
            test_elbos[epoch]   = np.mean(test_losses)
            test_mae[epoch]     = np.mean(test_maes)
            test_mad[epoch]     = np.mean(test_mads)
            test_mse[epoch]     = np.mean(test_mses)
            test_rmse[epoch]    = np.mean(np.sqrt(test_mses))
            epoch_rec[epoch]    = epoch
            tr_indices.append(idx_trn[:,0])
        
        # retransform sample noise scheme
        #s_noise = s_noise.numpy()
        
        print('Finished Training')
        
        
        torch.save(regressor.state_dict(), 'out/test_'+model_name+'_windows%s_VAR_I%i_%s.torch' %(
                        str(window_length), 
                        epochs, 
                        ('_'.join(np.array(h_layer).astype(str)))
                        )
                )
        
        train_info = pd.DataFrame(
                np.hstack([
                        epoch_rec[:epoch,np.newaxis],
                        train_elbos[:epoch,np.newaxis], 
                        train_mae[:epoch,np.newaxis], 
                        train_mad[:epoch,np.newaxis], 
                        train_mse[:epoch,np.newaxis], 
                        train_rmse[:epoch,np.newaxis], 
                        test_elbos[:epoch,np.newaxis], 
                        test_mae[:epoch,np.newaxis], 
                        test_mad[:epoch,np.newaxis], 
                        test_mse[:epoch,np.newaxis], 
                        test_rmse[:epoch,np.newaxis], 
                        S_scheme[:epoch,np.newaxis], 
                        K_scheme[:epoch,np.newaxis], 
                        s_noise[:epoch,np.newaxis], 
                        lr_capture[:epoch,:],
                        ]), 
                columns = [
                        'epoch',
                        'train_elbos',
                        'train_maes',
                        'train_mads',
                        'train_mses',
                        'train_rmses',
                        'test_elbos',
                        'test_maes',
                        'test_mads',
                        'test_mses',
                        'test_rmses',
                        'S_scheme',
                        'K_scheme',
                        's_noise',
                        ] + ['lr_%i' %p for p in range(len(optimizer.param_groups))]
                )
        
        train_info['feature_cols']  = list_stringification(feature_columns)
        train_info['target_cols']   = list_stringification(target_columns)
        train_info['log_flag']      = log_flag
        train_info['target_peak']   = list_stringification(target_peak)
        train_info['use_bias']      = use_bias
        train_info['layer_types']   = list_stringification(layer_types)
        train_info['activation_fcn']= list_stringification(activation_fcn)
        train_info['u_layer']       = u_layer
        train_info['u_activation']  = u_activation
        train_info['rec_type']      = rec_type
        train_info['opt']           = type(optimizer).__name__
        train_info['batch_size']    = batchsize
        train_info['n burn-in']     = n_burn_in
        train_info['full_S']        = train_info['S_scheme'] * batchsize
        train_info['full_K']        = train_info['K_scheme'] * batchsize
        train_info['window_strafe'] = wdL
        train_info['window_length'] = w_L
        train_info['filter_length'] = list_stringification(f_L)
        train_info['h_sizes']       = list_stringification(h_H)
        train_info['full_conv']     = full_conv
        train_info['zero_padd']     = zero_padd
        train_info['ar_length']     = ar_length
        train_info['ar_flag']       = ar_flag
        train_info['ar_activation'] = ar_activation
        train_info['n_tr_windows']  = idx_trn.shape[0]
        train_info['n_te_windows']  = idx_tst.shape[0]
        train_info['dropout_flags'] = list_stringification(dropout_flags) # uh_dropout_flag
        train_info['uh_dropout_flag'] = uh_dropout_flag # uh_dropout_flag
        train_info['hy_dropout_flag'] = hy_dropout_flag # uh_dropout_flag
        train_info['vrb3_flag']     = vrb3_flag
        train_info['conv_flag']     = conv_flag
        
        ar = pd.DataFrame(
                np.concatenate([a[np.newaxis] for a in tr_indices],axis=0), 
                columns=['wpos%i' %j for j in range(tr_indices[0].shape[0])]
        )
        
        train_info = pd.concat([train_info,ar], axis=1)
        
        train_info.to_csv(
                'out/test_'+model_name+'_elbos_windows_%s_VAR_I%i_%s.csv' %(
                        str(window_length), 
                        epochs, 
                        ('_'.join(np.array(h_layer).astype(str)))
                        )
                )
        try:
            torch.save(regressor,'out/test_'+model_name+'_windows%s_VAR_I%i_%s.model' %(
                    str(window_length), 
                    epochs, 
                    ('_'.join(np.array(h_layer).astype(str)))
                    )
            )
        except Exception as e:
            print('ERROR: Cannot pickle model object:')
            print(e)
