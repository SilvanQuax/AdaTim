#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:08:44 2017

@author: michele

Main for unsupervised time constants varying prediction synt signals"""

import sys
import numpy as np
import os
import test_predictor as tP
#import random
from shutil import copyfile
import scipy.signal
import matplotlib.pyplot as plt


device_id = -1

"""Data loading and pre processing------------------------------------------"""

n_input = 2
n_out = 2
n_hidden  = 10  # nr of context slow units
n_steps = 40
n_samples = 500
K=10 #amplitude of input signals

sample_input=np.zeros((n_samples,1,n_steps,n_input), dtype='float32')
for i in range(n_samples):
    X = np.random.random_sample((n_steps,n_input))
    #smooth signal, for slow varying input
    sample_input[i,0,:,:] = K*scipy.signal.savgol_filter(X, 7, 2, axis=0)
    
"""Model setup--------------------------------------------------------------"""
                                                                                                      
#choose mode 

mode = 'Static'
mode = 'LearnAlpha_local'
trial = int(sys.argv[2])
S=0.5
R=0.5
sd = float(sys.argv[1])
_min = 0.4
_max = +0.7

alphaS = np.zeros((1,n_hidden), dtype='float32')
alphaR = np.zeros((1,n_hidden), dtype='float32')

for i in xrange(n_hidden):
    while True:
        _aS = np.random.normal(S,sd)
        if((_aS>0) & (_aS<1)):
            break
    alphaS[0,i] = _aS
    while True:
        _aR = np.random.normal(R,sd)
        if((_aR>0) & (_aR<1)):
            break
    alphaR[0,i] = _aR

sR = np.std(alphaR)
sS = np.std(alphaS)
print sR
print sS
        
    # print('Generating required time constants...')
    # if sR>_min and sR<_max and sS>_min and sS<_max:
    #         break
print('Alphas standard dev are:')
print sR
print sS
    

#define model
model = tP.MTSRNN(mode, n_out, n_hidden, n_input, R, S)

#define predictor mode    
learning_rate = 0.002#float(sys.argv[5])
p = tP.RNNPredictor(model, learning_rate, device_id)
if mode == 'LearnAlpha_local':
    model.hidden.alphaS.initialize_alpha(alphaS)
    model.hidden.alphaR.initialize_alpha(alphaR)
#    
"""Train and Test-----------------------------------------------------------"""

#n_epochs = 1
##trial = int(sys.argv[7])
#filename = (mode+'_test1')
#if filename is not None:
#    os.makedirs('saved_models/'+filename)
#    
#train_loss, test_loss , batches_loss = p.train(train_data, train_input, val_data, val_input, n_epochs, filename, KL_loss=False)
#p.load('saved_models/'+filename+'/best')
#
#np.save('saved_models/'+filename+'/Train_loss_prim', train_loss)
#np.save('saved_models/'+filename+'/Test_loss_prim', test_loss)
##np.save('saved_models/'+filename+'/Wslow', model.slow.W.W.data)

"""Saving and Analysis phase ---------------------------------------------- """

#create analysis object
#    
#analisi.print_Loss(train_loss, test_loss, batch_size, n_batches, filename)
#
#index=1
#sample = val_data[index,:,:,:]
sample_output = np.zeros((n_samples,1,n_steps,n_out), dtype='float32')

sample=sample_input
for i in range(n_samples):
    act_pred, u_hidden, y_hidden = p.predict(sample[i,:,:,:], sample_input[i,:,:,:])
    sample_output[i,0,:,:] = act_pred
#analisi.print_Pred_vs_Truth(act_pred, y_hidden, u_hidden, sample, sample_input, filename=None, _sm=None)

#np.save('sim_inputs', sample_input)
#np.save('sim_outputs_R%lf_S%lf_hid%i' %(R, S, n_hidden), sample_output )

filename='sim_outputs_R%lf_S%lf_hid%i_std%5.3lf_trial%i' %(R, S, n_hidden, sd, trial)

os.makedirs('data2/'+filename)

np.save('data2/'+filename+'/Train_Input', sample_input[:400,:,:,:])
np.save('data2/'+filename+'/Train_Data.npy', sample_output[:400,:,:,:])
np.save('data2/'+filename+'/Val_Input.npy', sample_input[400:,:,:,:])
np.save('data2/'+filename+'/Val_Data.npy', sample_output[400:,:,:,:])
np.save('data2/'+filename+'/alphaS.npy', alphaS)
np.save('data2/'+filename+'/alphaR.npy', alphaR)

for i in range(6):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('input', color=color)
    ax1.plot(sample_input[i,0,:,0], color='darkred')
    ax1.plot(sample_input[i,0,:,1], color='r')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('output', color='navy')  # we already handled the x-label with ax1
    ax2.plot(sample_output[i,0,:,0], color='navy')
    ax2.plot(sample_output[i,0,:,1], color='b')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('data2/'+filename+'/sample_%i' %(i))

sR = np.std(alphaR)
sS = np.std(alphaS)
print sR
print sS