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
import random
from analysis_01 import Analysis
from shutil import copyfile
import scipy.io

device_id = -1

"""Data loading (preprocessed)------------------------------------------"""
tAlphaS=0.684000
tAlphaR=0.342000
path = 'data/sim_outputs_R%5.6lf_S%5.6lf_hid10/' %(tAlphaR, tAlphaS)
train_input = np.load(path+'Train_Input.npy')
train_data = np.load(path+'Train_Data.npy')
val_input = np.load(path+'Val_Input.npy')
val_data = np.load(path+'Val_Data.npy')
train_input=train_input[:,:,:20,:]
train_data=train_data[:,:,:20,:]
val_input=val_input[:,:,:20,:]
val_data=val_data[:,:,:20,:]

"""Model setup--------------------------------------------------------------"""

n_input = val_input.shape[3]
n_out = val_data.shape[3]
n_hidden  = int(sys.argv[1])                                                                                      

#choose mode 
mode = 'LearnAlpha_global'
#mode = 'LearnAlpha_local'
#mode = 'Static'

#define model
alphas=np.linspace(0.001, 1.3, num=20)
#alphas2=np.arange(1.3, 2, 0.06836842105)
#alphas = np.concatenate((alphas, alphas2[1:]))

indS= int(sys.argv[2])
indR= int(sys.argv[3])

if indS==999:
    indS=random.randint(0,14)
if indR==999:
    indR=random.randint(0,14)
alphaS= alphas[indS]
alphaR= alphas[indR]
model = tP.MTSRNN(mode, n_out, n_hidden, n_input, R=alphaR, S=alphaS)

#define predictor mode    
learning_rate = float(sys.argv[4])
p = tP.RNNPredictor(model, learning_rate, device_id)

if mode == 'LearnAlpha_local':
    model.hidden.alphaS.initialize_alpha(np.ones((1,n_hidden), dtype='float32')*alphaS)
    model.hidden.alphaR.initialize_alpha(np.ones((1,n_hidden), dtype='float32')*alphaR)
if mode == 'LearnAlpha_global':
    model.hidden.alphaS.initialize_alpha(np.ones((1), dtype='float32')*alphaS)
    model.hidden.alphaR.initialize_alpha(np.ones((1), dtype='float32')*alphaR)
    
   
"""Train and Test-----------------------------------------------------------"""

n_epochs = int(sys.argv[5])
trial = int(sys.argv[6])
if mode=='Static':
    filename = (mode+'_tS%5.6lf_tR%5.6lf_nhidd%i_S%lf_R%lf_trial%i_ReLu' %(tAlphaS, tAlphaR, n_hidden, alphaS, alphaR, trial))
else: 
    filename = (mode+'_tS%5.6lf_tR%5.6lf_nhidd%i_trial%i_ReLu' %(tAlphaS, tAlphaR, n_hidden, trial))

if filename is not None:
    os.makedirs('saved_models/'+filename)
train_loss, test_loss , batches_loss = p.train(train_data, train_input, val_data, val_input, n_epochs, filename, KL_loss=False)
#p.load('saved_models/'+filename+'/best')
#np.save('saved_models/'+filename+'/Train_loss_prim', train_loss)
#np.save('saved_models/'+filename+'/Test_loss_prim', test_loss)
#np.save('saved_models/'+filename+'/Wslow', model.slow.W.W.data)

"""Saving and Analysis phase ---------------------------------------------- """

#create analysis object
#analisi = Analysis(model)
#    
##analisi.print_Loss(train_loss, test_loss, batch_size, n_batches, filename)
#
#index=1
#sample = val_data[index,:,:,:]
#sample_input = val_input[index,:,:,:]
#act_pred, u_hidden, y_hidden = p.predict(sample, sample_input)
#analisi.print_Pred_vs_Truth(act_pred, y_hidden, u_hidden, sample, sample_input, filename=None, _sm=None)
