#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:28:44 2019

@author: michele
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import test_predictor as tP

tAlphaS=0.342000
tAlphaR=0.684000
path = 'data/sim_outputs_R%5.6lf_S%5.6lf_hid10/' %(tAlphaR, tAlphaS)
val_input = np.load(path+'Val_Input.npy')
val_data = np.load(path+'Val_Data.npy')
val_input=val_input[:,:,:20,:]
val_data=val_data[:,:,:20,:]
n_input = val_input.shape[3]
n_out = val_data.shape[3]
n_hidden  = 10  # nr of context slow units

""" plot barplot Elman vs best loss case """

# width of the bars
barWidth = 0.5

epocElman = np.empty(21)
lossElman = np.empty(21)
epocElmanEp = np.empty(21)
lossElmanEp = np.empty(21)
epocLearn = np.empty(21)
lossLearn = np.empty(21)
epocLearnEp = np.empty(21)
lossLearnEp = np.empty(21)

for T in range(21):
    _s = 1.000000
    _r = 1.000000
    filename = ('Static_tS0.342000_tR0.684000_S%lf_R%lf_trial%i' %(_s,_r,T))
    #epocElman[T]=np.load('saved_models/'+filename+'/conv_epoch.npy')
    _l=np.load('saved_models/'+filename+'/best_loss.npy')
    if _l == 4000:
        _l=np.nan
    else:
        #compute correlations val data
        mode = 'LearnAlpha_global'
        model = tP.MTSRNN(mode, n_out, n_hidden, n_input, R=1., S=1.)
        p = tP.RNNPredictor(model, 0.02, -1)
        
        
        
        
    lossElman[T]=_l
    
    filename = ('Static_tS0.342000_tR0.684000_S%lf_R%lf_trial%i_400ep' %(_s,_r,T))
    #epocElmanEp[T]=np.load('saved_models/'+filename+'/conv_epoch.npy')
    _l=np.load('saved_models/'+filename+'/best_loss.npy')
    if _l == 4000:
        _l=np.nan
    lossElmanEp[T]=_l
    
    filename = ('LearnAlpha_global_tS0.342000_tR0.684000_trial%i_BP_random_init' %(T))
    #epocLearn[T]=np.load('saved_models/'+filename+'/conv_epoch.npy')
    _l=np.load('saved_models/'+filename+'/best_loss.npy')
    if _l == 4000:
        _l=np.nan
    lossLearn[T]=_l
        
    filename = ('LearnAlpha_global_tS0.342000_tR0.684000_trial%i_BP_random_init_400ep' %(T))
    #epocLearnEp[T]=np.load('saved_models/'+filename+'/conv_epoch.npy')
    _l=np.load('saved_models/'+filename+'/best_loss.npy')
    if _l == 4000:
        _l=np.nan
    lossLearnEp[T]=_l
    

mode = 'LearnAlpha_global'
model = tP.MTSRNN(mode, n_out, n_hidden, n_input, R=1., S=1.)
p = tP.RNNPredictor(model, 0.02, -1)
filename = ('LearnAlpha_global_tS0.342000_tR0.684000_trial%i_BP_random_init_400ep' %(T))
p.load('saved_models/'+filename+'/best')
pred, u, y = p.predict(val_data[34,:,:,:], val_input[34,:,:,:])
plt.plot(pred)
plt.plot(val_data[34,0,:,:])
    
    
#AvEpocElman = np.nanmean(epocElman)
#_m2 = np.nanmean(lossElman)
#s2 = np.nanstd(lossElman, ddof=1)
#varEpocsElman = np.nanstd(epocElman, ddof=1)
#
#AvEpocElman = np.nanmean(epocElman)
#_m1 = np.nanmean(lossLearn)
#s1 = np.nanstd(lossLearn, ddof=1)
#varEpocsElman = np.nanstd(epocElman, ddof=1)
#
#t,p = stats.ttest_ind(lossElman, lossLearn, equal_var=False, nan_policy='omit')
#
#m1=1.0e-04
#m2=_m2-_m1+1.0e-04
#
## The x position of bars
#r1 = [0, 1]
#color = '#8B1A1A'
## Create blue bars
#plt.bar(r1, [m1,m2], width = barWidth, color = '#8B1A1A', edgecolor = 'black', yerr=[s1,s2], capsize=7)
#
## general layout
#plt.xticks([])
#plt.yticks([m1, m2], [round(_m1,7), round(_m2,7)])
#plt.ylabel('Mean loss')
##plt.ylim(ymax=0.002)
#plt.xlabel('')
#plt.title('T-test performance: p = %.2E' %p)
##plt.legend(loc=0)
#figurename = 'loss_elman_vs_DRU_bartdata'
##plt.savefig('figures_results/'+figurename+'.'+ext, format=ext, dpi=DPI)
##plt.show()
#plt.close()
