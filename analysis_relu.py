#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:06:28 2019

@author: michele
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

""" Figures extension and parameters """
ext = 'eps'
DPI = 1200

ntrials=40
#target alphaS, target alphaR, act funct if not sigmoid, hidden units, number of trials
target_alphas = [[0.342000, 0.684000, '', 10, ntrials],
                 [0.684000, 0.342000, '', 10, ntrials],
                 [0.342000, 0.684000, '_ReLu', 10, ntrials],
                 [0.684000, 0.342000, '_ReLu', 10, ntrials]]

alphas=np.linspace(0.001, 1.3, num=20)

epoc = np.empty((4,ntrials))
loss = np.empty((4,ntrials))
alphaS = np.empty((4,ntrials,501))
alphaR = np.empty((4,ntrials,501))
opt_aS = np.empty((4,ntrials))
opt_aR = np.empty((4,ntrials))
epoc[:,:] = np.nan
loss[:,:] = np.nan
alphaS[:,:] = np.nan
alphaR[:,:] = np.nan
opt_aS[:,:] = np.nan
opt_aR[:,:] = np.nan
i=0

for talphas in target_alphas:
    
    tAs=talphas[0]
    tAr=talphas[1]
    _des = talphas[2]
    nhidd=talphas[3]
    
    for trial in range(talphas[4]):

        path='saved_models_sigm_vs_relu/LearnAlpha_global_tS%lf_tR%lf_nhidd%i_trial%i' %(tAs, tAr, nhidd, trial+1) +_des
        epoc[i,trial] = np.load(path+'/conv_epoch.npy')
        loss[i,trial] = np.load(path+'/best_loss.npy')
        alphaS[i,trial,:] = np.squeeze(np.load(path+'/learning_alphaS.npy'))
        alphaR[i,trial,:] = np.squeeze(np.load(path+'/learning_alphaR.npy'))
        opt_aS[i,trial]=alphaS[i,trial,int(epoc[i,trial])]
        opt_aR[i,trial]=alphaR[i,trial,int(epoc[i,trial])]
    i=i+1
    
"plot grid and learned alphas for both panels"
plt.figure(figsize=(5,5))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.scatter(opt_aS[0,:], opt_aR[0,:], marker='+', label='sigmoid P1')
plt.scatter(opt_aS[1,:], opt_aR[1,:], marker='+', label='sigmoid P2')
plt.scatter(opt_aS[2,:], opt_aR[2,:], marker='+', label='ReLU P1')
plt.scatter(opt_aS[3,:], opt_aR[3,:], marker='+', label='ReLU P2')
plt.scatter(0.342000, 0.684000, c='yellow', marker='o', label='P1')
plt.scatter(0.684000, 0.342000, c='black', marker='o', label='P2')
plt.xlabel('alpha_s')
plt.ylabel('alpha_r')
plt.legend()
plt.title('Optimal learned time constants')
plt.plot()
plt.savefig('figures_results/optimal_alphas_relu.eps', format='eps', dpi=DPI)
#plt.close()

#"Case of weird learning"
#plt.figure(figsize=(5,5))
#plt.xlim(0, 1)
#plt.ylim(0, 1)
#plt.scatter(alphaS[3,6,:], alphaR[3,6,:], marker='+', label='BP ReLU')
##plt.scatter(opt_aS[1,:], opt_aR[1,:], marker='+', label='sigmoid P2')
##plt.scatter(opt_aS[2,:], opt_aR[2,:], marker='+', label='ReLU P1')
##plt.scatter(opt_aS[3,:], opt_aR[3,:], marker='+', label='ReLU P2')
##plt.scatter(0.342000, 0.684000, c='yellow', marker='o', label='P1')
#plt.scatter(0.684000, 0.342000, c='yellow', marker='o', label='P2')
#plt.xlabel('alpha_s')
#plt.ylabel('alpha_r')
#plt.legend()
#plt.title('Failed learning: dummy case')
#plt.plot()
#plt.savefig('figures_results/dummy_case.eps', format='eps', dpi=DPI)

   
"stats"
mean_opt_aR = np.nanmean(opt_aR, axis=1)
mean_opt_aS = np.nanmean(opt_aS, axis=1)
std_opt_aR = np.nanstd(opt_aR, axis=1)
std_opt_aS = np.nanstd(opt_aS, axis=1)


    
"Significancy: independent t-tes for the two time constants"

# target time constants 0.34, 0.68
t_S1,p_S1 = stats.ttest_ind(opt_aS[0,:], opt_aS[2,:], equal_var=False, nan_policy='omit')
t_R1,p_R1 = stats.ttest_ind(opt_aR[0,:], opt_aR[2,:], equal_var=False, nan_policy='omit')

# target time constants 0.68, 0.34
t_S2,p_S2 = stats.ttest_ind(opt_aS[1,:], opt_aS[3,:], equal_var=False, nan_policy='omit')
t_R2,p_R2 = stats.ttest_ind(opt_aR[1,:], opt_aR[3,:], equal_var=False, nan_policy='omit')
print('Significancy: independent t-tes for the two time constants')
print('target time constants s=0.34, r=0.68:')
print('alpha_s: p=%lf' %p_S1)
print('alpha_r: p=%lf' %p_R1)
print('target time constants s=0.68, r=0.34:')
print('alpha_s: p=%lf' %p_S2)
print('alpha_r: p=%lf' %p_R2)