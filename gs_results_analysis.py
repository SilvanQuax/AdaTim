#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:22:58 2018

Results analysis based on loss and epoch of best loss

@author: michele
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


""" Figures extension and parameters """
ext = 'eps'
DPI = 1200


"""Load results from grid searches"""

target_alphas = [[0.137282, 0.137230, '', 2],
                 [0.342000, 0.684000, '', 2],
                 [0.342000, 0.684000, '_hidd5', 2],
                 [0.342000, 0.684000, '_hidd30', 2],
                 [0.342000, 0.684000, '_hidd100', 2],
                 # [0.342000, 0.684000, '_Epochs', 1],
                 [0.342000, 0.684000, '_ReLu', 6],
                 [0.684000, 0.342000, '', 2],
                 [0.684000, 0.342000, '_ReLu', 2],
                 [0.342000, 1.000000, '', 2],
                 [0.684000, 0.684000, '', 2],
                 [0.889000, 0.889000, '', 2]]

alphas=np.linspace(0.001, 1.3, num=20)

for talphas in target_alphas:
    
    tAs=talphas[0]
    tAr=talphas[1]
    
    epoc = np.empty((10,10,10))
    loss = np.empty((10,10,10))
    epoc[:,:,:] = np.nan
    loss[:,:,:] = np.nan

    #Static_tS0.342000_tR0.684000_S0.001000_R0.001000_trial0

    for S in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]:
        for R in [ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]:
            for T in range(talphas[3]):
                _s = alphas[S]
                _r = alphas[R]
                _des = talphas[2]
                filename = ('Static_tS%lf_tR%lf_S%lf_R%lf_trial%i'%(tAs, tAr, _s,_r,T)+_des)
                epoc[S/2,R/2,T]=np.load('saved_models/'+filename+'/conv_epoch.npy')
                loss[S/2,R/2,T]=np.load('saved_models/'+filename+'/best_loss.npy')
    
    av_epochs = np.nanmean(epoc, axis=2)
    av_losses = loss[:,:,0]#np.nanmean(loss, axis=2)
    var_losses = np.nanstd(loss, axis=2)
    var_epochs = np.nanstd(epoc, axis=2)
    best_alphas_ind = None
    best_loss = 500
    for _s in range(10):
        for _r in range(10):
            if av_losses[_s,_r] < best_loss:
                best_alphas_ind = [2*_s, 2*_r]
                best_loss = av_losses[_s,_r]       
    best_alphas = [alphas[best_alphas_ind[0]], alphas[best_alphas_ind[1]]]

    """ plot gridsearch results"""

    log_av_losses = np.log(av_losses)
    if talphas==[0.342000, 0.684000, '', 2]:
        STD_log_av_losses = log_av_losses
    if talphas==[0.684000, 0.342000, '', 2]:
        STD_log_av_losses2 = log_av_losses

    fig, ax = plt.subplots()
    cax = ax.imshow(log_av_losses, interpolation='none', cmap='OrRd',  extent=[0.001,1.24,1.24, 0.001])
#    cax = ax.imshow(av_epochs, interpolation='none', cmap='OrRd',  extent=[0.001,1.24,1.24, 0.001])

    ax.set_title('log(Loss)')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    _min=np.nanmin(log_av_losses)
    _max=np.nanmax(log_av_losses)
    _med=(_max+_min)*0.5
    cbar = fig.colorbar(cax, ticks=[_min, _med, _max])
    cbar.ax.set_yticklabels(['%5.2f' %((_min)), '%5.2f' %((_med)), '%5.2f' %((_max))])  # vertically oriented colorbar
    ax.set_xlabel('Alpha_R')
    ax.set_ylabel('Alpha_S')
#    ax.scatter(best_alphas[1],best_alphas[0], c='b', marker='x')
    ax.scatter(tAr,tAs,  s=100, facecolors='none', edgecolors='g')
    figurename = 'grid_search_target_s%lf_r%lf'%(tAs, tAr)+_des
    #plt.savefig('figures_results/'+figurename+'.'+ext, format=ext, dpi=DPI)
    plt.close()



    
""" Load the larger gridsearch with target (1.,1.) """

target_alphas = [[1.000, 1.000, '', 2],
                 [1.000, 1.000, '_hidd30', 2],
                 [1.000, 1.000, '_ReLu', 3]]

alphas=np.linspace(0.001, 1.3, num=20)
alphas2=np.arange(1.3, 2, 0.06836842105)
alphas=np.concatenate((alphas,alphas2[1:]))
trials = [0,1]

for talphas in target_alphas:
    tAs=talphas[0]
    tAr=talphas[1]
    
    epoc = np.empty((15,15,5))
    loss = np.empty((15,15,5))
    epoc[:,:,:] = np.nan
    loss[:,:,:] = np.nan

    #Static_tS0.342000_tR0.684000_S0.001000_R0.001000_trial0

    for S in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]:
        for R in [ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]:
            for T in range(talphas[3]):
                if talphas[2]=='_ReLu':
                    T = T+1
                _s = alphas[S]
                _r = alphas[R]
                _des= talphas[2]
                filename = ('Static_tS%lf_tR%lf_S%lf_R%lf_trial%i'%(tAs, tAr, _s,_r,T)+_des)
                epoc[S/2,R/2,T]=np.load('saved_models/'+filename+'/conv_epoch.npy')
                loss[S/2,R/2,T]=np.load('saved_models/'+filename+'/best_loss.npy')
    
    av_epochs = np.nanmean(epoc, axis=2)
    av_losses = np.nanmean(loss, axis=2)
    var_losses = np.nanstd(loss, axis=2)
    var_epochs = np.nanstd(epoc, axis=2)
    
    best_alphas_ind = None
    best_loss = 500
    for _s in range(15):
        for _r in range(15):
            if av_losses[_s,_r] < best_loss:
                best_alphas_ind = [2*_s, 2*_r]
                best_loss = av_losses[_s,_r]       
    best_alphas = [alphas[best_alphas_ind[0]], alphas[best_alphas_ind[1]]]

    """ plot gridsearch results"""

    log_av_losses = np.log(av_losses)

    fig, ax = plt.subplots()
    cax = ax.imshow(log_av_losses, interpolation='none', cmap='OrRd',  extent=[0.001,1.92,1.92, 0.001])
    ax.set_title('log(Loss)')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    _min=np.nanmin(log_av_losses)
    _max=np.nanmax(log_av_losses)
    _med=(_max+_min)*0.5
    cbar = fig.colorbar(cax, ticks=[_min, _med, _max])
    cbar.ax.set_yticklabels(['%5.2f' %((_min)), '%5.2f' %((_med)), '%5.2f' %((_max))])  # vertically oriented colorbar
    ax.set_xlabel('Alpha_R')
    ax.set_ylabel('Alpha_S')
#    ax.scatter(best_alphas[1],best_alphas[0], c='b', marker='x')
    ax.scatter(tAr,tAs,  s=100, facecolors='none', edgecolors='g')
    figurename = 'grid_search_target_s%lf_r%lf'%(tAs, tAr)+_des
    #plt.savefig('figures_results/'+figurename+'.'+ext, format=ext, dpi=DPI)
    plt.close()    





""" Load and plot zoomed gridsearch """

target_alphas = [[0.137282, 0.137230, '']]

alphas=np.linspace(0.001, 0.3, num=20)
trials = [0,1]

for talphas in target_alphas:
    tAs=talphas[0]
    tAr=talphas[1]
    
    epoc = np.empty((10,10,5))
    loss = np.empty((10,10,5))
    epoc[:,:,:] = np.nan
    loss[:,:,:] = np.nan

    #Static_tS0.342000_tR0.684000_S0.001000_R0.001000_trial0

    for S in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]:
        for R in [ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]:
            for T in trials:
                _s = alphas[S]
                _r = alphas[R]
                _des = talphas[2]
                filename = ('Static_tS%lf_tR%lf_S%lf_R%lf_trial%i'%(tAs, tAr, _s,_r,T)+_des)
                epoc[S/2,R/2,T]=np.load('saved_models/'+filename+'/conv_epoch.npy')
                loss[S/2,R/2,T]=np.load('saved_models/'+filename+'/best_loss.npy')
    
    av_epochs = np.nanmean(epoc, axis=2)
    av_losses = np.nanmean(loss, axis=2)
    var_losses = np.nanstd(loss, axis=2)
    var_epochs = np.nanstd(epoc, axis=2)
    best_alphas_ind = None
    best_loss = 500
    for _s in range(10):
        for _r in range(10):
            if av_losses[_s,_r] < best_loss:
                best_alphas_ind = [2*_s, 2*_r]
                best_loss = av_losses[_s,_r]       
    best_alphas = [alphas[best_alphas_ind[0]], alphas[best_alphas_ind[1]]]

    """ plot gridsearch results"""

    log_av_losses = np.log(av_losses)

    fig, ax = plt.subplots()
    cax = ax.imshow(log_av_losses, interpolation='none', cmap='OrRd',  extent=[0.001,0.29,0.29, 0.001])
    ax.set_title('log(Loss)')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    _min=np.nanmin(log_av_losses)
    _max=np.nanmax(log_av_losses)
    _med=(_max+_min)*0.5
    cbar = fig.colorbar(cax, ticks=[_min, _med, _max])
    cbar.ax.set_yticklabels(['%5.2f' %((_min)), '%5.2f' %((_med)), '%5.2f' %((_max))])  # vertically oriented colorbar
    ax.set_xlabel('Alpha_R')
    ax.set_ylabel('Alpha_S')
#    ax.scatter(best_alphas[1],best_alphas[0], c='b', marker='x')
    ax.scatter(tAr,tAs,  s=100, facecolors='none', edgecolors='g')
    figurename = 'grid_search_target_s%lf_r%lf_zoom'%(tAs, tAr)
    #plt.savefig('figures_results/'+figurename+'.'+ext, format=ext, dpi=DPI)
    plt.close()






""" Load and plot GS results with Back Propagated alphas results on top """
alphas=np.linspace(0.001, 1.3, num=20)
alphaS_BP=np.empty((4,2,1,101))
alphaR_BP=np.empty((4,2,1,101))
filename = 'saved_models/'

ii=0
init_alphas=[[2,2],[10,2],[3,14],[12,14]]
#alphas seconda manche 3,10 ; 14
for i in init_alphas:
    S=i[0]
    R=i[1]
    for j in range(1):
        alphaS_BP[ii,0,j,:]=np.reshape(np.load(filename+'LearnAlpha_global_tS0.342000_tR0.684000_S%1.6lf_R%1.6lf_trial%i/learning_alphaS.npy' %(alphas[S], alphas[R], j)), (101))  
        alphaR_BP[ii,0,j,:]=np.reshape(np.load(filename+'LearnAlpha_global_tS0.342000_tR0.684000_S%1.6lf_R%1.6lf_trial%i/learning_alphaR.npy' %(alphas[S], alphas[R], j)), (101))
    ii = ii+1
    
fig1, ax = plt.subplots()
cax = ax.imshow(STD_log_av_losses, interpolation='nearest', cmap='OrRd',  extent=[0.001,1.3,1.3, 0.001])
ax.set_title('log(Loss)')
# Add colorbar, make sure to specify tick locations to match desired ticklabels
_min=np.nanmin(STD_log_av_losses)
_max=np.nanmax(STD_log_av_losses)
_med=(_max+_min)*0.5
cbar = fig1.colorbar(cax, ticks=[_min, _med, _max])
cbar.ax.set_yticklabels(['%5.2f' %_min, '%5.2f' %_med, '%5.2f' %_max])  # vertically oriented colorbar
ax.set_xlabel('Alpha_R')
ax.set_ylabel('Alpha_S')
color_idx = np.linspace(0, 1, 6)
iii=0
ax.scatter(0.684,0.342,  s=100, facecolors='none', edgecolors='g')
for i in range(4):     
    plt.scatter(alphaR_BP[i,0,0,1:], alphaS_BP[i,0,0,1:], marker='x', color=plt.cm.Greens(color_idx[i]))
    plt.scatter(alphaR_BP[i,0,0,0], alphaS_BP[i,0,0,0], marker='o', color=plt.cm.Greens(color_idx[i]))        

figurename = 'grid_search_BackProp_synth'
plt.savefig('figures_results/'+figurename+'.'+ext, format=ext, dpi=DPI)
plt.close()



""" Load and plot GS results with Back Propagated alphas results on top """
alphas=np.linspace(0.001, 1.3, num=20)
alphaS_BP=np.empty((3,2,1,101))
alphaR_BP=np.empty((3,2,1,101))
filename = 'saved_models/'

ii=0
init_alphas=[[4,3],[3,14],[12,14]]
#alphas seconda manche 3,10 ; 14
for i in init_alphas:
    S=i[0]
    R=i[1]
    for j in range(1):
        
        alphaS_BP[ii,0,j,:]=np.reshape(np.load(filename+'LearnAlpha_global_tS0.684000_tR0.342000_S%1.6lf_R%1.6lf_trial%i/learning_alphaS.npy' %(alphas[S], alphas[R], 1)), (101))  
        alphaR_BP[ii,0,j,:]=np.reshape(np.load(filename+'LearnAlpha_global_tS0.684000_tR0.342000_S%1.6lf_R%1.6lf_trial%i/learning_alphaR.npy' %(alphas[S], alphas[R], 1)), (101))
    ii = ii+1
    
fig1, ax = plt.subplots()
cax = ax.imshow(STD_log_av_losses2, interpolation='none', cmap='OrRd',  extent=[0.001,1.3,1.3, 0.001])
ax.set_title('log(Loss)')
# Add colorbar, make sure to specify tick locations to match desired ticklabels
_min=np.nanmin(STD_log_av_losses2)
_max=np.nanmax(STD_log_av_losses2)
_med=(_max+_min)*0.5
cbar = fig1.colorbar(cax, ticks=[_min, _med, _max])
cbar.ax.set_yticklabels(['%5.2f' %_min, '%5.2f' %_med, '%5.2f' %_max])  # vertically oriented colorbar
ax.set_xlabel('Alpha_R')
ax.set_ylabel('Alpha_S')
color_idx = np.linspace(0, 1, 6)
iii=0
ax.scatter(0.342,0.684,  s=100, facecolors='none', edgecolors='g')
for i in range(3):     
    plt.scatter(alphaR_BP[i,0,0,1:], alphaS_BP[i,0,0,1:], marker='x', color=plt.cm.Greens(color_idx[i]))
    plt.scatter(alphaR_BP[i,0,0,0], alphaS_BP[i,0,0,0], marker='o', color=plt.cm.Greens(color_idx[i]))        

figurename = 'grid_search_BackProp_synth2'
#plt.savefig('figures_results/'+figurename+'.'+ext, format=ext, dpi=DPI)
plt.close()











""" plot barplot Elman vs best loss case """

# width of the bars
barWidth = 0.3

#_m1 = best_loss
#s1 = var_losses[best_alphas_ind[0], best_alphas_ind[1]]

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
    _ep=np.nan
    if _l == 4000:
        _l=np.nan
    else:
        _ep=np.load('saved_models/'+filename+'/conv_epoch.npy')
    lossElman[T]=_l
    epocElman[T]=_ep
    
    filename = ('Static_tS0.342000_tR0.684000_S%lf_R%lf_trial%i_400ep' %(_s,_r,T))
    #epocElmanEp[T]=np.load('saved_models/'+filename+'/conv_epoch.npy')
    _l=np.load('saved_models/'+filename+'/best_loss.npy')
    _ep=np.nan
    if _l == 4000:
        _l=np.nan
    else:
        _ep=np.load('saved_models/'+filename+'/conv_epoch.npy')
    lossElmanEp[T]=_l
    epocElman[T]=_ep
    
    filename = ('LearnAlpha_global_tS0.342000_tR0.684000_trial%i_BP_random_init' %(T))
    #epocLearn[T]=np.load('saved_models/'+filename+'/conv_epoch.npy')
    _l=np.load('saved_models/'+filename+'/best_loss.npy')
    _ep=np.nan
    if _l == 4000:
        _l=np.nan
    else:
        _ep=np.load('saved_models/'+filename+'/conv_epoch.npy')
    lossLearn[T]=_l
    epocLearn[T]=_ep
    
    
    filename = ('LearnAlpha_global_tS0.342000_tR0.684000_trial%i_BP_random_init_400ep' %(T))
    #epocLearnEp[T]=np.load('saved_models/'+filename+'/conv_epoch.npy')
    _l=np.load('saved_models/'+filename+'/best_loss.npy')
    _ep=np.nan
    if _l == 4000:
        _l=np.nan
    else:
        _ep=np.load('saved_models/'+filename+'/conv_epoch.npy')        
    lossLearnEp[T]=_l
    epocLearnEp[T]=_ep
    
AvEpocElman = np.nanmean(epocElman)
_m2 = np.nanmean(lossElman)
s2 = np.nanstd(lossElman, ddof=1)/np.sqrt(np.sum(1-np.isnan(lossElman)))
varEpocElman = np.nanstd(epocElman, ddof=1)

AvEpocLearn = np.nanmean(epocLearn)
_m1 = np.nanmean(lossLearn)
s1 = np.nanstd(lossLearn, ddof=1)/np.sqrt(np.sum(1-np.isnan(lossLearn)))
varEpocLearn = np.nanstd(epocLearn, ddof=1)

# compute p-value
#def numberOfNonNans(data):
#    count = 0
#    for i in data:
#        if not np.isnan(i):
#            count += 1
#    return count 
#
#N1=numberOfNonNans(lossLearn)
#N2=numberOfNonNans(lossElman)
#s = np.sqrt( (s1**2/N1) + (s2**2/N2) )
#t = abs((_m1-_m2) / s)
#dof = (((s1**2/N1)+(s2**2/N2))**2) / ( ((s1**2/N1)**2/(N1-1)) + ((s2**2/N2)**2/(N2-1)) )
#p = 1 - stats.t.cdf(t,df=dof)
t,p = stats.ttest_ind(lossElman, lossLearn, equal_var=False, nan_policy='omit')

m1=1.0e-04
m2=_m2-_m1+1.0e-04

# The x position of bars
r1 = [0.25, 0.75]
color = '#8B1A1A'
# Create blue bars
plt.figure(figsize=[4.2,4])
plt.bar(r1, [_m1,_m2], width = barWidth, color = '#8B1A1A', edgecolor = 'black', yerr=[s1,s2], capsize=7)

# general layout
plt.xlim(-0.4,1.4)
plt.ylim(0,0.0015)
plt.xticks([0.25,0.75],['DRU','Elman'])
plt.ylabel('Loss')
#plt.ylim(ymax=0.002)
plt.xlabel('')
plt.ticklabel_format(axis='y', style='sci',scilimits=(-2,2))
# plt.title('T-test performance: p = %.2E' %p)
#plt.legend(loc=0)
figurename = 'loss_elman_vs_DRU_bartdata'
plt.tight_layout()
# plt.savefig('figures_results/'+figurename+'.'+ext, format=ext, dpi=DPI)
# plt.close()



AvEpocElmanEp = np.nanmean(epocElmanEp)
_m2 = np.nanmean(lossElmanEp)
s2 = np.nanstd(lossElmanEp, ddof=1)
varEpocElman = np.nanstd(epocElmanEp, ddof=1)

AvEpocLearnEp = np.nanmean(epocLearnEp)
_m1 = np.nanmean(lossLearnEp)
s1 = np.nanstd(lossLearnEp, ddof=1)
varEpocLearnEp = np.nanstd(epocLearnEp, ddof=1)

t,p = stats.ttest_ind(lossElmanEp, lossLearnEp, equal_var=False, nan_policy='omit')

m1=1.0e-04
m2=_m2-_m1+1.0e-04

# The x position of bars
r1 = [0, 1]
color = '#8B1A1A'
# Create blue bars
plt.figure()

plt.bar(r1, [m1,m2], width = barWidth, color = '#8B1A1A', edgecolor = 'black', yerr=[s1,s2], capsize=7)

# general layout
plt.xticks([])
plt.yticks([m1, m2], [round(_m1,7), round(_m2,7)])
plt.ylabel('Mean loss')
#plt.ylim(ymax=0.002)
plt.xlabel('')
plt.title('T-test performance: p = %.2E' %p)
#plt.legend(loc=0)
figurename = 'loss_elman_vs_DRU_bartdata_400ep'
#plt.savefig('figures_results/'+figurename+'.'+ext, format=ext, dpi=DPI)
# plt.close()







""" Baseline: plot barplot H0 (straight line signal) vs best loss case """

#??



