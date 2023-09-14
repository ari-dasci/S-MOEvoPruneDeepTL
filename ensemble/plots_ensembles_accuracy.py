# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:54:31 2022

@author: garru
"""

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

results = {}

datasets = ['ojos','rps','pinturas']#,'plantas']

for dataset in datasets:
    
    results[dataset]={}
    
    # =============================================================================
    # LEO PREDICCIONES
    # =============================================================================
    
    fid = open('./PREDICTIONS/predicciones_'+dataset+'.txt','r')
    
    predicciones = {} 
    
    index = 0
    
    for line in fid:
            
        lineSplit = line.strip().split(';')
        predicciones[index] = {'inD':[],'OoD':[],'acc':None,'neurons':None,'auroc':None}
        predicciones[index]['inD'] = list(map(int,[label for label in lineSplit[0]]))
        for indexOoD in range(len(lineSplit)-4):
            predicciones[index]['OoD'].append(list(map(int,[label for label in lineSplit[1+indexOoD]])))
        predicciones[index]['acc'] = 1-float(lineSplit[-3])
        predicciones[index]['neurons'] = float(lineSplit[-2])
        predicciones[index]['auroc'] = 1-float(lineSplit[-1])
        index = index+1
        
    fid.close()
    
    # =============================================================================
    # LEO REAL
    # =============================================================================
    
    fid = open('./PREDICTIONS/reales_'+dataset+'.txt','r')
    
    Ytrue = []
    
    for line in fid:
            
        lineSplit = line.strip()
        Ytrue = list(map(int,[label for label in lineSplit]))
        
    fid.close()
    
    Ytrue = np.array(Ytrue)
    
    # =============================================================================
    # COMPUTE ENSEMBLE
    # =============================================================================
    
    allAccs = [predicciones[index]['acc'] for index in range(len(predicciones))]
    
    for (minP,maxP) in [(50,60),(55,65),(60,70),(65,75),(70,80),(75,85),(80,90),(85,95)]:
    
        minAcc = np.percentile(allAccs, minP)
        maxAcc = np.percentile(allAccs, maxP)
        
        selectedIndices = [index for index in range(len(predicciones)) if predicciones[index]['acc']>=minAcc and predicciones[index]['acc']<=maxAcc]
        
        prediccionesEnsemble = []
        
        for indexSelected in selectedIndices:
            prediccionesEnsemble.append(predicciones[indexSelected]['inD'])
            
        maj_vote = np.apply_along_axis(lambda x:np.argmax(np.bincount(x)), axis=0, arr=prediccionesEnsemble)
        
        results[dataset][(minP,maxP)]={'predictionsEnsemble':maj_vote,'accEnsemble':sum(maj_vote==Ytrue)/len(Ytrue),'accModels':[predicciones[indexSelected]['acc'] for indexSelected in selectedIndices]}
        
        print('*'*50)
        print(dataset)
        print('*'*50)
        print('-'*50)
        print('Minimum Accuracy',minAcc,'corresponding to percentile',minP)
        print('Max Accuracy',maxAcc,'corresponding to percentile',maxP)
        print('Maximum accuracy of selected models:',max([predicciones[indexSelected]['acc'] for indexSelected in selectedIndices]))
        print('Ensemble accuracy:',sum(maj_vote==Ytrue)/len(Ytrue))
        
# =============================================================================
# Depict results
# =============================================================================

from matplotlib import pyplot as plt

fig, axs = plt.subplots(1,3)

indexDataset = 0

for dataset in datasets:

    indexP = 0
    
    labelsXticks = []
    
    Qs = list(results[dataset].keys())
    
    for (minP,maxP) in Qs:
        
        if minP==Qs[-1][0] and maxP==Qs[-1][1] and dataset == datasets[-1]:
    
            bpplot = axs[indexDataset].boxplot(results[dataset][(minP,maxP)]['accModels'],positions=[indexP],widths=0.45, patch_artist=True, boxprops=dict(facecolor="C0"))
            maxplot = axs[indexDataset].scatter(indexP+1,max(results[dataset][(minP,maxP)]['accModels']),c='C0',s=50,marker='s',label='Max individual learner accuracy',edgecolor='k')
            ensembleaccplot = axs[indexDataset].scatter(indexP+2,results[dataset][(minP,maxP)]['accEnsemble'],c='C0',s=100,marker='*',label='Accuracy of the ensemble',edgecolor='k')
            
        else:
            axs[indexDataset].boxplot(results[dataset][(minP,maxP)]['accModels'],positions=[indexP],widths=0.45, patch_artist=True, boxprops=dict(facecolor="C0"))
            axs[indexDataset].scatter(indexP+1,max(results[dataset][(minP,maxP)]['accModels']),c='C0',s=50,marker='s',edgecolor='k')
            axs[indexDataset].scatter(indexP+2,results[dataset][(minP,maxP)]['accEnsemble'],c='C0',s=100,marker='*',edgecolor='k')
            
        axs[indexDataset].axvline(indexP+3,0,1)
        indexP = indexP+4
        labelsXticks.append(str((minP,maxP)))
    
    axs[indexDataset].set_xticks([4*i+1 for i in range(len(results[dataset].keys()))])
    axs[indexDataset].set_xticklabels(labelsXticks)
    axs[indexDataset].set_xlim(-1,indexP-1)
    axs[indexDataset].set_xlabel('Quantiles for minimum and\n maximum individual accuracy',fontsize=18)
    axs[indexDataset].set_ylabel('Accuracy',fontsize=18)
    # axs[indexDataset].set_title('Dataset: '+dataset)
    indexDataset = indexDataset + 1

plt.figlegend([bpplot["boxes"][0], maxplot, ensembleaccplot], ['Distribution of individual accuracies', 'Maximum individual accuracy in ensemble', 'Accuracy of the ensemble'],loc='upper center', fancybox=True, shadow=True, fontsize=16)
plt.show()