# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:54:31 2022

@author: garru
"""

import numpy as np
import os
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

# =============================================================================
# COMPUTE AUROC
# =============================================================================

def computeAUROC(predictions_ind, predictions_ood):

    # 1 si se decide que es InD y 0 si es OoD

    truePositives = np.sum(predictions_ind,axis=1)
    falsePositives = np.sum(predictions_ood,axis=1)

    tpr_values = truePositives/len(predictions_ind[0])
    fpr_values = falsePositives/len(predictions_ood[0])

    tpr_values = np.append(tpr_values, 1)    
    fpr_values = np.append(fpr_values, 1)
    
    auroc = np.trapz(tpr_values, fpr_values)
    return auroc

results = {}
predicciones_OoD = {}

datasets = ['ojos','rps','pinturas']#,'plantas']

for dataset in datasets:
    
    results[dataset]={}
    predicciones_OoD[dataset] = {}
    
    # =============================================================================
    # LEO PREDICCIONES
    # =============================================================================
    
    allModelsFiles = os.listdir('./OoD/'+dataset+'/')
    allModelsFiles = [f for f in allModelsFiles if '_ood_' in f]
    index = 0
    
    for fileName in allModelsFiles:
    
        fid = open('./OoD/'+dataset+'/'+fileName,'r')
            
        for line in fid:
                
            lineSplit = line.strip().split(';')
            predicciones_OoD[dataset][index] = {'OoD':[],'acc':None,'neurons':None,'auroc':None,'InD':[]}
            for indexOoD in range(len(lineSplit)):
                if lineSplit[indexOoD]!='':
                    predicciones_OoD[dataset][index]['OoD'].append(list(map(int,[label for label in lineSplit[indexOoD]])))
            predicciones_OoD[dataset][index]['acc'] = 1-float(fileName.split('_')[3])
            predicciones_OoD[dataset][index]['neurons'] = float(fileName.split('_')[4])
            predicciones_OoD[dataset][index]['auroc'] = 1-float(fileName.split('_')[5][:-4])
        
        fid.close()
        
        fileName_ind = fileName.replace('ood', 'ind')
        
        fid = open('./OoD/'+dataset+'/'+fileName_ind,'r')
            
        for line in fid:
                
            lineSplit = line.strip().split(';')
            for indexOoD in range(len(lineSplit)):
                if lineSplit[indexOoD]!='':
                    predicciones_OoD[dataset][index]['InD'].append(list(map(int,[label for label in lineSplit[indexOoD]])))

        fid.close()
        
        index = index+1        

# =============================================================================
# VEMOS DISTRIBUCION DE OBJ1 y OBJ3
# =============================================================================

# import seaborn as sns

allAccs = {}
allAurocs = {}

for dataset in datasets:

    allAccs[dataset] = [predicciones_OoD[dataset][index]['acc'] for index in range(len(predicciones_OoD[dataset]))]
    allAurocs[dataset] = [predicciones_OoD[dataset][index]['auroc'] for index in range(len(predicciones_OoD[dataset]))]

from matplotlib import pyplot as plt

# fig, axs = plt.subplots(1,3)
index_col = 0

prediccionesEnsemble_OoD = {}

for dataset in datasets:
    
    prediccionesEnsemble_OoD[dataset]={}
    
    # sns.kdeplot(allAccs[dataset], allAurocs[dataset], color='b', shade=True, cmap="Blues", shade_lowest=False,ax=axs[index_col])
    # axs[index_col].scatter(allAccs[dataset], allAurocs[dataset], s=50,marker='s')
    # axs[index_col].set_xlabel('Accuracy')
    # axs[index_col].set_ylabel('AUROC')
    # axs[index_col].set_title('Dataset: '+dataset)
    
    for (minP,maxP) in [(50,60),(55,65),(60,70),(65,75),(70,80),(75,85),(80,90),(85,95)]:
        
        prediccionesEnsemble_OoD[dataset][(minP,maxP)]={'individuals_ind':[],'individuals_ood':[],'ensemble_ind':[],'ensemble_ood':[], 'aurocEnsemble':None,'aurocIndividuals':[]}
        
        minAUROC = np.percentile(allAurocs[dataset], minP)
        maxAUROC = np.percentile(allAurocs[dataset], maxP)
        
        selectedIndices = [index for index in range(len(predicciones_OoD[dataset])) if predicciones_OoD[dataset][index]['auroc']>=minAUROC and predicciones_OoD[dataset][index]['auroc']<=maxAUROC]
        # axs[index_col].axhline(minAUROC,0,1,c='k',lw=2)
        # axs[index_col].axhline(maxAUROC,0,1,c='k',lw=2)
        # axs[index_col].text(min(allAccs[dataset]),0.5*(minAUROC+maxAUROC),str(len(selectedIndices)),verticalalignment='center')
        
        for indexSelected in selectedIndices:
            prediccionesEnsemble_OoD[dataset][(minP,maxP)]['individuals_ind'].append(predicciones_OoD[dataset][indexSelected]['InD'])
            prediccionesEnsemble_OoD[dataset][(minP,maxP)]['individuals_ood'].append(predicciones_OoD[dataset][indexSelected]['OoD'])
            prediccionesEnsemble_OoD[dataset][(minP,maxP)]['aurocIndividuals'].append(predicciones_OoD[dataset][indexSelected]['auroc'])

        prediccionesEnsemble_OoD[dataset][(minP,maxP)]['ensemble_ind'] = np.apply_along_axis(lambda x:np.argmax(np.bincount(x)), axis=0, arr=prediccionesEnsemble_OoD[dataset][(minP,maxP)]['individuals_ind'])
        prediccionesEnsemble_OoD[dataset][(minP,maxP)]['ensemble_ood'] = np.apply_along_axis(lambda x:np.argmax(np.bincount(x)), axis=0, arr=prediccionesEnsemble_OoD[dataset][(minP,maxP)]['individuals_ood'])
        
        prediccionesEnsemble_OoD[dataset][(minP,maxP)]['aurocEnsemble'] = computeAUROC(prediccionesEnsemble_OoD[dataset][(minP,maxP)]['ensemble_ind'],prediccionesEnsemble_OoD[dataset][(minP,maxP)]['ensemble_ood'])
        
                
    index_col = index_col + 1
    

# =============================================================================
# PLOT SAME PLOTS THAN IN ACC
# =============================================================================

fig, axs = plt.subplots(1,3)

indexDataset = 0

for dataset in datasets:

    indexP = 0
    
    labelsXticks = []
    
    Qs = list(prediccionesEnsemble_OoD[dataset].keys())
    
    for (minP,maxP) in Qs:
        
        if minP==Qs[-1][0] and maxP==Qs[-1][1] and dataset == datasets[-1]:
    
            bpplot = axs[indexDataset].boxplot(prediccionesEnsemble_OoD[dataset][(minP,maxP)]['aurocIndividuals'],positions=[indexP],widths=0.45, patch_artist=True, boxprops=dict(facecolor="lightcoral"))
            maxplot = axs[indexDataset].scatter(indexP+1,max(prediccionesEnsemble_OoD[dataset][(minP,maxP)]['aurocIndividuals']),c='lightcoral',s=50,marker='s',label='Max individual learner accuracy',edgecolor='k')
            ensembleaccplot = axs[indexDataset].scatter(indexP+2,prediccionesEnsemble_OoD[dataset][(minP,maxP)]['aurocEnsemble'],c='lightcoral',s=80,marker='*',label='AUROC of the ensemble',edgecolor='k')
            
        else:
            axs[indexDataset].boxplot(prediccionesEnsemble_OoD[dataset][(minP,maxP)]['aurocIndividuals'],positions=[indexP],widths=0.45, patch_artist=True, boxprops=dict(facecolor="lightcoral"))
            axs[indexDataset].scatter(indexP+1,max(prediccionesEnsemble_OoD[dataset][(minP,maxP)]['aurocIndividuals']),c='lightcoral',s=50,marker='s',edgecolor='k')
            axs[indexDataset].scatter(indexP+2,prediccionesEnsemble_OoD[dataset][(minP,maxP)]['aurocEnsemble'],c='lightcoral',s=80,marker='*',edgecolor='k')
            
        axs[indexDataset].axvline(indexP+3,0,1)
        indexP = indexP+4
        labelsXticks.append(str((minP,maxP)))
    
    axs[indexDataset].set_xticks([4*i+1 for i in range(len(prediccionesEnsemble_OoD[dataset].keys()))])
    axs[indexDataset].set_xticklabels(labelsXticks)
    axs[indexDataset].set_xlim(-1,indexP-1)
    axs[indexDataset].set_xlabel('Quantiles for minimum and\n maximum individual AUROC',fontsize=18)
    axs[indexDataset].set_ylabel('AUROC',fontsize=18)
    # axs[indexDataset].set_title('Dataset: '+dataset)
    indexDataset = indexDataset + 1

plt.figlegend([bpplot["boxes"][0], maxplot, ensembleaccplot], ['Distribution of individual AUROC values', 'Maximum individual AUROC', 'AUROC of the ensemble'],loc='upper center', fancybox=True, shadow=True, fontsize=16)
plt.show()
