# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 22:51:59 2022

@author: garru
"""
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from matplotlib import pyplot as plt
import os
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

def fuseGradCam(img,heatmap,alpha=0.4,colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, colormap)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img

from matplotlib import cm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
# plt.style.use('default')

# =============================================================================
# PARAMS
# =============================================================================

DATASET_TARGET = 'srsmas'
imageFiles = os.listdir('./toPlot - '+DATASET_TARGET + '/')
TOPK = 10

# =============================================================================
# FIN PARAMS
# =============================================================================

# =============================================================================
# READ DATA
# =============================================================================

fileNames = os.listdir('./')

objectives = {}
soluciones = {}

for filename in fileNames:
    if 'txt' in filename and 'soluciones' not in filename:
        dataset = filename.split('_')[-1].split('.')[0]
        objectives[dataset] = np.loadtxt(filename)
    elif 'txt' in filename and 'soluciones' in filename:
        dataset = filename.split('_')[2]
        soluciones[dataset] = {'soluciones':np.loadtxt(filename)}
        

# =============================================================================
# Contabilizamos apariciones de las neuronas y pintamos histogramas
# =============================================================================

for dataset in objectives.keys():
    
    freqs = np.sum(soluciones[dataset]['soluciones'],axis=0)/len(soluciones[dataset]['soluciones'])
    soluciones[dataset]['freqs'] = freqs
    
    soluciones[dataset]['topk-freqs'] = []
    
    for neuron in np.argsort(freqs)[::-1][0:TOPK]:
        
        soluciones[dataset]['topk-freqs'].append((neuron,freqs[neuron]))

    soluciones[dataset]['objectives_topk'] = []

    for neuron in np.argsort(freqs)[::-1][0:TOPK]:
        objectives_neuron = [objectives[dataset][instance] for instance in range(len(objectives[dataset])) if soluciones[dataset]['soluciones'][instance,neuron]==1.]
        soluciones[dataset]['objectives_topk'].append(objectives_neuron)

TARGETNEURONS = [entry[0] for entry in soluciones[DATASET_TARGET]['topk-freqs'][:TOPK]]

# =============================================================================
# COMPUTE GRADCAM++
# =============================================================================

# grid = plt.GridSpec(3+len(imageFiles), 2+2*TOPK)

# ax_bars = plt.subplot(grid[0:3,1:10])
# ax_boxplots = plt.subplot(grid[0:3,12:21])

fig, ax_bars = plt.subplots(1,1,tight_layout=True)

colors_bars = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']

ax_bars.bar(range(TOPK),[entry[1] for entry in soluciones[DATASET_TARGET]['topk-freqs']],align='center',color=colors_bars,edgecolor='k')
# ax_bars.set_title(DATASET_TARGET)
ax_bars.set_xlabel('Index of relevant neuron',fontsize=18)
ax_bars.set_ylabel('Relative frequency in\n agg. Pareto front',fontsize=18)
ax_bars.set_xticks(range(TOPK))
ax_bars.set_xticklabels([entry[0] for entry in soluciones[DATASET_TARGET]['topk-freqs']])
ax_bars.set_ylim(0,1)
ax_bars.set_xlim(-0.6,9.6)

plt.savefig('XAI_'+DATASET_TARGET + '_bars.pdf')  

fig, ax_boxplots = plt.subplots(1,1,tight_layout=True)

i1 = -0.3
i2 = 0
i3 = 0.3
ticks = []

for ineuron in range(TOPK):
    ax_boxplots.boxplot(np.array(soluciones[DATASET_TARGET]['objectives_topk'][ineuron])[:,0],positions=[i1],patch_artist=True,boxprops=dict(facecolor='gray', color='k'))
    ax_boxplots.boxplot(np.array(soluciones[DATASET_TARGET]['objectives_topk'][ineuron])[:,1],positions=[i2],patch_artist=True,boxprops=dict(facecolor='gray', color='k'))
    ax_boxplots.boxplot(np.array(soluciones[DATASET_TARGET]['objectives_topk'][ineuron])[:,2],positions=[i3],patch_artist=True,boxprops=dict(facecolor='gray', color='k'))
    if ineuron<TOPK-1:
        ax_boxplots.axvline(i3+0.2,0,1)
    
    ticks.extend([i2])

    i1 = i1+1
    i2 = i2+1
    i3 = i3+1
    
# ax_boxplots.set_xticklabels(['Accuracy','% neurons','AUROC']*int(len(ticks)/3),rotation=90,fontsize=12)
ax_boxplots.set_ylabel('Distribution of objectives among\n solutions with active neuron',fontsize=18)
ax_boxplots.set_xlim(-0.6,9.6)
    
ax_boxplots.set_xticks(ticks)
ax_boxplots.set_xticklabels([entry[0] for entry in soluciones[DATASET_TARGET]['topk-freqs']])
ax_boxplots.set_xlabel('Index of relevant neuron',fontsize=18)
ax_boxplots.set_xlim(-0.5,i3-1+0.2)
ax_boxplots.set_xlim(-0.5,i3-1+0.2)

plt.savefig('XAI_'+DATASET_TARGET + '_boxplots.pdf')  



fig = plt.figure(tight_layout=True)
grid = plt.GridSpec(len(imageFiles), 1+TOPK)

indexFile = 0

for imageFile in imageFiles:

    image = cv2.imread('./toPlot - '+DATASET_TARGET + '/'+imageFile)
    gray = 0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]
    imageExpanded = np.expand_dims(image, axis=0)
    imageInput = preprocess_input_resnet(imageExpanded)

    base_model = keras.applications.ResNet50(include_top=False, input_shape=np.shape(image), weights="imagenet", pooling="avg")

    # Create GradCAM++ object
    gradcam = GradcamPlusPlus(base_model,
                               model_modifier=ReplaceToLinear(),
                               clone=True)
    
    ax_image_original = plt.subplot(grid[indexFile,0])
    ax_image_original.imshow(image[:,:,::-1])
    ax_image_original.set_xticks([])
    ax_image_original.set_yticks([])
    # ax_image_original.set_title('Original')

    for n in range(len(TARGETNEURONS)):
        
        # Generate cam with GradCAM++
        cam = gradcam(CategoricalScore(TARGETNEURONS[n]),imageInput,'conv5_block3_out')        
        heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
        ax_heatmap = plt.subplot(grid[indexFile,1+n])
        ax_heatmap.imshow(gray,cmap='gray', vmin=0, vmax=255)
        ax_heatmap.imshow(heatmap, cmap='jet', alpha=0.7) # overlay
        ax_heatmap.set_xticks([])
        ax_heatmap.set_yticks([])
        [ax_heatmap.spines[i].set_linewidth(3) for i in ax_boxplots.spines]
        [ax_heatmap.spines[i].set_edgecolor(colors_bars[n]) for i in ax_boxplots.spines]
        
    indexFile = indexFile + 1

fig = plt.gcf()
fig.set_size_inches([11.09,  8.22])
plt.subplots_adjust(top=0.957,
bottom=0.024,
left=0.014,
right=0.986,
hspace=0.2,
wspace=0.0)

plt.savefig('XAI_'+DATASET_TARGET + '_heatmaps.pdf')  
    
# =============================================================================
# https://keisen.github.io/tf-keras-vis-docs/examples/attentions.html#SmoothGrad
# =============================================================================
# from tf_keras_vis.saliency import Saliency      
# saliency = Saliency(base_model,
#                     model_modifier=ReplaceToLinear(),
#                     clone=True)

# cam = saliency(score_function,imageInput,smooth_samples=20, # The number of calculating gradients iterations.
#                         smooth_noise=0.20)
