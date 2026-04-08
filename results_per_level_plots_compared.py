import os
import importlib
os.environ["DATA_FOLDER"] = "./"

import argparse

import time

import torch
import torch.utils.data
import torch.nn as nn

import random

from utils.parser import *
from utils import datasets

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, jaccard_score

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc

import ast
import csv

datasets_names = ["cellcycle_FUN"]

for dataset_name in datasets_names:

    results_per_level = [] #list of lists, one per seed

    # already sorted list of the levels we have a non Nan result for (see GO)
    keys = []

    with open("results_per_level/"+dataset_name+".csv", 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if idx==0:
                keys = line.split(",")[2:-1] # skip seed, epoch and \n and take the levels (already strings and sorted)
            else:
                if len(line.strip()) != 0: # for all non empty lines
                    nums = line.split(",")[2:-1] # skip seed, epoch and \n
                    nums = [float(numeric_string) for numeric_string in nums]
                    results_per_level.append(nums)

    results_per_level = np.array(results_per_level)

    mean_results_per_level = np.mean(results_per_level, axis=0)

    print(dataset_name + " " + str(mean_results_per_level))

    # read other results per level files:

    results_per_level_chmcnn = []

    with open("results_per_level_C_HMCNN/"+dataset_name+".csv", 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if len(line.strip()) != 0: # for all non empty lines
                nums = line.split(",")[:-1] # skip \n
                nums = [float(numeric_string) for numeric_string in nums]
                results_per_level_chmcnn.append(nums)

    results_per_level_chmcnn = np.array(results_per_level_chmcnn)

    mean_results_per_level_chmcnn = np.mean(results_per_level_chmcnn, axis=0)

    print(list(mean_results_per_level_chmcnn))

    #Clus:
    results_per_level_Clus = []
    with open("results_per_level_Clus/"+dataset_name+".csv", 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if idx!=0:
                if len(line.strip()) != 0: # for all non empty lines
                    nums = line.split(",")[:-1] # skip \n
                    nums = [float(numeric_string) for numeric_string in nums]
                    results_per_level_Clus = nums

    print(results_per_level_Clus)

    #HMC_LMLP
    results_per_level_HMC_LMLP = []
    with open("results_per_level_HMC_LMLP/"+dataset_name+".csv", 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if len(line.strip()) != 0: # for all non empty lines
                nums = line.split(",")[:-1] # skip \n
                nums = [float(numeric_string) for numeric_string in nums]
                results_per_level_HMC_LMLP = nums

    print(results_per_level_HMC_LMLP)

    all_results = {
        'DC_HMCNN' : list(mean_results_per_level),
        'C-HMCNN' : list(mean_results_per_level_chmcnn),
        'Clus-Ens' : results_per_level_Clus,
        'HMC_LMLP' : results_per_level_HMC_LMLP
    }

    
    # Create folder if it does not exist
    if not os.path.exists('results_per_level_plots_compared'):
            os.makedirs('results_per_level_plots_compared')

    x = np.arange(len(keys))  # the label locations
    width = 0.20  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange']
    hatches = ['xxxxx', '/////', 'ooooo', '+++++']
    iter = 0

    for attribute, measurement in all_results.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color='none', edgecolor=colors[iter], hatch=hatches[iter])
        #ax.bar_label(rects, padding=3)
        multiplier += 1
        iter+=1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(dataset_name)
    ax.set_ylabel('AU(PRC)')
    ax.set_xlabel('Levels')
    ax.set_xticks(x + width, keys)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, max(list(mean_results_per_level)+results_per_level_HMC_LMLP+results_per_level_Clus) + 0.08)

    plt.savefig('results_per_level_plots_compared/'+dataset_name+'.pdf', format='pdf', bbox_inches='tight')

    plt.savefig('results_per_level_plots_compared/'+dataset_name+'.png', format='png', bbox_inches='tight')
