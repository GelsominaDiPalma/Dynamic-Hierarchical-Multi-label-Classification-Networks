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

datasets_names = ["gasch1_FUN"]

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
    
    # Create folder if it does not exist
    if not os.path.exists('results_per_level_plots'):
            os.makedirs('results_per_level_plots')

    values = list(mean_results_per_level)

    fig, ax = plt.subplots()
    ax.bar(keys, values, color="tab:blue")
    ax.set_title(dataset_name)
    ax.set_ylabel('mean AUPRC per level')
    ax.set_xlabel('levels')

    #increase tick frequency on y axis
    _ = plt.yticks(np.arange(round(min(mean_results_per_level),3), round(max(mean_results_per_level),3)+0.06, 0.05))

    if "GO" in dataset_name:
        #adjust tick size
        _ = plt.xticks(fontsize=6)

    plt.savefig('results_per_level_plots/'+dataset_name+'.pdf', format='pdf', bbox_inches='tight')

    plt.savefig('results_per_level_plots/'+dataset_name+'.png', format='png', bbox_inches='tight')
