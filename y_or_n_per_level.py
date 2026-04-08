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
from utils import parser_ontology
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
    # Load train, val and test set
    data = dataset_name.split('_')[0]
    ontology = dataset_name.split('_')[1]

    # Load the datasets
    if ('others' in dataset_name):
        _, test = initialize_other_dataset(dataset_name, datasets)
    elif ('ontology' in dataset_name):
        dataset_info = {}
        dataset_info['random_seed'] = 0
        dname = dataset_name.split('_')[0]
        dataset_info['folderpath'] = 'HMC_data/ontology/'+dname     
        _, _, test = parser_ontology.initialize_DBPedia_dataset(dataset_info)
    else:
        _, _, test = initialize_dataset(dataset_name, datasets)

    g = nx.DiGraph(test.A)
    # see main.py
    inv_g = g.reverse()
    generations = [sorted(generation) for generation in nx.topological_generations(inv_g)]

    num_predictions, _ = test.Y.shape
    # we have one row per prediction (sample) and one column per class.
    Y = test.Y # np array
    # compute the mean of the 1 values occurrences for each column (=class).
    Y_mean = np.mean(Y, axis=0)

    print(Y_mean)

    y_or_n = []

    with open("y_or_n/"+dataset_name+".csv", 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if len(line.strip()) != 0: # for all non empty lines
                nums = line.split(",")[1:-1] # skip seed and \n
                nums = [float(numeric_string) for numeric_string in nums]
                nums[:] = [x / num_predictions for x in nums]
                y_or_n.append(nums) # create a list fo 10 (one per seed) lists. Each inner list contains the percentage of the "y over n" quantity for each class.

    y_or_n_lev = {}

    # at the end of this loop I want y_or_n_lev to be a dictionary with an entry per level. Each entry will be a list where each element
    # is the sum of the "y_or_n" quantity for all the classes of that level FOR ONE SEED. Later, when I print / plot the y_or_n per level for 
    # each entry of the dictionary I will take the mean of the values in the list.

    for idx2, y_or_n_seed in enumerate(y_or_n):

        evaluated = 0
        y_or_n_lev_temp = {} #for each seed I use a temporary dictionary, otherwise I have problems in the management of new seeds and new levels per seed (if the levels are not presented in order).

        for idx, to_evaluate in enumerate(test.to_eval):

            for gen_idx, gen in enumerate(generations):
                if idx in gen:
                    level = gen_idx

            if to_evaluate == 1:

                if not np.isnan(y_or_n_seed[evaluated]):

                    # for each seed I always work with the temporary dictionary (which is initialized at every iteration of the outer loop)
                    if level in y_or_n_lev_temp.keys(): 
                        y_or_n_lev_temp[level].append(y_or_n_seed[evaluated])
                    else:
                        y_or_n_lev_temp[level] = [y_or_n_seed[evaluated]]

                evaluated += 1

        # I put the correct results in the final dictionary (summing the values for each entry).            
        for key in y_or_n_lev_temp:
            if idx2==0: # every level will be a new entry so I need to create the list
                y_or_n_lev[key] = [np.mean(y_or_n_lev_temp[key])]
            else: # for the other seeds I just append the new sum
                y_or_n_lev[key].append(np.mean(y_or_n_lev_temp[key]))


    # Create folder if it does not exist
    if not os.path.exists('y_or_n_per_level'):
            os.makedirs('y_or_n_per_level')

    # add ground truth plot:
    # in the end this dictionary will have one entry per level, each entry contains a list of numbers where each number is the sum of 1 values 
    # over all predictions for that specific class of the level.
    # In the end we plot the mean of all these lists
    ground_truths = {}

    #evaluated_gt = 0

    for idx_gt, to_evaluate_gt in enumerate(test.to_eval):

        for gen_idx, gen in enumerate(generations):
            if idx_gt in gen:
                level_gt = gen_idx
        #level_gt = len(list(nx.descendants(g, idx_gt)))

        if to_evaluate_gt == 1:
            
            # directly use idx_gt, Y is "raw"
            if not np.isnan(Y_mean[idx_gt]):

                if level_gt in ground_truths.keys(): 
                    ground_truths[level_gt].append(Y_mean[idx_gt])
                else:
                    ground_truths[level_gt] = [Y_mean[idx_gt]]


    # sort the keys (levels)
    keys = list(y_or_n_lev.keys())
    keys.sort()
    sorted_y_or_n_lev = {i: y_or_n_lev[i] for i in keys}
    keys = [str(lev) for lev in keys]
    #print(keys)

    keys_gt = list(ground_truths.keys())
    keys_gt.sort()
    sorted_ground_truths = {i: ground_truths[i] for i in keys_gt}
    keys_gt = [str(lev) for lev in keys_gt]
    print(keys_gt)

    # in the final file we will have only one line (with the average per level among all the seeds)
    with open('y_or_n_per_level/'+dataset_name+'.csv', 'w') as file:
        for key in sorted_y_or_n_lev.keys():
            file.write(str(np.mean(sorted_y_or_n_lev[key])) + ",")
            # substitute the list with the mean value for each dict entry
            sorted_y_or_n_lev[key] = np.mean(sorted_y_or_n_lev[key])
        file.write("\n")

    with open('y_or_n_per_level/'+dataset_name+'_ground_truths.csv', 'w') as file:
        for key in sorted_ground_truths.keys():
            file.write(str(np.mean(sorted_ground_truths[key])) + ",")
            # substitute the list with the mean value for each dict entry
            sorted_ground_truths[key] = np.mean(sorted_ground_truths[key])
        file.write("\n")

    values = list(sorted_y_or_n_lev.values())
    #print(values)

    values_gt = list(sorted_ground_truths.values())
    print(values_gt)

    fig, ax = plt.subplots()
    ax.bar(keys, values, color="tab:orange")
    ax.set_title(dataset_name)
    ax.set_ylabel('y over n')
    ax.set_xlabel('levels')

    if "GO" in dataset_name:
        #adjust tick size
        _ = plt.xticks(fontsize=6)

    plt.savefig('y_or_n_per_level/'+dataset_name+'.pdf', format='pdf', bbox_inches='tight')
    
    plt.savefig('y_or_n_per_level/'+dataset_name+'.png', format='png', bbox_inches='tight')

    fig_gt, ax_gt = plt.subplots()
    ax_gt.bar(keys_gt, values_gt, color="tab:orange")
    ax_gt.set_title(dataset_name + " (ground truth)")
    ax_gt.set_ylabel('mean gt per level')
    ax_gt.set_xlabel('levels')

    if "GO" in dataset_name:
        #adjust tick size
        _ = plt.xticks(fontsize=6)

    plt.savefig('y_or_n_per_level/'+dataset_name+'_ground_truths.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('y_or_n_per_level/'+dataset_name+'_ground_truths.png', format='png', bbox_inches='tight')