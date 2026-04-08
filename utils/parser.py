"""
This code was adapted from https://github.com/lucamasera/AWX
"""

import numpy as np
import networkx as nx
import keras
from itertools import chain
import os
from pathlib import Path

# Skip the root nodes 
to_skip = ['root', 'GO0003674', 'GO0005575', 'GO0008150']


class arff_data():
    def __init__(self, arff_file, is_GO, is_test=False):
        self.X, self.Y, self.A, self.terms, self.g = parse_arff(arff_file=arff_file, is_GO=is_GO, is_test=is_test)
        self.to_eval = [t not in to_skip for t in self.terms]
        r_, c_ = np.where(np.isnan(self.X))
        m = np.nanmean(self.X, axis=0)
        for i, j in zip(r_, c_):
            self.X[i,j] = m[j]

            
def parse_arff(arff_file, is_GO=False, is_test=False):
    with open(arff_file) as f:
        read_data = False
        X = []
        Y = []
        g = nx.DiGraph()
        feature_types = []
        d = []
        cats_lens = []
        for num_line, l in enumerate(f):
            if l.startswith('@ATTRIBUTE'):
                if l.startswith('@ATTRIBUTE class'):
                    h = l.split('hierarchical')[1].strip()
                    for branch in h.split(','):
                        terms = branch.split('/')
                        if is_GO:
                            g.add_edge(terms[1], terms[0])
                        else:
                            if len(terms)==1:
                                g.add_edge(terms[0], 'root')
                            else:
                                for i in range(2, len(terms) + 1):
                                    g.add_edge('.'.join(terms[:i]), '.'.join(terms[:i-1]))
                    nodes = sorted(g.nodes(), key=lambda x: (nx.shortest_path_length(g, x, 'root'), x) if is_GO else (len(x.split('.')),x))
                    nodes_idx = dict(zip(nodes, range(len(nodes))))
                    g_t = g.reverse()
                else:
                    _, f_name, f_type = l.split()
                    
                    if f_type == 'numeric' or f_type == 'NUMERIC':
                        d.append([])
                        cats_lens.append(1)
                        feature_types.append(lambda x,i: [float(x)] if x != '?' else [np.nan])
                        
                    else:
                        cats = f_type[1:-1].split(',')
                        cats_lens.append(len(cats))
                        d.append({key:keras.utils.to_categorical(i, len(cats)).tolist() for i,key in enumerate(cats)})
                        feature_types.append(lambda x,i: d[i].get(x, [0.0]*cats_lens[i]))
            elif l.startswith('@DATA'):
                read_data = True
            elif read_data:
                y_ = np.zeros(len(nodes))
                d_line = l.split('%')[0].strip().split(',')
                lab = d_line[len(feature_types)].strip()
                
                X.append(list(chain(*[feature_types[i](x,i) for i, x in enumerate(d_line[:len(feature_types)])])))
                
                for t in lab.split('@'): 
                    y_[[nodes_idx.get(a) for a in nx.ancestors(g_t, t.replace('/', '.'))]] =1
                    y_[nodes_idx[t.replace('/', '.')]] = 1
                Y.append(y_)
        X = np.array(X)
        Y = np.stack(Y)

    return X, Y, np.array(nx.to_numpy_matrix(g, nodelist=nodes)), nodes, g


def initialize_dataset(name, datasets):
    is_GO, train, val, test = datasets[name]
    return arff_data(train, is_GO), arff_data(val, is_GO), arff_data(test, is_GO, True)

def initialize_other_dataset(name, datasets):
    is_GO, train, test = datasets[name]
    return arff_data(train, is_GO), arff_data(test, is_GO, True)


def parse_logs(dataset_name):

    best_scores = {}

    directory = "../logs/"+dataset_name

    path = Path(__file__).parent / directory

    num_files=0
    short_files=0

    for filename in os.listdir(path):
        
        num_files=num_files+1
        f = os.path.join(path, filename)

        with open(f, 'r') as f_in:

            #list with all file lines
            lines = f_in.readlines()
            #number of lines
            num_lines = len(lines)
            
            if num_lines>=41:
                # 41 = patience*2 + 1 (in the files we have the real lines separated by empty lines, with a finel empty line)
                score = lines[num_lines-41].split('Precision score: ')[1].strip()
                score = float(score[score.find("(")+1:score.find(")")])

                epoch = int(lines[num_lines-41].split('-')[0].split(':')[1].strip())

                best_scores[filename] = (epoch, score)
            else:
                print("short file:" + filename + "\n")
                short_files=short_files+1

    print("total short files:" + str(short_files))

    #after all files were investigated we search for the max score
    best_params = max(best_scores.items(), key=lambda x: x[1][1])

    #best_params[0] contains the filename, best_params[1][0] contains the epoch and best_params[1][1] contains the score value
    return best_params[0], best_params[1][0], best_params[1][1], num_files


def mean_results(dataset_name, idx):

    results_file = "../results/"+dataset_name+".csv"

    path = Path(__file__).parent / results_file

    with open(path, 'r') as res_file:
        lines = res_file.readlines()
        num_lines = 0
        sum = 0
        scores = []
        for line in lines:
            if len(line.strip()) != 0: # for all non empty lines
                num_lines = num_lines + 1
                score = float(line.split(",")[idx].strip())
                sum = sum + score
                scores.append(score)

    avg = sum/num_lines
    std = np.std(scores)

    return avg, std

def mean_results_per_level(dataset_name):

    results_file = "../results_per_level/"+dataset_name+".csv"

    path = Path(__file__).parent / results_file

    results = []
    with (open(path, "r")) as rf:
        for line in rf.readlines():
            split_line = line.split(",")
            # skip seed and epoch numbers
            results.append(np.array([float(s.strip()) for s in split_line[2:-1]]))
    results = np.array(results)
    mean_result = str(results.shape) + ', '.join([str(r) for r in np.mean(results, axis=0)])

    return mean_result