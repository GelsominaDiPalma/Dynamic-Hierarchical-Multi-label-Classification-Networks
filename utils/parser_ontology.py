
import numpy as np
import sys
import pickle as pkl
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split

import keras
#import arff
import os
from itertools import chain

to_skip = ['root', 'go0003674', 'go0005575', 'go0008150']

class Node():

    def __init__(self, nodeID):

        self.nodeID = nodeID

        self.parents = []
        self.children = []

        self.root = None
        self.isRoot = False

        self.isSeen = False
        self.depth = 0

    def __repr__(self):
        return "Node(%s)"%self.nodeID

    def __str__(self):
        return self.nodeID

    def getParents(self):
        parents = []

        if self.isRoot:
            return []

        else:
            queue = self.parents[:]

            while len(queue)>0:
                node = queue.pop(0)
                parents.append(node)
                queue += node.parents

            return parents



    def getChildren(self):
        childs = []

        if self.children is None:
            return []
        else:
            queue = self.children[:]

            while len(queue)>0:
                node = queue.pop(0)
                childs.append(node)
                queue += node.children

            return childs


    def getTreeJson(self):

        #name, children, aatid, inKeyword

        if len(self.children) >0:

            return {"name": self.nodeID,  "children":[x.getTreeJson() for x in self.children]}
        else:
            return {"name": self.nodeID}

class Tree():


    def __init__(self, ontology: list, dataset: str):

        self.roots = []
        self.nodes = {}
        self.edges = []

        self.isRagged = False

        self.dataset = dataset
        self.ontology = ontology

        self.buildTree(ontology)
        self.depth = self.return_depth()
        self.g = nx.DiGraph()
        for edge in self.edges:
            self.g.add_edge(edge[0],edge[1])


    def exportTree(self, color_values = None):


        norm = plt.Normalize()
        colors = plt.cm.Reds(norm([c for n, c in color_values.items()]))   


        for i, n in enumerate(color_values):

            self.g.nodes[n]['color'] = matplotlib.colors.rgb2hex(colors[i])


        net = nx.Network(notebook=False, height="750px", width="100%")
        net.from_nx(self.g)
        net.show('tree_violation_graph.html')

    

    def buildTree(self, ontology):


        if self.dataset in ['Enron_corr', 'ImCLEF07A', 'ImCLEF07D', 'Diatoms'] or "FUN" in self.dataset:
            self.nodes['root'] = Node('root')

        for relation in ontology:

            # relation = relation.split('/')


            if len(relation) == 1:

                if self.dataset in ['Enron_corr', 'ImCLEF07A', 'ImCLEF07D', 'Diatoms'] or "FUN" in self.dataset:
                    nodeID = relation[0]
                    if nodeID not in self.nodes:
                        self.nodes[nodeID] = Node(nodeID)
                        self.nodes[nodeID].parents.append(self.nodes['root'])

                        self.nodes['root'].children.append(self.nodes[nodeID])
                        self.nodes['root'].root = True

                        self.edges.append(('root', nodeID))
                    else:
                        self.nodes[nodeID].parents.append(self.nodes['root'])
                        self.nodes['root'].children.append(self.nodes[nodeID])
                        self.edges.append(('root', nodeID))                     


                else:
                    if relation[0] not in self.nodes:
                        self.nodes[relation[0]] = Node(relation[0])
            else:

                index = 1

                while index<len(relation):

                    if self.dataset in ['Enron_corr', 'ImCLEF07A', 'ImCLEF07D', 'Diatoms'] or "FUN" in self.dataset:

                        parent = '.'.join(relation[:index])
                        child = '.'.join(relation[:index+1])

                        self.edges.append(('.'.join(relation[:index]), '.'.join(relation[:index+1])))


                    else:
                        parent = relation[index-1]
                        child = relation[index]
                        self.edges.append((relation[index-1], relation[index]))


                    if parent not in self.nodes:
                        self.nodes[parent] = Node(parent)
                        parent = self.nodes[parent]
                    else:
                        parent = self.nodes[parent]

                    if child not in self.nodes:
                        self.nodes[child] = Node(child)
                        child = self.nodes[child]
                    else:
                        child = self.nodes[child]

                    if child not in parent.children:
                        parent.children.append(child)

                    if parent not in child.parents:
                        child.parents.append(parent)


                    index+=1

        for nodeID, node in self.nodes.items():

            if len(node.parents) == 0:
                self.roots.append(node)





    def return_tree_violations(self, preds, nodes_list):
        seen = []
        violations = 0


        preds = sorted(preds, key=lambda x: -self.nodes[nodes_list[x]].depth)


        for pred in preds:
            parents = self.nodes[nodes_list[pred]].parents[:]
            while len(parents) != 0:

                parent = parents.pop()
                if parent not in seen:


                    if nodes_list.index(parent.nodeID) not in preds and parent not in seen and not parent.root:
                        violations +=1
                    parents += parent.parents
                    if parent not in seen:
                        seen.append(parent)

        if violations >2:
            test = [1 if i in preds else 0 for i in range(len(nodes_list))]
            self.exportTree(dict(zip(nodes_list, test)))


        return violations


    def return_depth(self):

        visited = set()
        queue = [self.roots[0]]
        max_depth = 0


        while len(queue)!=0:

            node = queue.pop(0)
            if node.depth > max_depth:
                max_depth = node.depth 

            if node not in visited:
                visited.add(node)
                for child in node.children:

                    child.depth = node.depth+1
                    queue.append(child)
        
        return max_depth





class arff_data():
    def __init__(self, arff_file, args, is_GO,  is_train=False):



        self.X, self.Y, self.A, self.terms, self.g, (self.fea, self.adjacency_matrix)= parse_arff(arff_file=arff_file, args = args, is_GO=is_GO, is_train=is_train)


        self.to_eval = [t not in to_skip for t in self.terms]

        r_, c_ = np.where(np.isnan(self.X))
        m = np.nanmean(self.X, axis=0)
        for i, j in zip(r_, c_):
            self.X[i,j] = m[j]


def parse_DBPedia(folderpath, args):

    with open('%s/data.pkl'%folderpath, 'rb') as handle:
        data = pkl.load(handle)

    print (folderpath)

    g = nx.read_edgelist("%s/graph.edgelist"%folderpath, create_using=nx.DiGraph)


    X = []
    Y = []
    feature_types = []
    d = []
    cats_lens = []
    counts_unseen = 0

    # nodes = sorted(g.nodes(), key=lambda x: (nx.shortest_path_length(g, x, 'root'), x) if is_GO else (len(x.split('.')),x))

    nodes = list(g.nodes())

    nodes_idx = dict(zip(nodes, range(len(nodes))))
    adjacency_matrix = nx.adjacency_matrix(g, nodelist=nodes)
    fea = sp.identity(len(nodes)) 

    args['tree'] = Tree(g.edges(), folderpath.split('/')[-1])
    # for i, label in enumerate(nodes):

    #     children = g.out_edges(label)
    #     adjacency_matrix

        # if label != 'root':
        #     label = label.split('.')
        #     if len(label) == 1:

        #         adjacency_matrix[0][i] = 1
        #         adjacency_matrix[i][0] = 1
        #     else:
        #         for index in range(len(label)):
        #             if index == 0:
        #                 adjacency_matrix[0][nodes_idx.get(label[0])] = 1
        #                 adjacency_matrix[nodes_idx.get(label[0])][0] = 1  
        #             else:

        #                 node1 = '.'.join(label[:index])
        #                 node2 = '.'.join(label[:index+1])
                        
        #                 adjacency_matrix[nodes_idx.get(node1)][nodes_idx.get(node2)] = 1
        #                 adjacency_matrix[nodes_idx.get(node2)][nodes_idx.get(node1)] = 1

    
    for sub, info in data.items():



        y_ = np.zeros(len(nodes))
        X.append(info['bert_abstract'])
        for cat in info['y']:
            y_[[nodes_idx.get(a) for a in nx.ancestors(g, cat)]] =1
            y_[nodes_idx[cat]] = 1

        
        Y.append(y_)

        # print (unseen_nodes)
        # print (counts_unseen)
    X = np.array(X)
    Y = np.stack(Y)


    args['nodes_list'] = nodes
    args['nodes_idx'] = nodes_idx

    return X, Y, np.array(nx.to_numpy_matrix(g, nodelist=nodes)), nodes, g, (fea, sp.csr_matrix(adjacency_matrix))


def parse_arff(arff_file, args, is_GO=False, is_train=False):



    with open(arff_file) as f:
        read_data = False
        X = []
        Y = []
        g = nx.DiGraph()
        feature_types = []
        d = []
        cats_lens = []
        counts_unseen = 0

                # ontology = ty.split()[1].split(',')
                # ontology = [x.lower() for x in ontology]
                # split_ontology = [x.split('/') for x in ontology]
        for num_line, l in enumerate(f):

            if l.startswith('@ATTRIBUTE'):
                if l.startswith('@ATTRIBUTE class'):

                    h = l.split('hierarchical')[1].strip()


                    ontology = h.split(',')
                    ontology = [x.lower() for x in ontology]
                    split_ontology = [x.split('/') for x in ontology]



                    if 'GO' not in arff_file and 'FUN' not in arff_file:
                        tree = Tree(split_ontology, '_'.join(arff_file.split('/')[-1].split('.')[0].split('_')[:-1]))
                    else:
                        tree = Tree(split_ontology, arff_file.split('/')[-1].split('.')[0])

                    

                    args['tree'] = tree

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
                    # if args['rand'] is None:
                    #     rand = np.random.permutation(len(nodes))
                    #     nodes = list(np.asarray(nodes)[rand])
                    #     args['rand'] = rand
                    # else:
                    #     nodes = list(np.asarray(nodes)[args['rand']])
                    nodes_idx = dict(zip(nodes, range(len(nodes))))


                    nodes = [x.lower() for x in nodes]
                    adjacency_matrix = np.zeros((len(nodes), len(nodes)), dtype = np.int64)
                    fea = sp.identity(len(nodes)) 
                    for i, label in enumerate(nodes):
                        if label != 'root':
                            label = label.split('.')
                            if len(label) == 1:

                                adjacency_matrix[0][i] = 1
                                adjacency_matrix[i][0] = 1
                            else:
                                for index in range(len(label)):
                                    if index == 0:
                                        adjacency_matrix[0][nodes_idx.get(label[0])] = 1
                                        adjacency_matrix[nodes_idx.get(label[0])][0] = 1  
                                    else:
                                        node1 = '.'.join(label[:index])
                                        node2 = '.'.join(label[:index+1])
                                        adjacency_matrix[nodes_idx.get(node1)][nodes_idx.get(node2)] = 1
                                        adjacency_matrix[nodes_idx.get(node2)][nodes_idx.get(node1)] = 1





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
                        d.append({key:keras.utils.np_utils.to_categorical(i, len(cats)).tolist() for i,key in enumerate(cats)})
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

        # print (unseen_nodes)
        # print (counts_unseen)
        X = np.array(X)
        Y = np.stack(Y)


    args['nodes_list'] = nodes
    args['nodes_idx'] = nodes_idx


    return X, Y, np.array(nx.to_numpy_matrix(g, nodelist=nodes)), nodes, g, (fea, sp.csr_matrix(adjacency_matrix))

class DBPedia_data():

    def __init__(self, X, Y, A, terms, g, fea, adjacency_matrix):

        self.X = X
        self.Y = Y
        self.A = A
        self.terms = terms
        self.g = g
        self.fea = fea
        self.adjacency_matrix = adjacency_matrix
        self.to_eval = [True for t in terms]




def initialize_DBPedia_dataset(args):

    X, Y, A, terms, g, (fea, adjacency_matrix) = parse_DBPedia(args['folderpath'], args)


    X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y, test_size=0.2, random_state = args['random_seed'])
    X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size = 0.5, random_state = args['random_seed'])

    return DBPedia_data(X_train, Y_train, A, terms, g, fea, adjacency_matrix), DBPedia_data(X_val, Y_val, A, terms, g, fea, adjacency_matrix), DBPedia_data(X_test, Y_test, A, terms, g, fea, adjacency_matrix)


def initialize_dataset(name, args):

    dir_path = args['arff_dir_path']
    print (name)

    if name not in ['Enron_corr', 'ImCLEF07A', 'ImCLEF07D', 'Diatoms']:
        train = dir_path + name + '.train.arff'
        val = dir_path + name + '.valid.arff'
        test = dir_path + name + '.test.arff'
    else:
        train = dir_path + name + '_train.arff'
        val = dir_path + name + '_train.arff'
        test = dir_path + name + '_test.arff'


    if 'GO' in name:
        is_GO = True
    else:
        is_GO = False
    

    # is_GO, train, val, test = datasets_paths[name]
    # print (datasets_paths[name])
    return arff_data(train, args, is_GO, True), arff_data(val,args, is_GO), arff_data(test, args, is_GO)

