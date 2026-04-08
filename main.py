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

import numpy

from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc

import ast
import csv


def average_score_per_level(seed, epoch, dataset_name, test_eval, score_per_level, g):
    # g has the connections "inverted", i.e. the descendants of a node are really its ancestors.
    # To correctly compute the levels (aka generations) of every node we have to first invert the
    # connections (and end un with the "correct" graph)

    inv_g = g.reverse()

    generations = [sorted(generation) for generation in nx.topological_generations(inv_g)]

    #print(generations[0])

    AUPRC_lev = {}

    evaluated = 0
    for idx, to_evaluate in enumerate(test_eval):
        for gen_idx, gen in enumerate(generations):
            if idx in gen:
                level = gen_idx
        #level = len(list(nx.descendants(g, idx)))
        if to_evaluate == 1:
            if not np.isnan(score_per_level[evaluated]):
                if level in AUPRC_lev.keys():
                    AUPRC_lev[level].append(score_per_level[evaluated])
                else:
                    AUPRC_lev[level] = [score_per_level[evaluated]]
            evaluated += 1

    #print(AUPRC_lev)

    keys = list(AUPRC_lev.keys())
    keys.sort()
    sorted_AUPRC_lev = {i: AUPRC_lev[i] for i in keys}
    keys = [str(lev) for lev in keys]

    # Create folder if it does not exist
    if not os.path.exists('results_per_level'):
         os.makedirs('results_per_level')

    path_per_level = Path('results_per_level/'+dataset_name+'_inter_step_BCE_withRoot.csv')
    num_lines_per_level = 0
    
    if path_per_level.is_file():
        with open('results_per_level/'+dataset_name+'_inter_step_BCE_withRoot.csv', 'r') as file:
            lines = file.readlines()
            for line in lines:
                if len(line.strip()) != 0: # for all non empty lines
                    num_lines_per_level = num_lines_per_level + 1   

    # add an header with the used levels if I have to start from scratch writing the files or the file is empty
    if (num_lines_per_level==11) or (num_lines_per_level == 0):
        af = open('results_per_level/'+dataset_name+'_inter_step_BCE_withRoot.csv', 'w')
        af.write("seed" + ',' + "epoch" + ',')
        for key in keys:
            af.write(key + ",")
        af.write("\n")
    else:
        af = open('results_per_level/'+dataset_name+'_inter_step_BCE_withRoot.csv', 'a')

    af.write(str(seed)+ ',' +str(epoch) + ',')
    for key in sorted_AUPRC_lev.keys():
        af.write(str(np.mean(sorted_AUPRC_lev[key]))+ ",")
    af.write("\n")
    af.close()


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy)
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def get_best_parameters(dataset_name): 
    if 'ontology' in dataset_name:
        with open('logs/'+str(dataset_name)+'_bayes_inter_step_BCE_withRoot/bayes_result.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            data = [row for row in reader] # get all rows in the csv (header + one data row)
            num_epochs = int(data[1][2]) 
            # convert the the formatted string into a dictionary
            params = ast.literal_eval(data[1][1]) 
    else:
        with open('logs/'+str(dataset_name)+'_bayes_inter_step_BCE/bayes_result.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            data = [row for row in reader] # get all rows in the csv (header + one data row)
            num_epochs = int(data[1][2]) 
            # convert the the formatted string into a dictionary
            params = ast.literal_eval(data[1][1]) 
            
    return num_epochs, params


def get_constr_out(out, R, alfa):
    """ Given the output of the neural network x returns the output of the final and intermediate constraint modules
    given the hierarchy constraint expressed in the matrix R """
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j (it means that j is the subclass)
    
    # In the bottom module h we have 2 outputs for each class.
    # Also, we separate the computations for c_out_y and c_out_n.

    # c_out_y:
    c_out_y = out[0].double()   
    c_out_y = c_out_y.unsqueeze(1)
    c_out_y = c_out_y.expand(len(out[0]), R.shape[1], R.shape[1])
    R_batch_y = R.expand(len(out[0]), R.shape[1], R.shape[1])
    
    # c_out_n
    c_out_n = out[1].double()
    R_t = R.transpose(1, 2)
    c_out_n = c_out_n.unsqueeze(1)
    c_out_n = c_out_n.expand(len(out[1]), R_t.shape[1], R_t.shape[1])
    R_batch_n = R_t.expand(len(out[1]), R_t.shape[1], R_t.shape[1])

    # Now the intermediate CM is obtained separately for the y outputs and the n ones.
    inter_out_y, _ = torch.max(R_batch_y*c_out_y.double(), dim = 2)
    inter_out_n, _ = torch.max(R_batch_n*c_out_n.double(), dim = 2)

    # NOW for final_out we directly use alfa in the computation
    final_out = alfa*inter_out_y + (1-alfa)*(1-inter_out_n)

    #final_out has the "original" dimension, with one output per class
    return final_out, inter_out_y, inter_out_n


class ConstrainedFFNNModel(nn.Module):
    """ our model - during training it returns two fake intermediate outputs (for computational ease) 
    and the not-constrained output that is then passed to CLoss """
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R, alfa):
        super(ConstrainedFFNNModel, self).__init__()
        
        self.nb_layers = hyperparams['num_layers']
        self.R = R
        self.alfa = alfa
        
        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers-1:
                # We have 2 outputs for each class, so the final dimension will be 2*output_dim
                fc.append(nn.Linear(hidden_dim, 2*output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)
        
        self.drop = nn.Dropout(hyperparams['dropout'])
        
        
        self.sigmoid = nn.Sigmoid()
        if hyperparams['non_lin'] == 'tanh':
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()
        
    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers-1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        # At this point the output is x.
        # To make computations easier, we consider the output to have first all the hy ouputs, then all the hn ones. We make the 
        # net itself return the y outputs and the n ones separately.

        out_y = x[:, 0:self.R.size(1)]
        out_n = x[:, self.R.size(1):2*self.R.size(1)]

        # out[0] corresponds to out_y, out[1] corresponds to out_n.
        out = torch.stack((out_y, out_n), dim=0)

        if self.training:   
            constrained_out = out
            inter_out_y = 0
            inter_out_n = 0
        else:
            constrained_out, inter_out_y, inter_out_n = get_constr_out(out, self.R, self.alfa)
        return constrained_out, inter_out_y, inter_out_n


def main():

    parser = argparse.ArgumentParser(description='Train neural network on train and validation set')

    # Required  parameter
    parser.add_argument('--dataset', type=str, default=None, required=True,
                        help='dataset name, must end with: "_GO", "_FUN", or "_others"' )
    # Other parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU (default:0)')
    args = parser.parse_args()

    assert('_' in args.dataset)
    assert('FUN' in args.dataset or 'GO' in args.dataset or 'others' in args.dataset or 'ontology' in args.dataset)

    # Load train, val and test set
    dataset_name = args.dataset
    data = dataset_name.split('_')[0]
    ontology = dataset_name.split('_')[1]

    # Dictionaries with number of features and number of labels for each dataset
    input_dims = {'comedy': 384, 'engineering':384, 'law': 384, 'main': 384, 'people': 384,'culture': 384, 'information': 384,'philosophy': 384,'mathematics': 384,'energy': 384,'diatoms':371, 'enron':1001,'imclef07a': 80, 'imclef07d': 80,'cellcycle':77, 'church':27, 'derisi':63, 'eisen':79, 'expr':561, 'gasch1':173, 'gasch2':52, 'hom':47034, 'seq':529, 'spo':86}
    output_dims_FUN = {'cellcycle':499, 'church':499, 'derisi':499, 'eisen':461, 'expr':499, 'gasch1':499, 'gasch2':499, 'hom':499, 'seq':499, 'spo':499}
    output_dims_GO = {'cellcycle':4122, 'church':4122, 'derisi':4116, 'eisen':3570, 'expr':4128, 'gasch1':4122, 'gasch2':4128, 'hom':4128, 'seq':4130, 'spo':4116}
    output_dims_others = {'diatoms':398,'enron':56, 'imclef07a': 96, 'imclef07d': 46, 'reuters':102}
    output_dims_ontology = {'comedy': 395, 'engineering' : 587, 'law': 958, 'main': 147, 'people': 177,'culture': 1429,'information': 754,'philosophy': 305,'mathematics':407,'energy': 879}
    output_dims = {'FUN':output_dims_FUN, 'GO':output_dims_GO, 'others':output_dims_others, 'ontology': output_dims_ontology}

    # use get_best_parameters function to directly extract the best parameters for each dataset from the corresponding file under the 
    # logs directory
    num_epochs, hyperparams = get_best_parameters(dataset_name)

    # Set the hyperparameters 
    num_epochs = num_epochs + 1 # take into account the use of range(arg) which will stop at arg-1
    batch_size = hyperparams['batch_size']
    hidden_dim = hyperparams['hidden_dim']
    lr = hyperparams['lr']
    weight_decay = hyperparams['weight_decay']


    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available:
        pin_memory = True


    # Load the datasets
    if ('others' in args.dataset):
        train, test = initialize_other_dataset(dataset_name, datasets)
    elif ('ontology' in dataset_name):
        dataset_info = {}
        dataset_info['random_seed'] = seed
        dname = dataset_name.split('_')[0]
        dataset_info['folderpath'] = 'HMC_data/ontology/'+dname     
        train, val, test = parser_ontology.initialize_DBPedia_dataset(dataset_info)
    else:
        train, val, test = initialize_dataset(dataset_name, datasets)
    

    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
    R = np.zeros(train.A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(train.A) # train.A is the matrix where the direct connections are stored 
    for i in range(len(train.A)):
        ancestors = list(nx.descendants(g, i)) #here we need to use the function nx.descendants() because in the directed graph the edges have source from the descendant and point towards the ancestor 
        if ancestors:
            R[i, ancestors] = 1
    R = torch.tensor(R)
    #Transpose to get the ancestors for each node 
    if 'ontology' not in dataset_name:
        R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(device)


    # Rescale data and impute missing data
    if ('others' in args.dataset):
        scaler = preprocessing.StandardScaler().fit((train.X.astype(float)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit((train.X.astype(float)))
    else:
        scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, val.X)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X, val.X)))
        val.X, val.Y = torch.tensor(scaler.transform(imp_mean.transform(val.X))).to(device), torch.tensor(val.Y).to(device)
    train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(device)        
    test.X, test.Y = torch.tensor(scaler.transform(imp_mean.transform(test.X))).to(device), torch.tensor(test.Y).to(device)

    #Create loaders 
    train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]
    if ('others' not in args.dataset):
        val_dataset = [(x, y) for (x, y) in zip(val.X, val.Y)]
        for (x, y) in zip(val.X, val.Y):
            train_dataset.append((x,y))
    test_dataset = [(x, y) for (x, y) in zip(test.X, test.Y)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)

    # We do not evaluate the performance of the model on the 'roots' node (https://dtai.cs.kuleuven.be/clus/hmcdatasets/)
    if 'GO' in dataset_name: 
        num_to_skip = 4
    else:
        num_to_skip = 1 

    # Create the model
    model = ConstrainedFFNNModel(input_dims[data], hidden_dim, output_dims[ontology][data]+num_to_skip, hyperparams, R, hyperparams['alfa'])
    model.to(device)
    print("Model on gpu", next(model.parameters()).is_cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    if 'ontology' not in dataset_name:
        labels_sum = torch.zeros(1,output_dims[ontology][data]+num_to_skip) 
    else:
        labels_sum = torch.zeros(1,output_dims[ontology][data]) # per gli ontology non vale il to_skip?
    labels_sum = labels_sum.to(device)

    for epoch in range(num_epochs):
        model.train()

        for i, (x, labels) in enumerate(train_loader):

            x = x.to(device)
            labels = labels.to(device)

            labels_sum += torch.sum(labels, dim=0)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # output[0] corresponds to the y outputs, output[1] corresponds to the n outputs.
            output, _ , _ = model(x.float())

            # custom constrained loss
            out_1_2 = torch.stack((labels*output[0], output[1]), dim=0)
            _ , inter_out_y1, inter_out_n2 = get_constr_out(out_1_2, R, hyperparams['alfa'])

            inter_out_y1 = inter_out_y1
            inter_out_n2 = inter_out_n2

            out_3_4 = torch.stack((output[0], (1-labels)*output[1]), dim=0)
            _ , inter_out_y3, inter_out_n4 = get_constr_out(out_3_4, R, hyperparams['alfa'])

            inter_out_y3 = inter_out_y3
            inter_out_n4 = inter_out_n4
            train_output_y = labels*inter_out_y1 + (1-labels)*inter_out_y3
            loss_y = hyperparams['alfa']*criterion(train_output_y, labels)

            train_output_n = labels*(1-inter_out_n2) + (1-labels)*(1-inter_out_n4)
            loss_n = (1-hyperparams['alfa'])*criterion(train_output_n, labels)

            loss = loss_y + loss_n

            loss.backward()
            optimizer.step()

    if 'ontology' not in dataset_name:
        bigger_y = torch.zeros(1,output_dims[ontology][data]+num_to_skip) 
    else:
        bigger_y = torch.zeros(1,output_dims[ontology][data]) # per gli ontology non vale il to_skip?
    bigger_y = bigger_y.to(device)

    # Create folder if it does not exist
    if not os.path.exists('outputs'):
            os.makedirs('outputs')

    iter=0

    for i, (x,y) in enumerate(test_loader):

        model.eval()
                
        x = x.to(device)
        y = y.to(device)

        constrained_output, inter_out_y, inter_out_n = model(x.float())

        # write in a csv file the first 7 datapoints with (for each class):
        # final predicion
        # Y prediction & N prediction
        # ground truth

        # if iter=0 we empty any previous content of the file
        if (iter==0):
            # opening the file with w+ mode truncates the file
            f = open('outputs/'+dataset_name+'_'+str(seed)+'_inter_step_BCE_withRoot.csv', "w+")
            f.close()

        # we don't use any header but we know that each class (not to skip) occupies two columns.
        with open('outputs/'+dataset_name+'_'+str(seed)+'_inter_step_BCE_withRoot.csv', 'a') as file:

            for row_co,row_y,row_n,row_gt in zip(constrained_output,inter_out_y,inter_out_n,y):
                
                if(iter==7):
                    break
            
                for final_pred in row_co:
                    file.write(str(float(final_pred))+","+""+",")
                file.write("\n")
                for y_pred,n_pred in zip(row_y,row_n):
                    file.write(str(float(y_pred))+","+str(float(n_pred))+",")
                file.write("\n")
                for gt in row_gt:
                    file.write(str(float(gt))+","+""+",")
                file.write("\n")
                
                iter = iter + 1

        predicted = constrained_output.data > 0.5
        # count the number of times inter_out_y was chosen in place of inter_out_n (in test.to_eval)
        bigger_y = bigger_y + torch.sum((inter_out_y > inter_out_n), dim=0)

        # Move output and label back to cpu to be processed by sklearn
        predicted = predicted.to('cpu')
        cpu_constrained_output = constrained_output.to('cpu')
        y = y.to('cpu')

        if i == 0:
            predicted_test = predicted
            constr_test = cpu_constrained_output
            y_test = y
        else:
            predicted_test = torch.cat((predicted_test, predicted), dim=0)
            constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
            y_test = torch.cat((y_test, y), dim =0)


    score = average_precision_score(y_test[:,test.to_eval], constr_test.data[:,test.to_eval], average='micro')
    # additional score metrics:
    score_f1 = f1_score(y_test[:,test.to_eval], predicted_test[:,test.to_eval], average='micro')
    # In multilabel classification, this function computes subset accuracy
    score_sub_acc = accuracy_score(y_test[:,test.to_eval], predicted_test[:,test.to_eval])
    # add also hamming loss and a custom computed hamming score for accuracy:
    score_hamm_loss = hamming_loss(y_test[:,test.to_eval], predicted_test[:,test.to_eval])
    score_hamm_score = hamming_score(y_test[:,test.to_eval], predicted_test[:,test.to_eval])

    
    if not os.path.exists('results'):
         os.makedirs('results')

    # we want to open the file with w option if the file already had 10 non empty lines (to overwrite it).
    # if the file is empty/doesn't exist or has less than 10 lines we can open it with a option to directly start/keep appending
    path = Path('results/'+dataset_name+'_inter_step_BCE_withRoot.csv')
    num_lines = 0
    
    if path.is_file():
        with open('results/'+dataset_name+'_inter_step_BCE_withRoot.csv', 'r') as file:
            lines = file.readlines()
            for line in lines:
                if len(line.strip()) != 0: # for all non empty lines
                    num_lines = num_lines + 1   

    if num_lines==10:
        f = open('results/'+dataset_name+'_inter_step_BCE_withRoot.csv', 'w')
    else:
        f = open('results/'+dataset_name+'_inter_step_BCE_withRoot.csv', 'a')

    #scores order: average precision score, f1 score, subset accuracy, hamming loss, hamming score (=label based accuracy)
    f.write(str(seed)+ ',' +str(epoch) + ',' + str(score) + ',' + str(score_f1) + ',' + str(score_sub_acc) + ',' + str(score_hamm_loss) + ',' + str(score_hamm_score) + '\n')
    f.close()

    # Code to save bigger_y data

    if not os.path.exists('y_or_n'):
         os.makedirs('y_or_n')

    path_y_n = Path('y_or_n/'+dataset_name+'_inter_step_BCE_withRoot.csv')
    num_lines_y_n = 0
    
    if path_y_n.is_file():
        with open('y_or_n/'+dataset_name+'_inter_step_BCE_withRoot.csv', 'r') as file:
            lines = file.readlines()
            for line in lines:
                if len(line.strip()) != 0: # for all non empty lines
                    num_lines_y_n = num_lines_y_n + 1   

    if num_lines_y_n==10:
        f_y_n = open('y_or_n/'+dataset_name+'_inter_step_BCE_withRoot.csv', 'w')
    else:
        f_y_n = open('y_or_n/'+dataset_name+'_inter_step_BCE_withRoot.csv', 'a')
        
    f_y_n.write(str(seed) + ",")
    for count in bigger_y[0].tolist():
        f_y_n.write(str(count)+',')
    f_y_n.write("\n")
    f_y_n.close()

    
    # Code to compute the average precision score per level
    score_per_level = average_precision_score(y_test[:,test.to_eval], constr_test.data[:,test.to_eval], average=None)

    average_score_per_level(seed, epoch, dataset_name, test.to_eval, score_per_level, g)


    small_pos_classes = torch.where(labels_sum<5, True, False)

    y_test_to_eval = y_test[:,test.to_eval]
    constr_test_to_eval = constr_test.data[:,test.to_eval]
    small_pos_classes = small_pos_classes[:,test.to_eval][0].tolist()

    if 'ontology' in dataset_name:
        print(y_test_to_eval[:,small_pos_classes], constr_test_to_eval[:,small_pos_classes])

    small_pos_score = average_precision_score(y_test_to_eval[:,small_pos_classes], constr_test_to_eval[:,small_pos_classes], average=None)

    if not os.path.exists('small_pos_results'):
         os.makedirs('small_pos_results')

    path_s_p = Path('small_pos_results/'+dataset_name+'_inter_step_BCE_withRoot.csv')
    num_lines_s_p = 0
    
    if path_s_p.is_file():
        with open('small_pos_results/'+dataset_name+'_inter_step_BCE_withRoot.csv', 'r') as file:
            lines = file.readlines()
            for line in lines:
                if len(line.strip()) != 0: # for all non empty lines
                    num_lines_s_p = num_lines_s_p + 1   

    if num_lines_s_p==10:
        f_s_p = open('small_pos_results/'+dataset_name+'_inter_step_BCE_withRoot.csv', 'w')
    else:
        f_s_p = open('small_pos_results/'+dataset_name+'_inter_step_BCE_withRoot.csv', 'a')
        
    f_s_p.write(str(seed) + "," + str(epoch) + ',')
    for small_score in small_pos_score:
        f_s_p.write(str(small_score)+',')
    f_s_p.write("\n")
    f_s_p.close()

if __name__ == "__main__":
    main()
