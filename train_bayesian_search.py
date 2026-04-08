import os
import importlib
os.environ["DATA_FOLDER"] = "./"

import argparse
import csv

import torch
import torch.utils.data
import torch.nn as nn

from utils.parser import *
from utils import parser_ontology
from utils import datasets
import random
import sys

from sklearn.impute import SimpleImputer

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc

from hyperopt import STATUS_OK, hp, tpe, Trials, fmin

from timeit import default_timer as timer

import pickle


# Training settings
parser = argparse.ArgumentParser(description='Train neural network')

# Required  parameters (now the hyperparameters should be managed by hyperopt, we just need a couple parameters for our 
# internal "machinery". We also want them where everyone can see tem so we don't have to add parameters to the objective function)
parser.add_argument('--dataset', type=str, required=True,
                    help='dataset')
parser.add_argument('--device', type=int, default=0,
                    help='device (default:0)')
parser.add_argument('--num_epochs', type=int, default=2000,
                    help='Max number of epochs to train (default:2000)')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default:0)')
# add parameter for MAX_EVALS (so it can be easily set to be different for different datasets)
parser.add_argument('--max_evals', type=int, required=True,
                    help='max number of evaluations to perform')
# add required parameter to discriminate between a new search and an already started one
parser.add_argument('--new_search', type=int, required=True,
                    help='value indicating if the current search is a new one or not. The only admissible values are 1 (True) and 0 (False)')

args = parser.parse_args()

dataset_name = args.dataset
data = dataset_name.split('_')[0]
ontology = dataset_name.split('_')[1]
num_epochs = args.num_epochs
new_search = args.new_search

# Pick device
device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

# Set seed
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

iteration = 0

# the NEW evaluations (possible previous evaluations are automatically detected)
MAX_EVALS = args.max_evals


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

    # The final comparison is done directly between inter_out_y and inter_out_n
    #final_out = torch.where(inter_out_y>inter_out_n, inter_out_y, 1-inter_out_n)

    # NOW for final_out we directly use alfa in the computation
    # CM = alfa*CMY + (1-alfa)*(1-CMN)
    final_out = alfa*inter_out_y + (1-alfa)*(1-inter_out_n)

    #final_out has the "not_doubled" dimension
    return final_out, inter_out_y, inter_out_n


class ConstrainedFFNNModel(nn.Module):
    """ our model - during training it returns the not-constrained output that is then passed to MCLoss """
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
        # To make computations easier, we consider the output to have first all the hy ouputs, then all the hn ones. We make the 
        # net itself return the y outputs and the n ones separately.

        out_y = x[:, 0:self.R.size(1)]
        out_n = x[:, self.R.size(1):2*self.R.size(1)]

        # out[0] corresponds to out_y, out[1] corresponds to out_n.
        out = torch.stack((out_y, out_n), dim=0)

        if self.training:
            # At training time constrained_out[0] corresponds to out_y, constrained_out[1] corresponds to out_n.      
            constrained_out = out
        else:
            constrained_out, _, _ = get_constr_out(out, self.R, self.alfa)
        return constrained_out

def objective(hyperparams):
    """ Returns validation score from hyperparameters """
    global iteration
    iteration += 1

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['batch_size', 'hidden_dim', 'num_layers']:
        hyperparams[parameter_name] = int(hyperparams[parameter_name])


    # Dictionaries with number of features and number of labels for each dataset
    input_dims = {'comedy': 384, 'engineering':384, 'law': 384, 'main': 384, 'people': 384,'culture': 384, 'information': 384,'philosophy': 384,'mathematics': 384,'energy': 384,'diatoms':371, 'enron':1001,'imclef07a': 80, 'imclef07d': 80,'cellcycle':77, 'church':27, 'derisi':63, 'eisen':79, 'expr':561, 'gasch1':173, 'gasch2':52, 'hom':47034, 'seq':529, 'spo':86}
    output_dims_FUN = {'cellcycle':499, 'church':499, 'derisi':499, 'eisen':461, 'expr':499, 'gasch1':499, 'gasch2':499, 'hom':499, 'seq':499, 'spo':499}
    output_dims_GO = {'cellcycle':4122, 'church':4122, 'derisi':4116, 'eisen':3570, 'expr':4128, 'gasch1':4122, 'gasch2':4128, 'hom':4128, 'seq':4130, 'spo':4116}
    output_dims_others = {'diatoms':398,'enron':56, 'imclef07a': 96, 'imclef07d': 46, 'reuters':102}
    output_dims_ontology = {'comedy': 395, 'engineering' : 587, 'law': 958, 'main': 147, 'people': 177,'culture': 1429,'information': 754,'philosophy': 305,'mathematics':407,'energy': 879}
    output_dims = {'FUN':output_dims_FUN, 'GO':output_dims_GO, 'others':output_dims_others, 'ontology': output_dims_ontology}

    # Load the datasets
    if ('others' in dataset_name):
        train, test = initialize_other_dataset(dataset_name, datasets)
        train.X, valX, train.Y, valY = train_test_split(train.X, train.Y, test_size=0.30, random_state=seed)
    elif ('ontology' in dataset_name):
        dataset_info = {}
        dataset_info['random_seed'] = seed
        dname = dataset_name.split('_')[0]
        dataset_info['folderpath'] = 'HMC_data/ontology/'+dname     
        train, val, test = parser_ontology.initialize_DBPedia_dataset(dataset_info)
    else:
        train, val, test = initialize_dataset(dataset_name, datasets)

    
    R = np.zeros(train.A.shape)
    np.fill_diagonal(R, 1) 
    g = nx.DiGraph(train.A)
    for i in range(len(train.A)):
        descendants = list(nx.descendants(g, i))
        if descendants:
            R[i, descendants] = 1
    R = torch.tensor(R)
    #Transpose to get the ancestors for each node 
    if 'ontology' not in dataset_name:
        R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(device)


    # Rescale dataset and impute missing data
    if ('others' in dataset_name):
        scaler = preprocessing.StandardScaler().fit((train.X))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit((train.X))
        train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(device)
        valX, valY = torch.tensor(scaler.transform(imp_mean.transform(valX))).to(device), torch.tensor(valY).to(device)
    else:
        scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, val.X)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X, val.X)))
        val.X, val.Y = torch.tensor(scaler.transform(imp_mean.transform(val.X))).to(device), torch.tensor(val.Y).to(device)
        train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(device)        

    # Create loaders 
    train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]
    if ('others' in args.dataset):
        val_dataset = [(x, y) for (x, y) in zip(valX, valY)]
    else:
        val_dataset = [(x, y) for (x, y) in zip(val.X, val.Y)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=hyperparams['batch_size'],
                                            shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=hyperparams['batch_size'],
                                            shuffle=False)


    if 'GO' in dataset_name: 
        num_to_skip = 4
    else:
        num_to_skip = 1 

    # Set patience 
    patience, max_patience = 20, 20
    max_score = 0.0

    # start timing the train iteration
    start = timer()

    # Create the model
    model = ConstrainedFFNNModel(input_dims[data], hyperparams['hidden_dim'], output_dims[ontology][data]+num_to_skip, hyperparams, R, hyperparams['alfa'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay']) 
    criterion = nn.BCELoss()

    # Create folder for the dataset (if it does not exist)
    if not os.path.exists('logs/'+str(dataset_name)+'_bayes_inter_step_BCE_withRoot/'):
         os.makedirs('logs/'+str(dataset_name)+'_bayes_inter_step_BCE_withRoot/')

    for epoch in range(num_epochs):
        total_train = 0.0
        correct_train = 0.0
        model.train()

        for i, (x, labels) in enumerate(train_loader):

            x = x.to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # output[0] corresponds to the y outputs, output[1] corresponds to the n outputs.
            output = model(x.float())

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
            constr_output, _, _ = get_constr_out(output, R, hyperparams['alfa'])

            predicted = constr_output.data > 0.5
            # Total number of labels
            total_train += labels.size(0) * labels.size(1)
            # Total correct predictions
            correct_train += (predicted == labels.byte()).sum()
                    
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
            
        model.eval()

        for i, (x,y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            constrained_output = model(x.float())

            predicted = constrained_output.data > 0.5
            # Total number of labels
            total = y.size(0) * y.size(1)
            # Total correct predictions
            correct = (predicted == y.byte()).sum()

            #Move output and label back to cpu to be processed by sklearn
            cpu_constrained_output = constrained_output.to('cpu')
            y = y.to('cpu')

            if i == 0:
                constr_val = cpu_constrained_output
                y_val = y
            else:
                constr_val = torch.cat((constr_val, cpu_constrained_output), dim=0)
                y_val = torch.cat((y_val, y), dim =0)

        score = average_precision_score(y_val[:,train.to_eval], constr_val.data[:,train.to_eval], average='micro') 
        
        if score >= max_score:
            patience = max_patience
            max_score = score
        else:
            patience = patience - 1
        
        floss= open('logs/'+str(dataset_name)+'_bayes_inter_step_BCE_withRoot/measures_batch_size_'+str(hyperparams['batch_size'])+'_lr_'+str(hyperparams['lr'])+'_weight_decay_'+str(hyperparams['weight_decay'])+'_seed_'+str(seed)+'_num_layers_'+str(hyperparams['num_layers'])+'_hidden_dim_'+str(hyperparams['hidden_dim'])+'_dropout_'+str(hyperparams['dropout'])+'_alfa_'+str(hyperparams['alfa'])+'_'+hyperparams['non_lin'], 'a')
        floss.write('\nEpoch: {} - Loss: {:.4f}, Accuracy train: {:.5f}, Accuracy: {:.5f}, Precision score: ({:.5f})\n'.format(epoch,
                    loss, float(correct_train)/float(total_train), float(correct)/float(total), score))
        floss.close()

        if patience == 0:
            break

    # stop timing the train iteration (run_time expressed in seconds)
    run_time = timer() - start

    best_score=0
    best_epoch=0
    
    with open('logs/'+str(dataset_name)+'_bayes_inter_step_BCE_withRoot/measures_batch_size_'+str(hyperparams['batch_size'])+'_lr_'+str(hyperparams['lr'])+'_weight_decay_'+str(hyperparams['weight_decay'])+'_seed_'+str(seed)+'_num_layers_'+str(hyperparams['num_layers'])+'_hidden_dim_'+str(hyperparams['hidden_dim'])+'_dropout_'+str(hyperparams['dropout'])+'_alfa_'+str(hyperparams['alfa'])+'_'+hyperparams['non_lin'], 'r') as f_in:

        #list with all file lines
        lines = f_in.readlines()
        #number of lines
        num_lines = len(lines)
        
        # 41 = patience*2 + 1 (in the files we have the real lines separated by empty lines, with a finel empty line)
        best_score = lines[num_lines-41].split('Precision score: ')[1].strip()
        best_score = float(best_score[best_score.find("(")+1:best_score.find(")")])

        best_epoch = int(lines[num_lines-41].split('-')[0].split(':')[1].strip())

    # hyperopt works with minimization
    bayes_loss = 1-best_score

    # Write to the csv file ('a' means append)
    f_trials= open('logs/'+str(dataset_name)+'_bayes_inter_step_BCE_withRoot/bayes_trials.csv', 'a')
    writer = csv.writer(f_trials)
    writer.writerow([bayes_loss, hyperparams, best_epoch, iteration, run_time])
    f_trials.close()

    return {'loss': bayes_loss, 'hyperparams': hyperparams, 'epoch': best_epoch, 'status': STATUS_OK}


def main():
    assert('_' in dataset_name)
    assert('FUN' in dataset_name or 'GO' in dataset_name or 'others' in dataset_name or 'ontology' in dataset_name)
    assert(new_search==0 or new_search==1)

    # domain space:
    # (note: hp.choice can be used with numbers too, instead of using hp.quniform)

    wider = {"seq_GO", "expr_GO", "spo_GO", "eisen_GO", "gasch2_GO", "spo_FUN", "gasch1_FUN", "cellcycle_FUN", "eisen_FUN", "seq_FUN", "expr_FUN", "enron_others", "comedy_ontology", "engineering_ontology", "law_ontology", "main_ontology",
             "people_ontology", "culture_ontology", "information_ontology", "philosophy_ontology", "mathematics_ontology", "energy_ontology"}

    # discriminate between the datasets needing higher hidden dimension directly in code
    if  dataset_name in wider:
        space = {
            'batch_size': hp.choice('batch_size', [4]),
            'dropout': hp.choice('dropout', [0.7]),
            'hidden_dim': hp.quniform('hidden_dim', 5000, 10000, 100),
            'num_layers': hp.choice('num_layers', [3]),
            # lr takes values between 1e-05 
            'lr': hp.choice('lr', [0.00001]),
            # weight decay = 1e-5
            'weight_decay':  hp.choice('weight_decay', [0.00001]),
            'non_lin': hp.choice('non_lin', ['relu']),
            'alfa': hp.uniform('alfa', 0.0, 1.0)
        }
    else:
        space = {
            'batch_size': hp.choice('batch_size', [4]),
            'dropout': hp.choice('dropout', [0.7]),
            'hidden_dim': hp.quniform('hidden_dim', 50, 4000, 50),
            'num_layers': hp.choice('num_layers', [3]),
            # lr takes values between 1e-05 and 1e-03
            'lr': hp.choice('lr', [0.0001]),
            # weight decay = 1e-5
            'weight_decay':  hp.choice('weight_decay', [0.00001]),
            'non_lin': hp.choice('non_lin', ['relu']),
            'alfa': hp.uniform('alfa', 0.0, 1.0)
        }

    
    log_dir = f'logs/{dataset_name}_bayes_inter_step_BCE_withRoot'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    trials_path = os.path.join(log_dir, 'trials_object.p')
    csv_path    = os.path.join(log_dir, 'bayes_trials.csv')
    rng_path = os.path.join(log_dir, 'rng_state.p')

    global iteration

    if new_search == 1 or (not os.path.exists(trials_path)):
        print(">>> NEW search")
        bayes_trials = Trials()
        iteration = 0
        random_state = np.random.default_rng(seed)

        with open(csv_path, 'w', newline='') as f_trials:
            writer = csv.writer(f_trials)
            writer.writerow(['loss', 'hyperparams', 'epoch', 'iteration', 'train_time'])
    else:
        print(">>> RESUME search")
        bayes_trials = pickle.load(open(trials_path, "rb"))
        iteration = len(bayes_trials.trials)
        
        if os.path.exists(rng_path):
            random_state = pickle.load(open(rng_path, "rb"))

    n_existing = len(bayes_trials.trials)
    max_evaluations = n_existing + MAX_EVALS 

    print(f"Existing trials: {n_existing}, target total trials: {max_evaluations}")

    

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evaluations,
        trials=bayes_trials,
        rstate=random_state
    )

    pickle.dump(bayes_trials, open(trials_path, "wb"))
    pickle.dump(random_state, open(rng_path, "wb"))
    
    with open(csv_path, 'r') as f_trials_sorted:
        reader = csv.reader(f_trials_sorted)
        header = next(reader)
        data = [row for row in reader if row]
    data.sort(key=lambda row: float(row[0]))

    best_path = os.path.join(log_dir, 'bayes_result.csv')
    with open(best_path, 'w', newline='') as f_best_result:
        writer = csv.writer(f_best_result)
        writer.writerow(header)
        writer.writerow(data[0])
    
if __name__ == "__main__":
    main()
