This folder will be populated by the `train_bayesian_search.py` script. Namely, the script will:
- Create a folder named `<dataset_name>_bayes` for each dataset considered for the nn training.
- In each `<dataset_name>_bayes` folder, it will:
  - Add several files logging training statistics for each hyperparameter configuration (i.e., epoch, loss, accuracy and precision score)
  - Add the script `bayes_trials.csv`. This will collect the best losses obtained during the hyperparameter search (taken from the files at the previous point). Namely, this file contains the following columns: loss, hyperparams, epoch, iteration and train time.
  - Add the script `bayes_result.csv`. This will contain the best row from the `bayes_trials.csv` script (i.e., the best hyperparameter configuration).
