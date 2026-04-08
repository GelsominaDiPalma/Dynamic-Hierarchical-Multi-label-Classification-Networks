This folder will be populated by the `main.py` script and will contain the results of the nn on the test set.

For each dataset the script will create a `dataset_name + '.csv'` file. Each row of these files corresponds to one triale and will contain:
- Seed
- Number of epochs the nn was trained on
- AUPRC score
- F1 score
- Subset accuracy score
- Hamming loss
- Hamming score


**NOTE**: each 10 trials the `dataset_name + '.csv'` will be overwritten (i.e., the script will start saving the results from scratch).