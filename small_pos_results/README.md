This folder will be populated by the `main.py` script.

For each dataset the script will create a `dataset_name + '.csv'` file. Each row of these files corresponds to one triale and will contain:
- Seed
- Number of epochs the nn was trained on
- AUPRC score computed (at test time) separately for each dataset class that counts less than 5 data points.


**NOTE**: each 10 trials the `dataset_name + '.csv'` will be overwritten (i.e., the script will start saving the results from scratch).