This folder will be populated by the `main.py` script and will contain the results of the nn on the test set, grouped per level.

For each dataset the script will create a `dataset_name + '.csv'` file. Each row of these files corresponds to one triale and will contain:
- Seed
- Number of epochs the nn was trained on
- AUPRC score for each level.
