This folder will be populated by the `main.py` script and will contain example outputs from the nn on the test set.

The script will save e `.csv` file for each dataset and seed the nn is run on: `dataset_name + '_' + str(seed) + '.csv'`.

In each of these files, the script will save the outputs of the first seven datapoints. Namely, to each datapoint correspond three rows respectively for the:
  - final prediction (FCM)
  - intermediate Y and N predictions (ICM)
  - ground truth values

Hence, each `dataset_name+ '_' + str(seed) + '.csv'` will contain 21 rows in total.

**NOTES**: 
- the number of columns correspond to the number of classes for dataset `dataset_name`.
- each time the `main.py` is run the `dataset_name + '_' + str(seed) + '.csv'` file will be overwritten.