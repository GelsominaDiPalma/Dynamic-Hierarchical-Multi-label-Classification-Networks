This folder will be populated by the `main.py` script.

For each dataset the script will create a `dataset_name + '.csv'` file. Each row of these files corresponds to one triale and will contain:
- Seed
- How many times (at test time), for each class, the output of CM_Y was bigger than CM_N (ICM outputs)
  - Having CM_Y > CM_N for a class A means that A was "more" predicted than not.


**NOTE**: each 10 trials the `dataset_name + '.csv'` will be overwritten (i.e., the script will start saving the results from scratch).