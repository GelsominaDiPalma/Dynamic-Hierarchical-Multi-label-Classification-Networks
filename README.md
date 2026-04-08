# DHN

This repository contains the code and data for the paper "When Constraints Go Both Ways: Dynamic Hierarchical Multi-label Classification Networks".

- [DHN](#dhn)
  - [Architecture and set up](#architecture-and-set-up)
  - [Reproduce hyperparameters search](#reproduce-hyperparameters-search)
    - [Details on hyperparameters search](#details-on-hyperparameters-search)
  - [Evaluate DHN](#evaluate-dhn)
  - [Statistics and plots](#statistics-and-plots)
  - [Results](#results)

## Architecture and set up 

**Hardware specification**: we ran our code on a NVIDIA A100 GPU with 80 GB memory.

**Python environment**: we ran our code in a conda environment. We provide the `DHN-env.yml` file that the user can use to create our same conda environment with the command `conda env create -f DHN-env.yml`

**Data**: Before running the python scripts in this repository and reproduce the steps reported below, the user needs to download the datasets's files that we used for our experiments. They can do so by following the instructions in the [data README](HMC_data/README.md).

## Reproduce hyperparameters search

We carried out the hyperparameters search using bayesian optimization as implemented by the [hyperopt library](https://github.com/hyperopt/hyperopt/).

The main entrypoint is the `train_bayesian_search.py` script. The user can run it with the command: 

`python train_bayesian_search.py --dataset <dataset> --device <device> --num_epochs <num_epochs> --seed <seed> --max_evals <max_evals> --new_search <new_search>`

We additionally provide the `bayes_search_script.sh` script as an example that shows how to run the hyperparameters search for different datasets on different devices. 

**Input arguments**
- `dataset`: name of the dataset on which the model is trained and validated.
- `device`: which device to put the nn on.
- `num_epochs`: max number of epochs to perform during training (in the case the early stopping doesn't interrupt the training). Defaults to 2000.
- `seed`: random seed. Defaults to 0.
- `max_evals`: max number of evaluations to perform (used by the objective optimization in hyperopt).
- `new_search`: whether the user is starting a new search (1) or not (0). If not, the search will continue starting from a log file that is automatically saved at the end of the script.

**Output**
- The script will save the hyperparameters search log files (including the best hyperparameter configuration found) under `logs/`, please refer to [this README](logs/README.md) for more details.

**NOTES**:
- The number of features of the datasets we considered are hardcoded in the script (`objective()` method). Hence, if the user wants to add new datasets they should manually add them.
- The parameters spaces for the bayesian search are defined at the beginning of the `main()` method. We split the datasets into two categories, for which we specified different parameters spaces. If needed, the user should manually update them.

### Details on hyperparameters search

Bayesian optimization is a type of informed search that keeps track of past evaluation results and later uses them to create a probabilistic model mapping hyperparameters to a score of an objective function. The model is a surrogate of the real objective function and is much easier to optimize. The goal of bayesian optimization is to find the next set of hyperparameters to evaluate on the real objective function by selecting the configuration that performs best on the surrogate.

We implemented bayesian optimization using the [hyperopt library](https://github.com/hyperopt/hyperopt/) and chose the Tree of Parzen Estimators (TPE) as optimization algorithm.

During our experiments, batch size (4), dropout (0.7), and weight decay (10^-5) were kept fixed (in line with those in [Coherent Hierarchical Multi-Label Classification Networks](https://proceedings.neurips.cc//paper/2020/file/6dd4e10e3296fa63738371ec0d5df818-Paper.pdf)). The remaining hyperparameters are the learning rate, mixing parameter alpha used by the FCM and *DLoss*, and the hidden dimension, and they were all chosen using bayesian optimization over the validaton set. Below we report the ranges chosen for each hyperparameter during the search:
- **learning rate**: sampled uniformly in [10^-5,10^-3]
- **alpha**: sampled uniformly in [0,1]
- **hidden dimension**: 
  - from 50 to 10000 with a step of 50 for datasets that benefited from **larger hidden layers**
  - from 50 to 4000 with a step of 50 for datasets that benefited from **smaller hidden layers**
  - from 5000 to 10000 with a step of 100 for all **Ontology** datasets

In particular, the datasets that benefited from **larger hidden layers** are:
`seq_GO`, `expr_GO`, `spo_GO`, `eisen_GO`, `gasch2_GO`, `spo_FUN`, `gasch1_FUN`, `cellcycle_FUN`, `eisen_FUN`, `seq_FUN`, `expr_FUN`, and `enron_others`.
All other datasets, excluding the **Ontology** datasets, benefited from **smaller hidden layers**.

- This was a decision taken from empirical observations.

Finally, we adopted a different number of bayesian search iterations for the different datasets, depending bith on their dimensions (i.e., execution time) and on the how fast the training seemed to converge. 

Below we report, for each dataset, the hyperparameter' values that were chosen by the optimization algorithm and how many iterations were performed.

**NOTES**: 
- Please not that the dataset marked with **_alt** indicate a parameter configuration that matches C-HMCNN (alpha set to 1).
- Our DHN uses a **ReLU** non linearity and **3** layers.

| **Dataset** | **alpha** | **batch size** | **dropout rate** | **hidden dimension** | **learning rate**  | **weight decay** | **Bayesian search iterations** |
|--------------------------|-------|---|-----|-------|----------------------|------|---|
| Cellcycle_FUN           | 0.892 | 4 | 0.7 | 450   | 8.18e^-5 | 1e^-5 | 600  |
| Derisi_FUN              | 0.596 | 4 | 0.7 | 350   | 1.49e^-4 | 1e^-5 | 200  |
| Eisen_FUNtextbf{_alt}   | 1 | 4 | 0.7 | 500   | 1e^-4 | 1e^-4 | / |
| Expr_FUNtextbf{_alt}    | 1 | 4 | 0.7 | 1250  | 1e^-4 | 1e^-5 | / |
| Gasch1_FUN              | 0.855 | 4 | 0.7 | 2600  | 4.41e^-5 | 1e^-5 | 2000 |
| Gasch2_FUN              | 0.834 | 4 | 0.7 | 650   | 8.35e^-5 | 1e^-5 | 200  |
| Seq_FUN                 | 0.752 | 4 | 0.7 | 6550  | 5.27e^-4 | 1e^-5 | 600  |
| Spo_FUNtextbf{_alt}     | 1 | 4 | 0.7 | 250   | 1e^-4 | 1e^-5 | / |
| Cellcyle_GO             | 0.791 | 4 | 0.7 | 850   | 7.13e^-5 | 1e^-5 | 200  |
| Derisi_GO               | 0.667 | 4 | 0.7 | 650   | 6.59e^-5 | 1e^-5 | 200  |
| Eisen_GO                | 0.849 | 4 | 0.7 | 750   | 5.63e^-4 | 1e^-6 | 800  |
| Expr_GO                 | 0.735 | 4 | 0.7 | 6050  | 1.19e^-5 | 1e^-5 | 800  |
| Gasch1_GO               | 0.999 | 4 | 0.7 | 1150  | 3.72e^-5 | 1e^-4 | 200  |
| Gasch2_GO               | 0.683 | 4 | 0.7 | 800   | 1.21e^-4 | 1e^-5 | 400  |
| Seq_GO                  | 0.831 | 4 | 0-7 | 10000 | 1.11e^-5 | 1e^-5 | 300  |
| Spo_GOtextbf{_alt}      | 1 | 4 | 0.7 | 500   | 1e^-4 | 1e^-5 | / |
| Diatoms                 | 0.895 | 4 | 0.7 | 2000  | 4.66e^-5 | 1e^-5 | 200  |
| Enron                   | 0.242 | 4 | 0.7 | 7800  | 1.20e^-5 | 1e^-5 | 300  |
| Imclef07a               | 0.844 | 4 | 0.7 | 2150  | 3.54e^-5 | 1e^-5 | 200  |
| Imclef07d               | 0.544 | 4 | 0.7 | 2450  | 4.02e^-5 | 1e^-5 | 200  |
| Comedy_ontology         | 0.997 | 4 | 0.7 | 6900  | 1.00e^-5 | 1e^-5 | 40  |
| Engineering_ontology    | 0.858 | 4 | 0.7 | 9900  | 1.00e^-5 | 1e^-5 | 40  |
| Law_ontology            | 0.993 | 4 | 0.7 | 9700  | 1.00e^-5 | 1e^-5 | 40  |
| Main_ontology           | 0.763 | 4 | 0.7 | 8300  | 1.00e^-5 | 1e^-5 | 40  |
| Information_ontology    | 0.929 | 4 | 0.7 | 8900  | 1.00e^-5 | 1e^-5 | 40  |
| Mathematics_ontology    | 0.881 | 4 | 0.7 | 8400  | 1.00e^-5 | 1e^-5 | 40  |
| People_ontology         | 0.478 | 4 | 0.7 | 8300  | 1.00e^-5 | 1e^-5 | 40  |
| Philosophy_ontology     | 0.702 | 4 | 0.7 | 7200  | 1.00e^-5 | 1e^-5 | 40  |
| Culture_ontology        | 0.991 | 4 | 0.7 | 8600  | 1.00e^-5 | 1e^-5 | 40  |
| Energy_ontology         | 0.930 | 4 | 0.7 | 8100  | 1.00e^-5 | 1e^-5 | 40  |

## Evaluate DHN

The main entrypoint to evaluate DHN with the parameters obtained from the hyperparameters search is the `main.py` script. The user can run it with th ecommand:

`python main.py --dataset <dataset> --device <device> --seed <seed>`

We additionally provide the `main_script.sh` script as an example that shows how to run 10 trials of evaluation on a dataset.

**Input arguments**
- `dataset`: name of the dataset on which the model is trained and validated.
- `device`: which device to put the nn on.
- `seed`: random seed. Defaults to 0.

**Output**
- The script will save the model outputs (at test time) under `outputs/`, please refer to [this README](outputs/README.md) for more details.
- The script will save the test-time results (scores) under `results/`, please refer to [this README](results/README.md) for more details.
- The script will save the test-time results (scores) computed per-level under `results_per_level/`, please refer to [this README](results_per_level/README.md) for more details.
- The script will save some additional (test-time) scores under `small_pos_results/` and `y_or_n/`. Please refer to the [`small_pos_results/`](small_pos_results/README.md) and [`y_or_n/`](y_or_n/README.md)'s READMEs.

**NOTES**
- The outputs under `outputs/` are overwritten at each run (i.e., trial).
- The script will append the obtained results for each trial in the same file up to 10 trials. After 10 trials the script will ovewrite the result files and start saving scores from scratch.
- The number of features of the datasets we considered are hardcoded in the script (beginning of the `main()` method). Hence, if the user wants to add new datasets they should manually add them.

## Statistics and plots

To compute some statistics related to DHN's performance we provide the following scripts:
- `friedman_nemenyi.py`: this script will consider the performance of our system and of the other baselines for the set of datasets we studied, and:
  - perform the Friedman test (obtaining a p-value)
  - generate a heatmap of the p values from the Friedman test (plot)
  - generate the critical distance diagram (plot) 
- `wilcoxon.py`: this script will calculate the Wilcoxon signed-rank test (obtaining a p-value) between DHN and the other models that appeared to be statistically indistinguishable given the critical distance diagram computed during our study.

We additionally provide the following scripts to plots the results obtained by DHN:
- `results_per_level_plots.py`: this script plots DHN' results computed per level and saved under `results_per_level/`. The plots will be saved under `results_per_level_plots/`. **NOTE**
  - At the beginning of the script the user should specify for which datasets they want to generate the plots.
  - The ontology datasets may present a cyclic structure, so the user may not be able to generate plots for all of them.
- **BONUS** `results_per_level_plots_compared.py`: this script can be used to generate a plot (grouped bar chart) comparing the results per level of DHN with the ones of C-HMCNN, Clus_-_Ens and HMC_LMLP. **NOTE**:
  - This script assumes that the baseline models' results per level are saved under `results_per_level_C_HMCNN`, `results_per_level_Clus` and `results_per_level_HMC_LMLP`.
  - We do not include the code needed to compute the baseline models' results per level in this repository. If the user wants to do it, they could adapt the `average_score_per_level()` method from the `main.py` script.
- `y_or_n_per_level.py`: this script will generate the plots explained in the following.
  - At the beginning of the script the user should specify for which datasets they want to generate the plots. For each specified dataset, the script will:
    - Read the y_or_n results under `y_or_n/` (please refer to [`y_or_n/`'s README](y_or_n/README.md))
    - Plot the mean among the n trials in `y_or_n/` of the **y_or_n percentage** for all the hierarchical levels (`<dataset_name>.png`).
      - **y_or_n percentage**: for each hierarchical level, the y_or_n percentage was computed dividing the y_or_n quantity of every class belonging to that level by the number of samples in the test set, and then computing the mean of the results.
    - Plot the **mean of the ground truth percentage** per level (`<dataset_name>_ground_truths.png`).
      - **mean of the ground truth percentage**: for each hierarchical level the quantity that was plotted was computed by first calculating the mean of the ground truth values set to True out of all the test samples (i.e., the sum of all the True occurrences divided by the number of test samples) for every class belonging to that level, and then computing the mean of the results.
  - **NOTE**: to each generated plot corresponds a csv file (with the same name) containing the values that are being plotted.

## Results

We ran 10 trials (using `main.py` and `main_script.sh`) and obtained the following mean AUPRC scores:

| Dataset | mean AUPRC | 
----------|------------|
| Cellcycle_FUN     | **0.256** |
| Derisi_FUN     | **0.198** |
| Eisen_FUN     | **0.307** |
| Expr_FUN     | <ins>0.301</ins> |
| Gasch1_FUN     | <ins>0.285</ins> |
| Gasch2_FUN     | **0.260**  |
| Seq_FUN     | **0.295**  |
| Spo_FUN     | **0.216**  |
| Cellcyle_GO     | **0.415**  |
| Derisi_GO     | **0.371**  |
| Eisen_GO     | **0.456**  |
| Expr_GO     |  0.448  |
| Gasch1_GO     | **0.440**  |
| Gasch2_GO    | 0.416  |
| Seq_GO      | **0.447**  |
| Spo_GO     | **0.382**  |
| Diatoms      | **0.768**  |
| Enron     | **0.764**  |
| Imclef07a     | **0.958**  |
| Imclef07d     | **0.928**  |
| Main     | **0.898**  |
| Law     | **0.843**  |
| Engineering     | **0.839**  |
| Comedy     | **0.904**  |
| Culture     | **0.702**  |
| Information     | **0.795**  |
| Mathematics     | **0.910**  |
| People     | **0.916**  |
| Philosophy     | **0.742**  |
| Energy     | <ins>0.858</ins>  |

