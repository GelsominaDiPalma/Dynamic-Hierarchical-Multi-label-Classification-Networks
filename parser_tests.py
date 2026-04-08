import os

os.environ["DATA_FOLDER"] = "./"

from utils.parser import *

dataset_name = ["spo_GO"]

# scores order: average precision score(2), f1 score(3), subset accuracy(4), hamming loss(5), hamming score(6) (=label based accuracy)
idx=6

mean_0, std_0 = mean_results(dataset_name[0], idx)

print(dataset_name[0]+"- avg: "+str(mean_0)+", std: "+str(std_0)+"\n")