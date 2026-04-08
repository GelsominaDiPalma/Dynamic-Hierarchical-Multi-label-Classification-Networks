import os

os.environ["DATA_FOLDER"] = "./"

from utils.parser import *

dataset_name = ["eisen_GO"]

filename, epoch, score, _ = parse_logs(dataset_name[0])

print(dataset_name[0] + " - params: " + filename + ", epoch: " + str(epoch) + ", score: " + str(score) + "\n")
