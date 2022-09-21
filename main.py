import csv

import numpy as np
import pandas as pd

from functions import *

spam_path = "/Users/mogamimorimodo/CSDS440-ML/440data/voting/voting.data"
train_data = pd.read_csv(spam_path)

tree = ID3(train_data, "1.1")
test_data_m = pd.read_csv(spam_path)

accuracy = evaluate(tree, test_data_m, "1.1")

print("accuracy is: ", accuracy)
