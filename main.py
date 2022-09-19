import csv

import numpy as np
import pandas as pd

from functions import *

train_data = pd.read_csv("file name")

tree = ID3(train_data, 'label')
test_data_m = pd.read_csv("file name")

accuracy = evaluate(tree, test_data_m, 'label')

