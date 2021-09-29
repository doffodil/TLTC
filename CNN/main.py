import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
import datetime
from utils import data_helpers, set_rand
from models.text_cnn_tf import TextCNN
from tensorflow.contrib import learn
import gc
gc.collect()

data_files_dir = '../data/semantic_data/'
domains = ['book', 'dvd', 'kitchen', 'electronics']

# =================================================================================================================== #
# train models in each domain
if not os.path.exists("./exp1_cache"):
    for domain in domains:
        os.system(f"python TC_CNN_train.py --domain={domain}")
    for source_domain in domains:
        for target_domain in domains:
            os.system(f"python TC_CNN_eval.py --source_domain={source_domain} --target_domain={target_domain}")

# =================================================================================================================== #
# load result
prediction_result = pd.DataFrame(index=domains, columns=domains)
for source_domain in domains:
    for target_domain in domains:
        dir = f"./exp1_cache/TC_CNN_{source_domain}/{target_domain}_prediction.csv"
        prediction_result[source_domain][target_domain] = pd.read_csv(dir,header=None).rename(columns={0:'text', 1:'label'})

# =================================================================================================================== #
# calculate score
# TODO: calculate score

#
# plot figure
# TODO: plot figure
