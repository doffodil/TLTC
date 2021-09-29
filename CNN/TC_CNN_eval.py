#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from utils import data_helpers
from models.text_cnn_tf import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_dir", default="../data/semantic_data/", help="data files location")
tf.flags.DEFINE_string("source_domain", 'book', "source domain")
tf.flags.DEFINE_string("target_domain", 'dvd', "target domain")

# model Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# load data
positive_data_file =  FLAGS.data_dir + FLAGS.target_domain + "_positive_1000.txt"
negative_data_file = FLAGS.data_dir + FLAGS.target_domain + "_negative_1000.txt"
x_text, y_test = data_helpers.load_data_and_labels(positive_data_file, negative_data_file)
y_test = np.argmax(y_test, axis=1)

# Map data into vocabulary
source_model_path = f'./exp1_cache/TC_CNN_{FLAGS.source_domain}/'
vocab_path = os.path.join(source_model_path,"vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_text)))

# checkpoint
checkpoint_dir = f'{source_model_path}/checkpoints/'

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print(f"{FLAGS.source_domain} ===> {FLAGS.target_domain}")
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_text), all_predictions))
out_path = os.path.join(source_model_path + f"{FLAGS.target_domain}_prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
