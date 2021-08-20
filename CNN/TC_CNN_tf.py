import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
import datetime
from utils import data_helpers, set_rand
from text_cnn import TextCNN
from tensorflow.contrib import learn
import gc


# =================================================================================================================== #
# data parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_dir", default="../data/semantic_data/", help="data files location")
tf.flags.DEFINE_string("source_domain", default="book", help="source domain of data")
tf.flags.DEFINE_string("target_domain", default="dvd", help="target domain of data")

# Model Hyper_parameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("SEED", 2, "random seed")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 50)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


# =================================================================================================================== #
# set random seed
set_rand.set_random_seed(FLAGS.SEED)


# =================================================================================================================== #
def data_preprocess(source_domain = 'book', target_domain = 'dvd'):
    # Load data
    print("Loading data...")

    source_positive_data_file =  FLAGS.data_dir + FLAGS.source_domain + "_positive_1000.txt"
    source_negative_data_file = FLAGS.data_dir + FLAGS.source_domain + "_negative_1000.txt"
    target_positive_data_file = FLAGS.data_dir + FLAGS.target_domain + "_positive_1000.txt"
    target_negative_data_file = FLAGS.data_dir + FLAGS.target_domain + "_negative_1000.txt"
    sourcce_x_text, source_y = data_helpers.load_data_and_labels(source_positive_data_file, source_negative_data_file)
    target_x_text, target_y = data_helpers.load_data_and_labels(target_positive_data_file, target_negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in (sourcce_x_text + target_x_text)])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(sourcce_x_text + target_x_text)))
    source_x = x[0:2000]
    target_x = x[-2000::]

    # Randomly shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(source_x)))
    x_shuffled = source_x[shuffle_indices]
    y_shuffled = source_y[shuffle_indices]

    x_train = x_shuffled
    y_train = y_shuffled
    x_dev = target_x
    y_dev = target_y

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print(f"source domain:{FLAGS.source_domain}  =>  target domain:{FLAGS.target_domain}")
    return x_train, y_train, vocab_processor, x_dev, y_dev


# =================================================================================================================== #
# train and eval
def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = int(time.time())
            cache_dir = 'TC_CNN_' + FLAGS.source_domain + '_to_' + FLAGS.target_domain + '_' + str(timestamp)
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "exp1", cache_dir))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # train
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            # eval
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """

                # generate batches
                step = []
                loss = []
                accuracy = []
                batches = data_helpers.batch_iter(
                    list(zip(x_train, y_train)), 200,1)
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0
                    }
                    stepi, summaries, lossi, accuracyi = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    step.append(stepi)
                    loss.append(lossi)
                    accuracy.append(accuracyi)
                step1 = np.mean(step)
                loss1 = np.mean(loss)
                accuracy1 = np.mean(accuracy)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step1, loss1, accuracy1))
                if writer:
                    writer.add_summary(summaries, step1)

            # Generate batches
            source_batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs,FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in source_batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = data_preprocess(FLAGS.source_domain, FLAGS.target_domain)
    gc.collect()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()