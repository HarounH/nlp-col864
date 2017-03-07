"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import read_dstc2_data, vectorize_data
from sklearn import cross_validation, metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 25, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 4, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 1000, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 128, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 10, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/dialog-bAbI-tasks/", "Directory containing dialog bAbI task")
FLAGS = tf.flags.FLAGS

#print("Started Task:", FLAGS.task_id)

# task data
train = read_dstc2_data(FLAGS.data_dir+'dialog-babi-task6-dstc2-trn.txt') + read_dstc2_data(FLAGS.data_dir+'dialog-babi-task6-dstc2-dev.txt')
test = read_dstc2_data(FLAGS.data_dir+'dialog-babi-task6-dstc2-tst.txt')
data = train + test

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q) for s, q, a in data)))
print(len(vocab))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

idx=1
candidate_idx={}
with open(FLAGS.data_dir+'dialog-babi-task6-dstc2-candidates.txt') as cFile:
    for line in cFile.readlines():
        nid, line = line.split(' ', 1)
        candidate_idx[line.rstrip()]=idx
        idx=idx+1

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)
answer_size = len(candidate_idx)+1

# Add time words/indexes
for i in range(memory_size):
    word_idx['time{}'.format(i+1)] = 'time{}'.format(i+1)

vocab_size = len(word_idx) + 1 # +1 for nil word
sentence_size = max(query_size, sentence_size) # for the position
sentence_size += 1  # +1 for time words

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
trainS, trainQ, trainA = vectorize_data(train, word_idx, sentence_size, memory_size, candidate_idx)
valS, valQ, valA = vectorize_data(test, word_idx, sentence_size, memory_size, candidate_idx)
testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size, candidate_idx)

print("Training set shape", trainS.shape)

# params
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size,answer_size,session=sess,hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm)

    #model.restore(100)

    for t in range(1, FLAGS.epochs+1):
        # Stepped learning rate
        if t - 1 <= FLAGS.anneal_stop_epoch:
            anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
        else:
            anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
        lr = FLAGS.learning_rate / anneal

        np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in batches:
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            cost_t = model.batch_fit(s, q, a, lr)
            total_cost += cost_t

        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s = trainS[start:end]
                q = trainQ[start:end]
                pred = model.predict(s, q)
                train_preds += list(pred)

            val_preds = model.predict(valS, valQ)
            train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
            val_acc = metrics.accuracy_score(val_preds, val_labels)

            model.save(t)

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')

    test_preds = model.predict(testS, testQ)
    test_acc = metrics.accuracy_score(test_preds, test_labels)
    print("Testing Accuracy:", test_acc)
