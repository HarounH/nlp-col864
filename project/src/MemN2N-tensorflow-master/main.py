import os
import pprint
import numpy as np
import tensorflow as tf

from data import read_data, read_dstc2_data, vectorize_data
from model import MemN2N
from itertools import chain
from six.moves import range, reduce

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 150, "internal state dimension [150]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 6, "number of hops [6]")
flags.DEFINE_integer("mem_size", 100, "memory size [100]")
flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_string("data_dir", "../weston_baseline/data/dialog-bAbI-tasks", "data directory [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
# flags.DEFINE_string("data_name", "ptb", "data set name [ptb]")
flags.DEFINE_string("data_name", "dialog-babi-task6-dstc2", "data set name [dialog-babi-task6-dstc2]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", False, "print progress [False]")
flags.DEFINE_string("candidate_filename","dialog-babi-task6-dstc2-candidates.txt","file containing valid candidates")
flags.DEFINE_integer("n_candidates", 1, "Number of candidates to pick from")

FLAGS = flags.FLAGS

def main(_):
	count = []
	word2idx = {}

	if not os.path.exists(FLAGS.checkpoint_dir):
	  os.makedirs(FLAGS.checkpoint_dir)
	'''
	train_data = read_data('%s/%s-trn.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
	valid_data = read_data('%s/%s-dev.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
	test_data = read_data('%s/%s-tst.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
	'''
	raw_train = read_dstc2_data('%s/%s-trn.txt' % (FLAGS.data_dir, FLAGS.data_name))
	raw_test = read_dstc2_data('%s/%s-tst.txt' % (FLAGS.data_dir, FLAGS.data_name))
	raw_val = read_dstc2_data('%s/%s-dev.txt' % (FLAGS.data_dir, FLAGS.data_name))

	raw_data = raw_train + raw_test + raw_val

	vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q) for s, q, a in raw_data)))
	print(len(vocab))
	extra_words = FLAGS.mem_size-1 + 2 # One for each position and one for system, one for user.
	word2idx = dict((c, i + extra_words) for i, c in enumerate(vocab,0))

	idx=1
	candidate2idx={}
	with open('%s/%s' % (FLAGS.data_dir, FLAGS.candidate_filename)) as cFile:
		for line in cFile.readlines():
			nid, line = line.split(' ', 1)
			candidate2idx[line.rstrip()]=idx
			idx=idx+1

	FLAGS.n_candidates = len(candidate2idx)


	max_story_size = max(map(len, (s for s, _, _ in raw_data)))
	mean_story_size = int(np.mean([ len(s) for s, _, _ in raw_data ]))
	sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in raw_data)))
	query_size = max(map(len, (q for _, q, _ in raw_data)))
	n_memory_cells = min(FLAGS.mem_size, max_story_size)
	answer_size = len(candidate2idx) # +1 TODO ... do we need +1???

	# We now have the candidates and everything else.

	idx2word = dict(zip(word2idx.values(), word2idx.keys()))
	FLAGS.nwords = len(word2idx)


	sentence_size = max(query_size, sentence_size) # for the position
	sentence_size += 2  # +1 for time words, +1 for utterer
	max_sentence_length = sentence_size

	# Vectorise data.
	train_data = vectorize_data(raw_train, max_sentence_length, n_memory_cells, word2idx, candidate2idx)
	test_data = vectorize_data(raw_test, max_sentence_length, n_memory_cells, word2idx, candidate2idx)
	val_data = vectorize_data(raw_val, max_sentence_length, n_memory_cells, word2idx, candidate2idx)

	print('Done loading data')

	pp.pprint(flags.FLAGS.__flags)

	# Do model things.
	with tf.Session() as sess:
		model = MemN2N(FLAGS, sess)
		model.build_model()

		if FLAGS.is_test:
			model.run(valid_data, test_data)
		else:
			model.run(train_data, valid_data)

if __name__ == '__main__':
	tf.app.run()
