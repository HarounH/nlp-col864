# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"

import tensorflow as tf
logging = tf.logging
logging.set_verbosity(logging.INFO)

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    logging.info('Creating vocabulary %s from data %s' % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 1000000 == 0:
          logging.info("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    logging.info("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            logging.info("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_seq_2_seq_data(data_dir, en_vocabulary_size, fr_vocabulary_size, tokenizer, normalize_digits=False):

  train_path = os.path.join(data_dir, 'data.train')
  dev_path = os.path.join(data_dir, 'data.dev')


  # Create vocabularies of the appropriate sizes.
  fr_vocab_path = os.path.join(data_dir, "vocab%d.fr" % fr_vocabulary_size)
  en_vocab_path = os.path.join(data_dir, "vocab%d.en" % en_vocabulary_size)
  create_vocabulary(fr_vocab_path, train_path + ".fr", fr_vocabulary_size, tokenizer, normalize_digits=normalize_digits)
  create_vocabulary(en_vocab_path, train_path + ".en", en_vocabulary_size, tokenizer, normalize_digits=normalize_digits)

  # Create token ids for the training data.
  fr_train_ids_path = train_path + (".ids%d.fr" % fr_vocabulary_size)
  en_train_ids_path = train_path + (".ids%d.en" % en_vocabulary_size)
  data_to_token_ids(train_path + ".fr", fr_train_ids_path, fr_vocab_path, tokenizer, normalize_digits=normalize_digits)
  data_to_token_ids(train_path + ".en", en_train_ids_path, en_vocab_path, tokenizer, normalize_digits=normalize_digits)

  # Create token ids for the development data.
  fr_dev_ids_path = dev_path + (".ids%d.fr" % fr_vocabulary_size)
  en_dev_ids_path = dev_path + (".ids%d.en" % en_vocabulary_size)
  data_to_token_ids(dev_path + ".fr", fr_dev_ids_path, fr_vocab_path, tokenizer, normalize_digits=normalize_digits)
  data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path, tokenizer, normalize_digits=normalize_digits)

  return (en_train_ids_path, fr_train_ids_path,
          en_dev_ids_path, fr_dev_ids_path,
          en_vocab_path, fr_vocab_path)

def create_dialog_data(data_file, encoder_file, decoder_file, only_supporting=False):
  
  outContext = open(encoder_file, "wb")
  outResponse = open(decoder_file, "wb")
  outContext.close()
  outResponse.close()
  outContext = open(encoder_file, "ab")
  outResponse = open(decoder_file, "ab")
  
  #to print statistics
  outcount = open('count.tsv', "wb")
  outcount.close()
  outcount = open('count.tsv', "ab")
  linedict={}
  respdict={}

  with open(data_file) as f:
      mem = []
      dbquery = []
      restaurant = ''
      for line in f.readlines():
          if line.strip():
              nid, line = line.split(' ', 1)
              nid = int(nid)
              line = line.rstrip()
              if '\t' in line:
                  q,a= line.split('\t',1)
                  linecount = 0
                  input_utterance = []
                  if len(dbquery) > 0:
                    mem.append(dbquery)
                    dbquery = []
                    restaurant = ""
                  for memElement in mem:
                    for memWord in memElement:
                      input_utterance.append(memWord)
                      linecount = linecount+1
                    input_utterance.append("<EOS> ")
                    linecount = linecount+1
                  q = q.split(' ')
                  input_utterance.extend(q)
                  if len(input_utterance) > 500:
                    outContext.write(" ".join(str(x) for x in input_utterance[-500:]))
                  else :
                     outContext.write(" ".join(str(x) for x in input_utterance))
                  outContext.write("\n")
                  cleaned_a = ' '.join(a.split()[:-1])
                  outResponse.write(cleaned_a + "\n")
                  
                  # to print statisctics
                  linecount = linecount+len(q)
                  if linecount in linedict:
                    linedict[linecount] = linedict[linecount]+1
                    respdict[linecount].append(len(cleaned_a.split(' ')))
                  else:
                    linedict[linecount] = 1
                    respdict[linecount] = []
                    respdict[linecount].append(len(cleaned_a.split(' ')))

                  mem.append(q)
                  mem.append(cleaned_a.split(' '))
              else:
                  if line == 'api_call no result':
                    dbquery.append('api_call_no_result')
                  else:
                    lsplit = line.split(' ')
                    if restaurant != lsplit[0]:
                      if len(dbquery) > 0:
                        mem.append(dbquery)
                        dbquery = []
                      restaurant = lsplit[0]
                      dbquery.append(restaurant)
                      dbquery.append(lsplit[2])
                    else:
                      dbquery.append(lsplit[2])
          else:
              mem = []
              dbquery = []
              restaurant = ''
      
  for length, count in linedict.iteritems():
    outcount.write(str(length) + "\t" + str(count) + "\t" + str(respdict[length]) + "\n")
  outcount.close()
  outContext.close()
  outResponse.close()


def main():
  
  # train
  create_dialog_data('../weston_baseline/data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-trn-template.txt', 'data/data.train.en', 'data/data.train.fr')
  # dev
  create_dialog_data('../weston_baseline/data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-dev-template.txt', 'data/data.dev.en', 'data/data.dev.fr')
  # test
  create_dialog_data('../weston_baseline/data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-tst-template.txt', 'data/data.test.en', 'data/data.test.fr')
  
  #create_dialog_data('../weston_baseline/data/dialog-bAbI-tasks/small-trn-template.txt', 'small/data.train.en', 'small/data.train.fr')
  #create_dialog_data('../weston_baseline/data/dialog-bAbI-tasks/small-trn-template.txt', 'small/data.dev.en', 'small/data.dev.fr')

if __name__ == "__main__":
  main()