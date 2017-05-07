import os, codecs, argparse
import tensorflow as tf
import cPickle as pkl
import numpy as np
import seq2seq_model
from data_utils import initialize_vocabulary, sentence_to_token_ids

from nltk.tokenize import word_tokenize as tokenizer

from commons import read_stopw, replace_line, replace_phrases, get_diff_map, merge_parts, \
    get_rev_unk_map, generate_missing_symbol_candidates, generate_new_q1_candidates, get_bleu_score, \
    execute_bleu_command, get_unk_map, convert_phrase,  generate_q2_anaphora_candidates, merge_and_sort_scores

#Constants
from commons import CONFIG_FILE, SUBTREE, LEAVES, RAW_CANDIDATES, DEV_INPUT, DEV_OUTPUT, ORIG_PREFIX, \
    ALL_HYP, ALL_REF, CANDIDATES_SUFFIX, STOPW_FILE, RESULTS_SUFFIX, NOT_SET, SCORES_SUFFIX, UNK_SET, \
    TEST_INPUT, TEST_OUTPUT, TEST


logging = tf.logging


class PendingWork:
    def __init__(self, prob, tree, prefix):
        self.prob = prob
        self.tree = tree
        self.prefix = prefix

    def __str__(self):
        return 'Str=%s(%f)'%(self.prefix, self.prob)


class Score(object):
    def __init__(self, candidate, candidate_unk):
        self.candidate = candidate.strip()
        self.candidate_unk = candidate_unk.strip()
        self.seq2seq_score = NOT_SET
        self.bleu_score = NOT_SET

    def set_seq2seq_score(self, prob):
        self.seq2seq_score = prob

    def set_bleu_score(self, gold_line, base_dir):
        self.bleu_score = get_bleu_score(gold_line, convert_phrase(self.candidate), base_dir=base_dir)

    def __str__(self):
        return 'C    : %s\nC_UNK: %s\nS:%f B:%f'%(self.candidate,
                                                 self.candidate_unk, self.seq2seq_score, self.bleu_score)

class NSUResult(object):
    def __init__(self, training_scores,
                 q1_scores, q2_scores, missing_scores, kw_scores):
        self.training_scores = training_scores
        self.q1_scores = q1_scores
        self.q2_scores = q2_scores
        self.missing_scores = missing_scores
        self.kw_scores = kw_scores

    def set_input(self, input_seq, input_seq_unk, unk_map, rev_unk_map, gold_line):
        self.input_seq = input_seq
        self.input_seq_unk = input_seq_unk
        self.unk_map = unk_map
        self.rev_unk_map = rev_unk_map
        if gold_line is not None:
            self.gold_line = gold_line.strip()

class Decode(object):
    def __init__(self, models_dir, debug=True):
        config_file_path = os.path.join(models_dir, CONFIG_FILE)
        if debug:
            logging.set_verbosity(tf.logging.DEBUG)
        else:
            logging.set_verbosity(tf.logging.INFO)
        self.debug = debug

        logging.info('Loading Pre-trained seq2model:%s' % config_file_path)
        config = pkl.load(open(config_file_path))
        logging.info(config)

        self.session = tf.Session()
        self.model_path = config['train_dir']
        self.data_path = config['data_dir']
        self.src_vocab_size = config['src_vocab_size']
        self.target_vocab_size = config['target_vocab_size']
        self._buckets = config['_buckets']

        self.model = seq2seq_model.Seq2SeqModel(
            source_vocab_size = config['src_vocab_size'],
            target_vocab_size = config['target_vocab_size'],
            buckets=config['_buckets'],
            size = config['size'],
            num_layers = config['num_layers'],
            max_gradient_norm = config['max_gradient_norm'],
            batch_size=1,
            learning_rate=config['learning_rate'],
            learning_rate_decay_factor=config['learning_rate_decay_factor'],
            forward_only=True)

        ckpt = tf.train.get_checkpoint_state(config['train_dir'])
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.model.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            logging.error('Model not found!')
            return None


        # Load vocabularies.
        en_vocab_path = os.path.join(self.data_path,
                                     "vocab%d.en" % self.src_vocab_size)
        fr_vocab_path = os.path.join(self.data_path,
                                     "vocab%d.fr" % self.target_vocab_size)

        self.en_vocab, _ = initialize_vocabulary(en_vocab_path)
        self.fr_vocab, self.rev_fr_vocab = initialize_vocabulary(fr_vocab_path)

    def set_output_tokens(self, output_token_ids, decoder_inputs):
        for index in range(len(output_token_ids)):
            if index + 1 >= len(decoder_inputs):
                logging.debug('Skip assignment Decoder_Size:%d Outputs_Size:%d'%(len(decoder_inputs), len(output_token_ids)))
                return
            decoder_inputs[index + 1] = np.array([output_token_ids[index]], dtype=np.float)

    def compute_fraction(self, logit, token_index):
        sum_all = np.sum(np.exp(logit))
        print '-----' + str(token_index) + '-----'
        print str(token_index) + ' : ' + str(np.exp(logit[token_index]) / sum_all)
        print '------------'
        return np.exp(logit[token_index]) / sum_all

    def normalize(self, logit, length):
        sum=0
        for i in range(length):
           sum = sum + np.exp(logit[i])
        print 'sum : ' + str(sum)
        normalized_prob=[]
        for i in range(length):
            print str(i) + " : " + str(np.exp(logit[i]) / sum)   
            normalized_prob.append(np.exp(logit[i]) / sum)
        return normalized_prob

    #Compute probability of output_sentence given an input sentence
    def compute_prob(self, sentence, output_sentence):
        token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), self.en_vocab, normalize_digits=False)
        output_token_ids = sentence_to_token_ids(tf.compat.as_bytes(output_sentence), self.fr_vocab, normalize_digits=False)

        bucket_ids = [b for b in xrange(len(self._buckets))
                      if self._buckets[b][0] > len(token_ids) and self._buckets[b][1] > len(output_token_ids)+1]

        if len(bucket_ids) == 0:
            bucket_id = len(self._buckets) - 1
        else:
            bucket_id = min(bucket_ids)

        print 'bucket ids : ' + str(bucket_id)

        encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)

        self.set_output_tokens(output_token_ids, decoder_inputs)
        _, _, output_logits = self.model.step(self.session, encoder_inputs, decoder_inputs,
                                              target_weights, bucket_id, True)

        if len(decoder_inputs) > len(output_token_ids):
            max_len = len(output_token_ids)
        else:
            max_len = len(decoder_inputs)

        prob = np.sum([self.compute_fraction(output_logits[index][0], output_token_ids[index])
                       for index in range(max_len)]) / max_len

        print prob
        #print len(output_logits[max_len][0])
        count=0
        normalized_output = self.normalize(output_logits[max_len][0], len(self.rev_fr_vocab))
        print 'normalized_output : ' + str(normalized_output)
        score_word = zip(normalized_output, self.rev_fr_vocab)
        print 'score_word : ' + str(score_word)
        score_word.sort(reverse=True)
        print '---------------------'
        print 'reversed score_word : ' + str(score_word)
        for score,word in score_word:
            print word + ' : ' + str(score)
            count = count+1
            if count == 25:
                break

        return prob

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Trained Model Directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = setup_args()
    decode = Decode(args.model_dir)
    #decode.compute_prob('<SILENCE>', 'Hello , welcome to the Cambridge restaurant system . You can ask for restaurants by area , price range or food type . How may I help you ?')
    decode.compute_prob('<SILENCE> <EOS> Hello , welcome to the Cambridge restaurant system . You can ask for restaurants by area , price range or food type . How may I help you ? <EOS> i want a moderately priced restaurant in the west part of town', 'api_call')