import os
import math
import random
import numpy as np
import tensorflow as tf
import pdb

from past.builtins import xrange

class MemN2N(object):
    def __init__(self, config, sess):
        self.nwords = config.nwords
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.nhop = config.nhop
        self.edim = config.edim
        self.mem_size = config.mem_size
        self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm

        self.show = config.show
        self.is_test = config.is_test
        self.checkpoint_dir = config.checkpoint_dir

        self.n_candidates = config.n_candidates
        self.max_sentence_length = config.max_sentence_length

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

        self.input = tf.placeholder(tf.int32, [None, self.max_sentence_length], name="input")
        # self.time = tf.placeholder(tf.int32, [None, self.mem_size], name="time")
        # self.target = tf.placeholder(tf.float32, [self.batch_size, self.n_candidates], name="target")
        self.target = tf.placeholder(tf.float32, [None, self.n_candidates], name="target")
        # self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size, self.edim], name="context")
        self.context = tf.placeholder(tf.int32, [None, self.mem_size, self.max_sentence_length], name="context")
        
        self.hid = []
        # self.hid.append(self.input)
        # self.hid must now have embedding on input
        self.share_list = []
        self.share_list.append([])

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        #self.log_perp = []

    def build_memory(self):
        self.global_step = tf.Variable(0, name="global_step")

        nil_word_embedding = tf.zeros([1, self.edim])

        # A=C.
        # self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.A = tf.Variable( tf.concat(axis=0, values=[ nil_word_embedding, tf.random_normal([self.nwords - 1, self.edim], stddev=self.init_std) ]) , name="A")
        # self.C = self.A Refer to the same thing for now.

        u0 = tf.reduce_sum(tf.nn.embedding_lookup(self.A, self.input), axis=1)

        self.hid.append(u0)
        
        # R
        self.R = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std), name="R")
    
        # W ... comes later.
        # self.W = tf.Variable(tf.random_normal([self.edim, self.n_candidates]), stddev=self.init_std)


        # self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        # self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        # self.C = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))

        # Temporal Encoding
        # self.T_A = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))
        # self.T_B = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))

        # m_i = sum A_ij * x_ij + T_A_i
        Ain_c = tf.reduce_sum( tf.nn.embedding_lookup(self.A, self.context), axis=2 )
        # Ain_t = tf.nn.embedding_lookup(self.T_A, self.time)
        # Ain = tf.add(Ain_c, Ain_t)
        Ain = Ain_c # 

        # Cin = Ain # Sharing the variable here.
        
        '''
            Ain: None * mem_size * edim .... represents embedded memory.
            u0 : None * 1 * edim
        '''

        # pdb.set_trace()

        # c_i = sum B_ij * u + T_B_i
        # Bin_c = tf.nn.embedding_lookup(self.B, self.context)
        # Bin_t = tf.nn.embedding_lookup(self.T_B, self.time)
        # Bin = tf.add(Bin_c, Bin_t)
        for h in xrange(self.nhop):
            print('\nStarting hop ', h, '\n')
            # pdb.set_trace()
            u_in = self.hid[-1]
            # Compute memory-u dot product.

            # hack to get around no reduce_dot
            u_temp = tf.transpose(tf.expand_dims(u_in, -1), [0, 2, 1])
            dotted = tf.reduce_sum(Ain * u_temp, 2)

            # Calculate probabilities
            probs = tf.nn.softmax(dotted)
            
            probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
            
            c_temp = tf.transpose(Ain, [0, 2, 1])

            o_h = tf.reduce_sum(c_temp * probs_temp, 2)

            # pdb.set_trace()

            ro_h = tf.matmul(o_h, self.R) # tf.matmul(self.R, o_h)
            
            # pdb.set_trace()

            u_h = u_in + o_h

            self.hid.append(u_h)
        
        '''
        for h in xrange(self.nhop):
            self.hid3dim = tf.reshape(self.hid[-1], [-1, 1, self.edim])
            Aout = tf.matmul(self.hid3dim, Ain, adjoint_b=True)
            Aout2dim = tf.reshape(Aout, [-1, self.mem_size])
            P = tf.nn.softmax(Aout2dim)

            probs3dim = tf.reshape(P, [-1, 1, self.mem_size])
            Bout = tf.matmul(probs3dim, Bin)
            Bout2dim = tf.reshape(Bout, [-1, self.edim])

            Cout = tf.matmul(self.hid[-1], self.C)
            Dout = tf.add(Cout, Bout2dim)

            self.share_list[0].append(Cout)

            if self.lindim == self.edim:
                self.hid.append(Dout)
            elif self.lindim == 0:
                self.hid.append(tf.nn.relu(Dout))
            else:
                F = tf.slice(Dout, [0, 0], [self.batch_size, self.lindim])
                G = tf.slice(Dout, [0, self.lindim], [self.batch_size, self.edim-self.lindim])
                K = tf.nn.relu(G)
                self.hid.append(tf.concat(axis=1, values=[F, K]))
        '''
    def build_model(self):
        self.build_memory()
        # IMPORTANT: self.hid[-1] is the thing that comes out of the last hop.
        self.W = tf.Variable(tf.random_normal([self.edim, self.n_candidates], stddev=self.init_std), name="W")
        z = tf.matmul(self.hid[-1], self.W)
        print(z)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=self.target)

        self.lr = tf.Variable(self.current_lr)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        params = [self.A, self.R, self.W]
        # pdb.set_trace()
        grads_and_vars = self.opt.compute_gradients(self.loss,var_list=params)
        # clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            # self.optim = self.opt.apply_gradients(clipped_grads_and_vars)
            self.optim = self.opt.apply_gradients(grads_and_vars)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

    def train(self, data):
        N = int(math.ceil(float(len(data[0])) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.max_sentence_length], dtype=np.float32)
        # time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.n_candidates]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size, self.max_sentence_length])

        # x.fill(self.init_hid) ... sad
        # for t in xrange(self.mem_size):
        #     time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Train', max=N)

        # shuffle data.
        # indices = list(range(0, len(data[0])))
        # random.shuffle(indices)
        # for i in range(0, len(data)):
        #     data[i] = data[i][indices]

        for idx in xrange(N):
            if self.show: bar.next()
            target.fill(0)
            x.fill(0)
            context.fill(0)
            for b in xrange(self.batch_size):
                indexIntoData = idx*self.batch_size + b
                if(indexIntoData >= len(data[0])):
                    break

                # pdb.set_trace()
                # m = random.randrange(self.mem_size, len(data))
                x[b] = data[1][indexIntoData]

                # pdb.set_trace()
                target[b] = data[2][indexIntoData]

                # pdb.set_trace()
                context[b] = data[0][indexIntoData]
                
            _, loss, self.step = self.sess.run([self.optim,
                                                self.loss,
                                                self.global_step],
                                                feed_dict={
                                                    self.input: x,
                                                    # self.time: time,
                                                    self.target: target,
                                                    self.context: context})
            cost += np.sum(loss)

        if self.show: bar.finish()
        return cost/N/self.batch_size

    def test(self, data, label='Test'):
        N = int(math.ceil(float(len(data[0])) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.max_sentence_length], dtype=np.float32)
        # time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.n_candidates]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size, self.max_sentence_length])

        # x.fill(self.init_hid)
        # for t in xrange(self.mem_size):
        #     time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar(label, max=N)

        m = self.mem_size

        for idx in xrange(N):
            if self.show: bar.next()
            target.fill(0)
            x.fill(0)
            context.fill(0)
            for b in xrange(self.batch_size):
                indexIntoData = idx*self.batch_size + b
                if(indexIntoData >= len(data[0])):
                    break

                # pdb.set_trace()
                # m = random.randrange(self.mem_size, len(data))
                x[b] = data[1][indexIntoData]

                # pdb.set_trace()
                target[b] = data[2][indexIntoData]

                # pdb.set_trace()
                context[b] = data[0][indexIntoData]
                
            loss = self.sess.run([self.loss], feed_dict={self.input: x,
                                                         # self.time: time,
                                                         self.target: target,
                                                         self.context: context})
            cost += np.sum(loss)

        if self.show: bar.finish()
        return cost/N/self.batch_size

    def run(self, train_data, test_data):
        if not self.is_test:
            for idx in xrange(self.nepoch):
                train_loss = np.sum(self.train(train_data))
                test_loss = np.sum(self.test(test_data, label='Validation'))

                # Logging
                self.log_loss.append([train_loss, test_loss])
                #self.log_perp.append([math.exp(train_loss), math.exp(test_loss)])

                state = {
                    'train_loss': train_loss,
                    'epoch': idx,
                    'learning_rate': self.current_lr,
                    'test_loss': test_loss
                }
                print(state)

                # Learning rate annealing
                if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx-1][1] * 0.9999:
                    self.current_lr = self.current_lr / 1.5
                    self.lr.assign(self.current_lr).eval()
                if self.current_lr < 1e-5: break

                if idx % 10 == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "MemN2N.model"),
                                    global_step = self.step.astype(int))
        else:
            self.load()

            valid_loss = np.sum(self.test(train_data, label='Validation'))
            test_loss = np.sum(self.test(test_data, label='Test'))

            state = {
                'valid_perplexity': math.exp(valid_loss),
                'test_perplexity': math.exp(test_loss)
            }
            print(state)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Trest mode but no checkpoint found")
