
import tensorflow as tf
import numpy as np
import misc

class model():
    def __init__(self, state_shape, n_actions):

        tf.set_random_seed(42)
        np.random.seed(42)

        def weight(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        def bias(shape):
            return tf.Variable(tf.constant(0.1, shape=shape))

        with tf.name_scope('input'):
            self.state_in = tf.placeholder(tf.float32, 
                    (None,) + state_shape)
            self.reward_in = tf.placeholder(tf.float32, [None])
            self.advantage_in = tf.placeholder(tf.float32, [None])
            self.action_in = tf.placeholder(tf.float32, [None, n_actions])
            self.keep_prob = tf.placeholder(tf.float32)
            self.explore = tf.placeholder(tf.float32)

        '''
        #3x3 conv2d, relu, 2x2 maxpool
        with tf.name_scope('conv_pool'):
            #filter shape = [height, width, in_channels, 
            #out_channels]
            state_in = tf.reshape(self.state_in, (-1,) + state_shape + (1,))
            out_channels = 4 #FIXME: out_channels hardcoded
            filter_shape = [3, 3, 1, out_channels]
            conv_w = tf.Variable(tf.truncated_normal(filter_shape, 
                    stddev=0.1))
            conv_b = tf.Variable(tf.constant(0.1, 
                    shape=[out_channels]))
            conv = tf.nn.conv2d(state_in, conv_w, 
                    strides=[1,1,1,1], padding='SAME')
            relu = tf.nn.relu(conv + conv_b)
            pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], 
                    strides=[1,2,2,1], padding='SAME')
        '''

        with tf.name_scope('dense_dropout'):
            n = 512
            flat = tf.contrib.layers.flatten(self.state_in)
            w_dense = weight([state_shape[0] * state_shape[1], n])
            b_dense = bias([n])

            dense = tf.nn.relu(tf.matmul(flat, w_dense) + b_dense)
            #FIXME: add dropout
            #drop = tf.nn.dropout(dense, self.keep_prob)
            drop = dense

        with tf.name_scope('policy'):
            w_class = weight([n, n_actions])
            b_class = bias([n_actions])

            logits_class = tf.matmul(drop, w_class) + b_class
            probs_class = tf.nn.softmax(logits_class) + 1e-8
            logprobs_class = tf.nn.log_softmax(logits_class) + 1e-8
            self.action = tf.argmax(probs_class, axis=1)

            '''
            #sample a 0 or 1 based on logprob in self.explore
                #[[no explore, explore]]
                #squeeze(shape([1,1])) --> shape([,])
            explore_distribution = tf.log([[1 - self.explore, 
                    self.explore]])
            explore_sample = tf.multinomial(explore_distribution, 1)
            explore_flag = tf.cast(tf.squeeze(explore_sample), tf.float32)

            #add random explore probs per action if explore_flag = 1
            rand = tf.random_uniform([n_actions])
            explore_probs = probs_class + tf.multiply(explore_flag, 
                    tf.random_shuffle(rand))
            '''

        with tf.name_scope('value'):
            w_val = weight([n,1])
            b_val = bias([1])

            logit_val = tf.matmul(drop, w_val) + b_val
            self.value = tf.reduce_sum(logit_val, axis=1)

            '''
            action_oh = tf.one_hot(self.action, n_actions)
            self.value = tf.reduce_sum(tf.multiply(probs_val, action_oh), 
                    axis=1)
            '''

        with tf.name_scope('loss'):
            #policy loss
            entropy = -tf.reduce_sum(probs_class * logprobs_class)
            action_probs = tf.reduce_sum(logprobs_class * self.action_in, 
                    [1])
            #policy_loss = tf.reduce_sum(-(action_probs * self.advantage_in)                     + 0.01 * entropy)
            
            policy_loss = -tf.reduce_sum(action_probs * self.advantage_in)

            #value loss
            #value_loss = tf.reduce_sum(tf.squared_difference(self.value, 
            #        self.reward_in))
            value_loss = 0.05 * tf.reduce_sum(tf.squared_difference(
                    self.value, self.reward_in))

            #total loss
            self.loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        with tf.name_scope('optimize'):
            self.step = tf.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            self.optimize = optimizer.minimize(self.loss, 
                    global_step=self.step)

        with tf.name_scope('summary'):
            tf.summary.scalar('1_total_loss', self.loss)
            tf.summary.scalar('2_value_loss', value_loss)
            tf.summary.scalar('3_policy_loss', policy_loss)
            tf.summary.scalar('4_entropy', entropy)
            self.summaries = tf.summary.merge_all()

        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter('./logs', self.loss.graph)
        self.sess.run(tf.global_variables_initializer())

    def act(self, state, explore=0.2):
        action, value = self.sess.run([self.action, self.value], 
                feed_dict={
                    self.state_in: [state],
                    self.keep_prob: 1.0,
                    self.explore: explore,
                })
        return action[0], value[0]

    def learn(self, batch):
        states, actions, rewards, values, dones, next_states = batch
        loss, _, step, summ = self.sess.run([self.loss, self.optimize,
                self.step, self.summaries],
                feed_dict={
                    self.state_in: states,
                    self.action_in: actions,
                    self.reward_in: rewards,
                    self.advantage_in: values,
                    self.keep_prob: 0.5,
                    self.explore: 0.0,
                })
        self.writer.add_summary(summ, step)
        return loss


#TODO: 
    #stats in train
    #epsilon exploration

