
import tensorflow as tf
import numpy as np
import math
import misc

class model():
    def __init__(self, state_shape, n_actions):

        tf.set_random_seed(42)
        np.random.seed(42)

        def weight(shape, conv=False):
            #xavier init from tensorflow contrib layers implementation
            f = 6.0 if conv else 2.3
            in_plus_out = shape[-2] + shape[-1]
            std = math.sqrt(f / in_plus_out)
            w = tf.truncated_normal(shape, mean=0.0, stddev=std)
            return tf.Variable(w)

        def bias(shape, v=0.0):
            #bias init to 0.0 in xavier paper
            return tf.Variable(tf.constant(v, shape=shape))

        def conv3x3_layer(x, out_channels):
            #filter shape = [height, width, in_channels, out_channels]
            state_in = tf.reshape(x, (-1,) + state_shape + (1,))
            out_channels = 16 #FIXME: out_channels hardcoded
            filter_shape = [3, 3, 1, out_channels]
            w_conv = weight(filter_shape, conv=True)
            b_conv = bias([out_channels])
            conv = tf.nn.conv2d(state_in, w_conv, 
                    strides=[1,1,1,1], padding='SAME')
            return tf.nn.relu(conv + b_conv)

        def dense_layer(x, n=512, drop=False, relu=False):
            w = weight([x.get_shape()[-1].value, n])
            b = bias([n])
            dense = tf.matmul(x, w) + b
            dense = tf.nn.relu(dense) if relu else dense
            dense = tf.nn.dropout(dense, self.keep_prob) if drop else dense
            return dense

        def minimize(x, rate=1e-3):
            step = tf.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(rate)
            return optimizer.minimize(x, global_step=step)

        with tf.name_scope('input'):
            self.state_in = tf.placeholder(tf.float32, 
                    (None,) + state_shape)
            self.reward_in = tf.placeholder(tf.float32, [None])
            self.advantage_in = tf.placeholder(tf.float32, [None])
            self.action_in = tf.placeholder(tf.float32, [None, n_actions])
            self.keep_prob = tf.placeholder(tf.float32)

        '''
        with tf.name_scope('working'):
            #focuses on overfitting to recent past
            dense = dense_relu(flat, 256)

            #FIXME: this needs loss so can train on last x steps (games?)
                #simple squared error of reward
            #rescale rewards to between [0,1]?
            prob_work = tf.nn.softmax(drop_work) + 1e-8
            logprob_work = tf.nn.log_softmax(drop_work) + 1e-8
            pred_work = tf.reduce_sum(logprobs_work * self.action_in, [1])
            self.loss_work = 0.05 * tf.reduce_sum(tf.square(
                    pred_work - self.reward_in))
            self. optimize = minimize(self.loss_work)
        '''

        '''
        with tf.name_scope('task'):
            #focuses on learning the current task via a3c
            n = 512
            #dense dropout layer
            w_task = weight([out_channels * state_shape[0]**2, n])
            b_task = bias([n])
            dense_task = tf.nn.relu(tf.matmul(flat, w_task) + b_task)
            #FIXME: simplify the net for debugging purposes
            #drop_task = tf.nn.dropout(dense_task, self.keep_prob)
            drop_task = dense_task

            #a3c loss
            w_class = weight([n, n_actions])
            b_class = bias([n_actions])
            logits_class = tf.matmul(drop, w_class) + b_class
            probs_class = tf.nn.softmax(logits_class) + 1e-8
            logprobs_class = tf.nn.log_softmax(logits_class) + 1e-8
            w_val = weight([n,1])
            b_val = bias([1])
            logit_val = tf.matmul(drop, w_val) + b_val
            self.value = tf.reduce_sum(logit_val, axis=1)
            entropy = - tf.reduce_sum(probs_class * logprobs_class)
            probs_act = tf.reduce_sum(logprobs_class * self.action_in, [1])
            policy_loss = - tf.reduce_sum(probs_act * self.advantage_in)
            value_loss = 0.05 * tf.reduce_sum(tf.square(
                    self.value - self.reward_in))
            self.loss_task = policy_loss + 0.5 * value_loss - 0.01 * entropy
            self.step_task = tf.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            self.optimize = optimizer.minimize(self.loss_task, 
                    global_step=self.step_task)
        '''
        '''
        with tf.name_scope('long'):
            #focuses on remembering the learned task
            #FIXME: input is weights of task
                #input is episode i weights, episode i reward sum
                #loss is squared difference (rewards - pred reward)
                #given reward input can also predict weight output??
                #that way task net can pull new weights at beginning
                    #of iteration
            w_long = weight([out_channels * state_shape[0]**2, n])
            b_long = bias([n])
            dense_long = tf.nn.relu(tf.matmul(flat, w_long) + b_long)
            #FIXME: simplify the net for debugging purposes
            #drop_long = tf.nn.dropout(dense_long, self.keep_prob)
            drop_long = dense_long

            #FIXME: totally different losses

        with tf.name_scope('filter'):
            w_filter = weight([n, 3])
            b_filter = bias([3])
            dense_filter = tf.nn.relu(tf.matmul(flat, w_filter) + b_filter)
            #FIXME: make sure this works...
            drop_stack = tf.concat([drop_work, drop_task, drop_long], 0)
            drop_filtered = tf.multiply(drop_stack, dense_filter)
            w = drop_work * dense_filter[0]
            i = drop_task * dense_filter[1]
            l = drop_long * dense_filter[2]
            drop = tf.add_n([w, i, l]) #sum along axis 1 somehow
            #tf.reduce_sum(tf.stack([w,i,l], axis=1)) #this should work
        '''
        ###############################################################
        ###############################################################
        ###############################################################
        ###############################################################
        ###############################################################

        #hidden size
        flat = tf.contrib.layers.flatten(self.state_in)

        with tf.name_scope('model'):
            drop = dense_layer(flat, n=512, relu=True)

        with tf.name_scope('policy'):
            logits_class = dense_layer(drop, n=n_actions)
            probs_class = tf.nn.softmax(logits_class) + 1e-8
            logprobs_class = tf.nn.log_softmax(logits_class) + 1e-8
            action_dist = tf.multinomial(logits_class - tf.reduce_max(
                    logits_class, [1], keepdims=True), 1)
            self.action = tf.squeeze(action_dist, [1])
            self.test_action = tf.argmax(probs_class, axis=1)

        with tf.name_scope('value'):
            logit_val = dense_layer(drop, n=1)
            self.value = tf.reduce_sum(logit_val, axis=1)

        with tf.name_scope('loss'):
            entropy = - tf.reduce_sum(probs_class * logprobs_class)
            probs_act = tf.reduce_sum(logprobs_class * self.action_in, [1])
            policy_loss = - tf.reduce_sum(probs_act * self.advantage_in)
            value_loss = 0.05 * tf.reduce_sum(tf.square(
                    self.value - self.reward_in))
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

    def act(self, state, explore=True):
        action_op = self.action if explore else self.test_action
        action, value = self.sess.run([action_op, self.value], 
                feed_dict={
                    self.state_in: [state],
                    self.keep_prob: 1.0,
                })
        return action[0], value[0]

    def learn(self, batch):
        states, actions, rewards, advantage, dones, next_states = batch
        loss, _, step, summ = self.sess.run([self.loss, self.optimize,
                self.step, self.summaries],
                feed_dict={
                    self.state_in: states,
                    self.action_in: actions,
                    self.reward_in: rewards,
                    self.advantage_in: advantage,
                    self.keep_prob: 0.5,
                })
        self.writer.add_summary(summ, step)
        return loss

