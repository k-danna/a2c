
import os
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

from tensor_utils import *
import params

class model():
    def __init__(self, state_shape, n_actions, recover=False):

        tf.set_random_seed(params.seed)
        np.random.seed(params.seed)
        
        with tf.name_scope('input'):
            self.state_in = tf.placeholder(tf.float32, 
                    (None,) + state_shape)
            self.action_in = tf.placeholder(tf.float32, [None, n_actions])
            self.reward_in = tf.placeholder(tf.float32, [None])
            self.advantage_in = tf.placeholder(tf.float32, [None])
            self.nextstate_in = tf.placeholder(tf.float32, 
                    (None,) + state_shape)
            self.keep_prob = tf.placeholder(tf.float32)

            #this is just for summary stats
            self.episode_reward_in = tf.placeholder(tf.float32)
            episode_count = tf.Variable(0, trainable=False)
            self.episode_count = tf.assign_add(episode_count, 1)

        #DEBUG
        x = self.state_in

        with tf.name_scope('model'):
            #x = conv_layer(x, (3,3), 16, 'elu')
            x = dense_layer(x, 512, 'elu', keep_prob=self.keep_prob, 
                    drop=True)
            #x = dense_layer(x, self.keep_prob, 256, 'elu', drop=True)
            #x = dense_layer(x, self.keep_prob, 64, 'elu', drop=True)

        with tf.name_scope('policy'):
            logits_class = dense_layer(x, n_actions)
            probs_class = tf.nn.softmax(logits_class)
            logprobs_class = tf.nn.log_softmax(logits_class)

            #openai universe starter agent distribution
            logits_max = tf.reduce_max(logits_class, [1], keepdims=True)
            dist = logits_class - logits_max

            #simple distribution with less exploration
            #dist = logprobs_class

            action_random = tf.multinomial(dist, 1)
            self.action = tf.squeeze(action_random, [1])
            self.test_action = tf.argmax(probs_class, axis=1)

        with tf.name_scope('value'):
            logit_val = dense_layer(x, 1)
            self.value = tf.reduce_sum(logit_val, axis=1)

        with tf.name_scope('loss'):
            entropy = - tf.reduce_sum(probs_class * logprobs_class)
            probs_act = tf.reduce_sum(logprobs_class * self.action_in, [1])
            policy_loss = - tf.reduce_sum(probs_act * self.advantage_in)
            value_loss = 0.5 * tf.reduce_sum(tf.square(
                    self.value - self.reward_in))
            self.loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        with tf.name_scope('optimize'):
            self.optimize, self.step = minimize(self.loss, 
                    params.learn_rate)

        with tf.name_scope('summary'):
            batch_size = tf.shape(self.state_in)[0]
            tf.summary.scalar('1_total_loss', tf.divide(self.loss, 
                    tf.cast(batch_size, tf.float32)))
            tf.summary.scalar('2_value_loss', value_loss)
            tf.summary.scalar('3_policy_loss', policy_loss)
            tf.summary.scalar('4_entropy', entropy)
            self.summaries = tf.summary.merge_all()
            self.episode_rewards = tf.summary.scalar('5_episode_reward', 
                    self.episode_reward_in)

        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(os.path.join(params.out_dir, 
                'model'), self.loss.graph)
        self.saver = tf.train.Saver()
        self.model_vars = tf.trainable_variables()
        self.sess.run(tf.global_variables_initializer())

        #load weights from disk if specified
        self.status = 'initialized'
        if recover:
            self.load()
            self.status = 'recovered'

    def add_episode_stat(self, reward):
        ep, summ = self.sess.run([self.episode_count, self.episode_rewards],
                feed_dict={self.episode_reward_in: np.float32(reward),})
        self.writer.add_summary(summ, ep)
        self.writer.flush()

    def act(self, state, explore=True):
        action_op = self.action if explore else self.test_action
        action, value = self.sess.run([action_op, self.value], 
                feed_dict={
                    self.state_in: [state],
                    self.keep_prob: 1.0,
                })
        return action[0], value[0]

    def learn(self, batch, sample=False):
        states, actions, rewards, advantage, dones, next_states = batch
        loss, _, step, summ = self.sess.run([self.loss, 
                self.optimize, self.step, self.summaries],
                feed_dict={
                    self.state_in: states,
                    self.action_in: actions,
                    self.reward_in: rewards,
                    self.advantage_in: advantage,
                    self.nextstate_in: next_states,
                    self.keep_prob: 0.5,
                })
        self.writer.add_summary(summ, step)
        self.writer.flush()
        return loss

    def save(self):
        self.saver.save(self.sess, os.path.join(params.out_dir, 
                'model', 'model.ckpt'), global_step=self.step)

    def load(self):
        #load sess, trained variables
        self.saver.restore(self.sess, tf.train.latest_checkpoint(
                os.path.join(params.out_dir, 'model')))
        recovered = tf.trainable_variables()

        #assign recovered variables over the initialized ones
        restore = [self.model_vars[i].assign(recovered[i]) for i in range(
                len(self.model_vars))]
        self.sess.run([restore])















