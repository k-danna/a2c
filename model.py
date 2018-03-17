
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
            self.action_in = tf.placeholder(tf.float32, [None, n_actions])
            self.keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope('dense_dropout'):
            n = 512
            flat = tf.contrib.layers.flatten(self.state_in)
            w_dense = weight([state_shape[0] * state_shape[1], n])
            b_dense = bias([n])

            dense = tf.nn.relu(tf.matmul(flat, w_dense) + b_dense)
            #FIXME: add dropout
            #drop = tf.nn.dropout(dense, self.keep_prob)

        with tf.name_scope('classify'):
            w_class = weight([n, n_actions])
            b_class = bias([n_actions])

            self.predictions = tf.matmul(dense, w_class) + b_class

        with tf.name_scope('optimize'):
            self.step = tf.Variable(0, trainable=False)
            reward_preds = tf.reduce_sum(tf.multiply(self.predictions, 
                    self.action_in))
            
            self.loss = tf.reduce_mean(tf.square(
                    self.reward_in - reward_preds))

            optimizer = tf.train.AdamOptimizer(1e-3)
            self.optimize = optimizer.minimize(self.loss, 
                    global_step=self.step)

        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            self.summaries = tf.summary.merge_all()

        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter('./logs', self.loss.graph)
        self.sess.run(tf.global_variables_initializer())

    def action(self, state, epsilon=0.14):
        #returns action and corresponding value
        action_values = self.sess.run([self.predictions], 
                feed_dict={
                    self.state_in: [state],
                    self.keep_prob: 1.0,
                })[0][0]

        #epsilon greedy exploration
        if np.random.random_sample() < epsilon:
            action = np.random.randint(0, len(action_values)) 
            return action, action_values[action]

        return np.argmax(action_values), np.amax(action_values)

    def learn(self, batch):
        states, actions, rewards, dones, next_states = batch
        loss, _, step, summ = self.sess.run([self.loss, self.optimize,
                self.step, self.summaries],
                feed_dict={
                    self.state_in: states,
                    self.action_in: actions,
                    self.reward_in: rewards,
                    self.keep_prob: 0.5,
                })
        self.writer.add_summary(summ, step)
        return loss


#TODO: 
    #stats in train
    #epsilon exploration

