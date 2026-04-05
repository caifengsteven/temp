import tensorflow as tf
import numpy as np
from net_config_loader import *

def norm_raw_values(values):
    return (values - np.min(values, axis=1).reshape(values.shape[0], 1)) \
                  / (np.max(values, axis=1).reshape(values.shape[0], 1)
                     - np.min(values, axis=1).reshape(values.shape[0], 1))


class NeuralNetwork:
    def __init__(self, net_cfg_name, load_checkpoint=False, checkpoint_path=''):
        '''
        Constructor of a neural network instance
        :param net_cfg_name: name of config file, which has the config of the network
        :param load_checkpoint: if you want to load an pre-training network, set this param as True
        :param checkpoint_path: the network param would automatically saved in this path during training,
                                if load_checkpoint is True, the network would restore network from this path first
        '''
        self.name = net_cfg_name
        self.net_cfg = load_net_config(net_cfg_name)
        self.checkpoint_path = checkpoint_path
        self.load_checkpoint = load_checkpoint
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
            # build neural network according to cfg file
            assert(len(self.net_cfg) >= 2)
            layerName, layerSize = self.net_cfg[0]
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, layerSize], name=layerName)
            x = self.input
            self.weights = []
            for i in range(1, len(self.net_cfg)):
                layerName, layerSize = self.net_cfg[i]
                if i < len(self.net_cfg) - 1:
                    x = tf.layers.dense(inputs=x, units=layerSize, activation=tf.sigmoid, name=layerName)
                    layer_weights = tf.get_variable('{}/kernel'.format(layerName))
                    self.weights.append(layer_weights)
                else:
                    self.logits = tf.layers.dense(inputs=x, units=layerSize, name=layerName)
                    self.output = tf.sigmoid(self.logits)
                    # self.output = tf.nn.softmax(self.logits)
                    layer_weights = tf.get_variable('{}/kernel'.format(layerName))
                    self.weights.append(layer_weights)

            # pass label into network as [None, ] shape, e.g. if you have three samples with pattern 1, 4 and 7,
            # pass [1, 4, 7], the code following would convert to zero-based one-hot vectors as
            # [[0 1 0 0 0 0 0 0 0 0 0 0]
            #  [0 0 0 0 1 0 0 0 0 0 0 0]
            #  [0 0 0 0 0 0 0 1 0 0 0 0]]
            self.label = tf.placeholder(dtype=tf.int32, shape=[None, ])
            self.one_hot_label = tf.one_hot(self.label, self.net_cfg[-1][1])
            # self.loss = tf.losses.sigmoid_cross_entropy(one_hot_label, self.logits)
            self.loss = tf.nn.l2_loss(self.output - self.one_hot_label)

            # define train step for train
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            # self.train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(self.loss, global_step=self.global_step_tensor)
            self.train_step = tf.train.AdamOptimizer(learning_rate=0.05).minimize(self.loss, global_step=self.global_step_tensor)
        self.model_saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

        if load_checkpoint:
            checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.model_saver.restore(self.session, checkpoint.model_checkpoint_path)
                print("Successfully loaded pre-trained model from ", checkpoint.model_checkpoint_path)
            else:
                print('Checkpoint path not exists, network needs to train')

    def train(self, input, label, max_epoch=10000, verbose=False, end_loss=0.0):
        losses = []
        l2_losses = []
        for i in range(max_epoch):
            _, loss, output, step = self.session.run((self.train_step, self.loss, self.output, self.global_step_tensor),
                             feed_dict={self.input:input, self.label:label})
            losses.append(loss)
            if verbose:
                print('Iteration {}, loss {}'.format(step, loss))
            if step % 500 == 0:
                self.model_saver.save(self.session, os.path.join(self.checkpoint_path, self.name),
                                      global_step=self.global_step_tensor)
            if loss < end_loss:
                break
        self.model_saver.save(self.session, os.path.join(self.checkpoint_path, self.name),
                              global_step=self.global_step_tensor)
        return losses

    def predict(self, samples_to_predict):
        ret = self.session.run(self.output, feed_dict={self.input: samples_to_predict})
        return ret

    def export_weights_value(self):
        weights = self.session.run(self.weights)
        return weights


class ClassifyNetowrk(NeuralNetwork):
    def __init__(self, load_checkpoint=False, checkpoint_path='log/classify_net'):
        self.cfg_filename = 'classify_net'
        super().__init__(self.cfg_filename, load_checkpoint, checkpoint_path)

    def predict_with_no_norm_values(self, values):
        '''
        predict class with raw values with no norm
        :param values: input raw value with [None, 10] shape
        :return: class predict with [None, 12] shape
        '''
        norm_values = norm_raw_values(values)
        input_samples = np.concatenate((norm_values, 1 - norm_values), axis=1)
        return self.predict(input_samples)

class PredictNetwork(NeuralNetwork):
    def __init__(self, load_checkpoint=False, checkpoint_path='log/predict_net'):
        self.cfg_filename = 'predict_net'
        super().__init__(self.cfg_filename, load_checkpoint, checkpoint_path)


if __name__ == '__main__':
    network = ClassifyNetowrk()
    inputs = np.random.random((12, 20))
    label = np.array(range(12)).reshape((12,))
    print('input\n', inputs)
    print('label\n', label)
    print('weights\n', network.export_weights_value())
    # input('press any key to continue...')
    network.train(inputs, label, verbose=True, max_epoch=1000)
    print('weights\n', network.export_weights_value())
    # input('press any key to continue...')
    print('predict')
    output = network.predict(inputs)
    for i in range(12):
        for j in range(12):
            print('{:.4f}'.format(output[i, j]), end='\t')
        print('\n')

    pnet = PredictNetwork(load_checkpoint=True)
    trend = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0]).reshape((12,))
    # pnet.train(output, trend, verbose=True, max_epoch=1000)
    trend_predict = pnet.predict(output)
    print(trend_predict)


