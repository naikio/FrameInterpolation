import tensorflow as tf


class ConvolutionalInterpolatorNetwork:

    def __init__(self, X, Y, learning_rate = 0.0001):
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate

        self._inference = None
        self._loss = None
        self._training = None

        self._inference = self.inference
        self._loss = self.loss
        self._training = self.training


    @property
    def inference(self):
        if self._inference is None:
            # filtri del modello salvato:
            # 16-16-16-32-64 conv
            # 64-32-16-16-16 deconv
            with tf.variable_scope('encoder'):
                conv1 = self.conv_block(self.X, filters=96)

                visualizer1 = conv1[:, :, :, :1]
                tf.summary.image('first layer', visualizer1)
                visualizer2 = conv1[:, :, :, 5:6]
                tf.summary.image('first layer', visualizer2)
                visualizer3 = conv1[:, :, :, 2:3]
                tf.summary.image('first layer', visualizer3)
                visualizer4 = conv1[:, :, :, 12:13]
                tf.summary.image('first layer', visualizer4)

                conv2 = self.conv_block(conv1, filters=96)
                conv3 = self.conv_block(conv2, filters=96)
                conv4 = self.conv_block(conv3, filters=96)
                conv5 = self.conv_block(conv4, filters=128)

                visualizer1 = conv5[:, :, :, :1]
                tf.summary.image('middle layer', visualizer1)
                visualizer2 = conv5[:, :, :, 31:32]
                tf.summary.image('middle layer', visualizer2)
                visualizer3 = conv5[:, :, :, 62:63]
                tf.summary.image('middle layer', visualizer3)
                visualizer4 = conv5[:, :, :, 25:26]
                tf.summary.image('middle layer', visualizer4)

            with tf.variable_scope('decoder'):
                deconv5 = self.deconv_block(conv5, filters=128)

                in4 = tf.concat((conv4, deconv5), axis=3)
                deconv4 = self.deconv_block(in4, filters=96)

                in3 = tf.concat((conv3, deconv4), axis=3)
                deconv3 = self.deconv_block(in3, filters=96)

                in2 = tf.concat((conv2, deconv3), axis=3)
                deconv2 = self.deconv_block(in2, filters=96)

                in1 = tf.concat((conv1, deconv2), axis=3)
                deconv1 = self.deconv_block(in1, filters=96)

            self._inference = tf.layers.conv2d(deconv1, filters=3, kernel_size=3, strides=1, activation=tf.tanh, padding='same')

        return self._inference

    @property
    def loss(self):
        if self._loss is None:
            self._loss = tf.reduce_mean(tf.square(self._inference - self.Y))
        return self._loss

    @property
    def training(self):
        if self._training is None:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self._training = optimizer.minimize(self._loss, global_step)
        return self._training

    @staticmethod
    def conv_block(input, ksize=3, filters=64):
        block1 = tf.layers.conv2d(input, filters=filters, kernel_size=ksize, strides=1,
                                  activation=tf.nn.relu, padding='same',
                                  bias_initializer=tf.constant_initializer(0.1))

        block2 = tf.layers.conv2d(block1, filters=filters, kernel_size=ksize, strides=1,
                                  activation=tf.nn.relu, padding='same',
                                  bias_initializer=tf.constant_initializer(0.1))
        block3 = tf.layers.max_pooling2d(block2, pool_size=2, strides=2, padding='same')

        return block3
    @staticmethod
    def deconv_block(input, filters=64):
        d_block1 = tf.layers.conv2d_transpose(input, filters=filters, kernel_size=4, strides=2,
                                              activation=tf.nn.relu, padding='same',
                                              bias_initializer=tf.constant_initializer(0.1))
        d_block2 = tf.layers.conv2d(d_block1, filters=filters, kernel_size=3, strides=1,
                                    activation=tf.nn.relu, padding='same',
                                    bias_initializer=tf.constant_initializer(0.1))

        return d_block2
