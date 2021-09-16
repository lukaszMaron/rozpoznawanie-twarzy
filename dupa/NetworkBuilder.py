import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class NetworkBuilder:
    def __init__(self):
        pass

    def attach_conv_layer(self, input_layer, output_size=32, feature_size=(5, 5), strides=[1, 1, 1, 1], padding='SAME',
                          summary=False):
        with tf.name_scope("Convolution") as scope:
            input_size = input_layer.get_shape().as_list()[-1]
            weights = tf.Variable(tf.random_normal([feature_size[0], feature_size[1], input_size, output_size]), name='conv_weights')
            if summary:
                tf.summary.histogram(weights.name, weights)
            biases = tf.Variable(tf.random_normal([output_size]),name='conv_biases')
            conv = tf.nn.conv2d(input_layer, weights, strides=strides, padding=padding)+biases
            return conv

    def attach_relu_layer(self, input_layer):
        with tf.name_scope("Activation") as scope:
            return tf.nn.relu(input_layer)

    def attach_sigmoid_layer(self, input_layer):
        with tf.name_scope("Activation") as scope:
            return tf.nn.sigmoid(input_layer)

    def attach_softmax_layer(self, input_layer):
        with tf.name_scope("Activation") as scope:
            return tf.nn.softmax(input_layer)

    def attach_pooling_layer(self, input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME'):
        with tf.name_scope("Pooling") as scope:
            return tf.nn.max_pool(input_layer, ksize=ksize, strides=strides, padding=padding)

    def flatten(self, input_layer):
        with tf.name_scope("Flatten") as scope:
            input_size = input_layer.get_shape().as_list()
            new_size = input_size[-1]*input_size[-2]*input_size[-3]
            return tf.reshape(input_layer, [-1, new_size])

    def attach_dense_layer(self, input_layer, size, summary=False):
        with tf.name_scope("Dense") as scope:
            input_size = input_layer.get_shape().as_list()[-1]
            weights = tf.Variable(tf.random_normal([input_size, size]), name='dense_weigh')
            if summary:
                tf.summary.histogram(weights.name, weights)
            biases = tf.Variable(tf.random_normal([size]), name='dense_biases')
            dense = tf.matmul(input_layer, weights) + biases
            return dense
    
    def attach_inception_module(self,input_layer, output_size):
        output_size_road1 = int(output_size*0.2)
        road1 = self.attach_conv_layer(input_layer=input_layer, output_size=output_size_road1,
                                       feature_size=(1, 1))
        road1 = self.attach_relu_layer(road1)

        output_size_road2 = int(output_size * 0.3)
        road2 = self.attach_conv_layer(input_layer=input_layer, output_size=output_size_road2,
                                       feature_size=(1, 1))
        road2 = self.attach_relu_layer(road2)
        road2 = self.attach_conv_layer(input_layer=road2, output_size=output_size_road2,
                                       feature_size=(3, 3))

        output_size_road3 = int(output_size * 0.3)
        road3 = self.attach_conv_layer(input_layer=input_layer, output_size=output_size_road3,
                                       feature_size=(1, 1))
        road3 = self.attach_relu_layer(road3)
        road3 = self.attach_conv_layer(input_layer=road3, output_size=output_size_road2,
                                       feature_size=(5, 5))

        output_size_road4 = output_size-(output_size_road1 + output_size_road2 + output_size_road3)
        road4 = self.attach_pooling_layer(input_layer=input_layer, strides=[1, 1, 1, 1])
        road4 = self.attach_conv_layer(input_layer=road4, output_size=output_size_road4,
                                       feature_size=(1, 1))

        with tf.name_scope("FilterConcat") as scope:
            concat = tf.concat([road1, road2, road3, road4], axis=3, name="FilterConcat")
            concat = self.attach_relu_layer(concat)
        return concat