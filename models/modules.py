import tensorflow as tf


def create_variable(name, shape, is_bias=False):
    if is_bias:
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.zeros_initializer())
    else:
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.glorot_normal_initializer())


class Conv1D(object):

    def __init__(self, in_channels, out_channels, kernel_size=1, use_bias=True, name='conv1d'):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_bias = use_bias

        with tf.variable_scope(name) as scope:
            self.scope = scope
            kernel_shape = (kernel_size, in_channels, out_channels)
            self.kernel = create_variable(name='kernel', shape=kernel_shape)
            if use_bias:
                self.bias = create_variable(name='bias', shape=(out_channels, ), is_bias=True)

    def __call__(self, inputs):
        with tf.name_scope(self.scope.original_name_scope):
            outputs = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID', data_format='NWC')
            if self.use_bias:
                outputs = tf.nn.bias_add(outputs, self.bias)
            return outputs


class FFTLayer(object):

    def __init__(self, in_channels, hidden_channels, layer_index, cin_channels=-1, pad_value=0, name='fft_layer'):
        self.receptive_field = 2**layer_index
        self.shift = self.receptive_field // 2
        self.pad_value = pad_value

        with tf.variable_scope(name) as scope:
            self.left_conv = Conv1D(in_channels, hidden_channels, kernel_size=1, name='left_conv')
            self.right_conv = Conv1D(in_channels, hidden_channels, kernel_size=1, name='right_conv')
            self.out_conv = Conv1D(hidden_channels, hidden_channels, kernel_size=1, name='out_conv')

            if cin_channels > 0:
                self.cin_left_conv = Conv1D(cin_channels, hidden_channels, kernel_size=1, name='cin_left_conv')
                self.cin_right_conv = Conv1D(cin_channels, hidden_channels, kernel_size=1, name='cin_right_conv')

            self.scope = scope

    def __call__(self, inputs,  c=None):
        with tf.name_scope(self.scope.original_name_scope):
            # apply zero padding to inputs
            padding = tf.constant([[0, 0], [self.shift, 0], [0, 0]])
            inputs = tf.pad(inputs, padding, constant_values=self.pad_value)

            left_out = self.left_conv(inputs[:, :-self.shift, :])
            right_out = self.right_conv(inputs[:, self.shift:, :])

            if c is not None:
                # apply zero padding to condition
                c = tf.pad(c, padding, constant_values=0)
                c = c[:, -tf.shape(inputs)[1]:, :]

                left_lc_out = self.cin_left_conv(c[:, :-self.shift, :])
                right_lc_out = self.cin_right_conv(c[:, self.shift:, :])
                left_out += left_lc_out
                right_out += right_lc_out

            output = tf.nn.relu(left_out + right_out)
            output = self.out_conv(output)
            return tf.nn.relu(output)


class ConvTransposed2d(object):
    """Use transposed conv to upsample"""
    def __init__(self, filters, kernel_size, freq_axis_kernel_size, padding, strides, scope):
        self.scope = scope
        self.conv = tf.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                              kernel_initializer=tf.constant_initializer(1 / freq_axis_kernel_size,
                                                                                         dtype=tf.float32),
                                              bias_initializer=tf.zeros_initializer(),
                                              data_format='channels_last')

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            return self.conv(inputs)



