import tensorflow as tf


def create_variable(name, shape, _type=0):
    if _type == 0:
        # kernel
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer())
    elif _type == 1:
        # bias
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.zeros_initializer())
    elif _type == 2:
        # weight norm
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(1.))
    else:
        raise ValueError("{} type is not supported".format(_type))


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
                self.bias = create_variable(name='bias', shape=(out_channels, ), _type=1)

    def __call__(self, inputs):
        with tf.name_scope(self.scope.original_name_scope):
            outputs = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='VALID', data_format='NWC')
            if self.use_bias:
                outputs = tf.nn.bias_add(outputs, self.bias)
            return outputs

    def incremental_forward(self, inputs):
        """"""
        with tf.name_scope(self.scope.original_name_scope):
            linearized_weight = self._get_linearized_weight()

            batch_size = tf.shape(inputs)[0]
            inputs = tf.reshape(inputs, [batch_size, -1])

            # compute step prediction
            output = tf.matmul(inputs, linearized_weight, transpose_b=True)
            if self.use_bias:
                output = tf.nn.bias_add(output, self.bias)
            return tf.reshape(output, [batch_size, 1, self.out_channels])

    def _get_linearized_weight(self,):
        # layers.Conv1D kw, in_channel, filters
        weight = tf.transpose(self.kernel, [2, 0, 1])
        assert weight.shape == (self.out_channels, self.kernel_size, self.in_channels)
        linearized_weight = tf.cast(tf.reshape(weight, [self.out_channels, -1]), dtype=weight.dtype)
        return linearized_weight


class FFTLayer(object):

    def __init__(self, in_channels, hidden_channels, layer_index, cin_channels=-1, pad_value=0, name='fft_layer'):
        self.receptive_field = 2**layer_index
        self.shift = self.receptive_field // 2
        self.pad_value = pad_value
        self.in_channels = in_channels
        self.cin_channels = cin_channels

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
            # get current condition
            if c is not None:
                c = c[:, -tf.shape(inputs)[1]:, :]

            # apply zero padding to inputs
            # when use mu-law, the input is one-hot, BxTx256
            padding = tf.constant([[0, 0], [self.shift, 0], [0, 0]])
            if self.pad_value != 0:
                input_pad = tf.ones(shape=(tf.shape(inputs)[0], self.shift), dtype=tf.int32) * self.pad_value
                input_pad = tf.one_hot(input_pad, depth=tf.shape(inputs)[-1])
                inputs = tf.concat([input_pad, inputs], axis=1)
            else:
                inputs = tf.pad(inputs, padding, constant_values=self.pad_value)

            left_out = self.left_conv(inputs[:, :-self.shift, :])
            right_out = self.right_conv(inputs[:, self.shift:, :])

            if c is not None:
                # apply zero padding to condition
                c = tf.pad(c, padding, constant_values=0)
                left_lc_out = self.cin_left_conv(c[:, :-self.shift, :])
                right_lc_out = self.cin_right_conv(c[:, self.shift:, :])
                left_out += left_lc_out
                right_out += right_lc_out

            output = tf.nn.relu(left_out + right_out)
            output = self.out_conv(output)
            return tf.nn.relu(output)

    def incremental_forward(self, inputs, c=None, input_buffers=None, c_buffers=None):
        input_buffers = input_buffers[:, 1:, :]
        input_buffers = tf.concat([input_buffers, tf.reshape(inputs, [1, 1, self.in_channels])], axis=1)

        # padding = tf.constant([[0, 0], [self.shift, 0], [0, 0]])
        # if self.pad_value != 0:
        #     input_pad = tf.ones(shape=(tf.shape(input_buffers)[0], self.shift), dtype=tf.int32) * self.pad_value
        #     input_pad = tf.one_hot(input_pad, depth=tf.shape(input_buffers)[-1])
        #     input_pad_buffers = tf.concat([input_pad, input_buffers], axis=1)
        # else:
        #     input_pad_buffers = tf.pad(input_buffers, padding, constant_values=self.pad_value)

        left_out = self.left_conv.incremental_forward(input_buffers[:, 0, :])
        right_out = self.right_conv.incremental_forward(input_buffers[:, -1, :])

        if c is not None:
            assert c_buffers is not None
            # append the new c into buffer
            c_buffers = c_buffers[:, 1:, :]
            c_buffers = tf.concat([c_buffers, tf.reshape(c, [1, 1, self.cin_channels])], axis=1)
            # pad zero to buffer
            # c_pad_buffers = tf.pad(c_buffers, padding, constant_values=0)

            # now choose the left final and right final to conv
            left_lc_out = self.cin_left_conv.incremental_forward(c_buffers[:, 0, :])
            right_lc_out = self.cin_right_conv.incremental_forward(c_buffers[:, -1, :])
            left_out += left_lc_out
            right_out += right_lc_out

        output = tf.nn.relu(left_out + right_out)
        output = self.out_conv.incremental_forward(output)
        if self.cin_channels > 0:
            return tf.nn.relu(output), input_buffers, c_buffers
        else:
            return tf.nn.relu(output), input_buffers


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



