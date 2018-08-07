import tensorflow as tf
from models.modules import FFTLayer, ConvTransposed2d
import utils
import numpy as np


class FFTNet(object):
    def __init__(self, hp):
        self.hp = hp
        self.receptive_filed = 2 ** hp.n_layers
        # fft layer
        self.fft_layers = []

        if utils.is_mulaw_quantize(self.hp.input_type):
            pad_value = 128
            self.in_channels = 256
        else:
            pad_value = 0
            self.in_channels = 1

        for idx in range(0, hp.n_layers):
            layer_index = hp.n_layers - idx
            if idx == 0:
                self.fft_layers += [FFTLayer(self.in_channels, hp.hidden_channels, layer_index, hp.cin_channels,
                                             pad_value, name='fft_layer_{}'.format(idx))]
            else:
                self.fft_layers += [FFTLayer(hp.hidden_channels, hp.hidden_channels, layer_index, hp.cin_channels,
                                             pad_value, name='fft_layer_{}'.format(idx))]
        self.out_layer = tf.layers.Dense(units=hp.quantize_channels, name='out_dense')

        # upsample conv
        if hp.upsample_conditional_features:
            self.upsample_conv = []
            for i, s in enumerate(hp.upsample_scales):
                convt = ConvTransposed2d(1, s, hp.freq_axis_kernel_size, padding='same', strides=(s, 1),
                                         scope='local_conditioning_upsample_{}'.format(i + 1))
                self.upsample_conv.append(convt)
        else:
            self.upsample_conv = None

        print('Receptive Field: %i samples' % self.receptive_filed)
        print('pad value: {}'.format(pad_value))

    def forward(self, inputs, targets=None, c=None, g=None):
        if g is not None:
            raise NotImplementedError("global condition is not added now!")

        # the rank of inputs is 2
        if utils.is_mulaw_quantize(self.hp.input_type):
            inputs = tf.one_hot(tf.cast(inputs, tf.int32), self.hp.quantize_channels)
        else:
            inputs = tf.expand_dims(inputs, axis=-1)

        with tf.control_dependencies([tf.assert_equal(tf.rank(inputs), 3)]):
            outputs = tf.identity(inputs)

        self.targets = tf.cast(targets, tf.int32)

        # check whether need to upsample local condition
        if c is not None and self.upsample_conv is not None:
            c = tf.expand_dims(c, axis=-1)  # [B T cin_channels 1]
            for transposed_conv in self.upsample_conv:
                c = transposed_conv(c)
            c = tf.squeeze(c, axis=-1)  # [B new_T cin_channels]

        # for training, we need to use previous samples and the condition of next sample (shift by one)
        outputs = outputs[:, :-1, :]
        if c is not None:
            c = c[:, 1:, :]

        with tf.control_dependencies([tf.assert_equal(tf.shape(outputs)[1], tf.shape(c)[1])]):
            c = tf.identity(c)

        for layer in self.fft_layers:
            outputs = layer(outputs, c=c)
        outputs = self.out_layer(outputs)
        self.outputs = outputs
        self.log_outputs = tf.argmax(tf.nn.softmax(self.outputs, axis=-1), axis=-1)

    def predict(self, c=None, g=None, test_inputs=None, targets=None):
        if g is not None:
            raise NotImplementedError("global condition is not added now!")

        # use the zero as inputs
        inputs = tf.zeros([1, self.receptive_filed], dtype=tf.float32)
        inputs = self._convert_type(inputs)

        # check whether need to upsample condition
        if c is not None and self.upsample_conv is not None:
            c = tf.expand_dims(c, axis=-1)  # [B T cin_channels 1]
            for transposed_conv in self.upsample_conv:
                c = transposed_conv(c)
            c = tf.squeeze(c, axis=-1)  # [B new_T cin_channels]

        # apply zero padding to condition
        if c is not None:
            c_shape = tf.shape(c)
            padding_c = tf.zeros([c_shape[0], self.receptive_filed, c_shape[-1]])
            c = tf.concat([padding_c, c], axis=1)

        synthesis_length = tf.shape(c)[1] - self.receptive_filed

        initial_time = tf.constant(0, dtype=tf.int32)

        initial_outputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        def condition(time, unused_initial_input, unused_final_outputs):
            return tf.less(time, synthesis_length)

        def body(time, current_inputs, final_outputs):
            # we need shift condition by one
            current_c = c[:, time + 1:time + 1 + self.receptive_filed, :] if c is not None else None

            current_outputs = current_inputs
            for layer in self.fft_layers:
                current_outputs = layer(current_outputs, c=current_c)
            current_outputs = self.out_layer(current_outputs)

            posterior = tf.nn.softmax(tf.reshape(current_outputs[:, -1, :], [1, -1]), axis=-1)

            # dist = tf.distributions.Categorical(probs=posterior)
            # sample = tf.cast(dist.sample(), tf.int32)

            sample = tf.py_func(np.random.choice,
                                [np.arange(self.hp.quantize_channels), 1, True, tf.reshape(posterior, [-1])], tf.int64)
            sample = tf.reshape(sample, [-1])

            # sample = tf.argmax(posterior, axis=-1)

            decode_sample = utils.inv_mulaw_quantize(sample, self.hp.quantize_channels)
            final_outputs = final_outputs.write(time, decode_sample)

            if utils.is_mulaw_quantize(self.hp.input_type):
                next_sample = tf.one_hot(tf.cast(sample, tf.int32), self.hp.quantize_channels)
            else:
                next_sample = decode_sample

            next_time = time + 1
            next_inputs = current_inputs[:, 1:, :]
            if test_inputs is not None:
                next_sample = tf.reshape(test_inputs[:, next_time], [1, 1, self.in_channels])
            else:
                next_sample = tf.reshape(next_sample, [1, 1, self.in_channels])

            next_inputs = tf.concat([next_inputs, tf.cast(next_sample, tf.float32)], axis=1)

            return next_time, next_inputs, final_outputs

        result = tf.while_loop(condition,
                               body,
                               loop_vars=[initial_time, inputs, initial_outputs_ta],
                               parallel_iterations=32,
                               swap_memory=True
                               )

        outputs_ta = result[2]
        outputs = outputs_ta.stack()
        self.eval_outputs = outputs
        self.eval_targets = utils.inv_mulaw_quantize(targets, self.hp.quantize_channels) if targets is not None else None

    def incremental_forward(self, c=None, g=None, test_inputs=None, targets=None):
        if g is not None:
            raise NotImplementedError("global condition is not added now!")

        # use the zero as inputs
        inputs = tf.zeros([1, 1], dtype=tf.float32)
        if utils.is_mulaw_quantize(self.hp.input_type):
            inputs = utils.mulaw_quantize(inputs, self.hp.quantize_channels)
            inputs = tf.one_hot(tf.cast(inputs, tf.int32), self.hp.quantize_channels)
        else:
            inputs = tf.expand_dims(inputs, axis=-1)

        # check whether need to upsample condition
        if c is not None and self.upsample_conv is not None:
            c = tf.expand_dims(c, axis=-1)  # [B T cin_channels 1]
            for transposed_conv in self.upsample_conv:
                c = transposed_conv(c)
            c = tf.squeeze(c, axis=-1)  # [B new_T cin_channels]

        # apply zero padding to condition
        if c is not None:
            c_shape = tf.shape(c)
            padding_c = tf.zeros([c_shape[0], self.receptive_filed, c_shape[-1]])
            c = tf.concat([padding_c, c], axis=1)
            # create c_buffers
            c_buffers = [tf.zeros([1, 2 ** i // 2+1, self.hp.cin_channels]) for i in range(self.hp.n_layers, 0, -1)]

        synthesis_length = tf.shape(c)[1]

        initial_time = tf.constant(0, dtype=tf.int32)

        initial_outputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        input_buffers = [self._convert_type(tf.zeros([1, 2 ** self.hp.n_layers // 2+1]))]
        for i in range(self.hp.n_layers - 1, 0, -1):
            input_buffers.append(
                self._convert_type(tf.zeros([1, 2 ** i // 2+1]))
            )

        def condition(time, unused_initial_input, unused_final_outputs, unused_input_buffers, unused_c_buffers):
            return tf.less(time, synthesis_length)

        def body(time, current_inputs, final_outputs, current_input_buffers, current_c_buffers):
            # we need shift condition by one
            current_c = c[:, time:time+1, :] if c is not None else None

            current_outputs = current_inputs
            new_input_buffers = []
            new_c_buffers = []

            for layer, current_input_buffer, current_c_buffer in zip(self.fft_layers, current_input_buffers,
                                                                     current_c_buffers):
                current_outputs, out_input_buffer, out_c_buffer = layer.incremental_forward(
                    inputs=current_outputs,
                    c=current_c,
                    input_buffers=current_input_buffer,
                    c_buffers=current_c_buffer,
                )
                new_input_buffers.append(out_input_buffer)
                new_c_buffers.append(out_c_buffer)

            current_outputs = self.out_layer(current_outputs)

            posterior = tf.nn.softmax(tf.reshape(current_outputs, [1, -1]), axis=-1)

            # dist = tf.distributions.Categorical(probs=posterior)
            # sample = tf.cast(dist.sample(), tf.int32)

            sample = tf.py_func(np.random.choice,
                                [np.arange(self.hp.quantize_channels), 1, True, tf.reshape(posterior, [-1])], tf.int64)
            sample = tf.reshape(sample, [-1])

            # sample = tf.argmax(posterior, axis=-1)

            decode_sample = utils.inv_mulaw_quantize(sample, self.hp.quantize_channels)
            final_outputs = final_outputs.write(time, decode_sample)

            if utils.is_mulaw_quantize(self.hp.input_type):
                next_sample = tf.one_hot(tf.cast(sample, tf.int32), self.hp.quantize_channels)
            else:
                next_sample = decode_sample

            next_time = time + 1
            next_inputs = current_inputs[:, 1:, :]
            if test_inputs is not None:
                next_sample = tf.reshape(test_inputs[:, next_time], [1, 1, self.in_channels])
            else:
                next_sample = tf.reshape(next_sample, [1, 1, self.in_channels])

            next_inputs = tf.concat([next_inputs, tf.cast(next_sample, tf.float32)], axis=1)

            return next_time, next_inputs, final_outputs, new_input_buffers, new_c_buffers

        result = tf.while_loop(condition,
                               body,
                               loop_vars=[initial_time, inputs, initial_outputs_ta, input_buffers, c_buffers],
                               parallel_iterations=32,
                               swap_memory=True
                               )

        outputs_ta = result[2]
        outputs = outputs_ta.stack()
        self.eval_outputs = outputs
        self.eval_targets = utils.inv_mulaw_quantize(targets, self.hp.quantize_channels) if targets is not None else None

    def add_loss(self):
        with tf.variable_scope("loss"):

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.outputs,
                labels=self.targets[:, 1:]
            )
            self.loss = tf.reduce_mean(loss)

    def add_optimizer(self, ema, global_step):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hp.init_lr)
        adam_optimize = optimizer.minimize(self.loss, global_step=global_step)

        with tf.control_dependencies([adam_optimize]):
            # Create the shadow variables and add ops to maintain moving averages
            # Also updates moving averages after each update step
            # This is the optimize call instead of traditional adam_optimize one.
            self.optimize = ema.apply(tf.trainable_variables())

    def _convert_type(self, inputs):
        if utils.is_mulaw_quantize(self.hp.input_type):
            inputs = utils.mulaw_quantize(inputs, self.hp.quantize_channels)
            inputs = tf.one_hot(tf.cast(inputs, tf.int32), self.hp.quantize_channels)
        else:
            inputs = tf.expand_dims(inputs, axis=-1)
        return inputs

