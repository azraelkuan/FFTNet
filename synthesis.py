import tensorflow as tf
import numpy as np
import time
import argparse
from tqdm import tqdm
from hparams import hparams, hparams_debug_string
from models.fftnet import FFTNet
import utils
import librosa
from utils.plot import waveplot


def create_placeholders():
    ph = {
        'inputs': tf.placeholder(dtype=tf.float32, shape=[1, None], name='inputs'),
        'local_condition': tf.placeholder(dtype=tf.float32, shape=[1, None, hparams.num_mels]),
        'test_inputs': tf.placeholder(dtype=tf.float32, shape=[1, None], name='test_inputs')
    }
    return ph


def create_model(ph):
    with tf.variable_scope("model"):
        model = FFTNet(hparams)
        model.predict(c=ph['local_condition'])
    return model


def main():
    checkpoint_path = tf.train.get_checkpoint_state('logs-test/checkpoints/').model_checkpoint_path
    print(checkpoint_path)

    ph = create_placeholders()

    model = create_model(ph)
    # apply ema to variable
    ema = tf.train.ExponentialMovingAverage(decay=hparams.ema_decay)

    audio = np.load('/mnt/lustre/sjtu/users/kc430/data/my/fftnet/cmu_arctic/cmu_arctic-audio-03139.npy')
    local_condition = np.load('/mnt/lustre/sjtu/users/kc430/data/my/fftnet/cmu_arctic/cmu_arctic-mel-03139.npy')
    audio = np.reshape(audio, [1, -1])
    # audio = np.pad(audio, [[0, 0], [2048, 0]], 'constant')
    local_condition = local_condition.reshape([1, -1, hparams.num_mels])
    saver = tf.train.Saver(ema.variables_to_restore())

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(force_gpu_compatible=True, allow_growth=True),
        log_device_placement=False,
    )
    with tf.Session(config=config) as sess:
        saver.restore(sess, checkpoint_path)
        start_time = time.time()
        outputs = sess.run(model.outputs, feed_dict={ph['local_condition']: local_condition})

        duration = time.time() - start_time
        print('Time Evaluation: Generation of {} audio samples took {:.3f} sec ({:.3f} frames/sec)'.format(
            len(outputs), duration, len(outputs) / duration))

        waveform = np.reshape(outputs, [-1])
        waveplot('test.png', waveform, None, hparams)

        librosa.output.write_wav('test.wav', waveform, sr=16000)


def main1():
    checkpoint_path = tf.train.get_checkpoint_state('logs-test/checkpoints/').model_checkpoint_path

    ph = create_placeholders()

    model = create_model(ph)
    # apply ema to variable
    ema = tf.train.ExponentialMovingAverage(decay=hparams.ema_decay)

    samples = [0] * model.receptive_filed

    audio = np.load('/mnt/lustre/sjtu/users/kc430/data/my/fftnet/cmu_arctic/cmu_arctic-audio-03139.npy')
    audio = np.reshape(audio, [1, -1])
    # audio = np.pad(audio, [[0, 0], [2048, 0]], 'constant')

    local_condition = np.load('/mnt/lustre/sjtu/users/kc430/data/my/fftnet/cmu_arctic/cmu_arctic-mel-03139.npy')
    local_condition = local_condition.reshape([1, -1, hparams.num_mels])

    saver = tf.train.Saver(ema.variables_to_restore())

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(force_gpu_compatible=True, allow_growth=True),
        log_device_placement=False,
    )
    out_samples = []
    with tf.Session(config=config) as sess:
        saver.restore(sess, checkpoint_path)
        upsample_local_condition = sess.run(model.upsample_c, feed_dict={ph['local_condition']: local_condition})

        for i in tqdm(range(upsample_local_condition.shape[1] - model.receptive_filed)):
            sample = np.array(samples[-model.receptive_filed:]).reshape(1, -1)
            h = upsample_local_condition[:, i+1:i+1+model.receptive_filed, :]
            output = sess.run(model.posterior, feed_dict={
                ph['inputs']: sample, ph['local_condition']: h})
            output = np.reshape(output, [-1])
            output = np.random.choice(np.arange(hparams.quantize_channels), p=output)
            output = utils.inv_mulaw_quantize(output).reshape(-1)
            samples.append(output[0])
            out_samples.append(output[0])

        waveform = np.array(out_samples)
        librosa.output.write_wav("test.wav", waveform, sr=16000)
        waveplot('test.png', waveform, None, hparams)


if __name__ == '__main__':
    print(hparams_debug_string())
    main()
    # main()