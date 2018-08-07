import tensorflow as tf
import numpy as np
import time
import argparse
import os
from hparams import hparams, hparams_debug_string
from models.fftnet import FFTNet
import librosa
from utils.plot import waveplot
from utils import audio


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default=None, type=str, help='the path of checkpoint', required=True)
    parser.add_argument('--local_path', default=None, type=str, help='the path of local condition', required=True)
    parser.add_argument('--global_id', default=None, type=int, help='the speaker id')
    parser.add_argument('--output_dir', default='output/', type=str, help='the output wav')
    parser.add_argument('--preset', default=None, type=str, help='the preset config json file')
    parser.add_argument('--hparams', default='',
                        help='Hyper parameter overrides as a comma-separated list of name=value pairs')
    return parser.parse_args()


def create_placeholders():
    ph = {
        'local_condition': tf.placeholder(dtype=tf.float32, shape=[1, None, hparams.num_mels]),
        'test_inputs': tf.placeholder(dtype=tf.float32, shape=[1, None], name='test_inputs')
    }
    return ph


def create_model(ph, hp):
    with tf.variable_scope("model"):
        model = FFTNet(hp)
        model.incremental_forward(c=ph['local_condition'], g=None, targets=None)
    return model


def synthesis(checkpoint_path, local_path, global_id, output_dir, hp):
    checkpoint_name = checkpoint_path.split('/')[-1]
    audio_dir = os.path.join(output_dir, checkpoint_name, 'wavs')
    plot_dir = os.path.join(output_dir, checkpoint_name, 'plots')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    ph = create_placeholders()

    model = create_model(ph, hp)
    # apply ema to variable
    ema = tf.train.ExponentialMovingAverage(decay=hp.ema_decay)

    local_condition = np.load(local_path)
    local_condition = local_condition.reshape([1, -1, hparams.num_mels])

    if not hp.upsample_conditional_features:
        local_condition = np.repeat(local_condition, audio.get_hop_size(), axis=1)

    index = local_path.split('-')[-1].split('.')[0]

    saver = tf.train.Saver(ema.variables_to_restore())

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        log_device_placement=False,
    )
    with tf.Session(config=config) as sess:
        saver.restore(sess, checkpoint_path)
        start_time = time.time()
        outputs = sess.run(model.eval_outputs, feed_dict={ph['local_condition']: local_condition})
        duration = time.time() - start_time
        print('Time Evaluation: Generation of {} audio samples took {:.3f} sec ({:.3f} frames/sec)'.format(
            len(outputs), duration, len(outputs) / duration))

        waveform = np.reshape(outputs, [-1])

        audio_path = os.path.join(audio_dir, '{}.wav'.format(index))
        plot_path = os.path.join(plot_dir, '{}.png'.format(index))
        waveplot(plot_path, waveform, None, hp)
        librosa.output.write_wav(audio_path, waveform, sr=hp.sample_rate)


def main():
    args = get_args()
    if args.preset is not None:
        with open(args.preset) as f:
            hparams.parse_json(f.read())

    modified_hp = hparams.parse(args.hparams)
    print(hparams_debug_string())
    synthesis(args.checkpoint_path, args.local_path, args.global_id, args.output_dir, modified_hp)

if __name__ == '__main__':
    main()