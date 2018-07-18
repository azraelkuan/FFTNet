import argparse
import os
import time
import sys
import librosa
import tensorflow as tf
from nnmnkwii import preprocessing as P

from models.fftnet import FFTNet
from hparams import hparams, hparams_debug_string
from models.feeder import get_dataset
from utils.plot import waveplot
from utils.window import ValueWindow


def add_stats(model):
    with tf.variable_scope('train_stats'):
        tf.summary.scalar('loss', model.loss)
        return tf.summary.merge_all()


def add_test_stats(summary_writer, step, val_loss):
    values = [
        tf.Summary.Value(tag='eval_stats/val_loss', simple_value=val_loss),
    ]
    test_summary = tf.Summary(value=values)
    summary_writer.add_summary(test_summary, step)


def create_train_model(feeder, ema, hp, global_step):
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        model = FFTNet(hp)
        local_condition = None if feeder[3][0].dtype.is_bool else feeder[3][0]
        global_condition = None if feeder[4][0].dtype.is_bool else feeder[4][0]
        model.forward(feeder[0][0], feeder[1][0], local_condition, global_condition)
        model.add_loss()
    model.add_optimizer(ema, global_step)
    return model


def create_eval_model(feeder, hp):
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        model = FFTNet(hp)
        local_condition = None if feeder[3][0].dtype.is_bool else feeder[3][0]
        global_condition = None if feeder[4][0].dtype.is_bool else feeder[4][0]
        model.predict(local_condition, global_condition)
    return model


def get_inputs(feeder, num_gpus):
    inputs, targets, input_lengths, local_conditions, global_conditions = feeder
    if num_gpus == 1:
        return [[inputs], [targets], [input_lengths], [local_conditions], [global_conditions]]
    tower_inputs = tf.split(inputs, num_or_size_splits=num_gpus, axis=0)
    tower_targets = tf.split(targets, num_or_size_splits=num_gpus, axis=0)
    tower_input_lengths = tf.split(input_lengths, num_or_size_splits=num_gpus, axis=0)
    tower_local_conditions = tf.split(local_conditions, num_or_size_splits=num_gpus, axis=0)
    tower_global_conditions = tf.split(global_conditions, num_or_size_splits=num_gpus, axis=0)
    return [tower_inputs, tower_targets, tower_input_lengths, tower_local_conditions, tower_global_conditions]


def save_log(sess, step, model, plot_dir, audio_dir, hp):
    predicts, targets = sess.run([model.log_outputs, model.targets])

    y_hat = P.inv_mulaw_quantize(predicts[0], hp.quantize_channels)
    y = P.inv_mulaw_quantize(targets[0], hp.quantize_channels)

    pred_wav_path = os.path.join(audio_dir, 'step-{}-pred.wav'.format(step))
    target_wav_path = os.path.join(audio_dir, 'step-{}-real.wav'.format(step))
    plot_path = os.path.join(plot_dir, 'step-{}-waveplot.png'.format(step))

    # Save audio
    librosa.output.write_wav(pred_wav_path, y_hat, sr=hp.sample_rate)
    librosa.output.write_wav(target_wav_path, y, sr=hp.sample_rate)

    # Save figure
    waveplot(plot_path, y_hat, y, hparams)


def eval_step(eval_model, sess, step, eval_plot_dir, eval_audio_dir):
    start_time = time.time()
    y_hat, y_target, loss = sess.run([model.y_eval_hat_log, model.y_eval_log, model.eval_loss])
    duration = time.time() - start_time
    print('Time Evaluation: Generation of {} audio frames took {:.3f} sec ({:.3f} frames/sec)'.format(
        len(y_target), duration, len(y_target) / duration))


def train(log_dir, args, hp):
    # create dir
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    event_dir = os.path.join(log_dir, 'events')
    os.makedirs(event_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_ckpt')
    audio_dir = os.path.join(log_dir, 'train_stats', 'wavs')
    plot_dir = os.path.join(log_dir, 'train_stats', 'plots')
    eval_audio_dir = os.path.join(log_dir, 'eval_stats', 'wavs')
    eval_plot_dir = os.path.join(log_dir, 'eval_stats', 'plots')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(eval_audio_dir, exist_ok=True)
    os.makedirs(eval_plot_dir, exist_ok=True)

    # create dataset and iterator
    train_dataset = get_dataset(args.train_file, True, hp, batch_size=hp.batch_size)
    val_dataset = get_dataset(args.val_file, False, hp, batch_size=hp.batch_size)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    # feeder: inputs, targets, input_lengths, local_condition, global_condition
    next_inputs = iterator.get_next()
    # To Do: multi gpu training
    feeder = get_inputs(next_inputs, 1)

    train_init = iterator.make_initializer(train_dataset)
    val_init = iterator.make_initializer(val_dataset)

    # global step
    global_step = tf.Variable(name='global_step', initial_value=-1, trainable=False)
    global_val_step = tf.Variable(name='global_val_step', initial_value=-1, trainable=False)
    global_val_step_add = tf.assign_add(global_val_step, 1, name='global_val_step_add')

    # apply ema to variable
    ema = tf.train.ExponentialMovingAverage(decay=hp.ema_decay)
    # create model
    # use multi gpu to train
    train_model = create_train_model(feeder, ema, hp, global_step)
    eval_model = create_eval_model(feeder, hp)

    # save info
    saver = tf.train.Saver(max_to_keep=5)
    train_stats = add_stats(train_model)
    train_loss_window = ValueWindow(100)
    val_loss_window = ValueWindow(100)

    # sess config
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(force_gpu_compatible=True, allow_growth=True),
        allow_soft_placement=True,
        log_device_placement=False,
    )
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(event_dir, sess.graph)

        # restore from checkpoint
        if args.restore_step is not None:
            restore_path = '{}-{}'.format(checkpoint_path, args.restore_step)
            # we don't load the ema to continue training, that is just for evaluating
            saver.restore(sess, restore_path)
            print('Resuming from checkpoint: {}...'.format(args.restore_step))
        else:
            print('Start new training....')

        for epoch in range(args.epochs):
            sess.run(train_init)
            while True:
                try:
                    start_time = time.time()
                    step, loss, _, = sess.run([global_step, train_model.loss, train_model.optimize])
                    train_loss_window.append(loss)
                    if step % 10 == 0:
                        message = 'Epoch {:4d} Train Step {:7d} [{:.3f} sec/step step_loss={:.5f} avg_loss={:.5f}]'.format(
                            epoch, step, time.time()-start_time, loss, train_loss_window.average)
                        print(message)

                    if step % args.checkpoint_interval == 0:
                        saver.save(sess, checkpoint_path, step)
                        save_log(sess, step, train_model, plot_dir, audio_dir, hp)

                    if step % args.summary_interval == 0:
                        print('Writing summary at step {}'.format(step))
                        summary_writer.add_summary(sess.run(train_stats), step)

                    sys.stdout.flush()
                except tf.errors.OutOfRangeError:
                    break

            sess.run(val_init)
            while True:
                try:
                    start_time = time.time()
                    step, loss, _ = sess.run([global_val_step, train_model.loss, global_val_step_add])
                    val_loss_window.append(loss)
                    if step % 10 == 0:
                        message = 'Epoch {:4d} Val Step {:7d} [{:.3f} sec/step step_loss={:.5f} avg_loss={:.5f}]'.format(
                            epoch, step, time.time() - start_time, loss, train_loss_window.average)
                        print(message)

                    if step % args.eval_interval:
                        eval_step(eval_model, sess, step, eval_plot_dir, eval_audio_dir)

                    if step % args.summary_val_interval == 0:
                        add_test_stats(summary_writer, step, loss)

                except tf.errors.OutOfRangeError:
                    break


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)
    print(hparams_debug_string())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir, modified_hp


def main():
    parser = argparse.ArgumentParser(description='Train FFTNet')
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='',
                        help='Hyper parameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--train_file', default='training_data/train.txt')
    parser.add_argument('--val_file', default='training_data/val.txt')
    parser.add_argument('--name', help='Name of logging directory.')
    parser.add_argument('--model', default='fftnet')
    parser.add_argument('--preset', default=None, type=str, help='the preset config json file')
    parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')

    parser.add_argument('--restore_step', default=None, type=int, help='the restore step')

    parser.add_argument('--summary_interval', type=int, default=1000,
                        help='Steps between running summary ops')
    parser.add_argument('--summary_val_interval', type=int, default=20,
                        help='Steps between running summary ops')
    parser.add_argument('--eval_interval', type=int, default=200,
                        help='Steps between train eval ops')
    parser.add_argument('--checkpoint_interval', type=int, default=2000,
                        help='Steps between writing checkpoints')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='total number of tacotron training steps')
    parser.add_argument('--tf_log_level', type=int, default=0, help='TensorFlow C++ log level.')
    args = parser.parse_args()

    # load preset config, so u don't need to change anything in the hparams
    if args.preset is not None:
        with open(args.preset) as f:
            hparams.parse_json(f.read())

    log_dir, hp = prepare_run(args)
    train(log_dir, args, hp)



if __name__ == "__main__":
    main()


