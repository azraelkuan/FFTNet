import matplotlib
matplotlib.use('Agg')
import librosa.display as dsp
import matplotlib.pyplot as plt


def waveplot(path, y_hat, y_target, hparams):
    sr = hparams.sample_rate

    if y_target is not None:
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(2, 1, 1)
        dsp.waveplot(y_target, sr=sr)
        ax.set_title('Target waveform')
        ax = plt.subplot(2, 1, 2)
        dsp.waveplot(y_hat, sr=sr)
        ax.set_title('Prediction waveform')
    else:
        plt.figure(figsize=(12, 3))
        ax = plt.subplot(1, 1, 1)
        dsp.waveplot(y_hat, sr=sr)
        ax.set_title('Generated waveform')

    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()
