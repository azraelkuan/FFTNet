import numpy as np
import tensorflow as tf


hparams = tf.contrib.training.HParams(
    # Input type:
    # 1. raw [-1, 1]
    # 2. mulaw [-1, 1]
    # 3. mulaw-quantize [0, mu]
    # If input_type is raw or mulaw, network assumes scalar input and
    # discretized mixture of logistic distributions output, otherwise one-hot
    # input and softmax output are assumed.
    # **NOTE**: if you change the one of the two parameters below, you need to
    # re-run preprocessing before training.
    input_type="raw",
    quantize_channels=256,  # 65536 or 256

    # Audio:
    sample_rate=16000,
    # this is only valid for mulaw is True
    silence_threshold=2,
    num_mels=80,
    fmin=125,
    fmax=7600,
    fft_size=1024,
    # shift can be specified by either hop_size or frame_shift_ms
    hop_size=256,
    frame_shift_ms=None,
    min_level_db=-100,
    ref_level_db=20,
    # whether to rescale waveform or not.
    # Let x is an input waveform, rescaled waveform y is given by:
    # y = x / np.abs(x).max() * rescaling_max
    rescaling=True,
    rescaling_max=0.999,
    # mel-spectrogram is normalized to [0, 1] for each utterance and clipping may
    # happen depends on min_level_db and ref_level_db, causing clipping noise.
    # If False, assertion is added to ensure no clipping happens.o0
    allow_clipping_in_normalization=True,

    # Mixture of logistic distributions:
    log_scale_min=-7.0,

    # =========================== model parameters =========================== #
    batch_size=8,
    hidden_channels=256,
    n_layers=11,
    freq_axis_kernel_size=3,

    init_lr=1e-3,
    ema_decay=0.9999,

    # Global conditioning (set negative value to disable)
    # currently limited for speaker embedding
    # this should only be enabled for multi-speaker dataset
    gin_channels=-1,  # i.e., speaker embedding dim
    n_speakers=7,  # 7 for CMU ARCTIC

    # Local conditioning (set negative value to disable))
    cin_channels=80,
    # If True, use transposed convolutions to upsample conditional features,
    # otherwise repeat features to adjust time resolution
    upsample_conditional_features=True,
    # should np.prod(upsample_scales) == hop_size
    upsample_scales=[4, 4, 4, 4],

    # max time steps can either be specified as sec or steps
    # if both are None, then full audio samples are used in a batch
    max_time_sec=None,
    max_time_steps=8000,


)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
    return 'Hyperparameters:\n' + '\n'.join(hp)
