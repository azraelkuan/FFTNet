# coding: utf-8
from __future__ import with_statement, print_function, absolute_import
import tensorflow as tf
import numpy as np
import six
from tensorflow.python.training import device_setter
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev


def _assert_valid_input_type(s):
    assert s == 'mulaw-quantize' or s == 'mulaw' or s == 'raw'


def is_mulaw_quantize(s):
    _assert_valid_input_type(s)
    return s == 'mulaw-quantize'


def is_mulaw(s):
    _assert_valid_input_type(s)
    return s == 'mulaw'


def is_raw(s):
    _assert_valid_input_type(s)
    return s == 'raw'


def is_scalar_input(s):
    return is_raw(s) or is_mulaw(s)


# From https://github.com/r9y9/nnmnkwii/blob/master/nnmnkwii/preprocessing/generic.py
def mulaw(x, mu=256):
    """Mu-Law companding"""
    return _sign(x) * _log1p(mu * _abs(x)) / _log1p(mu)


def inv_mulaw(y, mu=256):
    """Inverse of mu-law companding (mu-law expansion)"""
    return _sign(y) * (1.0 / mu) * ((1.0 + mu) ** _abs(y) - 1.0)


def mulaw_quantize(x, mu=256):
    """Mu-Law companding + quantize"""
    y = mulaw(x, mu)
    # scale [-1, 1] to [0, mu]
    return _asint((y + 1) / 2 * mu)


def inv_mulaw_quantize(y, mu=256):
    """Inverse of mu-law companding + quantize"""
    y = 2 * _asfloat(y) / mu - 1
    return inv_mulaw(y, mu)


def _sign(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sign(x) if isnumpy or isscalar else tf.sign(x)


def _log1p(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.log1p(x) if isnumpy or isscalar else tf.log1p(x)


def _abs(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.abs(x) if isnumpy or isscalar else tf.abs(x)


def _asint(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.int) if isnumpy else int(x) if isscalar else tf.cast(x, tf.int32)


def _asfloat(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.float32) if isnumpy else float(x) if isscalar else tf.cast(x, tf.float32)


def get_available_gpus(config):
    """Returns a list of the identifiers of all visible GPUs."""
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices(session_config=config)
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, v in grad_and_vars:
            if g is None:
                continue
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        if len(grads) == 0:
            average_grads.append((None, v))
            continue

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops == None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser