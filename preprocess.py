# coding: utf-8
import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
import importlib
from hparams import hparams


def preprocess(mod, hp, in_dir, out_root, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = mod.build_from_path(hp, in_dir, out_dir, num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    sr = hparams.sample_rate
    hours = frames / sr / 3600
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="", help='dataset name', required=True)
    parser.add_argument('--in_dir', default="", help='input dir', required=True)
    parser.add_argument('--out_dir', default="", help='output dir', required=True)
    parser.add_argument('--num_workers', default=cpu_count(), help='num workers')
    parser.add_argument('--preset', default=None, help='preset json')
    parser.add_argument('--hparams', default='', help='extra hparams, pair as key=value')

    args = parser.parse_args()
    name = args.name
    in_dir = args.in_dir
    out_dir = args.out_dir
    num_workers = args.num_workers
    preset = args.preset

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args.hparams)

    print("Sampling frequency: {}".format(hparams.sample_rate))

    assert name in ["cmu_arctic", "ljspeech"]
    mod = importlib.import_module("datasets.{}".format(name))
    preprocess(mod, hparams, in_dir, out_dir, num_workers)
