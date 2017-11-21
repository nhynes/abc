import argparse
import os
import pickle
import random
import sys

import torch
import numpy as np


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJ_ROOT, 'data')
RUN_DIR = os.path.join(PROJ_ROOT, 'run')

sys.path.insert(0, PROJ_ROOT)
import environ
import common


def _get_metadata(embs_path):
    embs_name = os.path.splitext(os.path.basename(embs_path))[0]
    embs_name_parts = embs_name.split('_')
    questions_name = '_'.join(embs_name_parts[1:-1])
    part = embs_name_parts[-1]
    questions = common.unpickle(
        os.path.join(DATA_DIR, questions_name, f'{part}.pkl'))
    return {
        'questions': questions,
        'dataset': [questions_name] * len(questions),
    }


def main():
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='g_ml')
    parser.add_argument('--subsample', default=1, type=int)
    parser.add_argument('--out-dir', default='wembs_projector')
    args = parser.parse_args()
    # --------------------------------------------------------------------------

    opts = argparse.Namespace(**common.unpickle(os.path.join(RUN_DIR, 'opts.pkl')))
    env = environ.create(opts.env, opts)
    env.state = torch.load(os.path.join(RUN_DIR, 'g_ml', 'state.pth'))

    wembs = env.g.tok_emb.weight.data.cpu().numpy()
    vocab = [tok for tok, _ in env.train_dataset.vocab.tok_counts]
    # wembs = np.load('../data/tok_vecs_pruned.npy')
    # vocab = common.EXTRA_VOCAB + [t for t, _ in common.unpickle('../data/qa/vocab.pkl').tok_counts]
    # vocab = vocab[:len(wembs)]
    # print(vocab)

    if args.subsample:
        wembs = wembs[::args.subsample]
        vocab = vocab[::args.subsample]

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    import tensorflow as tf
    from tensorflow.contrib.tensorboard.plugins import projector

    out_dir = os.path.join(RUN_DIR, args.out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    metadata_path = os.path.join(out_dir, 'metadata.tsv')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess, tf.device("/cpu:0"):
        with open(metadata_path, 'w') as f_md:
            for tok in vocab:
                print(tok, file=f_md)

        embs_var = tf.Variable(wembs, trainable=False, name='tok_embs')
        sess.run(tf.global_variables_initializer())
        print(sess.run(embs_var[0]) - wembs[0])

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(out_dir, 'model.ckpt'), global_step=42)

        projector_config = projector.ProjectorConfig()
        projector_config.embeddings.add(tensor_name=embs_var.name,
                                        metadata_path=metadata_path)

        summary_writer = tf.summary.FileWriter(out_dir, sess.graph)
        projector.visualize_embeddings(summary_writer, projector_config)


if __name__ == '__main__':
    main()
