"""
Shared CVAE construction + synthetic-generation helpers.

Factored out of train_CVAE.py so that both the CVAE stage (which trains and then
generates a fixed pool) and the TSRTR stage (which reuses a trained decoder to
generate exactly the number of spectra a ratio requires) build the model and
generate the same way.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path


def build_cvae_config(config, n_features):
    """
    Translate the experiment YAML's ``models.cvae`` block into the dict the
    ``CVAE`` class expects. ``input_shape`` is derived from the actual (possibly
    cropped) feature count so the architecture matches the data.

    gamma/delta (Suite 200 reconstruction-loss ablation) are passed through so
    the ablation actually takes effect during training; they don't affect the
    encoder/decoder topology, so decoder weights remain load-compatible
    regardless of their values.
    """
    c = config['models']['cvae']
    num_classes = config['models']['cnn']['num_classes']
    lw = c['loss_weights']

    return {
        'input_shape': (n_features, 1),
        'num_classes': num_classes,
        'latent_dim': c['latent_dim'],
        'loss_weights': {
            'beta': lw.get('beta', 1.0),
            'alpha': lw.get('alpha', 0.2),
            'gamma': lw.get('gamma', None),
            'delta': lw.get('delta', None),
        },
        'encoder': {
            'filters': c['encoder']['filters'],
            'kernels': c['encoder']['kernels'],
            'strides': c['encoder']['strides'],
            'use_global_pooling': False,
            'use_skip_gap': False,
        },
        'decoder': {
            'initial_filters': c['decoder']['initial_filters'],
            'use_dense_projection': True,
            'filters': c['decoder']['filters'],
            'kernels': c['decoder']['kernels'],
            'strides': c['decoder']['strides'],
        },
    }


def load_decoder(cvae_config, weights_dir, fold_name):
    """
    Rebuild a CVAE from ``cvae_config`` and load the saved decoder weights for
    ``fold_name`` from ``weights_dir`` (file: ``decoder_{fold_name}.weights.h5``).
    Returns the decoder Keras Model, ready for generation.
    """
    from src.models.cvae import CVAE

    dec_path = Path(weights_dir) / f"decoder_{fold_name}.weights.h5"
    if not dec_path.exists():
        raise FileNotFoundError(f"Decoder weights not found: {dec_path}")

    tf.keras.backend.clear_session()
    cvae = CVAE(cvae_config)
    cvae.decoder.load_weights(str(dec_path))
    return cvae.decoder


def load_encoder_decoder(cvae_config, weights_dir, fold_name):
    """
    Rebuild a CVAE from ``cvae_config`` and load BOTH the encoder and decoder
    weights for ``fold_name`` from ``weights_dir`` (files
    ``encoder_{fold}.weights.h5`` / ``decoder_{fold}.weights.h5``).
    Returns (encoder, decoder) Keras Models — used for Classification by
    Reconstruction in the evaluation stage.
    """
    from src.models.cvae import CVAE

    enc_path = Path(weights_dir) / f"encoder_{fold_name}.weights.h5"
    dec_path = Path(weights_dir) / f"decoder_{fold_name}.weights.h5"
    if not enc_path.exists() or not dec_path.exists():
        raise FileNotFoundError(f"CVAE weights not found for {fold_name} in {weights_dir}")

    tf.keras.backend.clear_session()
    cvae = CVAE(cvae_config)
    cvae.encoder.load_weights(str(enc_path))
    cvae.decoder.load_weights(str(dec_path))
    return cvae.encoder, cvae.decoder


def generate_synthetic(decoder, n_per_class, num_classes, latent_dim, batch_size=5000):
    """
    Generate ``n_per_class`` synthetic spectra for each class by decoding
    N(0, I) latent samples conditioned on the one-hot class label.

    Returns (X_synth, y_synth) with X_synth shape (n_per_class*num_classes,
    n_features) as float32 (2D — the decoder's trailing channel dim is squeezed)
    and y_synth the matching integer labels.
    """
    if n_per_class <= 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X_parts, y_parts = [], []
    for class_idx in range(num_classes):
        chunks = []
        remaining = n_per_class
        while remaining > 0:
            bs = int(min(batch_size, remaining))
            z = tf.random.normal(shape=(bs, latent_dim))
            label = tf.convert_to_tensor(
                tf.keras.utils.to_categorical([class_idx] * bs, num_classes=num_classes),
                dtype=tf.float32,
            )
            gen = decoder([z, label], training=False).numpy()
            if gen.ndim == 3:                       # (bs, n_features, 1) -> (bs, n_features)
                gen = gen.reshape(gen.shape[0], -1)
            chunks.append(gen)
            remaining -= bs
        X_parts.append(np.vstack(chunks))
        y_parts.append(np.full(n_per_class, class_idx, dtype=np.int64))

    return np.vstack(X_parts).astype(np.float32), np.concatenate(y_parts)
