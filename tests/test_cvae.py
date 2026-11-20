import copy

import numpy as np
import tensorflow as tf

from pipeline._cvae_gen import build_cvae_config
from src.models.cvae import CVAE
from src.utils.config_manager import ConfigManager

N_BANDS = 571  # fingerprint-region band count used by every paper experiment


def _paper_cvae_config(n_features=N_BANDS):
    """The CVAE config the pipeline builds for the paper's default experiment."""
    return build_cvae_config(ConfigManager.load_config('exp_608'), n_features)


def test_model_build():
    print("\n[1/2] Testing Dynamic Model Building...")

    # A smaller custom architecture on the production config structure, to test
    # that the encoder/decoder loops follow the config rather than fixed sizes.
    config = copy.deepcopy(_paper_cvae_config(n_features=100))
    config['encoder'].update({"filters": [16, 32], "kernels": [3, 3], "strides": [2, 2]})
    config['decoder'].update({"initial_filters": 32, "filters": [32, 16],
                              "kernels": [3, 3], "strides": [2, 2]})

    model = CVAE(config)

    # Dummy Input (Batch size 1, 100 bands, 1 channel)
    x = tf.random.normal((1, 100, 1))
    y = tf.one_hot([0], depth=config['num_classes'])

    # Run inference to build the graph
    model([x, y])

    # Check output shapes
    if model.decoder.output_shape != (None, 100, 1):
        raise ValueError(f"Decoder output shape mismatch! Got {model.decoder.output_shape}")

    print(" Model built successfully with dynamic config.")


def test_training_step():
    print("\n[2/2] Testing Training Step (Gradient Flow)...")

    config = _paper_cvae_config()
    num_classes = config['num_classes']
    batch_size = 4

    # Cast to Tensors explicitly to simulate model.fit() behavior
    X_dummy_np = np.random.rand(batch_size, N_BANDS, 1).astype(np.float32)
    y_dummy_np = np.eye(num_classes)[np.random.choice(num_classes, batch_size)].astype(np.float32)

    X_dummy = tf.convert_to_tensor(X_dummy_np)
    y_dummy = tf.convert_to_tensor(y_dummy_np)

    model = CVAE(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer)

    # Run one training step
    logs = model.train_step((X_dummy, y_dummy))

    required_keys = ['total_loss', 'recon_loss', 'kl_loss', 'class_loss']
    for k in required_keys:
        if k not in logs:
            raise ValueError(f"Missing metric '{k}' in training logs. Found: {list(logs.keys())}")
        if np.isnan(logs[k]):
            raise ValueError(f"Metric '{k}' is NaN! Check loss functions.")

    print(f" Training step passed. Total Loss: {logs['total_loss']:.4f}")
