"""Stage 6 (Tier 3) regression tests: gated-attention MIL
(docs/MIL_REFRAME_PLAN.md §4, Stage 6 backlog §13).

All synthetic. ``AttentionMIL`` calls its encoder inside a differentiable
Keras graph (``self.encoder(flat, training=False)``), not via ``.predict()``
the way Tier 4's fakes did (tests/test_stage6_generative_mil.py) -- so this
needs a genuine, if tiny, Keras functional model matching the real CVAE
encoder's 4-output signature ``[z_mean, z_log_var, z, class_pred]``
(``_tiny_encoder`` below), not a bare Python object.
"""
import numpy as np
import pytest
import tensorflow as tf

from src.mil.attention import (
    AttentionMIL, required_sub_bag_size, sample_sub_bags, sub_bag_null_rate,
)


# ---------------------------------------------------------------------------
# 1. required_sub_bag_size -- pure arithmetic
# ---------------------------------------------------------------------------

def test_required_sub_bag_size_matches_plans_worked_example():
    """docs/MIL_REFRAME_PLAN.md §4.1: pi_min=2%, epsilon=1% -> s>=228."""
    assert required_sub_bag_size(pi_min=0.02, epsilon=0.01) == 228


def test_required_sub_bag_size_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        required_sub_bag_size(pi_min=0.0, epsilon=0.01)
    with pytest.raises(ValueError):
        required_sub_bag_size(pi_min=0.02, epsilon=1.5)


def test_required_sub_bag_size_grows_as_pi_min_shrinks():
    """A rarer diagnostic fraction needs a bigger sub-bag to keep the same
    empty-sub-bag risk -- monotonicity is the whole point of tying size to
    pi_min rather than guessing a constant."""
    s_02 = required_sub_bag_size(pi_min=0.02, epsilon=0.01)
    s_01 = required_sub_bag_size(pi_min=0.01, epsilon=0.01)
    assert s_01 > s_02


# ---------------------------------------------------------------------------
# 2. sample_sub_bags -- uniform draws, correct bag membership, replacement
#    only when needed
# ---------------------------------------------------------------------------

def _toy_bags(n_bags_per_class=3, n_per_bag=50, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X, y_bag, bag_ids = [], [], []
    for cls in (0, 1):
        for b in range(n_bags_per_class):
            bag_id = f"bag_{cls}_{b}"
            X.append(rng.normal(size=(n_per_bag, n_features)).astype(np.float32))
            y_bag += [cls] * n_per_bag
            bag_ids += [bag_id] * n_per_bag
    return np.concatenate(X), np.array(y_bag), np.array(bag_ids, dtype=object)


def test_sample_sub_bags_shapes_and_bag_membership():
    X, y_bag, bag_ids = _toy_bags(n_bags_per_class=2, n_per_bag=100)
    X_sub, y_sub, ids_sub = sample_sub_bags(
        X, bag_ids, y_bag, n_per_bag=5, sub_bag_size=20, seed=1)

    n_bags = len(np.unique(bag_ids))
    assert X_sub.shape == (n_bags * 5, 20, X.shape[1])
    assert y_sub.shape == (n_bags * 5,)
    assert ids_sub.shape == (n_bags * 5,)

    # every sub-bag's instances actually came from its own labelled bag
    for i in range(len(ids_sub)):
        b = ids_sub[i]
        bag_rows = X[bag_ids == b]
        # each of the 20 sub-bag rows must be a row of its parent bag
        assert all(any(np.array_equal(row, br) for br in bag_rows) for row in X_sub[i])
        assert y_sub[i] == y_bag[bag_ids == b][0]


def test_sample_sub_bags_uses_replacement_only_for_small_bags():
    """A bag smaller than sub_bag_size must still produce a full-size
    sub-bag (via replacement), not silently truncate or skip it."""
    X, y_bag, bag_ids = _toy_bags(n_bags_per_class=1, n_per_bag=10)
    X_sub, y_sub, ids_sub = sample_sub_bags(
        X, bag_ids, y_bag, n_per_bag=3, sub_bag_size=50, seed=2)
    assert X_sub.shape[1] == 50  # full requested size despite bag size 10


def test_sample_sub_bags_draws_differ_across_calls():
    """Not a proof of uniformity, but a basic sanity check that repeated
    sampling doesn't return the same fixed subset every time (e.g. always
    the first sub_bag_size rows, which would silently be a contiguous,
    non-random draw)."""
    X, y_bag, bag_ids = _toy_bags(n_bags_per_class=1, n_per_bag=200)
    X_sub_a, _, _ = sample_sub_bags(X, bag_ids, y_bag, n_per_bag=1, sub_bag_size=20, seed=10)
    X_sub_b, _, _ = sample_sub_bags(X, bag_ids, y_bag, n_per_bag=1, sub_bag_size=20, seed=11)
    assert not np.array_equal(X_sub_a, X_sub_b)


# ---------------------------------------------------------------------------
# 3. sub_bag_null_rate -- matches the binomial expectation it's meant to
#    verify empirically
# ---------------------------------------------------------------------------

def test_sub_bag_null_rate_matches_binomial_expectation():
    """Build one bag with a KNOWN, exact diagnostic fraction (via a
    hand-built instance_posterior, not a fitted model) and check the
    empirical null rate lands near (1-pi_b)**s, the exact quantity
    required_sub_bag_size's own derivation is based on."""
    n, pi_b, sub_bag_size = 5000, 0.05, 50
    rng = np.random.default_rng(3)
    bag_ids = np.array(['only_bag'] * n, dtype=object)
    y_bag_per_instance = np.zeros(n, dtype=int)

    is_diag = rng.random(n) < pi_b
    posterior = np.zeros((n, 2))
    posterior[is_diag, 1] = 0.9     # own-class mass = 0.9 for diagnostic instances
    posterior[~is_diag, 1] = 0.05   # well below the 0.5 threshold otherwise
    posterior[:, 0] = 1 - posterior[:, 1]

    rate = sub_bag_null_rate(
        np.zeros((n, 1)), bag_ids, y_bag_per_instance, posterior,
        sub_bag_size=sub_bag_size, n_per_bag=2000, seed=4)
    expected = (1 - pi_b) ** sub_bag_size
    assert rate == pytest.approx(expected, abs=0.03)


# ---------------------------------------------------------------------------
# 4. AttentionMIL -- a genuine (tiny) Keras encoder, since this model calls
#    it inside a differentiable graph, not via .predict()
# ---------------------------------------------------------------------------

def _tiny_encoder(n_features, latent_dim=8, num_classes=2, seed=0):
    """A minimal real Keras functional model matching src/models/cvae.py's
    encoder output signature ([z_mean, z_log_var, z, class_pred]) -- small
    and fast, but a genuine differentiable graph, unlike the bare-.predict()
    fakes in test_stage6_generative_mil.py. Untrained/random weights: this
    test only needs the WIRING to be correct, the recovery test below trains
    AttentionMIL's own instance head on top of it regardless of what the
    (frozen, never-updated) trunk itself outputs.
    """
    tf.random.set_seed(seed)
    x_in = tf.keras.Input(shape=(n_features, 1))
    flat = tf.keras.layers.Flatten()(x_in)
    h = tf.keras.layers.Dense(16, activation="relu")(flat)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(h)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(h)
    z = z_mean  # deterministic stand-in; sampling noise isn't needed here
    class_pred = tf.keras.layers.Dense(num_classes, activation="softmax")(h)
    return tf.keras.Model(x_in, [z_mean, z_log_var, z, class_pred])


def _separable_sub_bags(n_features=6, seed=5):
    """A sub-bag-shaped synthetic dataset where a KNOWN subset of features
    linearly separates class 0 from class 1 -- something a trainable
    instance head + attention should be able to pick up regardless of what
    the frozen (random-weight) trunk above does to the raw features,
    because the trunk's Dense layers are linear-ish and initialized with
    nonzero weights, so class-separating signal in the input generally
    survives into z_mean even without training it."""
    rng = np.random.default_rng(seed)
    n_bags_per_class, n_per_bag = 6, 300
    X, y_bag, bag_ids = [], [], []
    for cls in (0, 1):
        center = np.zeros(n_features)
        center[0] = -4.0 if cls == 0 else 4.0
        for b in range(n_bags_per_class):
            X.append(rng.normal(center, 1.0, size=(n_per_bag, n_features)).astype(np.float32))
            y_bag += [cls] * n_per_bag
            bag_ids += [f"bag_{cls}_{b}"] * n_per_bag
    return np.concatenate(X), np.array(y_bag), np.array(bag_ids, dtype=object)


def test_attention_mil_call_shapes():
    n_features, latent_dim, num_classes = 6, 8, 2
    encoder = _tiny_encoder(n_features, latent_dim, num_classes)
    model = AttentionMIL(encoder, latent_dim, num_classes, attn_dim=8)

    x_sub = tf.random.normal((4, 20, n_features))  # (batch, sub_bag_size, n_features)
    bag_probs, instance_logits, attn = model(x_sub, training=False)
    assert bag_probs.shape == (4, num_classes)
    assert instance_logits.shape == (4, 20, num_classes)
    assert attn.shape == (4, 20)
    np.testing.assert_allclose(tf.reduce_sum(attn, axis=1).numpy(), 1.0, atol=1e-5)
    np.testing.assert_allclose(tf.reduce_sum(bag_probs, axis=1).numpy(), 1.0, atol=1e-5)


def test_attention_mil_trunk_stays_frozen_during_training():
    n_features, latent_dim, num_classes = 6, 8, 2
    encoder = _tiny_encoder(n_features, latent_dim, num_classes)
    encoder_weights_before = [w.numpy().copy() for w in encoder.trainable_weights]
    model = AttentionMIL(encoder, latent_dim, num_classes, attn_dim=8)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    X, y_bag, bag_ids = _separable_sub_bags(n_features)
    X_sub, y_sub, _ = sample_sub_bags(X, bag_ids, y_bag, n_per_bag=10, sub_bag_size=30, seed=6)
    y_oh = tf.keras.utils.to_categorical(y_sub, num_classes=num_classes)
    model.fit(X_sub, y_oh, epochs=2, batch_size=8, verbose=0)

    assert len(encoder.trainable_weights) == 0, "encoder.trainable=False should empty this list"
    # Confirm the actual VALUES never moved either, not just that the
    # trainable list is empty by convention.
    for before, after in zip(encoder_weights_before, [w.numpy() for w in encoder.weights]):
        np.testing.assert_array_equal(before, after)


def test_attention_mil_trains_and_recovers_separable_classes():
    n_features, latent_dim, num_classes = 6, 8, 2
    encoder = _tiny_encoder(n_features, latent_dim, num_classes, seed=7)
    model = AttentionMIL(encoder, latent_dim, num_classes, attn_dim=16)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-3))

    X, y_bag, bag_ids = _separable_sub_bags(n_features, seed=8)
    X_sub, y_sub, _ = sample_sub_bags(X, bag_ids, y_bag, n_per_bag=20, sub_bag_size=30, seed=9)
    y_oh = tf.keras.utils.to_categorical(y_sub, num_classes=num_classes)

    history = model.fit(X_sub, y_oh, epochs=15, batch_size=16, verbose=0)
    final_acc = history.history['bag_accuracy'][-1]
    assert final_acc > 0.85, f"expected the attention model to learn a linearly-separable " \
                             f"task, got bag_accuracy={final_acc}"


def test_attention_mil_instance_posterior_shape_and_normalisation():
    n_features, latent_dim, num_classes = 6, 8, 2
    encoder = _tiny_encoder(n_features, latent_dim, num_classes)
    model = AttentionMIL(encoder, latent_dim, num_classes, attn_dim=8)
    X, _, bag_ids = _separable_sub_bags(n_features)
    posterior = model.instance_posterior(X[:50], bag_ids[:50])
    assert posterior.shape == (50, num_classes + 1)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Regularisation added after the real exp_501 smoke test showed the
# default (unregularised, attn_dim=64) configuration overfitting to bag
# IDENTITY rather than diagnostic content (docs/MIL_REFRAME_PLAN.md backlog
# §13; sub-bag loss fell while 7-bag inner-val accuracy got WORSE with more
# training). Two independent fixes, both tested directly.
# ---------------------------------------------------------------------------

def _identity_encoder(n_features, num_classes, seed=0):
    """A LINEAR stand-in for _tiny_encoder, z_mean == x exactly (no hidden
    ReLU layer to confound the centering test below): a constant offset
    added to X must appear as the identical constant offset in z_mean, so
    the offset-cancellation property is tested cleanly in the space
    centering actually operates on, not distorted by encoder nonlinearity
    (_tiny_encoder's ReLU does NOT preserve additive offsets -- confirmed by
    this test initially failing against it, for that reason, not a bug in
    the centering logic itself)."""
    tf.random.set_seed(seed)
    x_in = tf.keras.Input(shape=(n_features, 1))
    flat = tf.keras.layers.Flatten()(x_in)
    z_mean = tf.keras.layers.Dense(
        n_features, use_bias=False,
        kernel_initializer=tf.keras.initializers.Identity(), trainable=False,
        name="z_mean")(flat)
    z_log_var = tf.keras.layers.Lambda(lambda t: tf.zeros_like(t))(z_mean)
    z = z_mean
    class_pred = tf.keras.layers.Dense(num_classes, activation="softmax")(flat)
    return tf.keras.Model(x_in, [z_mean, z_log_var, z, class_pred])


def test_center_by_bag_cancels_a_pure_bag_level_offset():
    """The mechanism the smoke test's failure pointed at directly: if two
    bags differ only by a CONSTANT per-bag offset added to every one of
    their instances (a stand-in for a bag-specific acquisition/baseline
    fingerprint the frozen trunk could otherwise leak into z_mean),
    centering must make the instance head see IDENTICAL features for
    instances that have the same WITHIN-bag relative position -- i.e. the
    bag-level offset must contribute nothing after centering, and
    everything before it."""
    n_features, latent_dim, num_classes = 4, 4, 2
    encoder = _identity_encoder(n_features, num_classes, seed=20)

    # bag A and bag B: identical relative structure, offset by a constant
    # that differs only by which bag it is -- not by class or content.
    rng = np.random.default_rng(21)
    base = rng.normal(size=(10, n_features)).astype(np.float32)
    x_bag_a = base + np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)   # bag A's fingerprint
    x_bag_b = base + np.array([-5.0, -5.0, -5.0, -5.0], dtype=np.float32)  # bag B's fingerprint
    X = np.concatenate([x_bag_a, x_bag_b])
    bag_ids = np.array(['A'] * 10 + ['B'] * 10, dtype=object)

    model_centered = AttentionMIL(encoder, latent_dim, num_classes, attn_dim=8,
                                  dropout_rate=0.0, l2=0.0, center_by_bag=True)
    post_centered = model_centered.instance_posterior(X, bag_ids)
    # Instance i of bag A and instance i of bag B have the SAME within-bag
    # relative content (both are `base[i]` relative to their own bag's
    # mean) -- centered, their posteriors should match closely.
    np.testing.assert_allclose(post_centered[:10], post_centered[10:], atol=1e-4)

    model_uncentered = AttentionMIL(encoder, latent_dim, num_classes, attn_dim=8,
                                    dropout_rate=0.0, l2=0.0, center_by_bag=False)
    model_uncentered.instance_posterior(X, bag_ids)  # force-build its layers first, so its
    # weight list matches model_centered's (already built by the call above) in length/order
    model_uncentered.set_weights(model_centered.get_weights())
    post_uncentered = model_uncentered.instance_posterior(X, bag_ids)
    # Same weights, but WITHOUT centering the +-5.0 bag fingerprint reaches
    # the instance head directly -- the two bags' posteriors should now
    # differ (this is exactly the leakage channel being closed).
    assert not np.allclose(post_uncentered[:10], post_uncentered[10:], atol=1e-4)


def test_attention_mil_regularised_defaults_still_train_shapes_correctly():
    """Dropout + L2 + centering active (the new defaults) must not break
    the training/shape contract -- a pure wiring check, not a convergence
    claim (see the recovery test above for that, run without them for a
    clean signal)."""
    n_features, latent_dim, num_classes = 6, 8, 2
    encoder = _tiny_encoder(n_features, latent_dim, num_classes)
    model = AttentionMIL(encoder, latent_dim, num_classes)  # all-default: dropout=0.3, l2=1e-3
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-3))

    X, y_bag, bag_ids = _separable_sub_bags(n_features)
    X_sub, y_sub, _ = sample_sub_bags(X, bag_ids, y_bag, n_per_bag=5, sub_bag_size=20, seed=22)
    y_oh = tf.keras.utils.to_categorical(y_sub, num_classes=num_classes)
    history = model.fit(X_sub, y_oh, epochs=3, batch_size=8, verbose=0)
    assert np.isfinite(history.history['loss']).all()

    posterior = model.instance_posterior(X[:30], bag_ids[:30])
    assert posterior.shape == (30, num_classes + 1)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-5)
