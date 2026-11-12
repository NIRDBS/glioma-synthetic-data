"""
Tier 4: generative MIL (docs/MIL_REFRAME_PLAN.md §4) -- promotes the existing
CVAE into instance classifiers, reusing weights the CVAE stage already
trained and saved per fold:

- 'reconstruction': Classification by Reconstruction's per-instance,
  class-conditional decoder MSE (pipeline/evaluation_stage.py::_cbr_predictions),
  turned into a posterior via softmax(-mse/T) instead of a hard argmin.
- 'encoder_head': the CVAE encoder's own unused auxiliary softmax classifier
  (src/models/cvae.py's ``class_output``, trained with the alpha=0.2 loss
  term as part of ordinary CVAE training) -- discarded by CbR today
  (evaluation_stage.py takes ``encoder.predict(...)[0]``, the z_mean, never
  ``[3]``, the class prediction).

Neither variant models a non-diagnostic class explicitly -- both are K-way,
over diagnostic classes only (the decoder is never conditioned on "not
diagnostic", and the auxiliary head was trained against y_bag one-hot). Both
are lifted to the store's (K+1)-way convention the same way src/mil/misvm.py's
Platt margins are: non-diagnostic mass = 1 - max_k p_k, renormalised.

Both score with the SAME per-fold CVAE weights the ``cvae`` pipeline stage
already produced (results/{exp}/models/cvae/{encoder,decoder}_{fold}.weights.h5),
with no new training -- what makes them "cheap" relative to Tier 1/3
(docs/MIL_REFRAME_PLAN.md Stage 6 reordering).
"""
import numpy as np


def _to_K1_posterior(class_probs):
    """
    Shared K -> K+1 lift (same convention as src/mil/misvm.py's combination
    rule): non-diagnostic mass = 1 - max_k p_k, renormalised to a valid
    probability vector per row.
    """
    class_probs = np.asarray(class_probs, dtype=np.float64)
    non_diag = np.clip(1.0 - class_probs.max(axis=1, keepdims=True), 0.0, None)
    raw = np.concatenate([non_diag, class_probs], axis=1)
    row_sums = raw.sum(axis=1, keepdims=True)
    return (raw / np.where(row_sums > 0, row_sums, 1.0)).astype(np.float32)


def mse_to_posterior(mse, temperature=1.0):
    """
    softmax(-mse/T) over K classes' per-instance, class-conditional
    reconstruction error -> (n, K+1). Pure numpy -- the piece worth
    unit-testing directly against hand-built MSE arrays, independent of any
    CVAE weights.

    ``temperature``: see docs/stage6_generative_mil.md §2 for how the default
    (0.001) was picked from a real smoke test on exp_501 fold_0 -- T=1 (a
    naive default) makes the posterior nearly uniform at this data's real MSE
    scale (mean top-class probability ~0.50 for K=2, i.e. no usable signal);
    T equal to the real median inter-class MSE gap gives a genuinely graded
    posterior without collapsing to a near-hard decision. Provisional, like
    every other project hyperparameter chosen this way (IARPLS lambda, CVAE
    epochs) -- not swept, documented rather than resolved.
    """
    mse = np.asarray(mse, dtype=np.float64)
    logits = -mse / max(float(temperature), 1e-12)
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    exp = np.exp(logits)
    class_probs = exp / exp.sum(axis=1, keepdims=True)
    return _to_K1_posterior(class_probs)


def class_probs_to_posterior(class_probs):
    """
    A K-way softmax (e.g. the CVAE encoder's auxiliary head) -> (n, K+1),
    same non-diagnostic-mass convention. Pure numpy -- no temperature, since
    the input is already a calibrated-by-training-objective softmax, not a
    quantity (like -mse) that needs one to become a probability at all.
    """
    return _to_K1_posterior(class_probs)


def instance_posteriors_from_cvae(X, encoder, decoder, num_classes, temperature=1.0,
                                  batch_size=256):
    """
    One encoder forward pass -> both Tier-4 variants' raw ingredients (the
    encoder's own output tuple is ``[z_mean, z_log_var, z, class_pred]`` --
    z_mean feeds 'reconstruction' via the decoder, class_pred IS
    'encoder_head' directly) -> both (n, K+1) posteriors, at the cost of one
    encoder call plus K decoder calls total, not two independent full passes.

    ``encoder``/``decoder``: Keras models with the same interface as
    ``pipeline._cvae_gen.load_encoder_decoder`` returns (or any object
    exposing a compatible ``.predict(...)`` -- deliberately not type-checked,
    so tests can pass a lightweight stand-in without building a real CVAE).

    Returns {'reconstruction': (n, K+1) float32, 'encoder_head': (n, K+1) float32}.
    """
    import tensorflow as tf

    n_features = X.shape[1]
    X_exp = X.reshape(-1, n_features, 1)
    z_mean, _, _, class_pred = encoder.predict(X_exp, batch_size=batch_size, verbose=0)

    mse = np.zeros((len(X), num_classes), dtype=np.float64)
    for k in range(num_classes):
        label = tf.keras.utils.to_categorical(
            [k] * len(X), num_classes=num_classes).astype('float32')
        rec = decoder.predict([z_mean, label], batch_size=batch_size, verbose=0)
        if rec.ndim == 3:                       # (n, n_features, 1) -> (n, n_features)
            rec = rec.reshape(rec.shape[0], -1)
        mse[:, k] = np.mean((X - rec) ** 2, axis=1)

    return {
        'reconstruction': mse_to_posterior(mse, temperature=temperature),
        'encoder_head': class_probs_to_posterior(class_pred),
    }
