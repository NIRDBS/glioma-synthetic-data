"""
Tier 3: gated-attention MIL (Ilse et al. 2018), over instance LOGITS rather
than pooled embeddings, on a frozen per-fold CVAE encoder trunk
(docs/MIL_REFRAME_PLAN.md §4, Stage 6).

Architecture, per bag (here: per *sub-bag*, see below):
  frozen encoder trunk  X (s, n_features) -> z_mean (s, latent_dim)
  instance head (trainable)      z_mean -> instance_logits (s, K)   -- a genuine,
      UNCONSTRAINED K-way logit per instance (never masked to the bag's own
      class -- see the module-level note below on why that matters)
  gated attention (trainable)    z_mean -> attention weights (s,), sum to 1
  bag score = sum_i( attention_i * instance_logits_i )              -- (K,)
  bag posterior = softmax(bag_score), trained against Y_b

"Attention over logits, not embeddings" is the plan's own instruction
(§4): pooling embeddings first and classifying the pooled result gives only
a bag score, with no interpretable per-instance posterior -- and Stage 7
needs one. Pooling logits keeps every instance's own K-way opinion directly
readable (`instance_logits`), with attention only deciding how much each
instance's opinion counts toward the bag decision.

**Why the instance head is never masked to the bag's own class (unlike
EM's Model (C) support restriction)**: Stage 6 found that a bag-classification
metric applied to a hard-masked posterior is circular -- it can only ever
recover the label it was given, not one it inferred
(docs/stage6_generative_mil.md §5). This tier's whole point is a genuinely
*learned* instance posterior, evaluable the same honest way Tier 4's is, so
the instance head stays a free K-way softmax throughout; only the
attention-pooled BAG score is supervised by Y_b (standard MIL attention
training), never a per-instance hard state constraint.

The (K+1)-way store posterior (non-diagnostic mass) uses the same
`1 - max_k p_k` convention as Tier 4/mi-SVM (src/mil/generative.py,
src/mil/misvm.py) for cross-tier comparability -- and inherits the exact
same K=2 boundedness caveat documented there
(docs/stage6_generative_mil.md §4): for K=2 it can never exceed 1/3.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


N_NULL_COMPONENTS = 3  # unused by this module directly, kept for reference --
                       # see the module docstring: the instance head is NOT
                       # restricted to Model (C)'s state space.


def required_sub_bag_size(pi_min, epsilon=0.01):
    """
    docs/MIL_REFRAME_PLAN.md §4.1: the smallest sub-bag size s such that
    P(a uniformly-drawn sub-bag from a bag at the pi_min floor contains zero
    diagnostic instances) <= epsilon, i.e. s >= ln(epsilon) / ln(1 - pi_min).
    At pi_min=0.02, epsilon=0.01 this gives s>=228 (the plan's own worked
    example) -- 256 is the calibrated default, not a round-number guess.
    """
    if not (0 < pi_min < 1):
        raise ValueError(f"pi_min must be in (0, 1), got {pi_min}")
    if not (0 < epsilon < 1):
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    return int(np.ceil(np.log(epsilon) / np.log(1 - pi_min)))


def sample_sub_bags(X, bag_ids, y_bag_per_instance, n_per_bag, sub_bag_size, seed=None):
    """
    Draw ``n_per_bag`` sub-bags per unique bag, each ``sub_bag_size``
    instances sampled UNIFORMLY AT RANDOM from that bag (never contiguous --
    §4.1: "Sample uniformly at random across the whole bag -- never
    contiguous spatial patches", which is also what keeps this sampler from
    ever touching ``coords`` at all, preserving the withheld-coordinates
    property Stage 5/6's validation depends on).

    Sampling is without replacement when the bag has >= sub_bag_size
    instances (true for every exp_501 bag -- smallest bag is 1,878
    instances, comfortably above the 256 default), with replacement
    otherwise (so a small bag still produces a full-size sub-bag rather than
    being silently skipped).

    Returns (X_sub, y_sub, bag_ids_sub): X_sub is
    (n_bags*n_per_bag, sub_bag_size, n_features); y_sub, bag_ids_sub are
    (n_bags*n_per_bag,) -- one label/bag-id per SUB-BAG, not per instance.
    """
    rng = np.random.default_rng(seed)
    bag_ids = np.asarray(bag_ids)
    y_bag_per_instance = np.asarray(y_bag_per_instance)

    X_parts, y_parts, id_parts = [], [], []
    for b in np.unique(bag_ids):
        idx = np.where(bag_ids == b)[0]
        y_b = y_bag_per_instance[idx[0]]
        replace = len(idx) < sub_bag_size
        for _ in range(n_per_bag):
            chosen = rng.choice(idx, size=sub_bag_size, replace=replace)
            X_parts.append(X[chosen])
            y_parts.append(y_b)
            id_parts.append(b)

    return (np.stack(X_parts).astype(np.float32),
            np.array(y_parts),
            np.array(id_parts, dtype=object))


def sub_bag_null_rate(X, bag_ids, y_bag_per_instance, instance_posterior, sub_bag_size,
                      n_per_bag=200, diagnostic_threshold=0.5, seed=None):
    """
    docs/MIL_REFRAME_PLAN.md §4.1's named diagnostic: the empirical fraction
    of sampled sub-bags that contain ZERO instances with own-class posterior
    mass above ``diagnostic_threshold``, using an already-fitted MIL
    formulation's instance posterior as the best available proxy for "which
    instances are diagnostic" (there is no ground truth to check against --
    see [[project_no_tumor_ground_truth]]). Compare the result against the
    epsilon `required_sub_bag_size` was calibrated to; materially above it
    means the sizing under-covers this bag's real diagnostic fraction and
    Tier 3's numbers from it should not be trusted at face value.

    ``instance_posterior``: (n, K+1) aligned to X/bag_ids/y_bag_per_instance,
    e.g. a Tier 1/2/4 store posterior's out-of-sample-merged rows, or this
    tier's own post-training instance_logits softmax (self-referential, but
    that is literally what the plan's "under the final model" means).
    """
    rng = np.random.default_rng(seed)
    bag_ids = np.asarray(bag_ids)
    y_bag_per_instance = np.asarray(y_bag_per_instance).astype(int)
    own_class_mass = instance_posterior[np.arange(len(bag_ids)), 1 + y_bag_per_instance]

    n_empty, n_total = 0, 0
    for b in np.unique(bag_ids):
        idx = np.where(bag_ids == b)[0]
        replace = len(idx) < sub_bag_size
        for _ in range(n_per_bag):
            chosen = rng.choice(idx, size=sub_bag_size, replace=replace)
            if (own_class_mass[chosen] < diagnostic_threshold).all():
                n_empty += 1
            n_total += 1
    return n_empty / n_total if n_total else float('nan')


class AttentionMIL(Model):
    """
    Trainable instance head + gated attention on top of a FROZEN encoder
    (passed in already built and weight-loaded -- this class never
    constructs or trains the trunk itself).
    """

    def __init__(self, encoder, latent_dim, num_classes, attn_dim=32,
                epsilon=0.0, dropout_rate=0.3, l2=1e-3, center_by_bag=True,
                name="AttentionMIL", **kwargs):
        """
        ``epsilon``: one-sided label-smoothing rate (docs/MIL_REFRAME_PLAN.md
        §4.1), computed by the caller from the actual sub-bag size via
        ``required_sub_bag_size``'s inverse, not guessed. For K=2, symmetric
        Keras label_smoothing=2*epsilon reproduces exactly the one-sided
        case (true class -> 1-epsilon, the other -> epsilon; there is only
        one "other" class to smooth toward), so no custom loss is needed --
        this only generalises cleanly to K=2 and should be revisited before
        reuse at 6-class.

        ``dropout_rate``/``l2``: real-data regularisation, added after a
        smoke test on exp_501 fold_0 showed this architecture overfitting to
        BAG IDENTITY rather than diagnostic content at the default (no
        regularisation, attn_dim=64) settings -- sub-bag training loss fell
        steadily while inner-validation bag accuracy on 7 HELD-OUT bags
        actively got WORSE (0.57 -> 0.29) as training continued, meaning the
        model was learning "which of these 38 bags is this" rather than
        anything that transfers to a new bag. attn_dim's default was also
        halved (64 -> 32) as part of the same fix -- fewer parameters,
        harder to memorise 38 bags' worth of identity.

        ``center_by_bag``: the more targeted half of that same fix. Every
        instance's z_mean is centred by the MEAN z_mean of its own
        (sub-)bag before the instance head and attention branch ever see it
        -- this removes exactly the "which bag is this" signal (a bag's own
        average acquisition/baseline fingerprint) while preserving relative
        WITHIN-bag differences, which is where diagnostic-vs-null contrast
        actually has to live under this project's own formulation (§1: a
        bag's instances differ from EACH OTHER, not from some absolute
        reference point). Applied identically at training (per sub-bag,
        ``call``) and inference (per whole bag, ``instance_posterior`` --
        which now needs ``bag_ids`` to know where one bag ends and the next
        begins).
        """
        super().__init__(name=name, **kwargs)
        self.encoder = encoder
        self.encoder.trainable = False
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        self.center_by_bag = center_by_bag

        reg = tf.keras.regularizers.l2(l2) if l2 else None
        # Each fold has its OWN independently-trained CVAE encoder, and they
        # do not produce z_mean on a consistent scale -- confirmed directly
        # on exp_501 (2026-09-28): fold_1's raw z_mean std was ~7.5x SMALLER
        # than fold_0's (0.011 vs 0.082), which alone was enough to make
        # fold_1's training stall completely (near-zero instance_logits,
        # near-uniform attention, loss frozen at ln(2) for 70+ epochs) at
        # weight-initialization scales tuned for a "normal" input variance.
        # Centering (below) does NOT fix this -- subtracting a per-bag mean
        # cannot change variance, only removes the mean -- so this
        # normalisation is the actual fix, applied regardless of
        # center_by_bag, making training robust to whatever scale a given
        # fold's frozen trunk happens to produce rather than assuming one.
        self.post_trunk_norm = layers.LayerNormalization(name="post_trunk_norm")
        self.pre_dropout = layers.Dropout(dropout_rate)
        self.instance_head = layers.Dense(num_classes, name="instance_logits",
                                          kernel_regularizer=reg)

        # Gated attention (Ilse et al. 2018, eq. 9): a_i = w^T (tanh(V h_i) * sigmoid(U h_i))
        self.attn_V = layers.Dense(attn_dim, activation="tanh", name="attn_V",
                                   kernel_regularizer=reg)
        self.attn_U = layers.Dense(attn_dim, activation="sigmoid", name="attn_U",
                                   kernel_regularizer=reg)
        self.attn_dropout = layers.Dropout(dropout_rate)
        self.attn_w = layers.Dense(1, name="attn_w", kernel_regularizer=reg)

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, label_smoothing=2.0 * epsilon if num_classes == 2 else epsilon)
        self.acc_tracker = tf.keras.metrics.CategoricalAccuracy(name="bag_accuracy")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def call(self, x_sub, training=False):
        """
        x_sub: (batch, sub_bag_size, n_features). Returns
        (bag_probs (batch, K), instance_logits (batch, sub_bag_size, K),
        attn (batch, sub_bag_size)).
        """
        batch = tf.shape(x_sub)[0]
        s = tf.shape(x_sub)[1]
        n_features = x_sub.shape[-1]

        flat = tf.reshape(x_sub, (batch * s, n_features, 1))
        z_mean = self.encoder(flat, training=False)[0]           # (batch*s, latent_dim)

        if self.center_by_bag:
            # (batch*s, latent_dim) -> (batch, s, latent_dim) and back, with
            # latent_dim passed EXPLICITLY (not inferred via -1): a dynamic
            # batch*s combined with an inferred trailing dim loses the
            # static latent_dim Keras needs to build instance_head's kernel
            # the first time this runs inside model.fit()'s traced graph
            # (confirmed by a real failure here, not a hypothetical).
            z_mean_3d = tf.reshape(z_mean, (batch, s, self.latent_dim))
            z_mean_3d = z_mean_3d - tf.reduce_mean(z_mean_3d, axis=1, keepdims=True)
            z_mean = tf.reshape(z_mean_3d, (batch * s, self.latent_dim))

        z_mean = self.post_trunk_norm(z_mean)
        z_mean = self.pre_dropout(z_mean, training=training)

        instance_logits_flat = self.instance_head(z_mean)         # (batch*s, K)
        instance_logits = tf.reshape(instance_logits_flat, (batch, s, self.num_classes))

        attn_feat = self.attn_dropout(
            self.attn_V(z_mean) * self.attn_U(z_mean), training=training)
        a = self.attn_w(attn_feat)                                 # (batch*s, 1)
        a = tf.reshape(a, (batch, s))
        attn = tf.nn.softmax(a, axis=1)                            # sums to 1 per bag

        bag_logits = tf.reduce_sum(attn[:, :, None] * instance_logits, axis=1)  # (batch, K)
        bag_probs = tf.nn.softmax(bag_logits, axis=1)
        return bag_probs, instance_logits, attn

    def train_step(self, data):
        x_sub, y_bag_onehot = data
        with tf.GradientTape() as tape:
            bag_probs, _, _ = self(x_sub, training=True)
            loss = self.loss_fn(y_bag_onehot, bag_probs)
            if self.losses:  # accumulated kernel_regularizer penalties
                loss = loss + tf.add_n(self.losses)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y_bag_onehot, bag_probs)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x_sub, y_bag_onehot = data
        bag_probs, _, _ = self(x_sub, training=False)
        loss = self.loss_fn(y_bag_onehot, bag_probs)
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y_bag_onehot, bag_probs)
        return {m.name: m.result() for m in self.metrics}

    def _encode_and_normalize(self, X, bag_ids, batch_size=256):
        """
        Shared by ``instance_posterior`` and ``bag_posterior``: frozen
        encoder -> per-(whole-)bag centering (if enabled) -> post_trunk_norm.
        Returns (n, latent_dim). ``bag_ids`` required whenever
        ``center_by_bag=True`` -- centering here is computed per WHOLE bag,
        the natural inference-time analogue of ``call``'s per-SUB-bag
        centering (same operation, subtract this bag's own mean, at a
        different granularity, since inference sees a bag's real instance
        count, not a fixed sub-bag size).
        """
        n_features = X.shape[1]
        X_exp = X.reshape(-1, n_features, 1)
        z_mean = self.encoder.predict(X_exp, batch_size=batch_size, verbose=0)[0]

        if self.center_by_bag:
            bag_ids = np.asarray(bag_ids)
            z_mean = z_mean.copy()
            for b in np.unique(bag_ids):
                idx = np.where(bag_ids == b)[0]
                z_mean[idx] -= z_mean[idx].mean(axis=0, keepdims=True)

        return self.post_trunk_norm(z_mean).numpy()

    def instance_posterior(self, X, bag_ids, batch_size=256):
        """
        Score arbitrary (whole-bag, not sub-bag) instances through the
        frozen encoder + trainable instance head only (no attention --
        deliberately: this gives one value PER INSTANCE, independent of
        which other instances happen to be scored alongside it, which is
        what Stage 5's spatial validation needs to reproject a per-pixel
        map -- attention weights are relative to a specific bag's instance
        SET and are not a well-defined per-instance quantity outside one)
        -- softmax over K, then lifted to (n, K+1) via the same 1-max_p
        convention as Tier 4 (src/mil/generative.py). This is what gets
        written to the latent store per fold, for spatial validation.

        For BAG-LEVEL classification (accuracy, the plan's "beats Tier 0"
        question), use ``bag_posterior`` instead -- see its docstring for
        why this attention-free posterior is a poor fit for that specific
        question, confirmed empirically, not just argued (docs/
        MIL_REFRAME_PLAN.md backlog §13).
        """
        from src.mil.generative import class_probs_to_posterior

        z_mean = self._encode_and_normalize(X, bag_ids, batch_size)
        logits = self.instance_head(z_mean).numpy()
        probs = tf.nn.softmax(logits, axis=1).numpy()
        return class_probs_to_posterior(probs)

    def bag_posterior(self, X, bag_ids, batch_size=256):
        """
        Attention-WEIGHTED bag-level posterior -- the aggregate this model
        is actually TRAINED to optimize (``call``'s ``bag_logits``), unlike
        ``instance_posterior``'s naive per-instance value. Runs the same
        gated-attention mechanism training uses, over each bag's FULL
        instance set (not a fixed-size sub-bag).

        Why this matters, confirmed on real exp_501 data (2026-09-28): an
        instance attention learns to downweight during training can develop
        an ARBITRARY individual prediction, since its contribution to the
        attention-pooled training loss is near zero regardless of what it
        predicts -- nothing in the objective ever shapes it. With this
        project's own pi_b findings (mostly a small minority of any bag is
        genuinely diagnostic), a naive unweighted pool over thousands of
        such instances is dominated by exactly the instances the model
        never learned to get right individually. Two real smoke tests
        (fold_0, fold_1) showed sub-bag training loss falling to a strong,
        well-behaved minimum while naive-pooled bag accuracy sat at or
        below chance on the SAME trained model -- this method is the fix,
        not a hypothetical.

        Returns (bag_ids_sorted, posterior (n_bags, K+1)) -- one row per
        unique bag in ``bag_ids``, K+1-lifted the same way.
        """
        from src.mil.generative import class_probs_to_posterior

        bag_ids = np.asarray(bag_ids)
        z_mean = self._encode_and_normalize(X, bag_ids, batch_size)
        instance_logits = self.instance_head(z_mean).numpy()             # (n, K)
        attn_feat = (self.attn_V(z_mean) * self.attn_U(z_mean)).numpy()
        raw_attn = self.attn_w(attn_feat).numpy()[:, 0]                  # (n,)

        bags_sorted = sorted(set(bag_ids.tolist()))
        class_probs = np.zeros((len(bags_sorted), self.num_classes))
        for i, b in enumerate(bags_sorted):
            idx = np.where(bag_ids == b)[0]
            a = raw_attn[idx] - raw_attn[idx].max()   # numerical stability
            w = np.exp(a)
            w /= w.sum()
            bag_logits = (w[:, None] * instance_logits[idx]).sum(axis=0)
            class_probs[i] = tf.nn.softmax(bag_logits).numpy()

        return np.array(bags_sorted, dtype=object), class_probs_to_posterior(class_probs)
