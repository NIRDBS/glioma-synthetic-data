# src/models/cvae.py
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, Input, metrics
from .losses import compute_reconstruction_loss, compute_kl_loss, compute_classification_loss

class CVAE(Model):
    def __init__(self, config):
        """
        Dynamic CVAE that builds itself based on a config dictionary.
        Supports 'use_skip_gap' to toggle multi-scale feature aggregation.
        """
        super(CVAE, self).__init__()
        # Handle both ExperimentConfig object or raw dict
        self.cfg = config.params if hasattr(config, 'params') else config
        
        self.input_shape_ = tuple(self.cfg['input_shape'])
        self.num_classes = self.cfg['num_classes']
        self.latent_dim = self.cfg['latent_dim']
        self.beta = self.cfg['loss_weights']['beta']
        self.alpha = self.cfg['loss_weights']['alpha']
        
        # Metrics
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.recon_loss_tracker = metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.class_loss_tracker = metrics.Mean(name="class_loss")
        self.accuracy_tracker = metrics.CategoricalAccuracy(name="accuracy")

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        x_in = Input(shape=self.input_shape_, name="encoder_input")
        enc_cfg = self.cfg['encoder']
        x = x_in
        
        pooled_features = []
        use_gap = enc_cfg.get('use_global_pooling', True) # Default to True (GAP)

        for i, (f, k, s) in enumerate(zip(enc_cfg['filters'], enc_cfg['kernels'], enc_cfg['strides'])):
            x = layers.Conv1D(f, k, strides=s, padding="same", activation="relu", name=f"enc_conv_{i}")(x)
            x = layers.BatchNormalization(name=f"enc_bn_{i}")(x)
            
            # Only collect intermediate features if we are using GAP
            if use_gap and enc_cfg.get('use_skip_gap', True):
                pooled_features.append(layers.GlobalAveragePooling1D()(x))
        
        # --- Aggregation Logic ---
        if use_gap:
            # OPTION A: Global Average Pooling (Shift Invariant)
            if enc_cfg.get('use_skip_gap', True):
                features = layers.Concatenate(name="multi_scale_concat")(pooled_features)
            else:
                features = layers.GlobalAveragePooling1D()(x)
        else:
            # OPTION B: Flatten (Position Aware) [Your proposed fix]
            # We flatten the FINAL feature map only (standard CNN approach)
            # This preserves spatial indices (e.g., peak at index 5 vs index 50)
            features = layers.Flatten(name="spatial_flatten")(x)
        
        # Latent Space (Z)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(features)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(features)
        
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
        
        # Proxy Classifier
        class_pred = layers.Dense(self.num_classes, activation="softmax", name="class_output")(features)
        
        return Model(x_in, [z_mean, z_log_var, z, class_pred], name="Encoder")

    def _build_decoder(self):
        z_in = Input(shape=(self.latent_dim,), name="z_input")
        label_in = Input(shape=(self.num_classes,), name="label_input")
        
        combined = layers.Concatenate()([z_in, label_in])
        
        # Dynamic Size Calculation
        total_stride = np.prod(self.cfg['encoder']['strides'])
        start_len = int(np.ceil(self.input_shape_[0] / total_stride))
        start_filters = self.cfg['decoder']['initial_filters']
        
        dec_cfg = self.cfg['decoder']
        
        # --- TOGGLE: Dense vs. Spatial Broadcast ---
        if dec_cfg.get('use_dense_projection', True):
            # OPTION A: Dense Layer
            # Good for absolute peak positions. High parameter count.
            x = layers.Dense(start_len * start_filters, activation="relu")(combined)
            x = layers.Reshape((start_len, start_filters))(x)
        else:
            # OPTION B: Spatial Broadcast (Lightweight)
            # Good for parameter efficiency. Relies on padding for position.
            x = layers.RepeatVector(start_len)(combined)
            x = layers.Conv1D(start_filters, kernel_size=1, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
        
        for i, (f, k, s) in enumerate(zip(dec_cfg['filters'], dec_cfg['kernels'], dec_cfg['strides'])):
            x = layers.Conv1DTranspose(f, k, strides=s, padding="same", activation="relu", name=f"dec_convT_{i}")(x)
            x = layers.BatchNormalization(name=f"dec_bn_{i}")(x)
        
        x = layers.Conv1D(1, 7, padding="same", activation="sigmoid", name="recon_out")(x)
        
        # Crop/Pad to match input exactly
        current_len = x.shape[1]
        target_len = self.input_shape_[0]
        if current_len > target_len:
            x = layers.Cropping1D(cropping=(0, current_len - target_len))(x)
        elif current_len < target_len:
            x = layers.ZeroPadding1D(padding=(0, target_len - current_len))(x)
            
        return Model([z_in, label_in], x, name="Decoder")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, 
                self.kl_loss_tracker, self.class_loss_tracker, self.accuracy_tracker]

    def call(self, inputs):
        x, label = inputs
        z_mean, z_log_var, z, class_pred = self.encoder(x)
        reconstruction = self.decoder([z, label])
        return reconstruction, class_pred, z_mean, z_log_var

    def train_step(self, data):
        if isinstance(data, tuple): x, y = data
        else: x, y = data, None

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, class_pred = self.encoder(x)
            reconstruction = self.decoder([z, y]) 
            
            recon_loss = compute_reconstruction_loss(x, reconstruction)
            kl_loss = compute_kl_loss(z_mean, z_log_var)
            class_loss = compute_classification_loss(y, class_pred)
            total_loss = recon_loss + (self.beta * kl_loss) + (self.alpha * class_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.accuracy_tracker.update_state(y, class_pred)
        
        return {m.name: m.result() for m in self.metrics}