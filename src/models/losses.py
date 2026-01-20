import tensorflow as tf
from tensorflow.keras import losses

def compute_reconstruction_loss(x, x_recon):
    """
    Computes Mean Squared Error between input and reconstruction.
    Sum over features (spectral bands), mean over batch.
    """
    # x shape: (batch, bands, 1) or (batch, bands)
    # Sum across the spectral dimension (axis 1)
    # We use MSE because the spectra are continuous signals.
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = mse(x, x_recon) 
    
    # If input is 3D (batch, bands, 1), MSE reduces to (batch, bands). 
    # We need to sum over bands to get total error per spectrum.
    if len(loss.shape) == 2:
        loss = tf.reduce_sum(loss, axis=1)
        
    return tf.reduce_mean(loss)

def compute_kl_loss(z_mean, z_log_var):
    """
    Computes Kullback-Leibler Divergence between the learned latent distribution
    and a standard Normal distribution N(0, 1).
    Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    # Sum over latent dimensions, mean over batch
    kl_loss = tf.reduce_sum(kl_loss, axis=1)
    return tf.reduce_mean(kl_loss)

def compute_classification_loss(y_true, y_pred):
    """
    Computes Categorical Cross Entropy for the Proxy Classifier.
    """
    # y_true: One-hot encoded labels
    return tf.reduce_mean(
        losses.categorical_crossentropy(y_true, y_pred)
    )