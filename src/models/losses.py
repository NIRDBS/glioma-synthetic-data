import tensorflow as tf
from tensorflow.keras import losses

def compute_mse_loss(x, x_recon):
    """
    Computes Mean Squared Error between input and reconstruction.
    Sum over features (spectral bands), mean over batch.
    """
    # x shape: (batch, bands, 1) or (batch, bands)
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = mse(x, x_recon) 
    
    if len(loss.shape) == 2:
        loss = tf.reduce_sum(loss, axis=1)
        
    return tf.reduce_mean(loss)

def compute_cosine_loss(x, x_recon):
    """
    Computes Cosine Distance between input and reconstruction.
    Encourages directional alignment (useful for normalized spectra).
    Sum over features, mean over batch.
    """
    # Flatten spatial dimension for cosine similarity
    x_flat = tf.reshape(x, [tf.shape(x)[0], -1])
    x_recon_flat = tf.reshape(x_recon, [tf.shape(x_recon)[0], -1])
    
    # Normalize vectors
    x_norm = tf.nn.l2_normalize(x_flat, axis=1)
    x_recon_norm = tf.nn.l2_normalize(x_recon_flat, axis=1)
    
    # Cosine distance = 1 - cosine_similarity
    cos_sim = tf.reduce_sum(x_norm * x_recon_norm, axis=1)
    cos_loss = 1.0 - cos_sim
    
    return tf.reduce_mean(cos_loss)

def compute_combined_reconstruction_loss(x, x_recon, gamma=0.5, delta=0.5):
    """
    Computes combined reconstruction loss: gamma*MSE + delta*Cosine.
    Used for Suite 200 ablation study to isolate reconstruction loss effects.
    
    Args:
        x: Input spectra (batch, bands, 1)
        x_recon: Reconstructed spectra (batch, bands, 1)
        gamma: Weight for MSE loss (0-1)
        delta: Weight for Cosine loss (0-1)
    
    Returns:
        Combined reconstruction loss
    """
    mse_loss = compute_mse_loss(x, x_recon)
    cos_loss = compute_cosine_loss(x, x_recon)
    
    combined_loss = gamma * mse_loss + delta * cos_loss
    return combined_loss

def compute_reconstruction_loss(x, x_recon):
    """
    Computes Mean Squared Error between input and reconstruction.
    Legacy function for backward compatibility. Use compute_mse_loss() directly.
    """
    return compute_mse_loss(x, x_recon)

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