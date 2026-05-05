"""
Stage 3: CVAE (Conditional Variational Autoencoder)
Trains generative model on tumor spectra and generates synthetic data
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from src.data.loader import GliomaDataLoader
from src.models.cvae import CVAE


def cvae_stage(config, output_dir):
    """
    Train CVAE on real tumor spectra and generate synthetic data.
    
    Args:
        config: Configuration dict
        output_dir: Output directory for synthetic data
    
    Returns:
        Dict with synthetic data path and generation metadata
    """
    
    print(f"\n{'='*80}")
    print(f"CVAE STAGE: Train Generative Model and Generate Synthetic Data")
    print(f"{'='*80}")
    
    # Load data
    data_path = Path(config['paths']['processed_file'])
    print(f"\nLoading data from: {data_path}")
    
    loader = GliomaDataLoader(
        data_path=data_path,
        memory_mapped=False,
        tumor_only=True,  # Only tumor spectra for generative training
        config=config
    )
    
    X = loader.X
    y = loader.y_class
    
    print(f"Dataset: {X.shape[0]} tumor spectra, {X.shape[1]} features")
    print(f"Classes distribution: {np.unique(y, return_counts=True)}")
    
    # CVAE config
    cvae_config = config['models'].get('cvae', {})
    latent_dim = cvae_config.get('latent_dim', 20)
    epochs = cvae_config.get('epochs', 50)
    batch_size = cvae_config.get('batch_size', 32)
    
    print(f"\nInitializing CVAE:")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Input dim: {X.shape[1]}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    # Train CVAE
    cvae = CVAE(
        input_dim=X.shape[1],
        latent_dim=latent_dim,
        config=config
    )
    
    print(f"\nTraining CVAE...")
    history = cvae.train(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    # Save encoder and decoder
    models_dir = Path(output_dir) / 'models' / 'cvae'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    cvae.encoder.save(models_dir / 'encoder.h5')
    cvae.decoder.save(models_dir / 'decoder.h5')
    print(f"\n✓ Models saved to {models_dir}")
    
    # Generate synthetic data
    print(f"\nGenerating synthetic data...")
    
    synthetic_dir = Path(output_dir) / 'synthetic'
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    synthetic_data = {}
    n_classes = len(np.unique(y))
    samples_per_class = cvae_config.get('samples_per_class', 200)
    
    print(f"  Classes: {n_classes}")
    print(f"  Samples per class: {samples_per_class}")
    
    for class_id in range(n_classes):
        # Generate samples
        synthetic_spectra = cvae.generate(
            n_samples=samples_per_class,
            condition=class_id
        )
        synthetic_data[f'class_{class_id}'] = synthetic_spectra
        print(f"    Class {class_id}: {synthetic_spectra.shape[0]} samples")
    
    # Save synthetic data
    synthetic_path = synthetic_dir / 'synthetic_spectra.npz'
    np.savez(synthetic_path, **synthetic_data)
    print(f"\n✓ Synthetic data saved to {synthetic_path}")
    
    # Return results
    cvae_results = {
        'synthetic_path': str(synthetic_path),
        'encoder_path': str(models_dir / 'encoder.h5'),
        'decoder_path': str(models_dir / 'decoder.h5'),
        'latent_dim': latent_dim,
        'n_classes': n_classes,
        'samples_per_class': samples_per_class,
        'history': {
            'loss': history.history.get('loss', []),
            'val_loss': history.history.get('val_loss', [])
        },
        'config': config
    }
    
    return cvae_results
