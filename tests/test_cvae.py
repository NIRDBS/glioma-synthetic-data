import os
import shutil
import numpy as np
import tensorflow as tf
from src.utils.config import ExperimentConfig
from src.models.cvae import CVAE

def test_config():
    print("\n[1/3] Testing Configuration Manager...")
    test_dir = "tests/temp_config_test"
    
    try:
        # 1. Init
        cfg = ExperimentConfig(experiment_name="test_run", latent_dim=99)
        if cfg['latent_dim'] != 99:
            raise ValueError("Config override failed!")
        
        # 2. Save
        cfg.save(test_dir)
        expected_path = os.path.join(test_dir, "config.json")
        if not os.path.exists(expected_path):
            raise FileNotFoundError("Config file was not saved!")
        
        # 3. Load
        loaded_cfg = ExperimentConfig.load(expected_path)
        if loaded_cfg['experiment_name'] != "test_run":
            raise ValueError("Loaded config does not match saved config!")
            
        print("✅ Config Manager passed.")
        
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def test_model_build():
    print("\n[2/3] Testing Dynamic Model Building...")
    
    # Define a custom Deep Architecture to test dynamic loops
    config = ExperimentConfig(
        input_shape=[100, 1],
        encoder={
            "filters": [16, 32], 
            "kernels": [3, 3],
            "strides": [2, 2],
            "use_skip_gap": True
        },
        decoder={
            "initial_filters": 32,
            "filters": [32, 16],
            "kernels": [3, 3],
            "strides": [2, 2]
        }
    )
    
    try:
        model = CVAE(config)
        
        # Dummy Input (Batch size 1, 100 bands, 1 channel)
        x = tf.random.normal((1, 100, 1))
        y = tf.one_hot([0], depth=7)
        
        # Run inference to build the graph
        model([x, y])
        
        # Check output shapes
        if model.decoder.output_shape != (None, 100, 1):
            raise ValueError(f"Decoder output shape mismatch! Got {model.decoder.output_shape}")
        
        print("✅ Model built successfully with dynamic config.")
        
    except Exception as e:
        print(f"❌ Model build failed: {e}")
        raise e

def test_training_step():
    print("\n[3/3] Testing Training Step (Gradient Flow)...")
    
    input_shape = (1738, 1)
    batch_size = 4
    
    # Cast to Tensors explicitly to simulate model.fit() behavior
    X_dummy_np = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_dummy_np = np.eye(7)[np.random.choice(7, batch_size)].astype(np.float32)
    
    X_dummy = tf.convert_to_tensor(X_dummy_np)
    y_dummy = tf.convert_to_tensor(y_dummy_np)

    # Init Model
    config = ExperimentConfig() # Default config
    model = CVAE(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer)
    
    try:
        # Run one training step
        logs = model.train_step((X_dummy, y_dummy))
        
        # Check logs
        # FIX: Changed 'loss' to 'total_loss' to match the CVAE model definition
        required_keys = ['total_loss', 'recon_loss', 'kl_loss', 'class_loss']
        for k in required_keys:
            if k not in logs:
                raise ValueError(f"Missing metric '{k}' in training logs. Found: {list(logs.keys())}")
            if np.isnan(logs[k]):
                raise ValueError(f"Metric '{k}' is NaN! Check loss functions.")
                
        print(f"✅ Training step passed. Total Loss: {logs['total_loss']:.4f}")
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        raise e

if __name__ == "__main__":
    print("--- Starting Integration Tests ---")
    try:
        test_config()
        test_model_build()
        test_training_step()
        print("\n🎉 ALL TESTS PASSED. You are ready to train.")
    except Exception as e:
        print(f"\n🚫 TESTS FAILED. See error above.")