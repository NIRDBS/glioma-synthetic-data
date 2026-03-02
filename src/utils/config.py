# src/utils/config.py
import json
import os

class ExperimentConfig:
    def __init__(self, **kwargs):
        # Default Configuration (The "Baseline")
        self.params = {
            "experiment_name": "default",
            "input_shape": [1738, 1], # List for JSON compatibility
            "num_classes": 7,
            "latent_dim": 16,
            "batch_size": 128,
            "epochs": 50,
            "learning_rate": 0.001,
            
            # Encoder Architecture
            "encoder": {
                "filters": [32, 64, 128],
                "kernels": [7, 7, 5],
                "strides": [2, 2, 2],
                "use_skip_gap": True,       # Used only if use_global_pooling=True
                "use_global_pooling": True  # (Toggle for GAP vs Flatten)
            },
            
            # Decoder Architecture
            "decoder": {
                "initial_filters": 128,
                "filters": [64, 32, 16],
                "kernels": [5, 7, 7],
                "strides": [2, 2, 2],
                "use_dense_projection": True  # (Default to True for better reconstructions)
            },
            
            # Loss Weights
            "loss_weights": {
                "beta": 1.0,  # KL Weight
                "alpha": 1.0  # Classification Weight
            }
        }
        # Update defaults with user overrides
        self.update(kwargs)

    def update(self, new_params):
        """Recursively updates the config dictionary."""
        for key, value in new_params.items():
            if isinstance(value, dict) and key in self.params:
                self.params[key].update(value)
            else:
                self.params[key] = value

    def save(self, output_dir):
        """Saves the config to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(self.params, f, indent=4)
        print(f"Config saved to {path}")

    @staticmethod
    def load(path):
        """Loads a config from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return ExperimentConfig(**data)

    def __getitem__(self, item):
        return self.params[item]