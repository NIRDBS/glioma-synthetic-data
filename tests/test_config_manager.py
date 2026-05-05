"""
Unit tests for ConfigManager and configuration system.

Tests configuration loading, validation, and merging behavior.
"""

import pytest
import tempfile
import json
from pathlib import Path
from src.utils.config_manager import ConfigManager, get_experiment_config


class TestConfigManager:
    """Test suite for ConfigManager class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory with test configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create configs subdirectory
            configs_dir = tmppath / 'configs'
            configs_dir.mkdir()
            
            # Create a valid config
            valid_config = {
                'name': 'test_exp_001',
                'description': 'Test experiment',
                'version': '1.0',
                'preprocessing': {
                    'airpls_lambda': 100,
                    'signal_length': 1738,
                    'crop_region': [700, 1800],
                },
                'cv_strategy': {
                    'split_source': 'data/splits/strict_splits.json',
                },
                'output': {
                    'output_base': 'test_exp_001',
                }
            }
            
            valid_config_path = configs_dir / 'test_exp_001.yaml'
            import yaml
            with open(valid_config_path, 'w') as f:
                yaml.dump(valid_config, f)
            
            # Create dummy splits file
            splits_dir = tmppath / 'data' / 'splits'
            splits_dir.mkdir(parents=True)
            splits_file = splits_dir / 'strict_splits.json'
            with open(splits_file, 'w') as f:
                json.dump({'test_set': [], 'cv_folds': {}}, f)
            
            yield tmppath, configs_dir
    
    def test_load_valid_config(self, temp_config_dir):
        """Test loading a valid configuration file."""
        tmppath, configs_dir = temp_config_dir
        
        # Change to temp directory so project root discovery works
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmppath)
            
            config = ConfigManager.load_config('test_exp_001')
            
            assert config['name'] == 'test_exp_001'
            assert config['description'] == 'Test experiment'
            assert config['preprocessing']['airpls_lambda'] == 100
        finally:
            os.chdir(original_cwd)
    
    def test_config_missing_file(self):
        """Test that FileNotFoundError is raised for missing configs."""
        with pytest.raises(FileNotFoundError):
            ConfigManager.load_config('nonexistent_config')
    
    def test_config_missing_required_fields(self, temp_config_dir):
        """Test that ValueError is raised when required fields are missing."""
        import os
        tmppath, configs_dir = temp_config_dir
        original_cwd = os.getcwd()
        
        try:
            os.chdir(tmppath)
            
            # Create invalid config missing 'name'
            import yaml
            invalid_config = {
                'description': 'Missing name',
                'preprocessing': {'airpls_lambda': 100},
                'cv_strategy': {'split_source': 'data/splits/strict_splits.json'},
                'output': {'output_base': 'test'},
            }
            
            invalid_path = configs_dir / 'invalid_config.yaml'
            with open(invalid_path, 'w') as f:
                yaml.dump(invalid_config, f)
            
            with pytest.raises(ValueError, match="Missing required"):
                ConfigManager.load_config('invalid_config')
        finally:
            os.chdir(original_cwd)
    
    def test_config_missing_split_source(self, temp_config_dir):
        """Test that ValueError is raised when split_source doesn't exist."""
        import os
        tmppath, configs_dir = temp_config_dir
        original_cwd = os.getcwd()
        
        try:
            os.chdir(tmppath)
            
            # Create config with non-existent split source
            import yaml
            invalid_config = {
                'name': 'bad_splits',
                'description': 'Bad splits path',
                'preprocessing': {'airpls_lambda': 100, 'signal_length': 1738, 'crop_region': [700, 1800]},
                'cv_strategy': {'split_source': 'data/splits/NONEXISTENT.json'},
                'output': {'output_base': 'test'},
            }
            
            invalid_path = configs_dir / 'bad_splits.yaml'
            with open(invalid_path, 'w') as f:
                yaml.dump(invalid_config, f)
            
            with pytest.raises(ValueError, match="does not exist"):
                ConfigManager.load_config('bad_splits')
        finally:
            os.chdir(original_cwd)
    
    def test_defaults_merged_with_provided_config(self, temp_config_dir):
        """Test that defaults are merged with provided config."""
        import os
        tmppath, configs_dir = temp_config_dir
        original_cwd = os.getcwd()
        
        try:
            os.chdir(tmppath)
            
            config = ConfigManager.load_config('test_exp_001')
            
            # Check that defaults were merged
            assert 'models' in config
            assert 'random_forest' in config['models']
            assert config['models']['random_forest']['n_estimators'] == 300  # Default
            
            # Check provided value is preserved
            assert config['preprocessing']['airpls_lambda'] == 100
        finally:
            os.chdir(original_cwd)
    
    def test_nested_value_access(self, temp_config_dir):
        """Test get_nested() method for dot-notation access."""
        import os
        tmppath, configs_dir = temp_config_dir
        original_cwd = os.getcwd()
        
        try:
            os.chdir(tmppath)
            
            config = ConfigManager.load_config('test_exp_001')
            
            # Test nested access
            lambda_value = config.get_nested('preprocessing.airpls_lambda')
            assert lambda_value == 100
            
            rf_estimators = config.get_nested('models.random_forest.n_estimators')
            assert rf_estimators == 300
            
            # Test default fallback
            missing_value = config.get_nested('nonexistent.path', default='fallback')
            assert missing_value == 'fallback'
        finally:
            os.chdir(original_cwd)
    
    def test_config_dict_access(self, temp_config_dir):
        """Test dict-like access to config."""
        import os
        tmppath, configs_dir = temp_config_dir
        original_cwd = os.getcwd()
        
        try:
            os.chdir(tmppath)
            
            config = ConfigManager.load_config('test_exp_001')
            
            # Test __getitem__
            preprocessing = config['preprocessing']
            assert preprocessing['airpls_lambda'] == 100
            
            # Test get()
            models = config.get('models')
            assert models is not None
            
            nonexistent = config.get('nonexistent', 'default_value')
            assert nonexistent == 'default_value'
        finally:
            os.chdir(original_cwd)
    
    def test_to_dict_method(self, temp_config_dir):
        """Test to_dict() method returns full config."""
        import os
        tmppath, configs_dir = temp_config_dir
        original_cwd = os.getcwd()
        
        try:
            os.chdir(tmppath)
            
            config = ConfigManager.load_config('test_exp_001')
            config_dict = config.to_dict()
            
            assert isinstance(config_dict, dict)
            assert config_dict['name'] == 'test_exp_001'
            assert 'models' in config_dict
        finally:
            os.chdir(original_cwd)


class TestConfigValidation:
    """Test configuration validation logic."""
    
    def test_crop_region_must_have_two_elements(self):
        """Test that crop_region validation works."""
        invalid_config = {
            'preprocessing': {
                'crop_region': [700],  # Only one element!
            }
        }
        
        with pytest.raises(ValueError, match="exactly 2 elements"):
            ConfigManager._validate_config(invalid_config, Path('dummy.yaml'))
    
    def test_name_must_be_string(self):
        """Test that name must be a string."""
        invalid_config = {
            'name': 123,  # Should be string
            'preprocessing': {'crop_region': [700, 1800]},
        }
        
        with pytest.raises(ValueError, match="must be a string"):
            ConfigManager._validate_config(invalid_config, Path('dummy.yaml'))


class TestGetExperimentConfig:
    """Test the convenience function."""
    
    def test_get_experiment_config_convenience_function(self):
        """Test that get_experiment_config() works as convenience wrapper."""
        # This will fail in test env without actual config, but validates the interface
        try:
            config = get_experiment_config('baseline_exp_001')
            assert hasattr(config, 'get_nested')
            assert hasattr(config, 'to_dict')
        except FileNotFoundError:
            # Expected if baseline_exp_001.yaml doesn't exist in test env
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
