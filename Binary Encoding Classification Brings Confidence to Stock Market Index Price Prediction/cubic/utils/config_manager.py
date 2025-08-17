"""
Configuration Manager for CUBIC framework
"""

import yaml
import logging
from typing import Any, Dict, Optional
import os

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration for CUBIC framework
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize Configuration Manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
            else:
                logger.warning(f"Configuration file {self.config_path} not found. Using default config.")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'data': {
                'bloomberg': {'timeout': 30000, 'max_retries': 3},
                'technical_indicators': {'lookback_window': 5}
            },
            'model': {
                'binary_encoding': {'precision_bits': 15, 'value_range': [-1, 1]},
                'embedding': {'stock_embedding_dim': 32, 'hidden_dim': 128}
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 100
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'data.bloomberg.timeout')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Configuration key '{key}' not found. Using default: {default}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        logger.debug(f"Configuration key '{key}' set to {value}")
    
    def save(self, filepath: Optional[str] = None) -> None:
        """
        Save configuration to file
        
        Args:
            filepath: Path to save configuration (uses original path if None)
        """
        save_path = filepath or self.config_path
        
        try:
            with open(save_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of updates
        """
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        logger.info("Configuration updated")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
