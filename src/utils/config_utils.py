"""
Configuration Management Utilities

This module provides utilities for managing experiment configurations,
including loading, saving, validation, and merging of configuration files.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging


@dataclass
class ExperimentConfig:
    """Configuration for IP protection experiments."""
    
    # Model configuration
    model_name: str = "google/vit-base-patch16-224"
    local_model_path: Optional[str] = None
    device: str = "cuda"
    
    # Dataset configuration
    imagenet_path: str = "dataset/imagenet/val"
    batch_size: int = 128
    num_workers: int = 8
    
    # Encryption configuration
    arnold_key: Optional[list] = None  # Will use default if None
    password: Optional[str] = None
    num_permutation_matrices: int = 6
    num_extra_layers: int = 4
    
    # Experiment configuration
    output_dir: str = "results"
    save_checkpoints: bool = True
    log_level: str = "INFO"
    random_seed: Optional[int] = None
    
    # Evaluation configuration
    evaluate_initial: bool = True
    evaluate_each_step: bool = True
    generate_reports: bool = True


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is unsupported
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    suffix = config_path.suffix.lower()
    
    try:
        with open(config_path, 'r') as f:
            if suffix == '.json':
                return json.load(f)
            elif suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {suffix}")
    except Exception as e:
        raise ValueError(f"Error loading config file {config_path}: {str(e)}")


def save_config(config: Dict[str, Any], 
               config_path: Union[str, Path],
               format: str = "json") -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
        format: File format ('json' or 'yaml')
        
    Raises:
        ValueError: If format is unsupported
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    format = format.lower()
    
    try:
        with open(config_path, 'w') as f:
            if format == 'json':
                json.dump(config, f, indent=4)
            elif format == 'yaml':
                yaml.dump(config, f, default_flow_style=False, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        raise ValueError(f"Error saving config to {config_path}: {str(e)}")


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Dict containing merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    logger = logging.getLogger(__name__)
    
    # Required fields
    required_fields = ['model_name', 'imagenet_path', 'output_dir']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required configuration field missing: {field}")
    
    # Validate model name
    if not isinstance(config['model_name'], str) or not config['model_name']:
        raise ValueError("model_name must be a non-empty string")
    
    # Validate paths (with permission handling)
    if 'imagenet_path' in config:
        imagenet_path = Path(config['imagenet_path'])
        try:
            if not imagenet_path.exists():
                logger.warning(f"ImageNet path does not exist: {imagenet_path}")
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot access ImageNet path {imagenet_path}: {e}")
            logger.info("Path validation skipped due to permission issues")
    
    # Validate numeric parameters
    numeric_params = {
        'batch_size': (1, 1024),
        'num_workers': (0, 32),
        'num_permutation_matrices': (1, 20),
        'num_extra_layers': (0, 12)
    }
    
    for param, (min_val, max_val) in numeric_params.items():
        if param in config:
            value = config[param]
            if not isinstance(value, int) or value < min_val or value > max_val:
                raise ValueError(f"{param} must be an integer between {min_val} and {max_val}")
    
    # Validate Arnold key if provided
    if 'arnold_key' in config and config['arnold_key'] is not None:
        arnold_key = config['arnold_key']
        if not isinstance(arnold_key, list) or len(arnold_key) != 5:
            raise ValueError("arnold_key must be a list of 5 integers")
        if not all(isinstance(x, int) for x in arnold_key):
            raise ValueError("arnold_key must contain only integers")
    
    # Validate device
    if 'device' in config:
        device = config['device']
        if device not in ['cuda', 'cpu', 'auto']:
            raise ValueError("device must be 'cuda', 'cpu', or 'auto'")
    
    return True


def create_default_config() -> ExperimentConfig:
    """
    Create a default experiment configuration.
    
    Returns:
        ExperimentConfig: Default configuration object
    """
    return ExperimentConfig()


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Convert ExperimentConfig object to dictionary.
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        Dict representation of the configuration
    """
    return asdict(config)


def dict_to_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """
    Convert dictionary to ExperimentConfig object.
    
    Args:
        config_dict: Dictionary containing configuration parameters
        
    Returns:
        ExperimentConfig: Configuration object
    """
    # Filter out unknown fields
    valid_fields = set(ExperimentConfig.__dataclass_fields__.keys())
    filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
    
    return ExperimentConfig(**filtered_dict)


def setup_logging_from_config(config: Dict[str, Any]) -> None:
    """
    Setup logging based on configuration.
    
    Args:
        config: Configuration dictionary containing log_level
    """
    log_level = config.get('log_level', 'INFO').upper()
    
    # Convert string to logging level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map.get(log_level, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def save_experiment_config(config: ExperimentConfig,
                          experiment_dir: Path) -> None:
    """
    Save experiment configuration to the experiment directory.
    
    Args:
        config: Experiment configuration to save
        experiment_dir: Directory where experiment results are stored
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    config_dict = config_to_dict(config)
    config_path = experiment_dir / "experiment_config.json"
    
    save_config(config_dict, config_path, format="json")
    
    # Also save as YAML for human readability
    yaml_path = experiment_dir / "experiment_config.yaml"
    save_config(config_dict, yaml_path, format="yaml")
