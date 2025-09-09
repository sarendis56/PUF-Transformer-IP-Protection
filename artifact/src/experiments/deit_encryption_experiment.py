#!/usr/bin/env python3
"""
DeiT IP Protection Experiment for CIFAR-100

This script provides a clean, reproducible implementation of the DeiT encryption
experiment on CIFAR-100 dataset, adapted from the ViT experiment for ImageNet.

Usage:
    python src/experiments/deit_encryption_experiment.py --config configs/deit_tiny.yaml
    python src/experiments/deit_encryption_experiment.py --model facebook/deit-tiny-patch16-224
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.deit_analyzer import DeiTEncryptionAnalyzer
from src.utils.config_utils import load_config
from src.utils.metrics import generate_evaluation_report


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DeiT IP Protection Experiment for CIFAR-100",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration file (JSON or YAML)'
    )
    
    # Model parameters
    parser.add_argument(
        '--model', type=str, default=None,
        help='HuggingFace model identifier (default: facebook/deit-tiny-patch16-224)'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        choices=['cuda', 'cpu', 'auto'],
        help='Device for computation (default: cuda)'
    )
    parser.add_argument(
        '--local-model-path', type=str, default=None,
        help='Path to local model files'
    )

    # Dataset parameters
    parser.add_argument(
        '--cifar100-path', type=str, default=None,
        help='Path to CIFAR-100 dataset'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size for evaluation (default: 64)'
    )
    parser.add_argument(
        '--num-workers', type=int, default=None,
        help='Number of data loading workers (default: 8)'
    )

    # Encryption parameters
    parser.add_argument(
        '--num-extra-layers', type=int, default=None,
        help='Number of additional security layers (default: 2)'
    )
    parser.add_argument(
        '--arnold-key', type=int, nargs=5, default=None,
        metavar=('N', 'a', 'b', 'c', 'd'),
        help='Arnold Cat Map key parameters [N, a, b, c, d]'
    )
    parser.add_argument(
        '--password', type=str, default=None,
        help='Password for permutation matrix generation'
    )

    # Experiment parameters
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results (default: results)'
    )
    parser.add_argument(
        '--save-checkpoints', action='store_true',
        help='Save model checkpoints during encryption'
    )
    parser.add_argument(
        '--no-save-checkpoints', dest='save_checkpoints', action='store_false',
        help='Do not save model checkpoints'
    )
    parser.set_defaults(save_checkpoints=None)

    # Logging
    parser.add_argument(
        '--log-level', type=str, default=None,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--random-seed', type=int, default=None,
        help='Random seed for reproducibility'
    )

    return parser.parse_args()


def create_experiment_config(args: argparse.Namespace, base_config: dict = None) -> dict:
    """Create experiment configuration from command line arguments."""
    config = {}

    # Only add model_name if explicitly provided on command line
    if args.model is not None:
        config['model_name'] = args.model
    elif not base_config:
        # No base config and no model specified, use default
        config['model_name'] = 'facebook/deit-tiny-patch16-224'

    # Handle local model path - check for local model first, then use provided path
    if args.local_model_path:
        config['local_model_path'] = args.local_model_path
    else:
        # Determine which model name to use for path resolution
        if base_config and 'model_name' in base_config and 'model_name' not in config:
            # Use model name from config file
            model_name = base_config['model_name']
        else:
            # Use model name from args or config or default
            model_name = config.get('model_name', 'facebook/deit-tiny-patch16-224')

        # Extract model identifier from HuggingFace model name
        if '/' in model_name:
            model_identifier = model_name.split('/')[-1]  # e.g., 'deit-tiny-patch16-224'
        else:
            model_identifier = model_name

        local_model_path = Path(f'model/{model_identifier}')
        if local_model_path.exists() and local_model_path.is_dir():
            config['local_model_path'] = str(local_model_path)

    # Only add values that were explicitly provided on command line
    if args.device is not None:
        config['device'] = args.device
    elif not base_config:
        config['device'] = 'cuda'

    if args.cifar100_path is not None:
        config['cifar100_path'] = args.cifar100_path
    elif not base_config:
        config['cifar100_path'] = 'dataset/cifar100'

    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    elif not base_config:
        config['batch_size'] = 64

    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    elif not base_config:
        config['num_workers'] = 8

    if args.num_extra_layers is not None:
        config['num_extra_layers'] = args.num_extra_layers
    elif not base_config:
        config['num_extra_layers'] = 2

    if args.arnold_key is not None:
        config['arnold_key'] = args.arnold_key

    if args.password is not None:
        config['password'] = args.password

    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    elif not base_config:
        config['output_dir'] = 'results'

    if args.save_checkpoints is not None:
        config['save_checkpoints'] = args.save_checkpoints
    elif not base_config:
        config['save_checkpoints'] = True

    if args.log_level is not None:
        config['log_level'] = args.log_level
    elif not base_config:
        config['log_level'] = 'INFO'

    if args.random_seed is not None:
        config['random_seed'] = args.random_seed

    return config


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """Merge base configuration with override configuration."""
    merged = base_config.copy()
    merged.update({k: v for k, v in override_config.items() if v is not None})
    return merged


def setup_experiment_directory(output_dir: str) -> Path:
    """Setup experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(output_dir) / f"deit_encryption_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def main() -> int:
    """Main experiment function."""
    args = parse_arguments()
    
    # Load base configuration if provided
    base_config = {}
    if args.config:
        try:
            base_config = load_config(args.config)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return 1
    
    # Create experiment configuration
    override_config = create_experiment_config(args, base_config)
    config_dict = merge_configs(base_config, override_config)
    
    # Setup experiment directory
    experiment_dir = setup_experiment_directory(config_dict['output_dir'])
    
    # Setup logging with custom configuration to avoid duplication
    log_level = config_dict.get('log_level', 'INFO').upper()
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    level = level_map.get(log_level, logging.INFO)

    # Clear any existing handlers to avoid duplication
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Setup console handler with simple format (no timestamp)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    )
    root_logger.addHandler(console_handler)

    # Setup file handler with full format (with timestamp)
    log_file = experiment_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    root_logger.addHandler(file_handler)

    # Set logging level
    root_logger.setLevel(level)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. File: {log_file}")
    
    # Save experiment configuration
    config_path = experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info("=== DeiT IP Protection Experiment ===")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Configuration: {config_dict}")
    
    try:
        # Initialize analyzer
        logger.info("Initializing DeiT Encryption Analyzer...")
        analyzer = DeiTEncryptionAnalyzer(
            model_name=config_dict['model_name'],
            batch_size=config_dict['batch_size'],
            num_workers=config_dict['num_workers'],
            num_extra_layers=config_dict['num_extra_layers'],
            password=config_dict.get('password'),
            arnold_key=config_dict.get('arnold_key'),
            device=config_dict['device'],
            cifar100_path=config_dict['cifar100_path'],
            local_model_path=config_dict.get('local_model_path')
        )
        
        logger.info(f"Initial model accuracy: {analyzer.initial_accuracy:.2%}")
        
        # Run encryption experiment
        logger.info("Starting progressive model encryption...")
        performance_tracker = analyzer.encrypt_model_progressive(
            output_dir=experiment_dir / "checkpoints",
            save_checkpoints=config_dict['save_checkpoints']
        )
        
        # Save final results
        logger.info("Saving final results...")
        performance_tracker.save_to_file(experiment_dir / "final_performance.json")
        
        # Generate evaluation report
        report = generate_evaluation_report(
            performance_tracker,
            output_path=experiment_dir / "evaluation_report.txt"
        )
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Final accuracy: {performance_tracker.get_current_accuracy():.2%}")
        logger.info(f"Total accuracy drop: {performance_tracker.get_total_accuracy_drop():.2%}")
        logger.info(f"Encrypted layers: {len(performance_tracker.encryption_steps)}")
        
        print(f"\nExperiment completed! Results saved to: {experiment_dir}")
        print(f"Final accuracy: {performance_tracker.get_current_accuracy():.2%}")
        print(f"Total accuracy drop: {performance_tracker.get_total_accuracy_drop():.2%}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        print(f"Experiment failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
