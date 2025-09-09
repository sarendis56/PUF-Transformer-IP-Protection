#!/usr/bin/env python3
"""
Vision Transformer IP Protection Experiment

This script provides a clean, reproducible implementation of the ViT encryption
experiment, replacing the functionality previously in Ex2-ViT.ipynb.

Usage:
    python src/experiments/vit_encryption_experiment.py --config configs/vit_base.yaml
    python src/experiments/vit_encryption_experiment.py --model google/vit-base-patch16-224
"""

import argparse
import logging
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.vit_analyzer import VitEncryptionAnalyzer
from src.utils.config_utils import (
    load_config, save_config, validate_config, 
    create_default_config, config_to_dict, dict_to_config,
    setup_logging_from_config, save_experiment_config
)
from src.utils.metrics import generate_evaluation_report


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vision Transformer IP Protection Experiment",
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
        help='HuggingFace model identifier (default: google/vit-base-patch16-224)'
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
        '--imagenet-path', type=str, default=None,
        help='Path to ImageNet validation dataset'
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
        help='Number of additional security layers (default: 4)'
    )
    parser.add_argument(
        '--arnold-key', type=int, nargs=5, default=None,
        metavar=('N', 'a', 'b', 'c', 'd'),
        help='Arnold Cat Map key parameters [N, a, b, c, d]'
    )
    parser.add_argument(
        '--num-permutation-matrices', type=int, default=None,
        help='Number of permutation matrices to generate (default: 6)'
    )

    
    # Experiment parameters
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results (default: results)'
    )
    parser.add_argument(
        '--experiment-name', type=str, default=None,
        help='Name for this experiment (auto-generated if not provided)'
    )
    parser.add_argument(
        '--save-checkpoints', action='store_true', default=None,
        help='Save model checkpoints during encryption'
    )
    parser.add_argument(
        '--no-save-checkpoints', dest='save_checkpoints', action='store_false',
        help='Do not save model checkpoints'
    )
    parser.add_argument(
        '--random-seed', type=int, default=None,
        help='Random seed for reproducibility (default: 42)'
    )

    # Logging
    parser.add_argument(
        '--log-level', type=str, default=None,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress console output (log to file only)'
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
        config['model_name'] = 'google/vit-base-patch16-224'

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
            model_name = config.get('model_name', 'google/vit-base-patch16-224')

        # Extract model identifier from HuggingFace model name
        if '/' in model_name:
            model_identifier = model_name.split('/')[-1]  # e.g., 'vit-base-patch16-224'
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

    if args.imagenet_path is not None:
        config['imagenet_path'] = args.imagenet_path
    elif not base_config:
        config['imagenet_path'] = 'dataset/imagenet/val'

    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    elif not base_config:
        config['batch_size'] = 64

    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    elif not base_config:
        config['num_workers'] = 8

    if args.arnold_key is not None:
        config['arnold_key'] = args.arnold_key

    if args.num_permutation_matrices is not None:
        config['num_permutation_matrices'] = args.num_permutation_matrices
    elif not base_config:
        config['num_permutation_matrices'] = 6

    if args.num_extra_layers is not None:
        config['num_extra_layers'] = args.num_extra_layers
    elif not base_config:
        config['num_extra_layers'] = 4

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
    elif not base_config:
        config['random_seed'] = 42

    return config


def setup_experiment_directory(output_dir: str, experiment_name: str = None) -> Path:
    """Setup experiment directory with timestamp."""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"vit_encryption_{timestamp}"
    
    experiment_dir = Path(output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir


def main():
    """Main experiment function."""
    args = parse_arguments()
    
    # Load configuration
    if args.config:
        config_dict = load_config(args.config)
        # Merge with command line arguments (CLI takes precedence)
        # But only for arguments that were explicitly provided
        cli_config = create_experiment_config(args, config_dict)
        config_dict.update(cli_config)
    else:
        config_dict = create_experiment_config(args, None)
    
    # Validate configuration
    validate_config(config_dict)
    
    # Setup experiment directory
    experiment_dir = setup_experiment_directory(
        config_dict['output_dir'], 
        args.experiment_name
    )
    
    # Setup logging
    setup_logging_from_config(config_dict)
    logger = logging.getLogger(__name__)
    
    # Add file handler for experiment-specific logging
    log_file = experiment_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    
    if args.quiet:
        # Remove console handler if quiet mode
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
    
    logger.info("Starting Vision Transformer IP Protection Experiment")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Configuration: {config_dict}")
    
    # Set random seeds
    if config_dict.get('random_seed'):
        set_random_seeds(config_dict['random_seed'])
        logger.info(f"Random seed set to: {config_dict['random_seed']}")
    
    # Save experiment configuration
    config_obj = dict_to_config(config_dict)
    save_experiment_config(config_obj, experiment_dir)
    
    try:
        # Initialize analyzer
        logger.info("Initializing ViT Encryption Analyzer...")
        analyzer = VitEncryptionAnalyzer(
            model_name=config_dict['model_name'],
            batch_size=config_dict['batch_size'],
            num_workers=config_dict['num_workers'],
            num_extra_layers=config_dict['num_extra_layers'],
            password=None,
            arnold_key=config_dict.get('arnold_key'),
            device=config_dict['device'],
            imagenet_path=config_dict['imagenet_path'],
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
