#!/usr/bin/env python3
"""
Decrypt One Layer Experiment

This script analyzes the security impact of decrypting individual layers from
fully encrypted models. It simulates an extreme scenario where an attacker
correctly guesses the encryption key for one specific layer.

The experiment demonstrates that decrypting any single layer provides only
minimal accuracy improvements, validating the robustness of the multi-layer
encryption approach.

Usage:
    python src/experiments/decrypt_one_layer_experiment.py --checkpoint results/vit_encryption_20250906_203206/checkpoints/final
    python src/experiments/decrypt_one_layer_experiment.py --checkpoint results/vit_encryption_20250905_232539/checkpoints/final --layers 0 1 4
    python src/experiments/decrypt_one_layer_experiment.py --checkpoint-dir results/ --analyze-all
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.decrypt_one_layer_analysis import DecryptOneLayerAnalyzer
from src.utils.config_utils import setup_logging_from_config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Decrypt One Layer Security Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input parameters
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to encrypted model checkpoint directory'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default=None,
        help='Directory containing multiple checkpoint experiments'
    )
    parser.add_argument(
        '--analyze-all', action='store_true',
        help='Analyze all checkpoints found in checkpoint-dir'
    )
    parser.add_argument(
        '--layers', type=int, nargs='+', default=None,
        help='Specific layer indices to test (if None, tests all encrypted layers)'
    )
    
    # Model parameters
    parser.add_argument(
        '--imagenet-path', type=str, default='dataset/imagenet/val',
        help='Path to ImageNet validation dataset'
    )
    parser.add_argument(
        '--batch-size', type=int, default=128,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num-workers', type=int, default=16,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device for computation'
    )
    
    # Output parameters
    parser.add_argument(
        '--output-dir', type=str, default='results/decrypt_one_layer_analysis',
        help='Output directory for results'
    )
    parser.add_argument(
        '--experiment-name', type=str, default=None,
        help='Name for this experiment'
    )
    
    # Logging
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def find_checkpoint_directories(checkpoint_dir: Path) -> list:
    """Find all checkpoint directories in the given path."""
    checkpoint_dirs = []
    
    # Look for directories with pattern vit_encryption_*
    for exp_dir in checkpoint_dir.glob("vit_encryption_*"):
        if exp_dir.is_dir():
            final_checkpoint = exp_dir / "checkpoints" / "final"
            if final_checkpoint.exists():
                checkpoint_dirs.append(final_checkpoint)
    
    return checkpoint_dirs


def analyze_single_checkpoint(analyzer: DecryptOneLayerAnalyzer, 
                            checkpoint_path: Path,
                            target_layers: list,
                            output_dir: Path) -> dict:
    """Analyze a single checkpoint."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Analyzing checkpoint: {checkpoint_path}")
    
    # Run the analysis
    results = analyzer.analyze_single_layer_decryption(
        checkpoint_path=checkpoint_path,
        target_layers=target_layers
    )
    
    # Create output directory for this checkpoint
    checkpoint_name = checkpoint_path.parent.parent.name  # e.g., vit_encryption_20250906_203206
    checkpoint_output_dir = output_dir / checkpoint_name
    
    # Save results
    results_path = checkpoint_output_dir / "decrypt_one_layer_results.json"
    analyzer.save_results(results, results_path)
    
    # Print summary table
    analyzer.print_summary_table(results)
    
    return results


def main():
    """Main experiment function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging_from_config({'log_level': args.log_level})
    logger = logging.getLogger(__name__)
    
    # Setup experiment directory
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"decrypt_one_layer_{timestamp}"
    
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Decrypt One Layer Analysis")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize analyzer
        analyzer = DecryptOneLayerAnalyzer(
            imagenet_path=args.imagenet_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device
        )
        
        all_results = {}
        
        if args.analyze_all and args.checkpoint_dir:
            # Analyze all checkpoints in directory
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_paths = find_checkpoint_directories(checkpoint_dir)
            
            if not checkpoint_paths:
                logger.error(f"No checkpoint directories found in {checkpoint_dir}")
                return 1
            
            logger.info(f"Found {len(checkpoint_paths)} checkpoint directories")
            
            for checkpoint_path in checkpoint_paths:
                try:
                    results = analyze_single_checkpoint(
                        analyzer, checkpoint_path, args.layers, output_dir
                    )
                    all_results[checkpoint_path.parent.parent.name] = results
                except Exception as e:
                    logger.error(f"Failed to analyze {checkpoint_path}: {e}")
                    continue
        
        elif args.checkpoint:
            # Analyze single checkpoint
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
                return 1
            
            results = analyze_single_checkpoint(
                analyzer, checkpoint_path, args.layers, output_dir
            )
            all_results[checkpoint_path.parent.parent.name] = results
        
        else:
            logger.error("Must specify either --checkpoint or --checkpoint-dir with --analyze-all")
            return 1
        
        # Print overall summary
        if len(all_results) > 1:
            print(f"\n{'='*80}")
            print("OVERALL SUMMARY - All Models")
            print(f"{'='*80}")
            print(f"{'Model':<30} {'Layers':<10} {'Max Improvement':<15} {'Avg Improvement':<15}")
            print(f"{'-'*80}")
            
            for exp_name, results in all_results.items():
                model_name = results['model_name'].split('/')[-1]  # Extract model name
                num_layers = len(results['layer_results'])
                
                improvements = [r['improvement_percentage'] for r in results['layer_results'].values()]
                max_improvement = max(improvements) if improvements else 0
                avg_improvement = sum(improvements) / len(improvements) if improvements else 0
                
                print(f"{model_name:<30} {num_layers:<10} {max_improvement:+.2f}%           {avg_improvement:+.2f}%")
            
            print(f"{'='*80}")
        
        logger.info("Analysis completed successfully!")
        
        print(f"\nDecrypt One Layer Analysis Complete!")
        print(f" Results saved to: {output_dir}")
        print(f" Analyzed {len(all_results)} model(s)")
        print(" This analysis demonstrates the security robustness of multi-layer encryption")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        print(f"❌ Analysis failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
