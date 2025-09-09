#!/usr/bin/env python3
"""
Dual Encryption Security Analysis Experiment

This script analyzes the security of dual encryption against key guessing attacks.
It tests whether an attacker who knows one key type can achieve higher accuracy
than with both keys unknown, validating the security of the dual encryption approach.

Usage:
    python src/experiments/dual_encryption_analysis_experiment.py --layers 0 1 2 --num-attack-variants 20
    python src/experiments/dual_encryption_analysis_experiment.py --config configs/analysis_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.dual_encryption_analysis import DualEncryptionAnalyzer
from src.analysis.visualization import AnalysisVisualizer
from src.utils.config_utils import load_config, setup_logging_from_config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dual Encryption Analysis Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration file'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--layers', type=int, nargs='+', default=[0, 1, 2, 3],
        help='Layer indices to analyze'
    )
    parser.add_argument(
        '--num-attack-variants', type=int, default=50,
        help='Number of key variants to test in attack simulation'
    )
    
    # Model parameters
    parser.add_argument(
        '--model', type=str, default='google/vit-base-patch16-224',
        help='HuggingFace model identifier'
    )
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
        '--output-dir', type=str, default='results/dual_encryption_analysis',
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


def main():
    """Main experiment function."""
    args = parse_arguments()
    
    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override with command line arguments
    if args.model:
        config['model_name'] = args.model
    if args.imagenet_path:
        config['imagenet_path'] = args.imagenet_path
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.num_workers:
        config['num_workers'] = args.num_workers
    if args.device:
        config['device'] = args.device
    
    # Setup logging
    setup_logging_from_config({'log_level': args.log_level})
    logger = logging.getLogger(__name__)
    
    # Setup experiment directory
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"dual_encryption_security_{timestamp}"
    
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Dual Encryption Security Analysis")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Layers: {args.layers}")
    logger.info(f"Attack variants: {args.num_attack_variants}")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Initialize analyzer
        analyzer = DualEncryptionAnalyzer(
            model_name=config.get('model_name', args.model),
            imagenet_path=config.get('imagenet_path', args.imagenet_path),
            batch_size=config.get('batch_size', args.batch_size),
            num_workers=config.get('num_workers', args.num_workers),
            device=config.get('device', args.device)
        )

        # Run security analysis
        logger.info("Running dual encryption security analysis...")
        results = analyzer.analyze_dual_encryption_security(
            layer_indices=args.layers,
            num_attack_variants=args.num_attack_variants
        )

        # Print security analysis summary
        logger.info("Security Analysis Results:")
        for layer_idx, layer_data in results['layer_results'].items():
            security_results = layer_data['security_results']

            # Calculate attack statistics
            acm_attack_accuracies = [r['accuracy'] for r in security_results['acm_attack_results']]
            perm_attack_accuracies = [r['accuracy'] for r in security_results['permutation_attack_results']]

            logger.info(f"Layer {layer_idx}:")
            logger.info(f"  Original accuracy: {security_results['original_accuracy']:.2%}")
            logger.info(f"  Dual encrypted accuracy: {security_results['dual_encrypted_accuracy']:.2%}")
            logger.info(f"  ACM attack - Mean: {sum(acm_attack_accuracies)/len(acm_attack_accuracies):.2%}, Max: {max(acm_attack_accuracies):.2%}")
            logger.info(f"  Permutation attack - Mean: {sum(perm_attack_accuracies)/len(perm_attack_accuracies):.2%}, Max: {max(perm_attack_accuracies):.2%}")
        
        # Add experiment metadata
        results.update({
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'layers': args.layers,
                'num_attack_variants': args.num_attack_variants
            }
        })
        
        # Generate visualizations and report (includes saving results as JSON)
        logger.info("Generating visualizations and report...")
        visualizer = AnalysisVisualizer()
        visualizer.create_comprehensive_report(
            results=results,
            output_dir=output_dir,
            report_name="dual_encryption_analysis"
        )
        
        logger.info("Security analysis completed successfully!")

        print(f"\nDual Encryption Security Analysis Complete!")
        print(f" Results saved to: {output_dir}")
        print(f" Analyzed {len(args.layers)} layers with {args.num_attack_variants} attack variants each")

        # Print overall security summary
        all_acm_attacks = []
        all_perm_attacks = []
        for layer_idx, layer_data in results['layer_results'].items():
            security_results = layer_data['security_results']
            all_acm_attacks.extend([r['accuracy'] for r in security_results['acm_attack_results']])
            all_perm_attacks.extend([r['accuracy'] for r in security_results['permutation_attack_results']])

        print(f" Overall attack success rates:")
        print(f"   ACM key guessing: {max(all_acm_attacks):.2%} max accuracy")
        print(f"   Permutation key guessing: {max(all_perm_attacks):.2%} max accuracy")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        print(f"❌ Analysis failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
