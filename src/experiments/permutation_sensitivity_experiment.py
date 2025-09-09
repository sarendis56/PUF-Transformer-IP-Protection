#!/usr/bin/env python3
"""
Permutation Sensitivity Experiment

This script runs sensitivity analysis for permutation matrices,
analyzing how different permutation matrices affect model performance.

Usage:
    python src/experiments/permutation_sensitivity_experiment.py --layers 0 --num-permutations 20
    python src/experiments/permutation_sensitivity_experiment.py --layers 0 1 2 --num-permutations 50
    python src/experiments/permutation_sensitivity_experiment.py --config configs/analysis_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.permutation_sensitivity import PermutationSensitivityAnalyzer
from src.analysis.visualization import AnalysisVisualizer
from src.utils.config_utils import load_config, setup_logging_from_config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Permutation Sensitivity Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration file'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--layers', type=int, nargs='+', default=[0],
        help='Layer indices to analyze'
    )
    parser.add_argument(
        '--num-permutations', type=int, default=20,
        help='Number of permutation matrices to test per layer'
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
        '--num-workers', type=int, default=8,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device for computation'
    )
    
    # Output parameters
    parser.add_argument(
        '--output-dir', type=str, default='results/permutation_sensitivity',
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
        # Extract permutation config if available
        perm_config = config.get('analysis', {}).get('permutation_sensitivity', {})
        # Use config values as defaults, but allow command-line overrides
        if perm_config.get('num_permutations') and args.num_permutations == 20:  # default value
            args.num_permutations = perm_config.get('num_permutations')
        if perm_config.get('layers_to_analyze') and args.layers == [0]:  # default value
            args.layers = perm_config.get('layers_to_analyze')
    
    # Override config with command line arguments
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
        if len(args.layers) == 1:
            experiment_name = f"permutation_sensitivity_layer_{args.layers[0]}_{timestamp}"
        else:
            experiment_name = f"permutation_sensitivity_layers_{'-'.join(map(str, args.layers))}_{timestamp}"
    
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Permutation Sensitivity Analysis")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Layers: {args.layers}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {config}")

    try:
        # Initialize analyzer
        analyzer = PermutationSensitivityAnalyzer(
            model_name=config.get('model_name', args.model),
            imagenet_path=config.get('imagenet_path', args.imagenet_path),
            batch_size=config.get('batch_size', args.batch_size),
            num_workers=config.get('num_workers', args.num_workers),
            device=config.get('device', args.device)
        )

        # Run analysis
        logger.info(f"Running permutation sensitivity analysis for layers {args.layers}")
        results = analyzer.analyze_permutation_sensitivity(
            layer_indices=args.layers,
            num_permutations=args.num_permutations
        )
        
        # Add experiment metadata
        results.update({
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'layers': args.layers,
                'num_permutations': args.num_permutations
            }
        })
        
        # Save results
        results_path = output_dir / "sensitivity_results.json"
        analyzer.save_results(results, results_path)
        
        # Generate visualizations and report
        logger.info("Generating visualizations and report...")
        visualizer = AnalysisVisualizer()
        visualizer.create_comprehensive_report(
            results=results,
            output_dir=output_dir,
            report_name="permutation_sensitivity_analysis"
        )
        
        # Print summary
        logger.info("Permutation Sensitivity Analysis Results:")
        logger.info(f"Initial accuracy: {results['initial_accuracy']:.2%}")

        for layer_idx, layer_data in results['layer_results'].items():
            accuracies = [r['accuracy'] for r in layer_data]
            accuracy_drops = [r['accuracy_drop'] for r in layer_data]
            logger.info(f"Layer {layer_idx} - Mean accuracy drop: {sum(accuracy_drops)/len(accuracy_drops):.2%}")

        print(f"\nPermutation Sensitivity Analysis Complete!")
        print(f" Results saved to: {output_dir}")
        print(f" Analyzed {len(args.layers)} layers with {args.num_permutations} permutations each")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        print(f"❌ Analysis failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
