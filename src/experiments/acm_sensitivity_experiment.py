#!/usr/bin/env python3
"""
Arnold Cat Map Key Sensitivity Experiment

This script runs comprehensive sensitivity analysis for ACM keys,
analyzing how different key parameters affect model performance.

Usage:
    python src/experiments/acm_sensitivity_experiment.py --layer 0 --num-variants 30
    python src/experiments/acm_sensitivity_experiment.py --config configs/analysis_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.acm_sensitivity import ACMKeySensitivityAnalyzer
from src.analysis.visualization import AnalysisVisualizer
from src.utils.config_utils import load_config, setup_logging_from_config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Arnold Cat Map Key Sensitivity Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration file'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--layer', type=int, default=0,
        help='Layer index to analyze'
    )
    parser.add_argument(
        '--num-variants', type=int, default=30,
        help='Number of key variants to test'
    )
    parser.add_argument(
        '--base-key', type=int, nargs=5, default=None,
        metavar=('N', 'a', 'b', 'c', 'd'),
        help='Base ACM key parameters'
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
        '--batch-size', type=int, default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num-workers', type=int, default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device for computation'
    )
    
    # Output parameters
    parser.add_argument(
        '--output-dir', type=str, default='results/acm_sensitivity',
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
        experiment_name = f"acm_sensitivity_layer_{args.layer}_{timestamp}"
    
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting ACM Key Sensitivity Analysis")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {config}")
    
    try:
        # Initialize analyzer
        analyzer = ACMKeySensitivityAnalyzer(
            model_name=config.get('model_name', args.model),
            imagenet_path=config.get('imagenet_path', args.imagenet_path),
            batch_size=config.get('batch_size', args.batch_size),
            num_workers=config.get('num_workers', args.num_workers),
            device=config.get('device', args.device)
        )
        
        # Run sensitivity analysis
        logger.info(f"Analyzing layer {args.layer} with {args.num_variants} key variants")
        
        results = analyzer.analyze_key_sensitivity(
            layer_idx=args.layer,
            base_key=args.base_key,
            num_variants=args.num_variants
        )
        
        # Add experiment metadata
        results.update({
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'acm_key_sensitivity',
            'parameters': {
                'layer_idx': args.layer,
                'num_variants': args.num_variants,
                'base_key': args.base_key
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
            report_name="acm_sensitivity_analysis"
        )
        
        # Print summary
        key_results = results['key_results']
        accuracies = [r['accuracy'] for r in key_results]
        accuracy_drops = [r['accuracy_drop'] for r in key_results]
        
        logger.info("Analysis completed successfully!")
        logger.info(f"Initial accuracy: {results['initial_accuracy']:.2%}")
        logger.info(f"Mean accuracy after encryption: {sum(accuracies)/len(accuracies):.2%}")
        logger.info(f"Mean accuracy drop: {sum(accuracy_drops)/len(accuracy_drops):.2%}")
        logger.info(f"Min accuracy: {min(accuracies):.2%}")
        logger.info(f"Max accuracy: {max(accuracies):.2%}")
        
        print(f"\n🎯 ACM Sensitivity Analysis Complete!")
        print(f"📊 Results saved to: {output_dir}")
        print(f"📈 Mean accuracy drop: {sum(accuracy_drops)/len(accuracy_drops):.2%}")
        print(f"📉 Most effective key achieved: {min(accuracies):.2%} accuracy")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        print(f"❌ Analysis failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
