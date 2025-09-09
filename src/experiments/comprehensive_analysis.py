#!/usr/bin/env python3
"""
Comprehensive Analysis Runner

This script runs all (three) available analysis experiments in sequence.

Usage:
    python src/experiments/comprehensive_analysis.py --config configs/analysis_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.acm_sensitivity import ACMKeySensitivityAnalyzer
from src.analysis.permutation_sensitivity import PermutationSensitivityAnalyzer
from src.analysis.dual_encryption_analysis import DualEncryptionAnalyzer
from src.analysis.visualization import AnalysisVisualizer
from src.utils.config_utils import load_config, setup_logging_from_config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Analysis Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', type=str, default='configs/analysis_config.yaml',
        help='Path to configuration file'
    )
    
    # Analysis selection
    parser.add_argument(
        '--analyses', type=str, nargs='+', 
        default=['acm_sensitivity', 'permutation_sensitivity', 'dual_encryption'],
        choices=['acm_sensitivity', 'permutation_sensitivity', 'dual_encryption'],
        help='Types of analyses to run'
    )
    
    # Model parameters
    parser.add_argument(
        '--model', type=str, default=None,
        help='HuggingFace model identifier (overrides config)'
    )
    parser.add_argument(
        '--imagenet-path', type=str, default=None,
        help='Path to ImageNet validation dataset (overrides config)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size for evaluation (overrides config)'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device for computation (overrides config)'
    )
    
    # Output parameters
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results (overrides config)'
    )
    parser.add_argument(
        '--experiment-name', type=str, default=None,
        help='Name for this experiment suite'
    )
    
    # Logging
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def run_acm_sensitivity_analysis(config: dict, output_dir: Path) -> dict:
    """Run ACM key sensitivity analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Running ACM Key Sensitivity Analysis...")

    analyzer = ACMKeySensitivityAnalyzer(
        model_name=config['model_name'],
        imagenet_path=config['imagenet_path'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        device=config['device']
    )

    acm_config = config.get('analysis', {}).get('acm_sensitivity', {})
    layers = acm_config.get('layers_to_analyze', [0])
    num_variants = acm_config.get('num_variants', 30)
    base_key = acm_config.get('base_key')

    results = {}
    for layer_idx in layers:
        logger.info(f"Analyzing ACM sensitivity for layer {layer_idx} with {num_variants} key variants")

        layer_results = analyzer.analyze_key_sensitivity(
            layer_idx=layer_idx,
            base_key=base_key,
            num_variants=num_variants
        )

        results[f'layer_{layer_idx}'] = layer_results

        # Save individual layer results (matching standalone experiment structure)
        layer_output_dir = output_dir / "acm_sensitivity" / f"layer_{layer_idx}"
        analyzer.save_results(layer_results, layer_output_dir / "results.json")

        # Print summary for this layer (matching standalone experiment)
        key_results = layer_results['key_results']
        accuracies = [r['accuracy'] for r in key_results]
        accuracy_drops = [r['accuracy_drop'] for r in key_results]

        logger.info(f"Layer {layer_idx} Analysis Results:")
        logger.info(f"  Initial accuracy: {layer_results['initial_accuracy']:.2%}")
        logger.info(f"  Mean accuracy after encryption: {sum(accuracies)/len(accuracies):.2%}")
        logger.info(f"  Mean accuracy drop: {sum(accuracy_drops)/len(accuracy_drops):.2%}")
        logger.info(f"  Min accuracy: {min(accuracies):.2%}")
        logger.info(f"  Max accuracy: {max(accuracies):.2%}")

    return results


def run_permutation_sensitivity_analysis(config: dict, output_dir: Path) -> dict:
    """Run permutation sensitivity analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Running Permutation Sensitivity Analysis...")

    analyzer = PermutationSensitivityAnalyzer(
        model_name=config['model_name'],
        imagenet_path=config['imagenet_path'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        device=config['device']
    )

    perm_config = config.get('analysis', {}).get('permutation_sensitivity', {})
    layers = perm_config.get('layers_to_analyze', [0])
    num_permutations = perm_config.get('num_permutations', 20)

    logger.info(f"Analyzing permutation sensitivity for layers {layers}")

    # Run permutation sensitivity analysis
    results = analyzer.analyze_permutation_sensitivity(
        layer_indices=layers,
        num_permutations=num_permutations
    )

    # Save results
    output_dir_perm = output_dir / "permutation_sensitivity"
    analyzer.save_results(results, output_dir_perm / "results.json")

    # Print summary (matching standalone experiment)
    logger.info("Permutation Sensitivity Analysis Results:")
    logger.info(f"Initial accuracy: {results['initial_accuracy']:.2%}")

    for layer_idx, layer_data in results['layer_results'].items():
        accuracies = [r['accuracy'] for r in layer_data]
        accuracy_drops = [r['accuracy_drop'] for r in layer_data]
        logger.info(f"Layer {layer_idx} - Mean accuracy drop: {sum(accuracy_drops)/len(accuracy_drops):.2%}")

    return results


def run_dual_encryption_analysis(config: dict, output_dir: Path) -> dict:
    """Run dual encryption security analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Running Dual Encryption Security Analysis...")

    analyzer = DualEncryptionAnalyzer(
        model_name=config['model_name'],
        imagenet_path=config['imagenet_path'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        device=config['device']
    )

    dual_config = config.get('analysis', {}).get('dual_encryption', {})
    layers = dual_config.get('layers_to_analyze', [0, 1, 2])
    num_attack_variants = dual_config.get('num_attack_variants', 10)

    logger.info(f"Running dual encryption security analysis for layers {layers}")

    # Run security analysis
    results = analyzer.analyze_dual_encryption_security(
        layer_indices=layers,
        num_attack_variants=num_attack_variants
    )

    # Save results
    output_dir_dual = output_dir / "dual_encryption"
    analyzer.save_results(results, output_dir_dual / "results.json")

    # Print security analysis summary (matching standalone experiment)
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

    return results


def main():
    """Main comprehensive analysis function."""
    args = parse_arguments()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Override config with command line arguments
    if args.model:
        config['model_name'] = args.model
    if args.imagenet_path:
        config['imagenet_path'] = args.imagenet_path
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.device:
        config['device'] = args.device
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Setup logging
    setup_logging_from_config({'log_level': args.log_level})
    logger = logging.getLogger(__name__)
    
    # Setup experiment directory
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"comprehensive_analysis_{timestamp}"
    
    output_dir = Path(config.get('output_dir', 'results/analysis')) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Comprehensive Analysis Suite")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Analyses: {args.analyses}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save experiment configuration
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump({
            'config': config,
            'args': vars(args),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    all_results = {}
    
    try:
        # Run selected analyses
        if 'acm_sensitivity' in args.analyses:
            all_results['acm_sensitivity'] = run_acm_sensitivity_analysis(
                config, output_dir
            )
        
        if 'permutation_sensitivity' in args.analyses:
            all_results['permutation_sensitivity'] = run_permutation_sensitivity_analysis(
                config, output_dir
            )
        
        if 'dual_encryption' in args.analyses:
            all_results['dual_encryption'] = run_dual_encryption_analysis(
                config, output_dir
            )
        
        # Generate comprehensive report
        logger.info("Generating comprehensive analysis report...")
        
        # Save all results
        with open(output_dir / "comprehensive_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate visualizations for each analysis type
        visualizer = AnalysisVisualizer()

        for analysis_type, results in all_results.items():
            if analysis_type == 'dual_encryption':
                # Handle dual encryption results based on structure
                if isinstance(results, dict) and 'method_comparison' in results:
                    # Old structure with both method_comparison and sensitivity
                    for sub_analysis, sub_results in results.items():
                        try:
                            visualizer.create_comprehensive_report(
                                results=sub_results,
                                output_dir=output_dir / analysis_type / sub_analysis,
                                report_name=f"{analysis_type}_{sub_analysis}"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to create report for {analysis_type}_{sub_analysis}: {e}")
                else:
                    # New structure - single analysis type result
                    try:
                        visualizer.create_comprehensive_report(
                            results=results,
                            output_dir=output_dir / analysis_type,
                            report_name=f"{analysis_type}_analysis"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create report for {analysis_type}: {e}")

            elif analysis_type == 'permutation_sensitivity':
                # Handle permutation sensitivity results - single unified structure
                try:
                    visualizer.create_comprehensive_report(
                        results=results,
                        output_dir=output_dir / analysis_type,
                        report_name=f"{analysis_type}_analysis"
                    )
                except Exception as e:
                    logger.warning(f"Failed to create report for {analysis_type}: {e}")

            else:
                # Handle ACM sensitivity (layer-based results)
                for layer_name, layer_results in results.items():
                    try:
                        visualizer.create_comprehensive_report(
                            results=layer_results,
                            output_dir=output_dir / analysis_type / layer_name,
                            report_name=f"{analysis_type}_{layer_name}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create report for {analysis_type}_{layer_name}: {e}")
        
        logger.info("Comprehensive analysis completed successfully!")
        
        print(f"\nAnalysis Suite Complete!")
        print(f" Results saved to: {output_dir}")
        print(f" Analyses completed: {', '.join(args.analyses)}")
        print(f" Check individual analysis directories for detailed results!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {str(e)}", exc_info=True)
        print(f" Analysis failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
