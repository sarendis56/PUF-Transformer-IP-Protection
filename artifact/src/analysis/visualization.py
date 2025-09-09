"""
Analysis Visualization and Reporting for Transformer IP Protection Experiments.

This module provides comprehensive visualization capabilities for:
1. ACM Key Sensitivity Analysis - Tests how key variations affect model accuracy
2. Permutation Sensitivity Analysis - Tests how permutation variations affect model accuracy
3. Dual Encryption Security Analysis - Tests resistance to key guessing attacks

Each analysis type has dedicated visualization functions that create appropriate plots
and generate comprehensive reports.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import json
import logging
from scipy import stats


def plot_sensitivity_results(results: Dict, 
                           output_path: Optional[Path] = None,
                           title: str = "Encryption Sensitivity Analysis") -> None:
    """
    Plot sensitivity analysis results.

    For ACM Key Sensitivity Analysis, creates a 2x3 subplot layout:
    - Top row: Accuracy vs Distance, Accuracy Drop vs Distance, Relative Accuracy vs Distance (with trend)
    - Bottom row: Accuracy Distribution, Accuracy Drop Distribution, Relative Accuracy Distribution

    The Relative Accuracy vs Distance plot includes a linear trend line to show correlation
    (or lack thereof) between key distance and relative accuracy.

    Args:
        results: Analysis results dictionary
        output_path: Path to save the plot (if None, displays plot)
        title: Title for the plot
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Handle different result structures
    if 'key_results' in results:
        # ACM Key Sensitivity Analysis
        key_results = results['key_results']
        distances = [r['key_distance'] for r in key_results]
        accuracies = [r['accuracy'] for r in key_results]
        accuracy_drops = [r['accuracy_drop'] for r in key_results]
        relative_accuracies = [r['relative_accuracy'] for r in key_results]

        # Plot 1: Accuracy vs Key Distance
        axes[0, 0].scatter(distances, accuracies, alpha=0.7, c='blue')
        axes[0, 0].set_xlabel('Key Distance')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs Key Distance')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Accuracy Drop vs Key Distance
        axes[0, 1].scatter(distances, accuracy_drops, alpha=0.7, c='red')
        axes[0, 1].set_xlabel('Key Distance')
        axes[0, 1].set_ylabel('Accuracy Drop')
        axes[0, 1].set_title('Accuracy Drop vs Key Distance')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: NEW - Relative Accuracy vs Key Distance with Linear Trend
        axes[0, 2].scatter(distances, relative_accuracies, alpha=0.7, c='purple', s=30)

        # Fit linear trend line
        if len(distances) > 1 and len(set(distances)) > 1:  # Need at least 2 different distances
            slope, intercept, r_value, p_value, std_err = stats.linregress(distances, relative_accuracies)
            line_x = np.array([min(distances), max(distances)])
            line_y = slope * line_x + intercept
            axes[0, 2].plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2,
                           label=f'Linear fit (R²={r_value**2:.3f})')
            axes[0, 2].legend()

        axes[0, 2].set_xlabel('Euclidean Distance from Correct Key')
        axes[0, 2].set_ylabel('Relative Accuracy')
        axes[0, 2].set_title('Relative Accuracy vs Key Distance')
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Accuracy Distribution
        axes[1, 0].hist(accuracies, bins=20, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Accuracy')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Accuracy Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Accuracy Drop Distribution
        axes[1, 1].hist(accuracy_drops, bins=20, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Accuracy Drop')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Accuracy Drop Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Relative Accuracy Distribution
        axes[1, 2].hist(relative_accuracies, bins=20, alpha=0.7, color='purple')
        axes[1, 2].set_xlabel('Relative Accuracy')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Relative Accuracy Distribution')
        axes[1, 2].grid(True, alpha=0.3)
    
    elif 'sensitivity_results' in results:
        # Dual Encryption Sensitivity Analysis
        sensitivity_results = results['sensitivity_results']
        if sensitivity_results and len(sensitivity_results) > 0:
            # Extract data for dual encryption sensitivity
            key_distances = [r.get('key_distance', 0) for r in sensitivity_results]
            accuracies = [r['accuracy'] for r in sensitivity_results]
            accuracy_drops = [r['accuracy_drop'] for r in sensitivity_results]
            relative_accuracies = [r['relative_accuracy'] for r in sensitivity_results]
            perm_seeds = [r.get('permutation_seed', 0) for r in sensitivity_results]

            # Plot 1: Accuracy vs Key Distance
            axes[0, 0].scatter(key_distances, accuracies, alpha=0.7, c='blue', s=30)
            axes[0, 0].set_xlabel('Key Distance')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Accuracy vs Key Distance')
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Accuracy Drop vs Key Distance
            axes[0, 1].scatter(key_distances, accuracy_drops, alpha=0.7, c='red', s=30)
            axes[0, 1].set_xlabel('Key Distance')
            axes[0, 1].set_ylabel('Accuracy Drop')
            axes[0, 1].set_title('Accuracy Drop vs Key Distance')
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Relative Accuracy vs Key Distance with Linear Trend
            axes[0, 2].scatter(key_distances, relative_accuracies, alpha=0.7, c='purple', s=30)

            # Fit linear trend line if we have varying distances
            if len(set(key_distances)) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(key_distances, relative_accuracies)
                line_x = np.array([min(key_distances), max(key_distances)])
                line_y = slope * line_x + intercept
                axes[0, 2].plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2,
                               label=f'Linear fit (R²={r_value**2:.3f})')
                axes[0, 2].legend()

            axes[0, 2].set_xlabel('Key Distance')
            axes[0, 2].set_ylabel('Relative Accuracy')
            axes[0, 2].set_title('Relative Accuracy vs Key Distance')
            axes[0, 2].grid(True, alpha=0.3)

            # Plot 4: Accuracy Distribution
            axes[1, 0].hist(accuracies, bins=min(20, len(set(accuracies))), alpha=0.7, color='green')
            axes[1, 0].set_xlabel('Accuracy')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Accuracy Distribution')
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 5: Accuracy Drop Distribution
            axes[1, 1].hist(accuracy_drops, bins=min(20, len(set(accuracy_drops))), alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('Accuracy Drop')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Accuracy Drop Distribution')
            axes[1, 1].grid(True, alpha=0.3)

            # Plot 6: Relative Accuracy Distribution
            axes[1, 2].hist(relative_accuracies, bins=min(20, len(set(relative_accuracies))), alpha=0.7, color='purple')
            axes[1, 2].set_xlabel('Relative Accuracy')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Relative Accuracy Distribution')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            # No data to plot
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No dual encryption sensitivity results available',
                       ha='center', va='center', transform=ax.transAxes)

    elif 'permutation_results' in results or isinstance(results, list) or 'layer_results' in results:
        # Handle different types of layer results
        perm_results = None
        decrypt_layer_results = None

        if 'permutation_results' in results:
            perm_results = results['permutation_results']
        elif isinstance(results, list):
            perm_results = results  # Direct list of results
        elif 'layer_results' in results:
            layer_results = results['layer_results']
            if isinstance(layer_results, dict) and layer_results:
                # Check if this is decrypt one layer analysis (layer_results contains dicts with accuracy metrics)
                first_layer_key = list(layer_results.keys())[0]
                first_layer_data = layer_results[first_layer_key]

                if isinstance(first_layer_data, dict) and 'decrypted_accuracy' in first_layer_data:
                    # This is decrypt one layer analysis
                    decrypt_layer_results = layer_results
                elif isinstance(first_layer_data, dict) and 'security_results' in first_layer_data:
                    # This is dual encryption security analysis - handled by dedicated function
                    pass  # Skip here, will be handled by plot_dual_encryption_security
                elif isinstance(first_layer_data, list):
                    # This is permutation sensitivity with layer_results structure
                    perm_results = first_layer_data

        # Handle decrypt one layer analysis results
        if decrypt_layer_results:
            layers = list(decrypt_layer_results.keys())
            # Sort layers by index for better visualization
            layers_sorted = sorted(layers, key=lambda x: int(x))
            layer_indices = [int(layer) for layer in layers_sorted]
            decrypted_accuracies = [decrypt_layer_results[layer]['decrypted_accuracy'] for layer in layers_sorted]
            improvements = [decrypt_layer_results[layer]['accuracy_improvement'] for layer in layers_sorted]
            improvement_percentages = [decrypt_layer_results[layer]['improvement_percentage'] for layer in layers_sorted]

            baseline_accuracy = results.get('baseline_accuracy', 0)
            initial_accuracy = results.get('initial_accuracy', 0)

            # Plot 1: Accuracy by Layer
            axes[0, 0].bar(layer_indices, decrypted_accuracies, alpha=0.7, color='blue')
            axes[0, 0].axhline(y=baseline_accuracy, color='red', linestyle='--', alpha=0.8, label=f'Baseline (encrypted): {baseline_accuracy:.4f}')
            axes[0, 0].axhline(y=initial_accuracy, color='green', linestyle='--', alpha=0.8, label=f'Original (unencrypted): {initial_accuracy:.4f}')
            axes[0, 0].set_xlabel('Layer Index')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Accuracy After Decrypting Single Layer')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Accuracy Improvement by Layer
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            axes[0, 1].bar(layer_indices, improvements, alpha=0.7, color=colors)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[0, 1].set_xlabel('Layer Index')
            axes[0, 1].set_ylabel('Accuracy Improvement')
            axes[0, 1].set_title('Accuracy Improvement by Layer')
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Improvement Percentage by Layer
            axes[0, 2].bar(layer_indices, improvement_percentages, alpha=0.7, color=colors)
            axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[0, 2].set_xlabel('Layer Index')
            axes[0, 2].set_ylabel('Improvement (%)')
            axes[0, 2].set_title('Accuracy Improvement Percentage by Layer')
            axes[0, 2].grid(True, alpha=0.3)

            # Plot 4: Accuracy Distribution
            axes[1, 0].hist(decrypted_accuracies, bins=min(10, len(set(decrypted_accuracies))), alpha=0.7, color='blue')
            axes[1, 0].axvline(x=baseline_accuracy, color='red', linestyle='--', alpha=0.8, label='Baseline')
            axes[1, 0].axvline(x=initial_accuracy, color='green', linestyle='--', alpha=0.8, label='Original')
            axes[1, 0].set_xlabel('Accuracy')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Accuracy Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 5: Improvement Distribution
            axes[1, 1].hist(improvements, bins=min(10, len(set(improvements))), alpha=0.7, color='orange')
            axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 1].set_xlabel('Accuracy Improvement')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Improvement Distribution')
            axes[1, 1].grid(True, alpha=0.3)

            # Plot 6: Summary Statistics
            axes[1, 2].axis('off')
            stats_text = f"""Decrypt One Layer Analysis Summary

Layers Analyzed: {len(layers)}
Baseline Accuracy: {baseline_accuracy:.4f}
Original Accuracy: {initial_accuracy:.4f}

Improvement Statistics:
Max Improvement: {max(improvements):.4f} ({max(improvement_percentages):.2f}%)
Min Improvement: {min(improvements):.4f} ({min(improvement_percentages):.2f}%)
Mean Improvement: {np.mean(improvements):.4f} ({np.mean(improvement_percentages):.2f}%)
Std Improvement: {np.std(improvements):.4f} ({np.std(improvement_percentages):.2f}%)
"""

            axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')

        elif perm_results and len(perm_results) > 0:
            # Use permutation_idx if available, otherwise use index as seed
            seeds = [r.get('permutation_idx', r.get('seed', i)) for i, r in enumerate(perm_results)]
            accuracies = [r['accuracy'] for r in perm_results]
            accuracy_drops = [r['accuracy_drop'] for r in perm_results]
            relative_accuracies = [r['relative_accuracy'] for r in perm_results]

            # Get permutation intensities if available
            intensities = [r.get('permutation_intensity', 0) for r in perm_results]
            has_intensity = any(intensity > 0 for intensity in intensities)

            # Plot 1: Accuracy vs Intensity (if available) or Index
            if has_intensity:
                axes[0, 0].scatter(intensities, accuracies, alpha=0.7, c='blue', s=30)
                axes[0, 0].set_xlabel('Permutation Intensity')
                axes[0, 0].set_title('Accuracy vs Permutation Intensity')
            else:
                axes[0, 0].plot(seeds, accuracies, 'b-o', alpha=0.7, markersize=4)
                axes[0, 0].set_xlabel('Permutation Index')
                axes[0, 0].set_title('Accuracy vs Permutation Index')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Accuracy Drop vs Intensity (if available) or Index
            if has_intensity:
                axes[0, 1].scatter(intensities, accuracy_drops, alpha=0.7, c='red', s=30)
                axes[0, 1].set_xlabel('Permutation Intensity')
                axes[0, 1].set_title('Accuracy Drop vs Permutation Intensity')
            else:
                axes[0, 1].plot(seeds, accuracy_drops, 'r-o', alpha=0.7, markersize=4)
                axes[0, 1].set_xlabel('Permutation Index')
                axes[0, 1].set_title('Accuracy Drop vs Permutation Index')
            axes[0, 1].set_ylabel('Accuracy Drop')
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Relative Accuracy vs Permutation Intensity
            axes[0, 2].scatter(intensities, relative_accuracies, alpha=0.7, c='purple', s=30)

            # Fit linear trend line if we have varying intensities
            if len(set(intensities)) > 1:
                slope, intercept, r_value, _, _ = stats.linregress(intensities, relative_accuracies)
                line_x = np.array([min(intensities), max(intensities)])
                line_y = slope * line_x + intercept
                axes[0, 2].plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2,
                               label='Linear fit')
                axes[0, 2].legend()

            axes[0, 2].set_xlabel('Permutation Intensity')
            axes[0, 2].set_ylabel('Relative Accuracy')
            axes[0, 2].set_title('Relative Accuracy vs Permutation Intensity')
            axes[0, 2].grid(True, alpha=0.3)

            # Plot 4: Accuracy Distribution
            axes[1, 0].hist(accuracies, bins=min(15, len(set(accuracies))), alpha=0.7, color='green')
            axes[1, 0].set_xlabel('Accuracy')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Accuracy Distribution')
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 5: Accuracy Drop Distribution
            axes[1, 1].hist(accuracy_drops, bins=min(15, len(set(accuracy_drops))), alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('Accuracy Drop')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Accuracy Drop Distribution')
            axes[1, 1].grid(True, alpha=0.3)

            # Plot 6: Relative Accuracy Distribution
            axes[1, 2].hist(relative_accuracies, bins=min(15, len(set(relative_accuracies))), alpha=0.7, color='purple')
            axes[1, 2].set_xlabel('Relative Accuracy')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Relative Accuracy Distribution')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            # No data to plot
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No permutation results available',
                       ha='center', va='center', transform=ax.transAxes)
    
    else:
        # Unknown format - show debug info
        for ax in axes.flat:
            ax.text(0.5, 0.5, f'Unknown result format\nKeys: {list(results.keys()) if isinstance(results, dict) else "Not a dict"}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_dual_encryption_security_per_layer(layer_idx: int,
                                           layer_data: Dict,
                                           output_path: Optional[Path] = None) -> None:
    """
    Plot dual encryption security analysis results for a single layer.

    Creates a single figure showing attack accuracy distributions for ACM and permutation attacks.

    Args:
        layer_idx: Layer index being analyzed
        layer_data: Security analysis results for this layer
        output_path: Path to save the plot
    """
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    security_results = layer_data['security_results']

    # Extract attack accuracies
    acm_accuracies = [r['accuracy'] for r in security_results['acm_attack_results']]
    perm_accuracies = [r['accuracy'] for r in security_results['permutation_attack_results']]

    if acm_accuracies and perm_accuracies:
        # Create histogram/distribution plot with improved binning
        all_accuracies = acm_accuracies + perm_accuracies
        max_acc = max(all_accuracies)
        min_acc = min(all_accuracies)

        # Use adaptive binning strategy
        if max_acc < 0.01:  # Very low accuracies (< 1%)
            # Use more bins and extend range slightly for better visualization
            bins = np.linspace(0, max_acc * 1.5, 30)
        elif max_acc < 0.1:  # Low accuracies (< 10%)
            bins = np.linspace(0, max_acc * 1.2, 25)
        else:  # Higher accuracies
            bins = np.linspace(0, max_acc * 1.1, 20)

        # Debug information
        print(f"Layer {layer_idx} - ACM accuracies range: {min(acm_accuracies):.6f} to {max(acm_accuracies):.6f}")
        print(f"Layer {layer_idx} - Perm accuracies range: {min(perm_accuracies):.6f} to {max(perm_accuracies):.6f}")
        print(f"Layer {layer_idx} - Bins range: {bins[0]:.6f} to {bins[-1]:.6f}, num_bins: {len(bins)-1}")

        # Create histograms with better visibility for small values
        n1, bins1, patches1 = ax.hist(acm_accuracies, bins=bins, alpha=0.7, color='gold',
                                     label=f'ACM Key Guessing ({len(acm_accuracies)} variants)',
                                     density=True, edgecolor='black', linewidth=0.5)
        n2, bins2, patches2 = ax.hist(perm_accuracies, bins=bins, alpha=0.7, color='purple',
                                     label=f'Permutation Key Guessing ({len(perm_accuracies)} variants)',
                                     density=True, edgecolor='black', linewidth=0.5)

        print(f"Layer {layer_idx} - Histogram heights - ACM max: {max(n1) if len(n1) > 0 else 0:.2f}, Perm max: {max(n2) if len(n2) > 0 else 0:.2f}")

        # Set x-axis limits to focus on attack accuracy distribution
        ax.set_xlim(0, max_acc * 1.1)

        # Only add dual encrypted accuracy reference line if it's within reasonable range
        dual_encrypted_acc = security_results['dual_encrypted_accuracy']
        if dual_encrypted_acc <= max_acc * 2:  # Only show if not too far from attack range
            ax.axvline(dual_encrypted_acc, color='red', linestyle='--', linewidth=2,
                       label=f'Dual Encrypted Accuracy ({dual_encrypted_acc:.2%})')

        ax.set_xlabel('Attack Accuracy')
        ax.set_ylabel('Density')
        ax.set_title(f'Layer {layer_idx}: Key Guessing Attack Accuracy Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text
        original_acc = security_results['original_accuracy']
        stats_text = f"""Attack Statistics:
ACM: μ={np.mean(acm_accuracies):.2%}, max={max(acm_accuracies):.2%}
Perm: μ={np.mean(perm_accuracies):.2%}, max={max(perm_accuracies):.2%}
Security Gap: {original_acc - max(max(acm_accuracies), max(perm_accuracies)):.2%}"""

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax.text(0.5, 0.5, f'No attack data available for Layer {layer_idx}',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_dual_encryption_security(results: Dict,
                                 output_path: Optional[Path] = None) -> None:
    """
    Plot dual encryption security analysis results.

    Creates separate plots for each layer analyzed.

    Args:
        results: Dual encryption security analysis results
        output_path: Base path to save the plots (will create multiple files)
    """

    if 'layer_results' in results and isinstance(results['layer_results'], dict):
        # Create separate plots for each layer
        layer_results = results['layer_results']

        if len(layer_results) == 1:
            # Single layer - use the provided output path
            layer_idx, layer_data = next(iter(layer_results.items()))
            plot_dual_encryption_security_per_layer(int(layer_idx), layer_data, output_path)
        else:
            # Multiple layers - create separate files for each
            if output_path:
                base_path = output_path.parent
                base_name = output_path.stem
                extension = output_path.suffix

                for layer_idx, layer_data in layer_results.items():
                    layer_output_path = base_path / f"{base_name}_layer_{layer_idx}{extension}"
                    plot_dual_encryption_security_per_layer(int(layer_idx), layer_data, layer_output_path)
            else:
                # No output path - show plots for each layer
                for layer_idx, layer_data in layer_results.items():
                    plot_dual_encryption_security_per_layer(int(layer_idx), layer_data, None)
    else:
        # No layer results - create empty plot
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'No layer results found',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Dual Encryption Security Analysis')

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_method_comparison(results: Dict,
                         output_path: Optional[Path] = None) -> None:
    """
    Plot comparison of different encryption methods.

    Args:
        results: Dual encryption analysis results
        output_path: Path to save the plot
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Encryption Method Comparison', fontsize=16, fontweight='bold')

    # Check different possible result structures
    if 'layer_results' in results and isinstance(results['layer_results'], dict):
        # Check if this is method comparison results
        first_layer_data = list(results['layer_results'].values())[0]
        if isinstance(first_layer_data, dict) and 'method_results' in first_layer_data:
            # Method comparison results
            layers = []
            arnold_drops = []
            perm_drops = []
            dual_drops = []

            for layer_idx, layer_data in results['layer_results'].items():
                layers.append(f"Layer {layer_idx}")
                method_results = layer_data['method_results']

                arnold_drops.append(method_results['arnold_only']['accuracy_drop'])
                perm_drops.append(method_results['permutation_only']['accuracy_drop'])
                dual_drops.append(method_results['dual_encryption']['accuracy_drop'])
            
            # Plot 1: Accuracy Drop Comparison
            x = np.arange(len(layers))
            width = 0.25
            
            axes[0].bar(x - width, arnold_drops, width, label='Arnold Only', alpha=0.8)
            axes[0].bar(x, perm_drops, width, label='Permutation Only', alpha=0.8)
            axes[0].bar(x + width, dual_drops, width, label='Dual Encryption', alpha=0.8)
            
            axes[0].set_xlabel('Layer')
            axes[0].set_ylabel('Accuracy Drop')
            axes[0].set_title('Accuracy Drop by Method')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(layers)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Relative Accuracy Comparison
            arnold_rel = []
            perm_rel = []
            dual_rel = []

            for layer_idx, layer_data in results['layer_results'].items():
                method_results = layer_data['method_results']
                arnold_rel.append(method_results['arnold_only']['relative_accuracy'])
                perm_rel.append(method_results['permutation_only']['relative_accuracy'])
                dual_rel.append(method_results['dual_encryption']['relative_accuracy'])

            axes[1].bar(x - width, arnold_rel, width, label='Arnold Only', alpha=0.8)
            axes[1].bar(x, perm_rel, width, label='Permutation Only', alpha=0.8)
            axes[1].bar(x + width, dual_rel, width, label='Dual Encryption', alpha=0.8)

            axes[1].set_xlabel('Layer')
            axes[1].set_ylabel('Relative Accuracy')
            axes[1].set_title('Relative Accuracy by Method')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(layers)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            # Layer results but not method comparison - check if it's permutation sensitivity
            # This handles cases like permutation sensitivity with layer_results containing lists
            first_layer_key = list(results['layer_results'].keys())[0]
            first_layer_data = results['layer_results'][first_layer_key]

            if isinstance(first_layer_data, list) and len(first_layer_data) > 0:
                # This looks like permutation sensitivity data
                perm_results = first_layer_data

                # Extract data for plotting
                permutation_indices = [r.get('permutation_idx', i) for i, r in enumerate(perm_results)]
                accuracies = [r['accuracy'] for r in perm_results]
                accuracy_drops = [r['accuracy_drop'] for r in perm_results]

                # Plot 1: Accuracy vs Permutation Index
                axes[0].plot(permutation_indices, accuracies, 'b-o', alpha=0.7, markersize=4)
                axes[0].set_xlabel('Permutation Index')
                axes[0].set_ylabel('Accuracy')
                axes[0].set_title('Accuracy vs Permutation Index')
                axes[0].grid(True, alpha=0.3)

                # Plot 2: Accuracy Drop vs Permutation Index
                axes[1].plot(permutation_indices, accuracy_drops, 'r-o', alpha=0.7, markersize=4)
                axes[1].set_xlabel('Permutation Index')
                axes[1].set_ylabel('Accuracy Drop')
                axes[1].set_title('Accuracy Drop vs Permutation Index')
                axes[1].grid(True, alpha=0.3)
            else:
                # Unknown layer results format
                axes[0].text(0.5, 0.5, 'Layer results found but not method comparison format',
                            ha='center', va='center', transform=axes[0].transAxes)
                axes[1].text(0.5, 0.5, 'Check data structure',
                            ha='center', va='center', transform=axes[1].transAxes)

    elif 'sensitivity_results' in results and isinstance(results['sensitivity_results'], list):
        # Sensitivity analysis results - create scatter plots
        sensitivity_results = results['sensitivity_results']
        if sensitivity_results:
            accuracies = [r['accuracy'] for r in sensitivity_results]
            accuracy_drops = [r['accuracy_drop'] for r in sensitivity_results]
            key_distances = [r.get('key_distance', 0) for r in sensitivity_results]

            # Plot 1: Accuracy vs Key Distance (if available) or Parameter Combination
            if any(kd > 0 for kd in key_distances):
                axes[0].scatter(key_distances, accuracies, alpha=0.7, c='blue', s=20)
                axes[0].set_xlabel('Key Distance')
                axes[0].set_ylabel('Accuracy')
                axes[0].set_title('Dual Encryption Sensitivity: Accuracy vs Key Distance')
            else:
                axes[0].scatter(range(len(accuracies)), accuracies, alpha=0.7, c='blue', s=20)
                axes[0].set_xlabel('Parameter Combination Index')
                axes[0].set_ylabel('Accuracy')
                axes[0].set_title('Dual Encryption Sensitivity: Accuracy vs Parameter')
            axes[0].grid(True, alpha=0.3)

            # Set y-axis to show the actual data range more clearly
            if accuracies:
                y_min, y_max = min(accuracies), max(accuracies)
                y_range = y_max - y_min
                if y_range > 0:
                    axes[0].set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            # Plot 2: Accuracy Drop vs Key Distance or Distribution
            if any(kd > 0 for kd in key_distances):
                axes[1].scatter(key_distances, accuracy_drops, alpha=0.7, c='red', s=20)
                axes[1].set_xlabel('Key Distance')
                axes[1].set_ylabel('Accuracy Drop')
                axes[1].set_title('Accuracy Drop vs Key Distance')
                axes[1].grid(True, alpha=0.3)
            else:
                # Use histogram if no key distance info
                unique_drops = len(set(accuracy_drops))
                bins = min(15, max(5, unique_drops))
                axes[1].hist(accuracy_drops, bins=bins, alpha=0.7, color='red')
                axes[1].set_xlabel('Accuracy Drop')
                axes[1].set_ylabel('Frequency')
                axes[1].set_title('Accuracy Drop Distribution')
                axes[1].grid(True, alpha=0.3)
        else:
            # No sensitivity results to plot
            axes[0].text(0.5, 0.5, 'No sensitivity results available',
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[1].text(0.5, 0.5, 'No sensitivity results available',
                        ha='center', va='center', transform=axes[1].transAxes)
    
    elif isinstance(results, list):
        # Direct list of results (permutation sensitivity case)
        if results and len(results) > 0:
            accuracies = [r['accuracy'] for r in results]
            accuracy_drops = [r['accuracy_drop'] for r in results]
            seeds = [r.get('seed', i) for i, r in enumerate(results)]
            
            # Plot 1: Accuracy vs Seed
            axes[0].plot(seeds, accuracies, 'b-o', alpha=0.7)
            axes[0].set_xlabel('Permutation Seed')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Permutation Sensitivity')
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Accuracy Drop Distribution
            axes[1].hist(accuracy_drops, bins=15, alpha=0.7, color='red')
            axes[1].set_xlabel('Accuracy Drop')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Accuracy Drop Distribution')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'No results in list',
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[1].text(0.5, 0.5, 'No results in list',
                        ha='center', va='center', transform=axes[1].transAxes)
    
    else:
        # Unknown result format - show debug info
        debug_info = f'Unknown result format\nType: {type(results)}\nKeys: {list(results.keys()) if isinstance(results, dict) else "Not a dict"}'
        axes[0].text(0.5, 0.5, debug_info, ha='center', va='center', transform=axes[0].transAxes, fontsize=10)
        axes[1].text(0.5, 0.5, debug_info, ha='center', va='center', transform=axes[1].transAxes, fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_analysis_report(results: Dict, 
                           output_path: Path,
                           include_plots: bool = True) -> str:
    """
    Generate a comprehensive analysis report.
    
    Args:
        results: Analysis results dictionary
        output_path: Path to save the report
        include_plots: Whether to include plots in the report
        
    Returns:
        str: Generated report content
    """
    report_lines = [
        "=" * 80,
        "COMPREHENSIVE ENCRYPTION ANALYSIS REPORT",
        "=" * 80,
        "",
        f"Analysis Type: {results.get('analysis_type', 'Unknown')}",
        f"Model: {results.get('model_name', 'Unknown')}",
        f"Generated: {results.get('timestamp', 'Unknown')}",
        "",
    ]
    
    # ACM Key Sensitivity Analysis
    if 'key_results' in results:
        report_lines.extend([
            "ARNOLD CAT MAP KEY SENSITIVITY ANALYSIS",
            "-" * 50,
            f"Base Key: {results.get('base_key', 'Unknown')}",
            f"Initial Accuracy: {results.get('initial_accuracy', 0):.2%}",
            f"Number of Key Variants Tested: {len(results['key_results'])}",
            "",
            "Key Sensitivity Statistics:",
        ])
        
        key_results = results['key_results']
        accuracies = [r['accuracy'] for r in key_results]
        accuracy_drops = [r['accuracy_drop'] for r in key_results]
        distances = [r['key_distance'] for r in key_results]
        
        if accuracies:  # Check if we have data
            report_lines.extend([
                f"  Mean Accuracy: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}",
                f"  Mean Accuracy Drop: {np.mean(accuracy_drops):.2%} ± {np.std(accuracy_drops):.2%}",
                f"  Mean Key Distance: {np.mean(distances):.2f} ± {np.std(distances):.2f}",
                f"  Min Accuracy: {np.min(accuracies):.2%}",
                f"  Max Accuracy: {np.max(accuracies):.2%}",
            ])
        else:
            report_lines.extend([
                f"  No key variant results available",
            "",
            "Top 5 Most Effective Keys (Lowest Accuracy):",
        ])
        
        if key_results:  # Check if we have results
            # Sort by accuracy (ascending)
            sorted_results = sorted(key_results, key=lambda x: x['accuracy'])
            for i, result in enumerate(sorted_results[:5]):
                report_lines.append(
                    f"  {i+1}. Key: {result['key']}, "
                    f"Accuracy: {result['accuracy']:.2%}, "
                    f"Drop: {result['accuracy_drop']:.2%}"
                )
        else:
            report_lines.append("  No key variant results to display")
    
    # Permutation Sensitivity Analysis
    elif 'permutation_results' in results:
        report_lines.extend([
            "PERMUTATION SENSITIVITY ANALYSIS",
            "-" * 40,
            f"Initial Accuracy: {results.get('initial_accuracy', 0):.2%}",
            f"Number of Permutations Tested: {len(results['permutation_results'])}",
            "",
            "Permutation Sensitivity Statistics:",
        ])
        
        perm_results = results['permutation_results']
        accuracies = [r['accuracy'] for r in perm_results]
        accuracy_drops = [r['accuracy_drop'] for r in perm_results]
        
        report_lines.extend([
            f"  Mean Accuracy: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}",
            f"  Mean Accuracy Drop: {np.mean(accuracy_drops):.2%} ± {np.std(accuracy_drops):.2%}",
            f"  Min Accuracy: {np.min(accuracies):.2%}",
            f"  Max Accuracy: {np.max(accuracies):.2%}",
            f"  Accuracy Range: {np.max(accuracies) - np.min(accuracies):.2%}",
        ])
    
    # Dual Encryption Analysis
    elif 'layer_results' in results:
        # Check what type of layer results we have
        first_layer_key = list(results['layer_results'].keys())[0]
        first_layer_data = results['layer_results'][first_layer_key]

        if isinstance(first_layer_data, dict) and 'security_results' in first_layer_data:
            # Dual encryption security analysis
            report_lines.extend([
                "DUAL ENCRYPTION SECURITY ANALYSIS",
                "-" * 45,
                f"Layers Analyzed: {list(results['layer_results'].keys())}",
                f"Analysis Type: Key Guessing Attack Resistance",
                "",
                "Security Analysis Results:",
            ])

            for layer_idx, layer_data in results['layer_results'].items():
                security_results = layer_data['security_results']

                # Calculate attack statistics
                acm_attack_accuracies = [r['accuracy'] for r in security_results['acm_attack_results']]
                perm_attack_accuracies = [r['accuracy'] for r in security_results['permutation_attack_results']]

                report_lines.extend([
                    f"",
                    f"Layer {layer_idx}:",
                    f"  Original Accuracy: {security_results['original_accuracy']:.2%}",
                    f"  Dual Encrypted Accuracy: {security_results['dual_encrypted_accuracy']:.2%}",
                    f"  Encryption Drop: {security_results['original_accuracy'] - security_results['dual_encrypted_accuracy']:.2%}",
                    "",
                    f"  ACM Key Guessing Attacks ({len(acm_attack_accuracies)} variants):",
                    f"    Mean Attack Accuracy: {np.mean(acm_attack_accuracies):.2%}",
                    f"    Max Attack Accuracy: {max(acm_attack_accuracies):.2%}",
                    f"    Min Attack Accuracy: {min(acm_attack_accuracies):.2%}",
                    "",
                    f"  Permutation Key Guessing Attacks ({len(perm_attack_accuracies)} variants):",
                    f"    Mean Attack Accuracy: {np.mean(perm_attack_accuracies):.2%}",
                    f"    Max Attack Accuracy: {max(perm_attack_accuracies):.2%}",
                    f"    Min Attack Accuracy: {min(perm_attack_accuracies):.2%}",
                    "",
                    f"  Security Assessment:",
                    f"    Best Attack vs Original: {max(max(acm_attack_accuracies), max(perm_attack_accuracies)) / security_results['original_accuracy']:.1%} relative success",
                    f"    Security Gap: {security_results['original_accuracy'] - max(max(acm_attack_accuracies), max(perm_attack_accuracies)):.2%}",
                ])

        elif isinstance(first_layer_data, dict) and 'method_results' in first_layer_data:
            # Method comparison analysis
            report_lines.extend([
                "DUAL ENCRYPTION METHOD COMPARISON",
                "-" * 45,
                f"Layers Analyzed: {list(results['layer_results'].keys())}",
                "",
                "Method Comparison Summary:",
            ])

            for layer_idx, layer_data in results['layer_results'].items():
                method_results = layer_data['method_results']

                report_lines.extend([
                    f"",
                    f"Layer {layer_idx}:",
                    f"  Initial Accuracy: {layer_data['initial_accuracy']:.2%}",
                    f"  Arnold Only: {method_results['arnold_only']['accuracy']:.2%} "
                    f"(Drop: {method_results['arnold_only']['accuracy_drop']:.2%})",
                    f"  Permutation Only: {method_results['permutation_only']['accuracy']:.2%} "
                    f"(Drop: {method_results['permutation_only']['accuracy_drop']:.2%})",
                    f"  Dual Encryption: {method_results['dual_encryption']['accuracy']:.2%} "
                    f"(Drop: {method_results['dual_encryption']['accuracy_drop']:.2%})",
                ])

        else:
            # Generic layer results
            report_lines.extend([
                "LAYER-BASED ANALYSIS",
                "-" * 25,
                f"Layers Analyzed: {list(results['layer_results'].keys())}",
                "",
                "Analysis Summary:",
            ])

            for layer_idx, layer_data in results['layer_results'].items():
                if isinstance(layer_data, dict):
                    report_lines.append(f"Layer {layer_idx}: {layer_data}")
                else:
                    report_lines.append(f"Layer {layer_idx}: {len(layer_data) if isinstance(layer_data, list) else 'Unknown'} results")
    
    # Handle dual encryption sensitivity results
    elif 'sensitivity_results' in results:
        report_lines.extend([
            "DUAL ENCRYPTION SENSITIVITY ANALYSIS",
            "-" * 45,
            f"Number of Parameter Combinations Tested: {len(results['sensitivity_results'])}",
            "",
            "Sensitivity Analysis Statistics:",
        ])
        
        sensitivity_results = results['sensitivity_results']
        if sensitivity_results:
            accuracies = [r['accuracy'] for r in sensitivity_results]
            accuracy_drops = [r['accuracy_drop'] for r in sensitivity_results]
            key_distances = [r.get('key_distance', 0) for r in sensitivity_results]
            
            report_lines.extend([
                f"  Mean Accuracy: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}",
                f"  Mean Accuracy Drop: {np.mean(accuracy_drops):.2%} ± {np.std(accuracy_drops):.2%}",
                f"  Mean Key Distance: {np.mean(key_distances):.2f} ± {np.std(key_distances):.2f}",
                f"  Min Accuracy: {np.min(accuracies):.2%}",
                f"  Max Accuracy: {np.max(accuracies):.2%}",
            ])
    
    report_lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80
    ])
    
    report_content = "\n".join(report_lines)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    return report_content


class AnalysisVisualizer:
    """
    Comprehensive visualization tool for analysis results.
    
    This class provides a unified interface for creating various types
    of plots and reports from analysis results.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.logger = logging.getLogger(__name__)
    
    def create_comprehensive_report(self,
                                  results: Dict,
                                  output_dir: Path,
                                  report_name: str = "analysis_report") -> None:
        """
        Create a comprehensive report with plots and text analysis.
        
        Args:
            results: Analysis results dictionary
            output_dir: Directory to save the report and plots
            report_name: Base name for the report files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots based on result type
        plot_path = output_dir / f"{report_name}_plots.png"
        comparison_plot_path = output_dir / f"{report_name}_comparison.png"

        # Determine the analysis type to choose appropriate plotting functions
        analysis_type = results.get('analysis_type', '')

        if analysis_type == 'dual_encryption_method_comparison':
            # For method comparison: plots should show method comparison, comparison should show method comparison
            try:
                plot_method_comparison(results, plot_path)
            except Exception as e:
                self.logger.warning(f"Could not generate method comparison plots: {e}")

            try:
                plot_method_comparison(results, comparison_plot_path)
            except Exception as e:
                self.logger.warning(f"Could not generate method comparison plots (comparison): {e}")

        elif analysis_type == 'dual_encryption_security_analysis':
            # For dual encryption security analysis: use dedicated security visualization
            try:
                plot_dual_encryption_security(results, plot_path)
            except Exception as e:
                self.logger.warning(f"Could not generate dual encryption security plots: {e}")

            # Skip comparison plot for dual encryption security analysis to avoid duplication
            # The main plot already shows all the necessary security analysis information

        elif analysis_type in ['dual_encryption_sensitivity', 'acm_key_sensitivity', 'single_layer_permutation_sensitivity']:
            # For sensitivity analyses: plots should show sensitivity, comparison should show method comparison if applicable
            try:
                plot_sensitivity_results(results, plot_path)
            except Exception as e:
                self.logger.warning(f"Could not generate sensitivity plots: {e}")

            # Try method comparison for comparison plot
            if ('layer_results' in results or 'sensitivity_results' in results):
                try:
                    plot_method_comparison(results, comparison_plot_path)
                except Exception as e:
                    self.logger.warning(f"Could not generate comparison plots: {e}")
        else:
            # Fallback: try both approaches
            try:
                plot_sensitivity_results(results, plot_path)
            except Exception as e:
                self.logger.warning(f"Could not generate sensitivity plots: {e}")

            if ('layer_results' in results or 'sensitivity_results' in results):
                try:
                    plot_method_comparison(results, comparison_plot_path)
                except Exception as e:
                    self.logger.warning(f"Could not generate comparison plots: {e}")
        
        # Generate text report
        report_path = output_dir / f"{report_name}.txt"
        generate_analysis_report(results, report_path)
        
        # Save results as JSON
        json_path = output_dir / f"{report_name}_data.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Comprehensive report created in {output_dir}")

        # Log files created based on analysis type
        files_created = [f"{report_name}.txt", f"{report_name}_plots.png", f"{report_name}_data.json"]
        if analysis_type != 'dual_encryption_security_analysis':
            files_created.append(f"{report_name}_comparison.png")

        self.logger.info(f"Files created: {', '.join(files_created)}")
