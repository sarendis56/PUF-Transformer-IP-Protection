"""
Analysis Package for Transformer Weight Encryption Testing

This package tests the robustness of a specific weight encryption scheme for Vision Transformers.
The scheme uses dual encryption where:
- Attention weights are encrypted using Arnold Cat Map (ACM)
- FFN weights are permuted using permutation matrices

Without the correct key, the encrypted model produces random predictions. The experiments
test various aspects of this encryption scheme to ensure it's secure and robust.

The experiments included:
- ACM Key Sensitivity: Tests how sensitive the encrypted model is to different ACM keys
- Permutation Sensitivity: Tests the impact of different permutation matrices on the encrypted model
- Dual Encryption Analysis: Compares individual encryption methods vs the combined dual approach
- Visualization: Creates charts and reports showing encryption effectiveness

These experiments validate that the encryption scheme provides strong IP protection while
allowing controlled access through proper key management.
"""

from .acm_sensitivity import (
    ACMKeySensitivityAnalyzer,
    generate_acm_key_variants,
    calculate_key_distance
)

from .permutation_sensitivity import (
    PermutationSensitivityAnalyzer,
    analyze_permutation_impact
)

from .dual_encryption_analysis import (
    DualEncryptionAnalyzer,
    analyze_dual_encryption_security
)

from .decrypt_one_layer_analysis import (
    DecryptOneLayerAnalyzer
)

from .visualization import (
    AnalysisVisualizer,
    plot_sensitivity_results,
    generate_analysis_report
)

__all__ = [
    'ACMKeySensitivityAnalyzer',
    'generate_acm_key_variants', 
    'calculate_key_distance',
    'PermutationSensitivityAnalyzer',
    'analyze_permutation_impact',
    'DualEncryptionAnalyzer',
    'compare_encryption_methods',
    'AnalysisVisualizer',
    'plot_sensitivity_results',
    'generate_analysis_report'
]
