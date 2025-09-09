"""
Neural Network Weight Encryption Package

This package provides comprehensive encryption capabilities for protecting
intellectual property in neural network models, specifically designed for
Vision Transformers.

Main Components:
    - Arnold Cat Map (ACM) encryption for attention weights
    - Permutation-based encryption for FFN weights
    - Combined dual encryption strategies
    - Key management and security utilities

Example Usage:
    >>> from src.encryption import DualEncryption
    >>> from src.encryption.arnold_transform import get_standard_key
    >>> 
    >>> # Initialize dual encryption
    >>> encryptor = DualEncryption(
    ...     arnold_key=get_standard_key('default'),
    ...     password="my_secret_password"
    ... )
    >>> 
    >>> # Encrypt a transformer layer
    >>> encrypted_weights = encryptor.encrypt_layer_weights(
    ...     attention_weights, ffn_weights
    ... )
"""

from .arnold_transform import (
    arnold,
    iarnold,
    generate_arnold_key,
    verify_arnold_invertibility,
    get_standard_key,
    STANDARD_ARNOLD_KEYS
)

from .permutation import (
    generate_permutation_matrix,
    encrypt_ffn_weight_row_permutation,
    decrypt_ffn_weight_row_permutation,
    encrypt_ffn_weight_col_permutation,
    decrypt_ffn_weight_col_permutation,
    generate_multiple_permutation_matrices,
    verify_permutation_invertibility
)

from .dual_encryption import (
    DualEncryption,
    EncryptionConfig,
    LayerEncryptionResult
)

__all__ = [
    # Arnold Transform
    'arnold',
    'iarnold', 
    'generate_arnold_key',
    'verify_arnold_invertibility',
    'get_standard_key',
    'STANDARD_ARNOLD_KEYS',
    
    # Permutation
    'generate_permutation_matrix',
    'encrypt_ffn_weight_row_permutation',
    'decrypt_ffn_weight_row_permutation', 
    'encrypt_ffn_weight_col_permutation',
    'decrypt_ffn_weight_col_permutation',
    'generate_multiple_permutation_matrices',
    'verify_permutation_invertibility',
    
    # Dual Encryption
    'DualEncryption',
    'EncryptionConfig',
    'LayerEncryptionResult'
]

__version__ = "1.0.0"
__author__ = "IP Protection Research Team"
__description__ = "Neural Network Weight Encryption for IP Protection"
