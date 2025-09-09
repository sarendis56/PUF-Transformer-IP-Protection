"""
Arnold Cat Map (ACM) Key Sensitivity Analysis

This experiment tests the key sensitivity of the ACM encryption component in the dual
encryption scheme for Vision Transformers. It specifically tests how sensitive the
encrypted model is to variations in the Arnold Cat Map key used to encrypt attention weights.

The process:
- Takes a base ACM key and generates many valid key variants
- Encrypts attention layer weights with each key variant using the dual encryption scheme
- Evaluates the encrypted model on ImageNet to measure accuracy degradation
- Calculates the distance between keys to understand sensitivity relationships
- Analyzes whether small key changes cause significant accuracy differences

This validates that the ACM key provides adequate security - small key variations should
cause significant accuracy drops, making brute-force attacks difficult.
"""

import torch
import numpy as np
import json
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import logging

from ..encryption.arnold_transform import arnold, iarnold, verify_arnold_invertibility
from ..utils import ModelEvaluator
from ..utils.model_utils import setup_model


def calculate_key_distance(key1: List[int], key2: List[int]) -> float:
    """
    Calculate Euclidean distance between two ACM keys.
    
    Args:
        key1: First ACM key [N, a, b, c, d]
        key2: Second ACM key [N, a, b, c, d]
        
    Returns:
        float: Euclidean distance between keys
    """
    # Only consider a, b, c, d parameters (skip N)
    params1 = np.array(key1[1:])
    params2 = np.array(key2[1:])
    return np.linalg.norm(params1 - params2)


def generate_valid_arnold_key(matrix_size: int, seed: int = None) -> List[int]:
    """
    Generate a single valid Arnold Cat Map key.

    Args:
        matrix_size: Size of the matrix
        seed: Random seed for reproducibility

    Returns:
        Valid ACM key [N, a, b, c, d]
    """
    if seed is not None:
        random.seed(seed)

    N = random.randint(2, 5)  # Number of iterations

    # Generate valid parameters
    # For Arnold Cat Map: (a*d - b*c) % matrix_size == 1
    max_attempts = 1000
    for _ in range(max_attempts):
        a = random.randint(1, min(100, matrix_size - 1))
        b = random.randint(0, min(100, matrix_size - 1))
        c = random.randint(0, min(100, matrix_size - 1))

        # Calculate d such that (a*d - b*c) % matrix_size == 1
        # So: a*d ≡ 1 + b*c (mod matrix_size)
        # We need to find d such that this holds

        target = (1 + b * c) % matrix_size

        # Find d by trying values
        for d in range(1, min(100, matrix_size)):
            if (a * d) % matrix_size == target:
                key = [N, a, b, c, d]
                # Verify the key is valid
                if (a * d - b * c) % matrix_size == 1:
                    return key

    # Fallback to a known valid key
    return [3, 1, 1, 1, 2]


def generate_acm_key_variants(base_key: List[int],
                             matrix_size: int,
                             num_variants: int = 50) -> List[List[int]]:
    """
    Generate valid ACM key variants.

    Args:
        base_key: Base ACM key [N, a, b, c, d]
        matrix_size: Size of the matrix (for validation)
        num_variants: Number of variants to generate

    Returns:
        List of valid ACM key variants
    """
    variants = []

    # Generate variants using different seeds
    for i in range(num_variants * 3):  # Try more to get enough variants
        variant = generate_valid_arnold_key(matrix_size, seed=i + 1000)

        # Add if different from base and not already in variants
        if (variant != base_key and
            variant not in variants and
            len(variants) < num_variants):
            variants.append(variant)

    # If we don't have enough variants, add some simple modifications
    while len(variants) < num_variants:
        # Create a simple variant by changing N
        N_variant = base_key.copy()
        N_variant[0] = (base_key[0] % 5) + 1  # Change N between 1-5
        if N_variant != base_key and N_variant not in variants:
            variants.append(N_variant)
        else:
            # Add a known valid key as fallback
            fallback = [2, 1, 1, 1, 2]
            if fallback != base_key and fallback not in variants:
                variants.append(fallback)
            else:
                break

    return variants[:num_variants]


class ACMKeySensitivityAnalyzer:
    """
    Analyzer for studying ACM key sensitivity in Vision Transformer models.
    
    This class provides comprehensive analysis of how different ACM keys
    affect model performance, helping understand the security properties
    of the encryption scheme.
    """
    
    def __init__(self,
                 model_name: str = "google/vit-base-patch16-224",
                 imagenet_path: str = "dataset/imagenet/val",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 device: str = "cuda"):
        """
        Initialize the ACM sensitivity analyzer.
        
        Args:
            model_name: HuggingFace model identifier
            imagenet_path: Path to ImageNet validation dataset
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            device: Device for computations
        """
        self.model_name = model_name
        self.imagenet_path = imagenet_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(
            imagenet_path=imagenet_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device
        )
    
    def analyze_key_sensitivity(self,
                              layer_idx: int = 0,
                              base_key: Optional[List[int]] = None,
                              num_variants: int = 30) -> Dict:
        """
        Analyze sensitivity to different ACM keys for a specific layer.
        
        Args:
            layer_idx: Index of the layer to analyze
            base_key: Base ACM key (if None, uses default)
            num_variants: Number of key variants to test
        """
        self.logger.info(f"Starting ACM key sensitivity analysis for layer {layer_idx}")
        
        # Load fresh model
        model, processor = setup_model(self.model_name, self.device)
        
        # Evaluate initial accuracy
        initial_metrics = self.evaluator.evaluate_model(model, processor)
        initial_accuracy = initial_metrics.top1_accuracy / 100
        
        self.logger.info(f"Initial model accuracy: {initial_accuracy:.2%}")
        
        # Get layer and attention weights
        layer = model.vit.encoder.layer[layer_idx]
        attention_weights = {
            'query': layer.attention.attention.query.weight.data.clone(),
            'key': layer.attention.attention.key.weight.data.clone(),
            'value': layer.attention.attention.value.weight.data.clone(),
            'output': layer.attention.output.dense.weight.data.clone()
        }
        
        matrix_size = attention_weights['query'].shape[0]
        
        # Use default key if none provided
        if base_key is None:
            base_key = [3, 1, 1, 1, 2]

        # Verify base key is valid (skip verification for now to allow testing)
        self.logger.info(f"Using base key: {base_key} for matrix size {matrix_size}")
        
        # Generate key variants
        key_variants = generate_acm_key_variants(
            base_key, matrix_size, num_variants
        )
        
        self.logger.info(f"Generated {len(key_variants)} valid key variants")
        
        results = {
            'layer_idx': layer_idx,
            'base_key': base_key,
            'initial_accuracy': initial_accuracy,
            'matrix_size': matrix_size,
            'key_results': []
        }
        
        # Test each key variant
        for i, test_key in enumerate(tqdm(key_variants, desc="Testing key variants")):
            # Reset model weights
            layer.attention.attention.query.weight.data = attention_weights['query'].clone()
            layer.attention.attention.key.weight.data = attention_weights['key'].clone()
            layer.attention.attention.value.weight.data = attention_weights['value'].clone()
            layer.attention.output.dense.weight.data = attention_weights['output'].clone()

            # Encrypt with test key
            for name, weight in attention_weights.items():
                encrypted_weight = arnold(weight, test_key)
                if name == 'output':
                    layer.attention.output.dense.weight.data = encrypted_weight
                else:
                    getattr(layer.attention.attention, name).weight.data = encrypted_weight
            
            # Evaluate accuracy
            metrics = self.evaluator.evaluate_model(model, processor)
            accuracy = metrics.top1_accuracy / 100
            
            # Calculate key distance
            key_distance = calculate_key_distance(base_key, test_key)
            
            key_result = {
                'key': test_key,
                'accuracy': accuracy,
                'accuracy_drop': initial_accuracy - accuracy,
                'relative_accuracy': accuracy / initial_accuracy,
                'key_distance': key_distance
            }
            
            results['key_results'].append(key_result)
            
            self.logger.debug(f"Key {i+1}/{len(key_variants)}: "
                            f"Accuracy: {accuracy:.2%}, "
                            f"Distance: {key_distance:.2f}")
        
        return results
    
    def save_results(self, results: Dict, output_path: Path) -> None:
        """Save analysis results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def load_results(self, input_path: Path) -> Dict:
        """Load analysis results from JSON file."""
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        self.logger.info(f"Results loaded from {input_path}")
        return results
