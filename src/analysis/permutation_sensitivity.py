"""
Permutation Sensitivity Analysis

This experiment tests the sensitivity of the dual encryption scheme to different permutation
matrices used for encrypting FFN weights. It evaluates how variations in the permutation
component affect the overall security and performance of the encrypted Vision Transformer.

The process:
- Generates different permutation matrices using various random seeds
- Applies permutations to FFN weights within the dual encryption scheme
- Evaluates the fully encrypted model on ImageNet to measure accuracy degradation
- Tests both single-layer and multi-layer encryption scenarios
- Analyzes controlled permutations with systematic seed variations

This validates the robustness of the permutation-based encryption component and ensures
that different permutation matrices provide consistent security guarantees.
"""

import torch
import numpy as np
import json
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import logging

from ..encryption.permutation import (
    generate_permutation_matrix,
    encrypt_ffn_weight_row_permutation,
    decrypt_ffn_weight_row_permutation
)
from ..utils import ModelEvaluator
from ..utils.model_utils import setup_model


def calculate_permutation_distance(P1: torch.Tensor, P2: torch.Tensor) -> float:
    """
    Calculate distance between two permutation matrices.

    Args:
        P1: First permutation matrix
        P2: Second permutation matrix

    Returns:
        float: Frobenius norm distance between matrices
    """
    return torch.norm(P1 - P2, p='fro').item()


def calculate_permutation_intensity(P: torch.Tensor) -> int:
    """
    Calculate permutation intensity (number of elements not in their original positions).

    For a permutation matrix P, this counts how many rows/columns are moved from their
    original positions. Higher intensity means more elements are displaced.

    Args:
        P: Permutation matrix of shape (size, size)

    Returns:
        int: Number of elements not in their original positions

    Example:
        >>> # Identity matrix has intensity 0
        >>> I = torch.eye(4)
        >>> calculate_permutation_intensity(I)  # 0
        >>>
        >>> # Swap first two rows: intensity = 2
        >>> P = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float)
        >>> calculate_permutation_intensity(P)  # 2
    """
    size = P.shape[0]
    identity = torch.eye(size, device=P.device, dtype=P.dtype)

    # Count positions where P differs from identity matrix
    # Each row that differs from identity means that element is displaced
    displaced_rows = 0
    for i in range(size):
        if not torch.allclose(P[i], identity[i], atol=1e-6):
            displaced_rows += 1

    return displaced_rows


def analyze_permutation_impact(model: torch.nn.Module,
                             evaluator: ModelEvaluator,
                             processor,
                             layer_indices: List[int],
                             num_permutations: int = 10) -> Dict:
    """
    Analyze the impact of different permutation matrices on multiple layers.
    
    Args:
        model: Vision Transformer model
        evaluator: Model evaluator
        processor: Image processor
        layer_indices: List of layer indices to analyze
        num_permutations: Number of different permutation matrices to test
        
    Returns:
        Dict containing analysis results
    """
    logger = logging.getLogger(__name__)
    
    # Get initial accuracy
    initial_metrics = evaluator.evaluate_model(model, processor)
    initial_accuracy = initial_metrics.top1_accuracy / 100
    
    results = {
        'initial_accuracy': initial_accuracy,
        'layer_results': {}
    }
    
    for layer_idx in layer_indices:
        logger.info(f"Analyzing permutation impact for layer {layer_idx}")
        
        layer = model.vit.encoder.layer[layer_idx]
        
        # Store original weights
        original_intermediate = layer.intermediate.dense.weight.data.clone()
        original_output = layer.output.dense.weight.data.clone()
        
        layer_results = []
        
        # Test different permutation matrices
        for perm_idx in range(num_permutations):
            # Generate permutation matrix with model-specific dimensions
            hidden_size = layer.attention.attention.query.weight.shape[0]
            P = generate_permutation_matrix(hidden_size, seed=perm_idx).to(model.device)

            # Calculate permutation intensity
            intensity = calculate_permutation_intensity(P)

            # Encrypt FFN weights
            encrypted_intermediate = encrypt_ffn_weight_row_permutation(
                original_intermediate.transpose(0, 1), P
            ).transpose(0, 1)

            encrypted_output = encrypt_ffn_weight_row_permutation(
                original_output, P
            )

            # Apply encrypted weights
            layer.intermediate.dense.weight.data = encrypted_intermediate
            layer.output.dense.weight.data = encrypted_output

            # Evaluate accuracy
            metrics = evaluator.evaluate_model(model, processor)
            accuracy = metrics.top1_accuracy / 100

            layer_results.append({
                'permutation_idx': perm_idx,
                'accuracy': accuracy,
                'accuracy_drop': initial_accuracy - accuracy,
                'relative_accuracy': accuracy / initial_accuracy,
                'permutation_intensity': intensity
            })
            
            # Restore original weights
            layer.intermediate.dense.weight.data = original_intermediate
            layer.output.dense.weight.data = original_output
        
        results['layer_results'][layer_idx] = layer_results
    
    return results


class PermutationSensitivityAnalyzer:
    """
    Analyzer for studying permutation sensitivity in Vision Transformer models.
    
    This class provides comprehensive analysis of how different permutation
    matrices affect model performance across different layers.
    """
    
    def __init__(self,
                 model_name: str = "google/vit-base-patch16-224",
                 imagenet_path: str = "dataset/imagenet/val",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 device: str = "cuda"):
        """
        Initialize the permutation sensitivity analyzer.
        
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
    
    def analyze_permutation_sensitivity(self,
                                      layer_indices: List[int] = [0],
                                      num_permutations: int = 20) -> Dict:
        """
        Analyze permutation sensitivity for specified layers.
        Each layer is analyzed independently with different permutation matrices.

        Args:
            layer_indices: List of layer indices to analyze
            num_permutations: Number of different permutation matrices to test per layer

        Returns:
            Dict containing analysis results
        """
        self.logger.info(f"Starting permutation sensitivity analysis for layers {layer_indices}")

        # Load fresh model
        model, processor = setup_model(self.model_name, self.device)

        # Analyze permutation impact
        results = analyze_permutation_impact(
            model=model,
            evaluator=self.evaluator,
            processor=processor,
            layer_indices=layer_indices,
            num_permutations=num_permutations
        )

        # Add metadata
        results.update({
            'analysis_type': 'permutation_sensitivity',
            'layer_indices': layer_indices,
            'num_permutations': num_permutations,
            'model_name': self.model_name
        })

        return results
    
    def analyze_layers_sensitivity(self,
                                 layer_indices: List[int] = [0, 1, 2],
                                 num_permutations: int = 10) -> Dict:
        """
        Analyze permutation sensitivity across specified layers.
        Each layer is analyzed independently with its own permutation matrices.

        Args:
            layer_indices: List of layer indices to analyze
            num_permutations: Number of different permutation matrices to test per layer

        Returns:
            Dict containing analysis results for all layers
        """
        self.logger.info(f"Starting permutation sensitivity analysis")
        self.logger.info(f"Analyzing layers: {layer_indices}")

        # Load fresh model
        model, processor = setup_model(self.model_name, self.device)

        # Analyze permutation impact
        results = analyze_permutation_impact(
            model=model,
            evaluator=self.evaluator,
            processor=processor,
            layer_indices=layer_indices,
            num_permutations=num_permutations
        )

        # Add metadata
        results.update({
            'analysis_type': 'permutation_sensitivity',
            'layer_indices': layer_indices,
            'num_permutations': num_permutations,
            'model_name': self.model_name
        })

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
