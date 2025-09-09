"""
Dual Encryption Security Analysis

This experiment evaluates the security of dual encryption against key guessing attacks.
It tests whether an attacker who knows one key (either ACM or permutation) can achieve
higher accuracy than with both keys unknown, which would indicate vulnerability to
partial key recovery attacks.

The process:
- Encrypts both attention weights (ACM) and FFN weights (permutation)
- Tests accuracy with both keys unknown (full encryption)
- Tests accuracy when attacker knows ACM key but guesses permutation key
- Tests accuracy when attacker knows permutation key but guesses ACM key
- Measures if partial key knowledge provides significant advantage

This validates that the dual encryption approach provides enhanced security through
complementary protection mechanisms, making the overall scheme more robust against
various attack vectors.
"""

import torch
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import logging

from ..encryption import DualEncryption
from ..encryption.arnold_transform import arnold, get_standard_key
from ..encryption.permutation import generate_permutation_matrix
from ..utils.cifar100_eval import CIFAR100ModelEvaluator
from ..utils.imagenet_eval import ModelEvaluator
from ..utils.model_utils import setup_model


def analyze_dual_encryption_security(model: torch.nn.Module,
                                    evaluator: ModelEvaluator,
                                    processor,
                                    layer_idx: int = 0,
                                    arnold_key: Optional[List[int]] = None,
                                    permutation_seed: int = 42,
                                    num_attack_variants: int = 10) -> Dict:
    """
    Analyze dual encryption security against key guessing attacks on a single layer.

    Tests three scenarios:
    1. Original model (no encryption)
    2. Dual encrypted model (both ACM + permutation)
    3. Key guessing attacks (attacker tries to guess one key type)

    Args:
        model: Vision Transformer model
        evaluator: Model evaluator
        processor: Image processor
        layer_idx: Index of the layer to analyze
        arnold_key: Arnold Cat Map key (if None, uses default)
        permutation_seed: Seed for permutation matrix generation
        num_attack_variants: Number of key variants to test in attacks

    Returns:
        Dict containing security analysis results
    """
    logger = logging.getLogger(__name__)

    if arnold_key is None:
        arnold_key = get_standard_key('default')

    # Get initial accuracy (unencrypted)
    initial_metrics = evaluator.evaluate_model(model, processor)
    initial_accuracy = initial_metrics.top1_accuracy / 100

    layer = model.vit.encoder.layer[layer_idx]

    # Store original weights
    original_attention = {
        'query': layer.attention.attention.query.weight.data.clone(),
        'key': layer.attention.attention.key.weight.data.clone(),
        'value': layer.attention.attention.value.weight.data.clone(),
        'output': layer.attention.output.dense.weight.data.clone()
    }

    original_ffn = {
        'intermediate': layer.intermediate.dense.weight.data.clone(),
        'output': layer.output.dense.weight.data.clone()
    }

    results = {
        'layer_idx': layer_idx,
        'initial_accuracy': initial_accuracy,
        'arnold_key': arnold_key,
        'permutation_seed': permutation_seed,
        'security_results': {
            'original_accuracy': initial_accuracy,
            'dual_encrypted_accuracy': 0.0,
            'acm_attack_results': [],
            'permutation_attack_results': []
        }
    }
    
    # Step 1: Apply dual encryption (both ACM + permutation)
    logger.info("Applying dual encryption...")

    # Extract hidden size from model
    hidden_size = layer.attention.attention.query.weight.shape[0]
    P = generate_permutation_matrix(hidden_size, seed=permutation_seed).to(model.device)

    # Encrypt attention weights with ACM
    for name, weight in original_attention.items():
        encrypted_weight = arnold(weight, arnold_key)
        if name == 'output':
            layer.attention.output.dense.weight.data = encrypted_weight
        else:
            getattr(layer.attention.attention, name).weight.data = encrypted_weight

    # Encrypt FFN weights with permutation
    from ..encryption.permutation import encrypt_ffn_weight_row_permutation

    encrypted_intermediate = encrypt_ffn_weight_row_permutation(
        original_ffn['intermediate'].transpose(0, 1), P
    ).transpose(0, 1)

    encrypted_output = encrypt_ffn_weight_row_permutation(
        original_ffn['output'], P
    )

    layer.intermediate.dense.weight.data = encrypted_intermediate
    layer.output.dense.weight.data = encrypted_output

    # Measure dual encrypted accuracy
    metrics = evaluator.evaluate_model(model, processor)
    dual_encrypted_accuracy = metrics.top1_accuracy / 100
    results['security_results']['dual_encrypted_accuracy'] = dual_encrypted_accuracy

    # Step 2: Simulate ACM key guessing attack
    logger.info("Simulating ACM key guessing attack...")
    # Attacker knows permutation key but tries to guess ACM key
    from .acm_sensitivity import generate_acm_key_variants
    base_arnold_key = get_standard_key('default')
    arnold_variants = generate_acm_key_variants(base_arnold_key, hidden_size, num_attack_variants)

    for i, attack_arnold_key in enumerate(arnold_variants):
        # Restore original attention weights first
        for name, weight in original_attention.items():
            if name == 'output':
                layer.attention.output.dense.weight.data = weight
            else:
                getattr(layer.attention.attention, name).weight.data = weight

        # Apply attacker's guessed ACM key to attention weights
        for name, weight in original_attention.items():
            encrypted_weight = arnold(weight, attack_arnold_key)
            if name == 'output':
                layer.attention.output.dense.weight.data = encrypted_weight
            else:
                getattr(layer.attention.attention, name).weight.data = encrypted_weight

        # FFN weights remain encrypted with known permutation
        # (they are already encrypted from Step 1)

        metrics = evaluator.evaluate_model(model, processor)
        attack_accuracy = metrics.top1_accuracy / 100

        results['security_results']['acm_attack_results'].append({
            'attack_variant': i,
            'arnold_key': attack_arnold_key,
            'accuracy': attack_accuracy,
            'accuracy_drop': initial_accuracy - attack_accuracy,
            'relative_accuracy': attack_accuracy / initial_accuracy
        })

    # Step 3: Simulate permutation key guessing attack
    logger.info("Simulating permutation key guessing attack...")
    # Attacker knows ACM key but tries to guess permutation key

    # First restore attention weights and apply correct ACM encryption
    for name, weight in original_attention.items():
        encrypted_weight = arnold(weight, arnold_key)
        if name == 'output':
            layer.attention.output.dense.weight.data = encrypted_weight
        else:
            getattr(layer.attention.attention, name).weight.data = encrypted_weight

    for attack_seed in range(num_attack_variants):
        # Restore original FFN weights first
        layer.intermediate.dense.weight.data = original_ffn['intermediate']
        layer.output.dense.weight.data = original_ffn['output']

        # Apply attacker's guessed permutation to FFN weights
        P_attack = generate_permutation_matrix(hidden_size, seed=attack_seed).to(model.device)

        encrypted_intermediate = encrypt_ffn_weight_row_permutation(
            original_ffn['intermediate'].transpose(0, 1), P_attack
        ).transpose(0, 1)

        encrypted_output = encrypt_ffn_weight_row_permutation(
            original_ffn['output'], P_attack
        )

        layer.intermediate.dense.weight.data = encrypted_intermediate
        layer.output.dense.weight.data = encrypted_output

        # Attention weights remain encrypted with known ACM key
        # (they are already encrypted from above)

        metrics = evaluator.evaluate_model(model, processor)
        attack_accuracy = metrics.top1_accuracy / 100

        results['security_results']['permutation_attack_results'].append({
            'attack_variant': attack_seed,
            'permutation_seed': attack_seed,
            'accuracy': attack_accuracy,
            'accuracy_drop': initial_accuracy - attack_accuracy,
            'relative_accuracy': attack_accuracy / initial_accuracy
        })

    # Restore all original weights
    for name, weight in original_attention.items():
        if name == 'output':
            layer.attention.output.dense.weight.data = weight
        else:
            getattr(layer.attention.attention, name).weight.data = weight

    layer.intermediate.dense.weight.data = original_ffn['intermediate']
    layer.output.dense.weight.data = original_ffn['output']
    
    return results


class DualEncryptionAnalyzer:
    """
    Analyzer for comprehensive dual encryption analysis.
    
    This class provides tools for comparing and analyzing the effectiveness
    of different encryption methods and their combinations.
    """
    
    def __init__(self,
                 model_name: str = "facebook/deit-tiny-patch16-224",
                 cifar100_path: str = "dataset/cifar100",
                 batch_size: int = 64,
                 num_workers: int = 4,
                 device: str = "cuda"):
        """
        Initialize the dual encryption analyzer.

        Args:
            model_name: HuggingFace model identifier
            cifar100_path: Path to CIFAR-100 dataset
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            device: Device for computations
        """
        self.model_name = model_name
        self.cifar100_path = cifar100_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize evaluator
        self.evaluator = CIFAR100ModelEvaluator(
            cifar100_path=cifar100_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device
        )
    
    def analyze_dual_encryption_security(self,
                                        layer_indices: List[int] = [0, 1, 2],
                                        arnold_keys: Optional[List[List[int]]] = None,
                                        permutation_seeds: Optional[List[int]] = None,
                                        num_attack_variants: int = 10) -> Dict:
        """
        Analyze dual encryption security against key guessing attacks across multiple layers.

        Args:
            layer_indices: List of layer indices to analyze
            arnold_keys: List of Arnold keys to test (if None, uses defaults)
            permutation_seeds: List of permutation seeds (if None, uses defaults)
            num_attack_variants: Number of key variants to test in attacks

        Returns:
            Dict containing security analysis results
        """
        self.logger.info("Starting dual encryption security analysis")

        if arnold_keys is None:
            arnold_keys = [get_standard_key('default')] * len(layer_indices)

        if permutation_seeds is None:
            permutation_seeds = list(range(len(layer_indices)))

        # Load fresh model (with CIFAR-100 configuration)
        model, processor = setup_model(self.model_name, self.device, num_labels=100)

        results = {
            'analysis_type': 'dual_encryption_security_analysis',
            'model_name': self.model_name,
            'layer_results': {}
        }

        for i, layer_idx in enumerate(layer_indices):
            arnold_key = arnold_keys[i % len(arnold_keys)]
            perm_seed = permutation_seeds[i % len(permutation_seeds)]

            self.logger.info(f"Analyzing security for layer {layer_idx}")

            layer_results = analyze_dual_encryption_security(
                model=model,
                evaluator=self.evaluator,
                processor=processor,
                layer_idx=layer_idx,
                arnold_key=arnold_key,
                permutation_seed=perm_seed,
                num_attack_variants=num_attack_variants
            )

            results['layer_results'][layer_idx] = layer_results

        return results
    
    def analyze_dual_encryption_sensitivity(self,
                                          layer_idx: int = 0,
                                          num_arnold_variants: int = 10,
                                          num_permutation_variants: int = 10) -> Dict:
        """
        Analyze dual encryption security against key guessing attacks (legacy method name).

        This method is kept for backward compatibility and calls the new security analysis.

        Args:
            layer_idx: Index of the layer to analyze
            num_arnold_variants: Number of Arnold key variants to test in attacks
            num_permutation_variants: Number of permutation variants to test in attacks

        Returns:
            Dict containing security analysis results
        """
        # Use the maximum of the two variant counts for attack simulation
        num_attack_variants = max(num_arnold_variants, num_permutation_variants)

        return self.analyze_dual_encryption_security(
            layer_indices=[layer_idx],
            num_attack_variants=num_attack_variants
        )['layer_results'][layer_idx]
    
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
