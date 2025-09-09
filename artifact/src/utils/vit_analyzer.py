"""
Vision Transformer Encryption Analyzer

This module provides a comprehensive analyzer for applying and evaluating
encryption techniques on Vision Transformer models. It combines dual encryption
methods with systematic evaluation to measure the impact on model performance.

Key Features:
    - Progressive layer encryption with impact analysis
    - Dual encryption (Arnold + Permutation)
    - Automatic best permutation matrix selection
    - Performance tracking and logging
    - Model saving/loading with encryption metadata
"""

import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import logging
from datetime import datetime
from transformers import ViTForImageClassification, ViTImageProcessor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random

from ..encryption import DualEncryption, EncryptionConfig
from . import ModelEvaluator, PerformanceTracker, AccuracyMetrics
from .model_utils import setup_model, save_encrypted_model, load_encrypted_model


class VitEncryptionAnalyzer:
    """
    Comprehensive analyzer for Vision Transformer encryption experiments.
    
    This class orchestrates the entire encryption process, from initial model
    setup through progressive layer encryption to final evaluation and saving.
    
    Attributes:
        model: The Vision Transformer model being analyzed
        processor: Image processor for the model
        dual_encryptor: Dual encryption system
        evaluator: Model evaluation system
        performance_tracker: Performance tracking system
        logger: Logging system for experiment tracking
    """
    
    def __init__(self,
                 model_name: str = "google/vit-base-patch16-224",
                 batch_size: int = 64,
                 num_workers: int = 8,
                 num_extra_layers: int = 4,
                 password: Optional[str] = None,
                 arnold_key: Optional[List[int]] = None,
                 device: str = "cuda",
                 imagenet_path: str = "dataset/imagenet/val",
                 local_model_path: Optional[str] = None):
        """
        Initialize the ViT encryption analyzer.

        Args:
            model_name: HuggingFace model identifier
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            num_extra_layers: Number of additional security layers to encrypt
            password: Password for permutation matrix generation
            arnold_key: Arnold Cat Map key parameters
            device: Device for computations ('cuda' or 'cpu')
            imagenet_path: Path to ImageNet validation dataset
            local_model_path: Optional path to local model files
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_extra_layers = num_extra_layers
        self.imagenet_path = imagenet_path
        
        # Setup logging
        self.logger = self._setup_logger()

        # Load model first
        self.model, self.processor = setup_model(model_name, self.device, local_model_path)

        # Initialize dual encryption system with model-specific dimensions
        self.dual_encryptor = DualEncryption.from_model(
            model=self.model,
            arnold_key=arnold_key,
            password=password,
            device=device
        )
        self.evaluator = ModelEvaluator(
            imagenet_path=imagenet_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device=self.device
        )
        
        # Initialize performance tracking
        initial_metrics = self.evaluator.evaluate_model(self.model, self.processor)
        self.initial_accuracy = initial_metrics.top1_accuracy / 100
        
        self.performance_tracker = PerformanceTracker(
            initial_accuracy=self.initial_accuracy,
            model_name=model_name,
            experiment_config={
                'batch_size': batch_size,
                'num_workers': num_workers,
                'num_extra_layers': num_extra_layers,
                'password_protected': password is not None,
                'arnold_key': arnold_key or self.dual_encryptor.config.arnold_key,
                'device': device
            }
        )
        
        self.logger.info(f"Initial model accuracy: {self.initial_accuracy:.2%}")
        
        # Track encrypted layers
        self.encrypted_layers = set()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging system for the analyzer."""
        logger = logging.getLogger('VitEncryptionAnalyzer')
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        log_filename = f'vit_encryption_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        fh = logging.FileHandler(log_filename)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def _get_layer_weights(self, layer_idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Extract attention and FFN weights from a specific transformer layer.
        
        Args:
            layer_idx: Index of the transformer layer
            
        Returns:
            Tuple of (attention_weights, ffn_weights) dictionaries
        """
        layer = self.model.vit.encoder.layer[layer_idx]
        
        # Get attention weights
        attention_weights = {
            'query': layer.attention.attention.query.weight.data,
            'key': layer.attention.attention.key.weight.data,
            'value': layer.attention.attention.value.weight.data,
            'output': layer.attention.output.dense.weight.data
        }
        
        # Get FFN weights
        ffn_weights = {
            'intermediate': layer.intermediate.dense.weight.data,
            'output': layer.output.dense.weight.data
        }
        
        return attention_weights, ffn_weights
    
    def _apply_encrypted_weights(self, 
                               layer_idx: int,
                               encrypted_attention: Dict[str, torch.Tensor],
                               encrypted_ffn: Dict[str, torch.Tensor]) -> None:
        """
        Apply encrypted weights to the model.
        
        Args:
            layer_idx: Index of the layer to update
            encrypted_attention: Encrypted attention weights
            encrypted_ffn: Encrypted FFN weights
        """
        layer = self.model.vit.encoder.layer[layer_idx]
        
        # Apply attention weights
        layer.attention.attention.query.weight.data = encrypted_attention['query']
        layer.attention.attention.key.weight.data = encrypted_attention['key']
        layer.attention.attention.value.weight.data = encrypted_attention['value']
        layer.attention.output.dense.weight.data = encrypted_attention['output']
        
        # Apply FFN weights
        layer.intermediate.dense.weight.data = encrypted_ffn['intermediate']
        layer.output.dense.weight.data = encrypted_ffn['output']
    
    def analyze_layer_impacts(self) -> List[Dict]:
        """
        Analyze the impact of encrypting each unencrypted layer.
        
        Returns:
            List of dictionaries containing layer impact analysis results
        """
        layer_impacts = []
        
        # Get the actual number of layers from the model
        num_layers = len(self.model.vit.encoder.layer)
        for layer_idx in range(num_layers):
            if layer_idx in self.encrypted_layers:
                continue
                
            self.logger.info(f"Analyzing impact of encrypting layer {layer_idx}...")
            
            # Get original weights
            attention_weights, ffn_weights = self._get_layer_weights(layer_idx)
            
            # Store original weights for restoration
            original_attention = {k: v.clone() for k, v in attention_weights.items()}
            original_ffn = {k: v.clone() for k, v in ffn_weights.items()}
            
            # Encrypt with default settings (first permutation matrix)
            encryption_result = self.dual_encryptor.encrypt_layer_weights(
                attention_weights, ffn_weights, permutation_matrix_idx=0
            )
            
            # Apply encrypted weights
            self._apply_encrypted_weights(
                layer_idx, 
                encryption_result.encrypted_attention,
                encryption_result.encrypted_ffn
            )
            
            # Evaluate impact
            metrics = self.evaluator.evaluate_model(self.model, self.processor)
            accuracy = metrics.top1_accuracy / 100
            accuracy_drop = self.initial_accuracy - accuracy
            
            layer_impacts.append({
                'layer_idx': layer_idx,
                'accuracy': accuracy,
                'accuracy_drop': accuracy_drop
            })
            
            self.logger.info(f"Layer {layer_idx}: Accuracy: {accuracy:.2%}, Drop: {accuracy_drop:.2%}")
            
            # Restore original weights
            self._apply_encrypted_weights(layer_idx, original_attention, original_ffn)
        
        return layer_impacts
    
    def find_best_permutation(self, layer_idx: int) -> Dict:
        """
        Find the best permutation matrix for a specific layer.
        
        Args:
            layer_idx: Index of the layer to analyze
            
        Returns:
            Dictionary containing the best permutation result
        """
        self.logger.info(f"Finding best permutation matrix for layer {layer_idx}...")
        
        attention_weights, ffn_weights = self._get_layer_weights(layer_idx)
        original_attention = {k: v.clone() for k, v in attention_weights.items()}
        original_ffn = {k: v.clone() for k, v in ffn_weights.items()}
        
        best_result = None
        best_accuracy_drop = -1
        
        for perm_idx in range(len(self.dual_encryptor.permutation_matrices)):
            # Encrypt with current permutation matrix
            encryption_result = self.dual_encryptor.encrypt_layer_weights(
                attention_weights, ffn_weights, permutation_matrix_idx=perm_idx
            )
            
            self._apply_encrypted_weights(
                layer_idx,
                encryption_result.encrypted_attention,
                encryption_result.encrypted_ffn
            )
            
            # Evaluate
            metrics = self.evaluator.evaluate_model(self.model, self.processor)
            accuracy = metrics.top1_accuracy / 100
            accuracy_drop = self.initial_accuracy - accuracy
            
            if accuracy_drop > best_accuracy_drop:
                best_accuracy_drop = accuracy_drop
                best_result = {
                    'perm_matrix_idx': perm_idx,
                    'accuracy': accuracy,
                    'accuracy_drop': accuracy_drop
                }
            
            # Restore original weights for next iteration
            self._apply_encrypted_weights(layer_idx, original_attention, original_ffn)
        
        self.logger.info(f"Best permutation for layer {layer_idx}: "
                        f"Matrix {best_result['perm_matrix_idx']}, "
                        f"Drop: {best_result['accuracy_drop']:.2%}")

        return best_result

    def encrypt_model_progressive(self,
                                output_dir: str = "encrypted_models",
                                save_checkpoints: bool = True) -> PerformanceTracker:
        """
        Progressively encrypt the model using a greedy approach.

        This method encrypts layers one by one, always choosing the layer
        that causes the maximum accuracy drop (most effective encryption).

        Args:
            output_dir: Directory to save encrypted model checkpoints
            save_checkpoints: Whether to save model after each encryption step

        Returns:
            PerformanceTracker: Complete performance tracking data
        """
        output_base = Path(output_dir)
        current_accuracy = self.initial_accuracy

        self.logger.info("Starting progressive model encryption...")

        # Main encryption loop - encrypt until very low accuracy
        while True:
            # Analyze impact of encrypting each remaining layer
            layer_impacts = self.analyze_layer_impacts()

            if not layer_impacts:
                self.logger.info("No more layers available for encryption")
                break

            # Find layer with highest impact (maximum accuracy drop)
            best_layer = max(layer_impacts, key=lambda x: x['accuracy_drop'])
            layer_idx = best_layer['layer_idx']

            # Find best permutation matrix for this layer
            best_perm_result = self.find_best_permutation(layer_idx)

            # Apply the encryption permanently
            attention_weights, ffn_weights = self._get_layer_weights(layer_idx)
            encryption_result = self.dual_encryptor.encrypt_layer_weights(
                attention_weights,
                ffn_weights,
                permutation_matrix_idx=best_perm_result['perm_matrix_idx']
            )

            self._apply_encrypted_weights(
                layer_idx,
                encryption_result.encrypted_attention,
                encryption_result.encrypted_ffn
            )

            # Update tracking
            self.encrypted_layers.add(layer_idx)
            current_accuracy = best_perm_result['accuracy']

            self.performance_tracker.add_encryption_step(
                layer_idx=layer_idx,
                accuracy=current_accuracy,
                permutation_matrix_idx=best_perm_result['perm_matrix_idx'],
                arnold_key=self.dual_encryptor.config.arnold_key
            )

            self.logger.info(f"Encrypted layer {layer_idx} with permutation matrix "
                           f"{best_perm_result['perm_matrix_idx']}")
            self.logger.info(f"Current accuracy: {current_accuracy:.2%}")
            self.logger.info(f"Total accuracy drop: {(self.initial_accuracy - current_accuracy):.2%}")

            # Save checkpoint if requested
            if save_checkpoints:
                checkpoint_dir = output_base / f"checkpoint_layer_{layer_idx}"
                self._save_model_checkpoint(checkpoint_dir)

            # Check stopping condition (very low accuracy)
            if current_accuracy < 0.005:  # 0.5% accuracy
                self.logger.info("Stopping main encryption due to very low accuracy")
                break

        # Add extra security layers
        self._add_extra_security_layers(output_base, save_checkpoints)

        # Save final model
        if save_checkpoints:
            final_dir = output_base / "final"
            self._save_model_checkpoint(final_dir)

        self.logger.info(f"Encryption complete. Final accuracy: {current_accuracy:.2%}")
        self.logger.info(f"Total layers encrypted: {len(self.encrypted_layers)}")

        return self.performance_tracker

    def _add_extra_security_layers(self,
                                 output_base: Path,
                                 save_checkpoints: bool) -> None:
        """
        Add extra security layers by encrypting additional random layers.

        Args:
            output_base: Base directory for saving checkpoints
            save_checkpoints: Whether to save checkpoints
        """
        self.logger.info("Adding extra security layers...")

        # Get the actual number of layers from the model
        num_layers = len(self.model.vit.encoder.layer)
        available_layers = [i for i in range(num_layers) if i not in self.encrypted_layers]

        if not available_layers:
            self.logger.info("No more layers available for extra security encryption")
            return

        # Randomly select layers for extra security
        num_extra = min(self.num_extra_layers, len(available_layers))
        extra_layers = random.sample(available_layers, num_extra)

        for idx, layer_idx in enumerate(extra_layers):
            # Randomly select a permutation matrix
            perm_matrix_idx = random.randint(0, len(self.dual_encryptor.permutation_matrices) - 1)

            # Encrypt the layer
            attention_weights, ffn_weights = self._get_layer_weights(layer_idx)
            encryption_result = self.dual_encryptor.encrypt_layer_weights(
                attention_weights,
                ffn_weights,
                permutation_matrix_idx=perm_matrix_idx
            )

            self._apply_encrypted_weights(
                layer_idx,
                encryption_result.encrypted_attention,
                encryption_result.encrypted_ffn
            )

            self.encrypted_layers.add(layer_idx)

            # Evaluate current accuracy
            metrics = self.evaluator.evaluate_model(self.model, self.processor)
            accuracy = metrics.top1_accuracy / 100

            self.performance_tracker.add_encryption_step(
                layer_idx=layer_idx,
                accuracy=accuracy,
                permutation_matrix_idx=perm_matrix_idx,
                arnold_key=self.dual_encryptor.config.arnold_key
            )

            self.logger.info(f"Extra security layer {idx + 1}/{num_extra}: "
                           f"Encrypted layer {layer_idx} with permutation matrix {perm_matrix_idx}")
            self.logger.info(f"Current accuracy: {accuracy:.2%}")

            # Save checkpoint
            if save_checkpoints:
                checkpoint_dir = output_base / f"extra_security_{idx}"
                self._save_model_checkpoint(checkpoint_dir)

    def _save_model_checkpoint(self, checkpoint_dir: Path) -> None:
        """
        Save model checkpoint with encryption metadata.

        Args:
            checkpoint_dir: Directory to save the checkpoint
        """
        save_encrypted_model(
            model=self.model,
            processor=self.processor,
            performance_tracker=self.performance_tracker,
            dual_encryptor=self.dual_encryptor,
            output_dir=checkpoint_dir
        )
