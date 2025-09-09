"""
DeiT Encryption Analyzer for CIFAR-100

This module provides encryption analysis functionality for DeiT models on CIFAR-100 dataset,
adapted from the ViT analyzer for ImageNet.
"""

import torch
import logging
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from transformers import DeiTForImageClassification, ViTForImageClassification, ViTImageProcessor

from .cifar100_eval import CIFAR100ModelEvaluator
from ..encryption.dual_encryption import DualEncryption
from ..utils.metrics import PerformanceTracker


class DeiTEncryptionAnalyzer:
    """
    Comprehensive encryption analyzer for DeiT models on CIFAR-100.
    
    This class provides functionality to encrypt DeiT models progressively and analyze
    the impact on CIFAR-100 classification performance.
    """
    
    def __init__(self,
                 model_name: str = "facebook/deit-tiny-patch16-224",
                 batch_size: int = 64,
                 num_workers: int = 8,
                 num_extra_layers: int = 2,
                 password: Optional[str] = None,
                 arnold_key: Optional[List[int]] = None,
                 device: str = "cuda",
                 cifar100_path: str = "dataset/cifar100",
                 local_model_path: Optional[str] = None):
        """
        Initialize the DeiT encryption analyzer.

        Args:
            model_name: HuggingFace model identifier
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            num_extra_layers: Number of additional security layers to encrypt
            password: Password for permutation matrix generation
            arnold_key: Arnold Cat Map key parameters
            device: Device for computations ('cuda' or 'cpu')
            cifar100_path: Path to CIFAR-100 dataset
            local_model_path: Optional path to local model files
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_extra_layers = num_extra_layers
        self.cifar100_path = cifar100_path
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Load model and processor
        self.logger.info(f"Loading model: {model_name}")
        if local_model_path and Path(local_model_path).exists():
            # For fine-tuned models, use ViTForImageClassification with CIFAR-100 config
            from transformers import ViTConfig
            config = ViTConfig.from_pretrained(local_model_path)
            config.num_labels = 100  # CIFAR-100 has 100 classes
            self.model = ViTForImageClassification.from_pretrained(local_model_path, config=config)
            self.processor = ViTImageProcessor.from_pretrained(local_model_path)
            self.logger.info(f"Loaded fine-tuned model from local path: {local_model_path}")
        else:
            # For original HuggingFace models, use DeiTForImageClassification
            self.model = DeiTForImageClassification.from_pretrained(model_name)
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.logger.info(f"Loaded model from HuggingFace: {model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize evaluator
        self.evaluator = CIFAR100ModelEvaluator(
            cifar100_path=cifar100_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device
        )
        
        # Initialize dual encryptor with correct matrix size for the model
        matrix_size = self.model.config.hidden_size
        self.dual_encryptor = DualEncryption(
            arnold_key=arnold_key,
            password=password,
            matrix_size=matrix_size,
            device=device
        )
        
        # Track encrypted layers
        self.encrypted_layers = set()
        
        # Get initial accuracy
        self.logger.info("Evaluating initial model accuracy...")
        initial_metrics = self.evaluator.evaluate_model(self.model, self.processor)
        self.initial_accuracy = initial_metrics.top1_accuracy / 100
        self.logger.info(f"Initial accuracy: {self.initial_accuracy:.2%}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the analyzer."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _get_model_backbone(self):
        """Get the model backbone (either 'deit' or 'vit')."""
        if hasattr(self.model, 'deit'):
            return self.model.deit
        elif hasattr(self.model, 'vit'):
            return self.model.vit
        else:
            raise AttributeError("Model does not have expected 'deit' or 'vit' attribute")
    
    def _get_layer_weights(self, layer_idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Extract attention and FFN weights from a specific layer."""
        backbone = self._get_model_backbone()
        layer = backbone.encoder.layer[layer_idx]
        
        # Attention weights
        attention_weights = {
            'query': layer.attention.attention.query.weight,
            'key': layer.attention.attention.key.weight,
            'value': layer.attention.attention.value.weight,
            'output': layer.attention.output.dense.weight
        }
        
        # FFN weights
        ffn_weights = {
            'intermediate': layer.intermediate.dense.weight,
            'output': layer.output.dense.weight
        }
        
        return attention_weights, ffn_weights
    
    def _apply_encrypted_weights(self, layer_idx: int,
                                encrypted_attention: Dict[str, torch.Tensor],
                                encrypted_ffn: Dict[str, torch.Tensor]):
        """Apply encrypted weights to a specific layer."""
        backbone = self._get_model_backbone()
        layer = backbone.encoder.layer[layer_idx]
        
        # Apply encrypted attention weights
        layer.attention.attention.query.weight.data = encrypted_attention['query']
        layer.attention.attention.key.weight.data = encrypted_attention['key']
        layer.attention.attention.value.weight.data = encrypted_attention['value']
        layer.attention.output.dense.weight.data = encrypted_attention['output']
        
        # Apply encrypted FFN weights
        layer.intermediate.dense.weight.data = encrypted_ffn['intermediate']
        layer.output.dense.weight.data = encrypted_ffn['output']
    
    def analyze_layer_impact(self) -> List[Dict]:
        """Analyze the impact of encrypting each individual layer."""
        layer_impacts = []
        
        # Get the actual number of layers from the model
        backbone = self._get_model_backbone()
        num_layers = len(backbone.encoder.layer)
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
        """Find the best permutation matrix for a given layer."""
        attention_weights, ffn_weights = self._get_layer_weights(layer_idx)
        
        # Store original weights for restoration
        original_attention = {k: v.clone() for k, v in attention_weights.items()}
        original_ffn = {k: v.clone() for k, v in ffn_weights.items()}
        
        best_accuracy_drop = 0
        best_result = None
        
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
        
        return best_result
    
    def encrypt_model_progressive(self, output_dir: Optional[Path] = None, 
                                save_checkpoints: bool = True) -> PerformanceTracker:
        """
        Progressively encrypt the model layers and track performance.
        
        Args:
            output_dir: Directory to save checkpoints
            save_checkpoints: Whether to save model checkpoints
            
        Returns:
            PerformanceTracker: Performance tracking results
        """
        performance_tracker = PerformanceTracker(self.initial_accuracy)
        
        # Get the actual number of layers from the model
        backbone = self._get_model_backbone()
        num_layers = len(backbone.encoder.layer)
        self.logger.info(f"Model has {num_layers} layers")
        
        # Main encryption loop - encrypt until very low accuracy
        current_accuracy = self.initial_accuracy
        step = 0

        self.logger.info("Starting progressive model encryption...")

        while True:
            step += 1

            # Find layer with highest impact
            layer_impacts = self.analyze_layer_impact()
            if not layer_impacts:
                self.logger.info("No more layers available for encryption")
                break

            # Find layer with highest impact (maximum accuracy drop)
            best_layer = max(layer_impacts, key=lambda x: x['accuracy_drop'])
            layer_idx = best_layer['layer_idx']

            self.logger.info(f"\n=== Encryption Step {step} ===")
            self.logger.info(f"Encrypting layer {layer_idx} (expected drop: {best_layer['accuracy_drop']:.2%})")

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

            performance_tracker.add_encryption_step(
                layer_idx=layer_idx,
                accuracy=current_accuracy,
                permutation_matrix_idx=best_perm_result['perm_matrix_idx'],
                arnold_key=self.dual_encryptor.config.arnold_key
            )

            self.logger.info(f"Encrypted layer {layer_idx} with permutation matrix {best_perm_result['perm_matrix_idx']}")
            self.logger.info(f"Current accuracy: {current_accuracy:.2%}")
            self.logger.info(f"Total accuracy drop: {(self.initial_accuracy - current_accuracy):.2%}")

            # Save checkpoint if requested
            if save_checkpoints and output_dir:
                checkpoint_dir = output_dir / f"checkpoint_layer_{layer_idx}"
                self._save_model_checkpoint(checkpoint_dir)

            # Check stopping condition (very low accuracy) - for CIFAR-100, 1.5% is near random
            if current_accuracy < 0.015:  # 1.5% accuracy
                self.logger.info("Stopping main encryption due to very low accuracy (near random level)")
                break

        # Add extra security layers
        self.logger.info(f"\n=== Adding {self.num_extra_layers} extra security layers ===")
        self._add_extra_security_layers(performance_tracker, output_dir, save_checkpoints)

        return performance_tracker
    
    def _add_extra_security_layers(self, performance_tracker: PerformanceTracker, 
                                 output_dir: Optional[Path], save_checkpoints: bool):
        """Add extra security layers by encrypting additional layers randomly."""
        backbone = self._get_model_backbone()
        num_layers = len(backbone.encoder.layer)
        available_layers = [i for i in range(num_layers) if i not in self.encrypted_layers]

        # Randomly select layers for extra security
        num_extra = min(self.num_extra_layers, len(available_layers))
        extra_layers = random.sample(available_layers, num_extra)

        for layer_idx in extra_layers:
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

            # Mark layer as encrypted
            self.encrypted_layers.add(layer_idx)

            # Evaluate performance
            metrics = self.evaluator.evaluate_model(self.model, self.processor)
            accuracy = metrics.top1_accuracy / 100

            # Record performance
            performance_tracker.add_encryption_step(
                layer_idx=layer_idx,
                accuracy=accuracy,
                permutation_matrix_idx=perm_matrix_idx,
                arnold_key=self.dual_encryptor.config.arnold_key
            )

            self.logger.info(f"Added extra security layer {layer_idx} with permutation matrix {perm_matrix_idx}")
            self.logger.info(f"Accuracy: {accuracy:.2%}")

        # Save final checkpoint
        if save_checkpoints and output_dir:
            final_checkpoint_path = output_dir / "final"
            final_checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.model.save_pretrained(final_checkpoint_path / "model")
            self.processor.save_pretrained(final_checkpoint_path / "model")
            
            # Save encryption metadata
            performance_tracker.save_to_file(final_checkpoint_path / "performance.json")
            
            self.logger.info(f"Saved final checkpoint to {final_checkpoint_path}")

    def _save_model_checkpoint(self, checkpoint_dir: Path):
        """Save model checkpoint to specified directory."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and processor
        self.model.save_pretrained(checkpoint_dir / "model")
        self.processor.save_pretrained(checkpoint_dir / "model")

        self.logger.info(f"Saved checkpoint to {checkpoint_dir}")
