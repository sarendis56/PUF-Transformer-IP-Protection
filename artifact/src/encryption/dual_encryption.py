"""
Combines Arnold Cat Map (ACM) encryption for attention weights with permutation-based
encryption for Feed-Forward Network (FFN) weights.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import logging

from .arnold_transform import arnold, iarnold, get_standard_key
from .permutation import (
    generate_multiple_permutation_matrices,
    encrypt_ffn_weight_row_permutation,
    decrypt_ffn_weight_row_permutation
)


@dataclass
class EncryptionConfig:
    """Configuration for dual encryption system."""
    arnold_key: List[int]
    password: Optional[str] = None
    num_permutation_matrices: int = 6
    matrix_size: int = 768
    device: str = 'cuda'
    dtype: torch.dtype = torch.float32


class LayerEncryptionResult(NamedTuple):
    """Result of encrypting a single transformer layer."""
    encrypted_attention: Dict[str, torch.Tensor]
    encrypted_ffn: Dict[str, torch.Tensor]
    permutation_matrix_idx: int
    arnold_key: List[int]


class DualEncryption:
    """
    Dual encryption system for Vision Transformer layers.
    
    This class provides a unified interface for encrypting and decrypting
    transformer layers using both Arnold Cat Map and permutation-based methods.
    
    Attributes:
        config: Encryption configuration
        permutation_matrices: List of generated permutation matrices
        logger: Logger for tracking operations
    """
    
    def __init__(self,
                 arnold_key: Optional[List[int]] = None,
                 password: Optional[str] = None,
                 num_permutation_matrices: int = 6,
                 matrix_size: int = 768,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the dual encryption system.

        Args:
            arnold_key: Arnold Cat Map key parameters [N, a, b, c, d]
            password: Password for permutation matrix generation
            num_permutation_matrices: Number of permutation matrices to generate
            matrix_size: Size of permutation matrices (hidden_size of the model)
                        e.g., 768 for ViT-base, 1024 for ViT-large
            device: Device for tensor operations
            dtype: Data type for tensors
        """
        # Use default Arnold key if none provided
        if arnold_key is None:
            arnold_key = get_standard_key('default')
            
        self.config = EncryptionConfig(
            arnold_key=arnold_key,
            password=password,
            num_permutation_matrices=num_permutation_matrices,
            matrix_size=matrix_size,
            device=device,
            dtype=dtype
        )
        
        # Generate permutation matrices
        self.permutation_matrices = generate_multiple_permutation_matrices(
            num_matrices=num_permutation_matrices,
            size=matrix_size,
            password=password,
            device=device
        )
        
        # Convert to specified dtype
        for i, matrix in enumerate(self.permutation_matrices):
            self.permutation_matrices[i] = matrix.to(dtype=dtype)
        
        self.logger = logging.getLogger(__name__)

    @classmethod
    def from_model(cls,
                   model,
                   arnold_key: Optional[List[int]] = None,
                   password: Optional[str] = None,
                   num_permutation_matrices: int = 6,
                   device: str = 'cuda',
                   dtype: torch.dtype = torch.float32):
        """
        Create a DualEncryption instance with dimensions extracted from a model.

        Args:
            model: Vision Transformer model (ViTForImageClassification)
            arnold_key: Arnold Cat Map key parameters [N, a, b, c, d]
            password: Password for permutation matrix generation
            num_permutation_matrices: Number of permutation matrices to generate
            device: Device for tensor operations
            dtype: Data type for tensors

        Returns:
            DualEncryption: Configured encryption system
        """
        # Extract hidden size from the model configuration
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            matrix_size = model.config.hidden_size
        else:
            # Fallback: extract from actual layer weights
            first_layer = model.vit.encoder.layer[0]
            matrix_size = first_layer.attention.attention.query.weight.shape[0]

        return cls(
            arnold_key=arnold_key,
            password=password,
            num_permutation_matrices=num_permutation_matrices,
            matrix_size=matrix_size,
            device=device,
            dtype=dtype
        )

    def encrypt_attention_weights(self,
                                attention_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encrypt attention weights using Arnold Cat Map.
        
        Args:
            attention_weights: Dictionary containing attention weight tensors
                             (query, key, value, output)
        
        Returns:
            Dict[str, torch.Tensor]: Encrypted attention weights
        """
        encrypted_attention = {}
        
        for name, weight in attention_weights.items():
            # Convert to numpy for Arnold transform, then back to tensor
            weight_np = weight.detach().cpu().numpy()
            encrypted_np = arnold(weight_np, self.config.arnold_key)

            # Convert back to tensor with original dtype and device
            encrypted_attention[name] = encrypted_np.clone().detach().to(
                dtype=self.config.dtype,
                device=self.config.device
            )
            
        return encrypted_attention
    
    def decrypt_attention_weights(self, 
                                encrypted_attention: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Decrypt attention weights using inverse Arnold Cat Map.
        
        Args:
            encrypted_attention: Dictionary containing encrypted attention weights
        
        Returns:
            Dict[str, torch.Tensor]: Decrypted attention weights
        """
        decrypted_attention = {}
        
        for name, weight in encrypted_attention.items():
            decrypted_tensor = iarnold(weight, self.config.arnold_key)
            decrypted_attention[name] = decrypted_tensor.clone().detach()
            
        return decrypted_attention
    
    def encrypt_ffn_weights(self, 
                          ffn_weights: Dict[str, torch.Tensor],
                          permutation_matrix_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Encrypt FFN weights using row permutation.
        
        Args:
            ffn_weights: Dictionary containing FFN weight tensors
                        (intermediate, output)
            permutation_matrix_idx: Index of permutation matrix to use
        
        Returns:
            Dict[str, torch.Tensor]: Encrypted FFN weights
        """
        if permutation_matrix_idx >= len(self.permutation_matrices):
            raise ValueError(f"Permutation matrix index {permutation_matrix_idx} "
                           f"out of range (0-{len(self.permutation_matrices)-1})")
        
        perm_matrix = self.permutation_matrices[permutation_matrix_idx]
        encrypted_ffn = {}
        
        # Encrypt intermediate layer (768 x 3072) -> transpose -> encrypt -> transpose back
        if 'intermediate' in ffn_weights:
            intermediate_weight = ffn_weights['intermediate']
            encrypted_ffn['intermediate'] = encrypt_ffn_weight_row_permutation(
                intermediate_weight.transpose(0, 1),
                perm_matrix.to(dtype=intermediate_weight.dtype)
            ).transpose(0, 1)
        
        # Encrypt output layer (3072 x 768) -> encrypt directly
        if 'output' in ffn_weights:
            output_weight = ffn_weights['output']
            encrypted_ffn['output'] = encrypt_ffn_weight_row_permutation(
                output_weight,
                perm_matrix.to(dtype=output_weight.dtype)
            )
            
        return encrypted_ffn
    
    def decrypt_ffn_weights(self, 
                          encrypted_ffn: Dict[str, torch.Tensor],
                          permutation_matrix_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Decrypt FFN weights using inverse row permutation.
        
        Args:
            encrypted_ffn: Dictionary containing encrypted FFN weights
            permutation_matrix_idx: Index of permutation matrix used for encryption
        
        Returns:
            Dict[str, torch.Tensor]: Decrypted FFN weights
        """
        if permutation_matrix_idx >= len(self.permutation_matrices):
            raise ValueError(f"Permutation matrix index {permutation_matrix_idx} "
                           f"out of range (0-{len(self.permutation_matrices)-1})")
        
        perm_matrix = self.permutation_matrices[permutation_matrix_idx]
        decrypted_ffn = {}
        
        # Decrypt intermediate layer
        if 'intermediate' in encrypted_ffn:
            intermediate_weight = encrypted_ffn['intermediate']
            decrypted_ffn['intermediate'] = decrypt_ffn_weight_row_permutation(
                intermediate_weight.transpose(0, 1),
                perm_matrix
            ).transpose(0, 1)
        
        # Decrypt output layer
        if 'output' in encrypted_ffn:
            output_weight = encrypted_ffn['output']
            decrypted_ffn['output'] = decrypt_ffn_weight_row_permutation(
                output_weight,
                perm_matrix
            )
            
        return decrypted_ffn
    
    def encrypt_layer_weights(self, 
                            attention_weights: Dict[str, torch.Tensor],
                            ffn_weights: Dict[str, torch.Tensor],
                            permutation_matrix_idx: int = 0) -> LayerEncryptionResult:
        """
        Encrypt both attention and FFN weights for a complete transformer layer.
        
        Args:
            attention_weights: Attention weight tensors
            ffn_weights: FFN weight tensors  
            permutation_matrix_idx: Index of permutation matrix to use
        
        Returns:
            LayerEncryptionResult: Complete encryption result
        """
        encrypted_attention = self.encrypt_attention_weights(attention_weights)
        encrypted_ffn = self.encrypt_ffn_weights(ffn_weights, permutation_matrix_idx)
        
        return LayerEncryptionResult(
            encrypted_attention=encrypted_attention,
            encrypted_ffn=encrypted_ffn,
            permutation_matrix_idx=permutation_matrix_idx,
            arnold_key=self.config.arnold_key.copy()
        )
    
    def decrypt_layer_weights(self, 
                            encryption_result: LayerEncryptionResult) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Decrypt both attention and FFN weights for a complete transformer layer.
        
        Args:
            encryption_result: Result from encrypt_layer_weights
        
        Returns:
            Tuple containing (decrypted_attention, decrypted_ffn)
        """
        decrypted_attention = self.decrypt_attention_weights(
            encryption_result.encrypted_attention
        )
        decrypted_ffn = self.decrypt_ffn_weights(
            encryption_result.encrypted_ffn,
            encryption_result.permutation_matrix_idx
        )
        
        return decrypted_attention, decrypted_ffn
    
    def verify_encryption_cycle(self, 
                              attention_weights: Dict[str, torch.Tensor],
                              ffn_weights: Dict[str, torch.Tensor],
                              permutation_matrix_idx: int = 0,
                              tolerance: float = 1e-5) -> bool:
        """
        Verify that encryption followed by decryption recovers original weights.
        
        Args:
            attention_weights: Original attention weights
            ffn_weights: Original FFN weights
            permutation_matrix_idx: Permutation matrix index to test
            tolerance: Numerical tolerance for comparison
        
        Returns:
            bool: True if encryption cycle is successful
        """
        # Store original weights
        original_attention = {k: v.clone() for k, v in attention_weights.items()}
        original_ffn = {k: v.clone() for k, v in ffn_weights.items()}
        
        # Encrypt
        encryption_result = self.encrypt_layer_weights(
            attention_weights, ffn_weights, permutation_matrix_idx
        )
        
        # Decrypt
        decrypted_attention, decrypted_ffn = self.decrypt_layer_weights(encryption_result)
        
        # Verify attention weights
        for name in original_attention:
            if not torch.allclose(original_attention[name], decrypted_attention[name], atol=tolerance):
                self.logger.warning(f"Attention weight {name} verification failed")
                return False
        
        # Verify FFN weights
        for name in original_ffn:
            if not torch.allclose(original_ffn[name], decrypted_ffn[name], atol=tolerance):
                self.logger.warning(f"FFN weight {name} verification failed")
                return False
        
        return True
