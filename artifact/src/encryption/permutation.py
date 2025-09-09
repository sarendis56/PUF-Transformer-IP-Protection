"""
Permutation-based encryption for Feed-Forward Network (FFN) weights.
- Password-based permutation matrix generation using HMAC-SHA256
- Support for different matrix dimensions (768x3072 for ViT-base)
- Invertible transformations for decryption
- Memory-efficient GPU operations
"""

import torch
import hmac
import hashlib
import numpy as np
from typing import Optional, Union


def generate_permutation_matrix(size: int = 768, 
                              seed: Optional[int] = 42, 
                              password: Optional[str] = None, 
                              index: Optional[int] = None) -> torch.Tensor:
    """
    Generate a permutation matrix for weight encryption.
    
    This function creates a square permutation matrix that can be used to
    scramble the rows or columns of neural network weight matrices. When a
    password is provided, it uses HMAC-SHA256 for cryptographically secure
    permutation generation.
    
    Args:
        size: Dimension of the square permutation matrix (default: 768 for ViT)
        seed: Random seed for reproducible generation (used when password is None)
        password: Password for HMAC-based secure generation
        index: Index for generating multiple different matrices from same password
        
    Returns:
        torch.Tensor: Square permutation matrix of shape (size, size)
        
    Example:
        >>> # Simple seed-based generation
        >>> P = generate_permutation_matrix(768, seed=42)
        >>> 
        >>> # Password-based secure generation
        >>> P = generate_permutation_matrix(768, password="my_secret", index=0)
    """
    if password is not None and index is not None:
        # Create HMAC object with the password
        h = hmac.new(password.encode(), digestmod=hashlib.sha256)
        # Update with the index to generate different matrices
        h.update(str(index).encode())
        # Get the hash value and convert to integer for seed
        seed = int.from_bytes(h.digest()[:8], byteorder='big')
        
    torch.manual_seed(seed)
    
    # Generate random permutation indices
    indices = torch.randperm(size)
    # Create permutation matrix
    P = torch.eye(size)  
    P = P[:, indices]
    return P


def encrypt_ffn_weight_row_permutation(weight_tensor: torch.Tensor,
                                     P: torch.Tensor) -> torch.Tensor:
    """
    Encrypt FFN weight matrix using row permutation.

    This function applies row permutation to a weight matrix, typically used
    for the intermediate and output layers of Vision Transformer FFN blocks.

    Args:
        weight_tensor: Weight matrix to encrypt, shape (hidden_size, intermediate_size)
                      e.g., (768, 3072) for ViT-base or (1024, 4096) for ViT-large
        P: Permutation matrix, shape (hidden_size, hidden_size)

    Returns:
        torch.Tensor: Encrypted weight matrix with same shape as input

    Raises:
        AssertionError: If tensor dimensions don't match permutation matrix

    Example:
        >>> weight = torch.randn(768, 3072)  # FFN intermediate weight for ViT-base
        >>> P = generate_permutation_matrix(768)
        >>> encrypted_weight = encrypt_ffn_weight_row_permutation(weight, P)
        >>>
        >>> # For ViT-large
        >>> weight_large = torch.randn(1024, 4096)  # FFN intermediate weight for ViT-large
        >>> P_large = generate_permutation_matrix(1024)
        >>> encrypted_weight_large = encrypt_ffn_weight_row_permutation(weight_large, P_large)
    """
    hidden_size = weight_tensor.shape[0]
    intermediate_size = weight_tensor.shape[1]

    # Validate dimensions
    assert len(weight_tensor.shape) == 2, f"Weight tensor must be 2D, got shape {weight_tensor.shape}"
    assert P.shape == (hidden_size, hidden_size), f"Permutation matrix must be {hidden_size} x {hidden_size}, got {P.shape}"

    # Ensure permutation matrix has same dtype as weight tensor
    P = P.to(dtype=weight_tensor.dtype, device=weight_tensor.device)

    # Apply row permutation: P @ weight_tensor
    encrypted_tensor = P @ weight_tensor

    # Ensure the result is contiguous for saving
    return encrypted_tensor.contiguous()


def decrypt_ffn_weight_row_permutation(encrypted_tensor: torch.Tensor,
                                     P: torch.Tensor) -> torch.Tensor:
    """
    Decrypt FFN weight matrix using inverse row permutation.

    Since P is a permutation matrix, its inverse is simply its transpose (P^T).
    This function applies the inverse permutation to recover the original weights.

    Args:
        encrypted_tensor: Encrypted weight matrix, shape (hidden_size, intermediate_size)
                         e.g., (768, 3072) for ViT-base or (1024, 4096) for ViT-large
        P: Original permutation matrix used for encryption, shape (hidden_size, hidden_size)

    Returns:
        torch.Tensor: Decrypted weight matrix with same shape as input

    Raises:
        AssertionError: If tensor dimensions don't match permutation matrix

    Example:
        >>> # Encrypt then decrypt for ViT-base
        >>> original = torch.randn(768, 3072)
        >>> P = generate_permutation_matrix(768)
        >>> encrypted = encrypt_ffn_weight_row_permutation(original, P)
        >>> decrypted = decrypt_ffn_weight_row_permutation(encrypted, P)
        >>> torch.allclose(original, decrypted)  # Should be True
        >>>
        >>> # For ViT-large
        >>> original_large = torch.randn(1024, 4096)
        >>> P_large = generate_permutation_matrix(1024)
        >>> encrypted_large = encrypt_ffn_weight_row_permutation(original_large, P_large)
        >>> decrypted_large = decrypt_ffn_weight_row_permutation(encrypted_large, P_large)
        >>> torch.allclose(original_large, decrypted_large)  # Should be True
    """
    hidden_size = encrypted_tensor.shape[0]
    intermediate_size = encrypted_tensor.shape[1]

    # Validate dimensions
    assert len(encrypted_tensor.shape) == 2, f"Encrypted tensor must be 2D, got shape {encrypted_tensor.shape}"
    assert P.shape == (hidden_size, hidden_size), f"Permutation matrix must be {hidden_size} x {hidden_size}, got {P.shape}"

    # For permutation matrix P, P^(-1) = P^T
    P_inv = P.t().to(dtype=encrypted_tensor.dtype, device=encrypted_tensor.device)

    # Apply inverse permutation: P^T @ encrypted_tensor
    decrypted_tensor = P_inv @ encrypted_tensor

    # Ensure the result is contiguous
    return decrypted_tensor.contiguous()


def encrypt_ffn_weight_col_permutation(weight_tensor: torch.Tensor,
                                     P: torch.Tensor) -> torch.Tensor:
    """
    Encrypt FFN weight matrix using column permutation.

    Args:
        weight_tensor: Weight matrix to encrypt, shape (intermediate_size, hidden_size)
                      e.g., (3072, 768) for ViT-base or (4096, 1024) for ViT-large
        P: Permutation matrix, shape (hidden_size, hidden_size)

    Returns:
        torch.Tensor: Encrypted weight matrix with same shape as input
    """
    intermediate_size = weight_tensor.shape[0]
    hidden_size = weight_tensor.shape[1]

    # Validate dimensions
    assert len(weight_tensor.shape) == 2, f"Weight tensor must be 2D, got shape {weight_tensor.shape}"
    assert P.shape == (hidden_size, hidden_size), f"Permutation matrix must be {hidden_size} x {hidden_size}, got {P.shape}"

    P = P.to(dtype=weight_tensor.dtype, device=weight_tensor.device)

    # Apply column permutation: weight_tensor @ P
    encrypted_tensor = weight_tensor @ P
    return encrypted_tensor


def decrypt_ffn_weight_col_permutation(encrypted_tensor: torch.Tensor,
                                     P: torch.Tensor) -> torch.Tensor:
    """
    Decrypt FFN weight matrix using inverse column permutation.

    Args:
        encrypted_tensor: Encrypted weight matrix, shape (intermediate_size, hidden_size)
                         e.g., (3072, 768) for ViT-base or (4096, 1024) for ViT-large
        P: Original permutation matrix used for encryption, shape (hidden_size, hidden_size)

    Returns:
        torch.Tensor: Decrypted weight matrix with same shape as input
    """
    intermediate_size = encrypted_tensor.shape[0]
    hidden_size = encrypted_tensor.shape[1]

    # Validate dimensions
    assert len(encrypted_tensor.shape) == 2, f"Encrypted tensor must be 2D, got shape {encrypted_tensor.shape}"
    assert P.shape == (hidden_size, hidden_size), f"Permutation matrix must be {hidden_size} x {hidden_size}, got {P.shape}"

    # For permutation matrix P, P^(-1) = P^T
    P_inv = P.t().to(dtype=encrypted_tensor.dtype, device=encrypted_tensor.device)

    # Apply inverse permutation: encrypted_tensor @ P^T
    decrypted_tensor = encrypted_tensor @ P_inv

    return decrypted_tensor


def generate_multiple_permutation_matrices(num_matrices: int = 6,
                                         size: int = 768,
                                         password: Optional[str] = None,
                                         device: str = 'cuda') -> list:
    """
    Generate multiple permutation matrices for enhanced security.
    
    This function creates a set of different permutation matrices that can be
    used to encrypt different layers with different permutations, increasing
    the overall security of the encryption scheme.
    
    Args:
        num_matrices: Number of permutation matrices to generate
        size: Dimension of each permutation matrix
        password: Password for HMAC-based generation (if None, uses index as seed)
        device: Device to place the matrices on ('cuda' or 'cpu')
        
    Returns:
        list: List of permutation matrices as torch.Tensors
        
    Example:
        >>> matrices = generate_multiple_permutation_matrices(
        ...     num_matrices=6, password="my_secret", device='cuda'
        ... )
        >>> len(matrices)  # 6
        >>> matrices[0].shape  # torch.Size([768, 768])
    """
    matrices = []
    for i in range(num_matrices):
        if password is not None:
            # Use password-based HMAC seeding
            P = generate_permutation_matrix(
                size=size, 
                password=password, 
                index=i
            )
        else:
            # Fallback to simple index-based seeding
            P = generate_permutation_matrix(
                size=size, 
                seed=i
            )
        matrices.append(P.to(device))
    return matrices


def verify_permutation_invertibility(P: torch.Tensor, 
                                   test_tensor: Optional[torch.Tensor] = None) -> bool:
    """
    Verify that a permutation matrix is properly invertible.
    
    Args:
        P: Permutation matrix to verify
        test_tensor: Optional test tensor to verify encryption/decryption cycle
        
    Returns:
        bool: True if the permutation is invertible
        
    Example:
        >>> P = generate_permutation_matrix(768)
        >>> is_invertible = verify_permutation_invertibility(P)
        >>> print(is_invertible)  # True
    """
    # Check if P @ P^T = I (identity matrix)
    identity_check = torch.allclose(P @ P.t(), torch.eye(P.shape[0], device=P.device))
    
    if test_tensor is not None and len(test_tensor.shape) == 2:
        # Test actual encryption/decryption cycle
        hidden_size = P.shape[0]

        # Check if this is a row permutation case (first dimension matches permutation matrix)
        if test_tensor.shape[0] == hidden_size:
            encrypted = encrypt_ffn_weight_row_permutation(test_tensor, P)
            decrypted = decrypt_ffn_weight_row_permutation(encrypted, P)
        # Check if this is a column permutation case (second dimension matches permutation matrix)
        elif test_tensor.shape[1] == hidden_size:
            encrypted = encrypt_ffn_weight_col_permutation(test_tensor, P)
            decrypted = decrypt_ffn_weight_col_permutation(encrypted, P)
        else:
            return identity_check

        cycle_check = torch.allclose(test_tensor, decrypted, atol=1e-6)
        return identity_check and cycle_check
    
    return identity_check
