"""
Arnold Cat Map (ACM) encryption implementation for neural network weight protection.

References:
    - Arnold, V.I. and Avez, A. (1968). Ergodic problems of classical mechanics.
    - Zhang, Y. et al. (2019). Image encryption using DNA addition combining with 
      chaotic maps. Mathematical and Computer Modelling.
"""

import torch
import numpy as np
from typing import List, Union, Tuple


def arnold(matrix: Union[torch.Tensor, np.ndarray], key: List[int]) -> torch.Tensor:
    """
    Apply Arnold Cat Map transformation to encrypt a matrix.
    
    The Arnold Cat Map is defined by the transformation:
    [x']   [1 1] [x]     [a b] [x]
    [y'] = [1 2] [y] or  [c d] [y]  (mod N)
    
    where the determinant (ad - bc) ≡ 1 (mod N) for invertibility.
    
    Args:
        matrix: Input matrix to encrypt. Must be square (H x W x ...).
        key: Arnold key parameters [N, a, b, c, d] where:
            - N: Number of iterations
            - a, b, c, d: Transformation matrix parameters
            
    Returns:
        torch.Tensor: Encrypted matrix with same shape as input
        
    Raises:
        ValueError: If matrix is not square or key parameters are invalid
        
    Example:
        >>> matrix = torch.randn(768, 768)
        >>> key = [3, 1, 1, 1, 2]  # N=3 iterations, standard Arnold map
        >>> encrypted = arnold(matrix, key)
    """
    N, a, b, c, d = key
    h, w = matrix.shape[:2]

    if h != w:
        raise ValueError("Matrix must be square")
    if (a * d - b * c) % w != 1:
        raise ValueError("Invalid Arnold key: determinant must be 1 (mod w)")

    # Ensure tensor is a torch tensor and get its device
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.from_numpy(matrix)

    device = matrix.device

    # Create original coordinate meshgrid on the same device as the matrix
    y_orig_grid, x_orig_grid = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.long),
        torch.arange(w, device=device, dtype=torch.long),
        indexing='ij'
    )
    # Flatten coordinates for efficient processing
    x_coords_to_transform = x_orig_grid.flatten()
    y_coords_to_transform = y_orig_grid.flatten()

    # Transform coordinates N times to find final destinations (no matrix operations yet)
    final_dest_x = x_coords_to_transform.clone()
    final_dest_y = y_coords_to_transform.clone()

    for _ in range(N):
        next_dest_x = (a * final_dest_x + b * final_dest_y) % w
        next_dest_y = (c * final_dest_x + d * final_dest_y) % h
        final_dest_x = next_dest_x
        final_dest_y = next_dest_y

    # Calculate final destination indices for scatter operation
    final_dest_flat_indices = (final_dest_y * w + final_dest_x).long()

    # Reshape original matrix and prepare output buffer
    original_matrix_flat = matrix.reshape(h * w, -1)
    output_flat = torch.zeros_like(original_matrix_flat)

    # Single efficient scatter: put original elements in their final positions
    output_flat.index_copy_(0, final_dest_flat_indices, original_matrix_flat)

    result = output_flat.reshape(matrix.shape)

    # Ensure the result is contiguous for saving
    return result.contiguous()


def iarnold(matrix: Union[torch.Tensor, np.ndarray], key: List[int]) -> torch.Tensor:
    """
    Apply inverse Arnold Cat Map transformation to decrypt a matrix.
    
    This function computes the inverse transformation by calculating the
    inverse of the Arnold transformation matrix modulo the matrix width.
    
    Args:
        matrix: Encrypted matrix to decrypt. Must be square.
        key: Arnold key parameters [N, a, b, c, d] (same as used for encryption)
        
    Returns:
        torch.Tensor: Decrypted matrix with same shape as input
        
    Raises:
        ValueError: If key parameters are invalid
        
    Example:
        >>> encrypted_matrix = arnold(original_matrix, key)
        >>> decrypted_matrix = iarnold(encrypted_matrix, key)
        >>> torch.allclose(original_matrix, decrypted_matrix)  # Should be True
    """
    N, a, b, c, d = key
    h, w = matrix.shape[:2]
    
    if (a * d - b * c) % w != 1:
        raise ValueError("Invalid Arnold key: determinant must be 1 (mod w)")

    # Compute inverse transformation matrix
    inv_key_matrix = np.array([[d, -b],
                               [-c, a]]) % w
    inv_key = [N,
               inv_key_matrix[0, 0],
               inv_key_matrix[0, 1],
               inv_key_matrix[1, 0],
               inv_key_matrix[1, 1]]
    
    return arnold(matrix, inv_key)


def generate_arnold_key(matrix_size: int, iterations: int = 3, 
                       seed: int = None) -> List[int]:
    """
    Generate a valid Arnold Cat Map key for a given matrix size.
    
    Args:
        matrix_size: Size of the square matrix (width/height)
        iterations: Number of Arnold iterations to apply
        seed: Random seed for reproducible key generation
        
    Returns:
        List[int]: Valid Arnold key [N, a, b, c, d]
        
    Example:
        >>> key = generate_arnold_key(768, iterations=5, seed=42)
        >>> print(key)  # [5, 1, 1, 1, 2] or similar valid key
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random parameters ensuring determinant = 1 (mod matrix_size)
    while True:
        a = np.random.randint(1, matrix_size)
        b = np.random.randint(1, matrix_size)
        c = np.random.randint(1, matrix_size)
        d = np.random.randint(1, matrix_size)
        
        if (a * d - b * c) % matrix_size == 1:
            return [iterations, a, b, c, d]


def verify_arnold_invertibility(key: List[int], matrix_size: int) -> bool:
    """
    Verify that an Arnold key produces invertible transformations.
    
    Args:
        key: Arnold key parameters [N, a, b, c, d]
        matrix_size: Size of the matrix the key will be applied to
        
    Returns:
        bool: True if the key is valid and invertible
        
    Example:
        >>> key = [3, 1, 1, 1, 2]
        >>> is_valid = verify_arnold_invertibility(key, 768)
        >>> print(is_valid)  # True
    """
    N, a, b, c, d = key
    determinant = (a * d - b * c) % matrix_size
    return determinant == 1


# Standard Arnold keys for common use cases
STANDARD_ARNOLD_KEYS = {
    'default': [3, 1, 1, 1, 2],
    'strong': [5, 2, 3, 3, 5],
    'extra_strong': [11, 1, 3, 2, 7],
}


def get_standard_key(key_name: str = 'default') -> List[int]:
    """
    Get a predefined standard Arnold key.
    
    Args:
        key_name: Name of the standard key ('default', 'strong', 'extra_strong')
        
    Returns:
        List[int]: Standard Arnold key parameters
        
    Raises:
        KeyError: If key_name is not recognized
    """
    if key_name not in STANDARD_ARNOLD_KEYS:
        available_keys = list(STANDARD_ARNOLD_KEYS.keys())
        raise KeyError(f"Unknown key name '{key_name}'. Available: {available_keys}")
    
    return STANDARD_ARNOLD_KEYS[key_name].copy()
