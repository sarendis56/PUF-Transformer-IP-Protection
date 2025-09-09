"""
Model Utilities for Vision Transformer IP Protection

This module provides utilities for loading, saving, and managing Vision
Transformer models with encryption metadata and security features.
"""

import torch
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from transformers import ViTForImageClassification, ViTImageProcessor

from .metrics import PerformanceTracker
from ..encryption.dual_encryption import DualEncryption


def setup_model(model_name: str, 
               device: torch.device,
               local_model_path: Optional[str] = None) -> Tuple[ViTForImageClassification, ViTImageProcessor]:
    """
    Setup Vision Transformer model and processor with proper error handling.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load the model on
        local_model_path: Optional local path to model files
        
    Returns:
        Tuple of (model, processor)
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        # Check for local model path
        if local_model_path and Path(local_model_path).exists():
            print(f"Loading local model from {local_model_path}...")
            model = ViTForImageClassification.from_pretrained(local_model_path)
            processor = ViTImageProcessor.from_pretrained(local_model_path)
        else:
            # If no local path provided, check for default local model directory based on model name
            if not local_model_path:
                # Extract model identifier from HuggingFace model name
                if '/' in model_name:
                    model_identifier = model_name.split('/')[-1]  # e.g., 'vit-base-patch16-224'
                else:
                    model_identifier = model_name

                default_local_path = Path(f'model/{model_identifier}')
                if default_local_path.exists() and default_local_path.is_dir():
                    print(f"Found local model at {default_local_path}, loading...")
                    model = ViTForImageClassification.from_pretrained(str(default_local_path))
                    processor = ViTImageProcessor.from_pretrained(str(default_local_path))
                else:
                    print(f"Local model not found at {default_local_path}, loading {model_name} from HuggingFace Hub...")
                    model = ViTForImageClassification.from_pretrained(model_name)
                    processor = ViTImageProcessor.from_pretrained(model_name)

                    # Save model locally for future use
                    try:
                        print(f"Saving model to {default_local_path} for future use...")
                        default_local_path.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(default_local_path)
                        processor.save_pretrained(default_local_path)
                        print(f"Model saved successfully to {default_local_path}")
                    except Exception as save_error:
                        print(f"Warning: Could not save model locally: {save_error}")
                        print("Model will be loaded from HuggingFace Hub on future runs")
            else:
                print(f"Local model path {local_model_path} not found, loading {model_name} from HuggingFace Hub...")
                model = ViTForImageClassification.from_pretrained(model_name)
                processor = ViTImageProcessor.from_pretrained(model_name)

                # Save model to the specified local path for future use
                try:
                    print(f"Saving model to {local_model_path} for future use...")
                    Path(local_model_path).mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(local_model_path)
                    processor.save_pretrained(local_model_path)
                    print(f"Model saved successfully to {local_model_path}")
                except Exception as save_error:
                    print(f"Warning: Could not save model locally: {save_error}")
                    print("Model will be loaded from HuggingFace Hub on future runs")
        
        model = model.to(device)
        model.eval()
        return model, processor
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {local_model_path or model_name}: {str(e)}")


def save_encrypted_model(model: ViTForImageClassification,
                        processor: ViTImageProcessor,
                        performance_tracker: PerformanceTracker,
                        dual_encryptor: DualEncryption,
                        output_dir: Path) -> None:
    """
    Save encrypted model with comprehensive metadata and security information.
    
    Args:
        model: The encrypted Vision Transformer model
        processor: Image processor for the model
        performance_tracker: Performance tracking data
        dual_encryptor: Dual encryption system used
        output_dir: Directory to save the model and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure all model parameters are contiguous before saving
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    # Save model and processor
    model.save_pretrained(output_dir / "model")
    processor.save_pretrained(output_dir / "processor")
    
    # Create comprehensive metadata
    metadata = {
        "model_info": {
            "base_model_name": performance_tracker.model_name,
            "model_type": "ViTForImageClassification",
            "save_timestamp": datetime.now().isoformat()
        },
        
        "performance_metrics": {
            "initial_accuracy": performance_tracker.initial_accuracy,
            "final_accuracy": performance_tracker.get_current_accuracy(),
            "total_accuracy_drop": performance_tracker.get_total_accuracy_drop(),
            "num_encrypted_layers": len(performance_tracker.encryption_steps)
        },
        
        "encryption_config": {
            "arnold_key": dual_encryptor.config.arnold_key,
            "num_permutation_matrices": dual_encryptor.config.num_permutation_matrices,
            "matrix_size": dual_encryptor.config.matrix_size,
            "password_protected": dual_encryptor.config.password is not None
        },
        
        "encryption_history": [
            {
                "step": i,
                "layer_idx": step.layer_idx,
                "accuracy": step.accuracy,
                "accuracy_drop": step.accuracy_drop,
                "permutation_matrix_idx": step.permutation_matrix_idx,
                "arnold_key": step.arnold_key,
                "timestamp": step.timestamp
            }
            for i, step in enumerate(performance_tracker.encryption_steps)
        ],
        
        "experiment_config": performance_tracker.experiment_config
    }
    
    # Add password hash if password was used
    if dual_encryptor.config.password is not None:
        password_hash = hashlib.sha256(dual_encryptor.config.password.encode()).hexdigest()
        metadata["security"] = {
            "password_hash": password_hash,
            "password_required_for_decryption": True
        }
    
    # Save metadata
    with open(output_dir / "encryption_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    # Save performance tracker separately for easy loading
    performance_tracker.save_to_file(output_dir / "performance_history.json")
    
    print(f"Encrypted model and metadata saved to {output_dir}")


def load_encrypted_model(model_dir: Path,
                        password: Optional[str] = None,
                        device: str = "cuda") -> Tuple[ViTForImageClassification, ViTImageProcessor, Dict[str, Any]]:
    """
    Load encrypted model with metadata and optional password verification.
    
    Args:
        model_dir: Directory containing the saved encrypted model
        password: Password for decryption (if model was password-protected)
        device: Device to load the model on
        
    Returns:
        Tuple of (model, processor, metadata)
        
    Raises:
        ValueError: If password verification fails
        FileNotFoundError: If required files are missing
    """
    model_dir = Path(model_dir)
    
    # Load metadata
    metadata_path = model_dir / "encryption_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Encryption metadata not found at {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Verify password if model is password-protected
    if metadata.get("security", {}).get("password_required_for_decryption", False):
        if password is None:
            raise ValueError("Password required for this encrypted model")
        
        provided_hash = hashlib.sha256(password.encode()).hexdigest()
        stored_hash = metadata["security"]["password_hash"]
        
        if provided_hash != stored_hash:
            raise ValueError("Invalid password provided")
    
    # Load model and processor
    model_path = model_dir / "model"
    processor_path = model_dir / "processor"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model files not found at {model_path}")
    if not processor_path.exists():
        raise FileNotFoundError(f"Processor files not found at {processor_path}")
    
    model = ViTForImageClassification.from_pretrained(model_path)
    processor = ViTImageProcessor.from_pretrained(processor_path)
    
    model = model.to(device)
    model.eval()
    
    return model, processor, metadata


def verify_model_integrity(model: ViTForImageClassification,
                          expected_encrypted_layers: set,
                          dual_encryptor: DualEncryption) -> bool:
    """
    Verify the integrity of an encrypted model by checking layer encryption status.
    
    Args:
        model: The model to verify
        expected_encrypted_layers: Set of layer indices that should be encrypted
        dual_encryptor: Dual encryption system for verification
        
    Returns:
        bool: True if model integrity is verified
    """
    try:
        # This is a simplified integrity check
        # In practice, you might want more sophisticated verification
        
        # Check that the model has a reasonable number of layers
        num_layers = len(model.vit.encoder.layer)
        if num_layers < 6 or num_layers > 48:  # Support ViT-base (12), ViT-large (24), etc.
            return False
        
        # Check that encrypted layers have different weight patterns
        # (This is a heuristic check - encrypted weights should look different)
        for layer_idx in expected_encrypted_layers:
            layer = model.vit.encoder.layer[layer_idx]
            
            # Check if attention weights have expected properties
            query_weight = layer.attention.attention.query.weight.data
            if torch.allclose(query_weight, torch.zeros_like(query_weight)):
                return False  # Weights shouldn't be all zeros
        
        return True
        
    except Exception:
        return False


def create_model_backup(model: ViTForImageClassification,
                       backup_path: Path) -> None:
    """
    Create a backup of the model before encryption.
    
    Args:
        model: Model to backup
        backup_path: Path to save the backup
    """
    backup_path = Path(backup_path)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), backup_path / "model_state_dict.pt")
    
    # Save model configuration
    model.save_pretrained(backup_path / "model_config")
    
    # Create backup metadata
    backup_metadata = {
        "backup_timestamp": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(model.device)
    }
    
    with open(backup_path / "backup_metadata.json", "w") as f:
        json.dump(backup_metadata, f, indent=4)
    
    print(f"Model backup saved to {backup_path}")


def restore_model_from_backup(backup_path: Path,
                             device: str = "cuda") -> ViTForImageClassification:
    """
    Restore model from backup.
    
    Args:
        backup_path: Path to the backup
        device: Device to load the model on
        
    Returns:
        ViTForImageClassification: Restored model
    """
    backup_path = Path(backup_path)
    
    # Load model from config
    model = ViTForImageClassification.from_pretrained(backup_path / "model_config")
    
    # Load state dict
    state_dict_path = backup_path / "model_state_dict.pt"
    if state_dict_path.exists():
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    return model
