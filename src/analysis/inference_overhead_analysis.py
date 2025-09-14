"""
Inference Overhead Analysis for Dual Encryption

This module analyzes the performance overhead of the dual encryption scheme
during inference, providing detailed timing breakdowns for different components:
- ACM (Arnold Cat Map) encryption/decryption
- FFN (Feed-Forward Network) permutation encryption/decryption
- Forward pass execution

The analysis measures overhead as a function of:
- Number of ACM iterations
- Number of encrypted layers
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
import sys
import os
import argparse
import glob
import random
from PIL import Image

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Global configuration for number of runs
# Adjust these values to control evaluation repetition for different devices
NUM_RUNS_CPU = 3      # Number of runs for CPU evaluation
NUM_RUNS_GPU = 10     # Number of runs for GPU evaluation

from encryption.arnold_transform import arnold, iarnold

def get_optimal_arnold_functions(device: torch.device):
    return arnold, iarnold

@dataclass
class TimingBreakdown:
    """Container for timing breakdown data."""
    total_time: float
    forward_time: float
    acm_enc_time: float = 0.0
    acm_dec_time: float = 0.0
    ffn_enc_time: float = 0.0
    ffn_dec_time: float = 0.0
    overhead_ratio: float = 1.0

    @property
    def encryption_time(self) -> float:
        return self.acm_enc_time + self.ffn_enc_time

    @property
    def decryption_time(self) -> float:
        return self.acm_dec_time + self.ffn_dec_time

    @property
    def total_overhead_time(self) -> float:
        return self.encryption_time + self.decryption_time


class InferenceOverheadAnalyzer:
    """Analyzer for measuring dual encryption inference overhead."""

    # Global font size configuration for all plots - centralized for easy maintenance
    # To change any font size, simply update the values here!
    FONT_SIZES = {
        'main_title': 55,      # Main figure title
        'axis_labels': 48,     # X and Y axis labels
        'subtitles': 40,       # Subplot titles (a, b, c, d)
        'ratio_text': 40,      # Overhead ratio text above bars
        'tick_labels': 28,     # X and Y tick labels
        'legend': 40           # Legend text
    }

    def __init__(self, model_name: str = "google/vit-base-patch16-224", device: str = "auto", batch_size: int = 1, skip_imagenet: bool = False):
        """Initialize the analyzer with model and device.

        Args:
            model_name: Name of the ViT model to use
            device: Device to run on ('auto', 'cuda', or 'cpu')
            batch_size: Batch size for inference (default: 1 for single sequential inference)
            skip_imagenet: Skip loading ImageNet dataset (useful for plot regeneration)
        """
        self.model_name = model_name

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.batch_size = batch_size  # Store batch size (default=1)

        # Initialize model and processor
        # Check if local model exists first, otherwise use HuggingFace Hub
        from pathlib import Path

        # Extract model identifier from HuggingFace model name
        if '/' in model_name:
            model_identifier = model_name.split('/')[-1]  # e.g., 'vit-base-patch16-224'
        else:
            model_identifier = model_name

        local_model_path = Path(f"model/{model_identifier}")

        if local_model_path.exists() and local_model_path.is_dir():
            print(f"Loading local model from {local_model_path}...")
            self.model = ViTForImageClassification.from_pretrained(str(local_model_path)).to(self.device)
            self.processor = ViTImageProcessor.from_pretrained(str(local_model_path))
        else:
            print(f"Local model not found at {local_model_path}, loading {model_name} from HuggingFace Hub...")
            self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
            self.processor = ViTImageProcessor.from_pretrained(model_name)

            # Save model locally for future use
            try:
                print(f"Saving model to {local_model_path} for future use...")
                local_model_path.mkdir(parents=True, exist_ok=True)
                # Save to CPU first to avoid device issues
                cpu_model = ViTForImageClassification.from_pretrained(model_name)
                cpu_model.save_pretrained(local_model_path)
                self.processor.save_pretrained(local_model_path)
                print(f"Model saved successfully to {local_model_path}")
            except Exception as save_error:
                print(f"Warning: Could not save model locally: {save_error}")
                print("Model will be loaded from HuggingFace Hub on future runs")

        # Sample real images from ImageNet dataset (skip if only regenerating plots)
        if not skip_imagenet:
            self.sample_input = self._load_imagenet_samples(num_samples=1000, batch_size=self.batch_size)
        else:
            print("Skipping ImageNet dataset loading for plot regeneration...")
            self.sample_input = None
            self.imagenet_samples = None

        # Select optimal Arnold functions for this device
        self.arnold_encrypt, self.arnold_decrypt = get_optimal_arnold_functions(self.device)

        print(f"Initialized analyzer with model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")

    def _load_imagenet_samples(self, num_samples: int = 1000, batch_size: int = 1) -> torch.Tensor:
        """Load and preprocess samples from ImageNet validation dataset.

        Args:
            num_samples: Number of images to sample from ImageNet
            batch_size: Batch size for the returned tensor

        Returns:
            Preprocessed tensor of shape (batch_size, 3, 224, 224)
        """
        imagenet_path = Path("dataset/imagenet/val")

        if not imagenet_path.exists():
            print(f"Warning: ImageNet dataset not found at {imagenet_path}")
            print("Falling back to random tensor generation...")
            return torch.randn(batch_size, 3, 224, 224).to(self.device)

        # Get all image files from all class directories
        image_files = []
        for class_dir in imagenet_path.iterdir():
            if class_dir.is_dir():
                class_images = list(class_dir.glob("*.JPEG"))
                image_files.extend(class_images)

        if len(image_files) == 0:
            print("Warning: No JPEG images found in ImageNet dataset")
            print("Falling back to random tensor generation...")
            return torch.randn(batch_size, 3, 224, 224).to(self.device)

        # Set seed for reproducible sampling
        random.seed(42)

        # Sample random images
        sampled_files = random.sample(image_files, min(num_samples, len(image_files)))
        print(f"Sampled {len(sampled_files)} images from ImageNet dataset")

        # Load and preprocess images
        processed_images = []
        for img_path in tqdm(sampled_files, desc="Loading ImageNet samples"):
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')

                # Preprocess using the ViT processor
                inputs = self.processor(images=image, return_tensors="pt")
                processed_images.append(inputs['pixel_values'].squeeze(0))

            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
                continue

        if len(processed_images) == 0:
            print("Warning: Failed to load any images from ImageNet dataset")
            print("Falling back to random tensor generation...")
            return torch.randn(batch_size, 3, 224, 224).to(self.device)

        # Stack all images and create batches
        all_images = torch.stack(processed_images)

        # For inference, we'll cycle through these images to create batches
        # For now, just return the first batch_size images
        if batch_size <= len(all_images):
            batch_images = all_images[:batch_size]
        else:
            # If we need more images than we have, repeat some
            indices = torch.randint(0, len(all_images), (batch_size,))
            batch_images = all_images[indices]

        # Store all images for later use in different batch configurations
        self.imagenet_samples = all_images.to(self.device)

        return batch_images.to(self.device)

    def _get_sample_input(self, batch_size: int = None) -> torch.Tensor:
        """Get sample input tensor with specified batch size.

        Args:
            batch_size: Desired batch size. If None, uses self.batch_size

        Returns:
            Tensor of shape (batch_size, 3, 224, 224)
        """
        if batch_size is None:
            batch_size = self.batch_size

        # If we have ImageNet samples, use them
        if hasattr(self, 'imagenet_samples') and self.imagenet_samples is not None:
            if batch_size <= len(self.imagenet_samples):
                return self.imagenet_samples[:batch_size]
            else:
                # If we need more images than we have, repeat some
                indices = torch.randint(0, len(self.imagenet_samples), (batch_size,))
                return self.imagenet_samples[indices]
        else:
            # Fallback to random tensor
            return torch.randn(batch_size, 3, 224, 224).to(self.device)

    def _time_forward_pass(self, model: torch.nn.Module, input_tensor: torch.Tensor, num_runs: int = 1) -> float:
        """Measure forward pass time for a model using realistic batch processing."""
        times = []

        # Warm-up runs
        warmup_runs = 3
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(input_tensor)

        # Actual timing runs
        for _ in range(num_runs):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                _ = model(input_tensor)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            times.append(end_time - start_time)

        return np.mean(times)

    def _time_acm_operation(self, matrix: torch.Tensor, key: List[int], is_encrypt: bool = True, num_runs: int = 1) -> float:
        """Measure ACM encryption/decryption time."""
        times = []

        # Warm-up runs (especially important for n=1)
        warmup_runs = 3
        for _ in range(warmup_runs):
            if is_encrypt:
                _ = arnold(matrix, key)
            else:
                _ = iarnold(matrix, key)

        # Actual timing runs
        for _ in range(num_runs):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            if is_encrypt:
                _ = arnold(matrix, key)
            else:
                _ = iarnold(matrix, key)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            times.append(end_time - start_time)

        return np.mean(times)

    def _time_ffn_operation(self, matrix: torch.Tensor, perm_indices: torch.Tensor, is_encrypt: bool = True, num_runs: int = 1) -> float:
        """Measure FFN permutation encryption/decryption time using efficient direct permutation."""
        times = []

        # Warm-up runs
        warmup_runs = 3
        for _ in range(warmup_runs):
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            if is_encrypt:
                _ = matrix[:, perm_indices]  # Direct column permutation
            else:
                inv_perm = torch.argsort(perm_indices)  # Get inverse permutation
                _ = matrix[:, inv_perm]  # Direct inverse permutation
            torch.cuda.synchronize() if self.device.type == 'cuda' else None

        # Actual timing runs
        for _ in range(num_runs):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            if is_encrypt:
                _ = matrix[:, perm_indices]  # Direct column permutation
            else:
                inv_perm = torch.argsort(perm_indices)  # Get inverse permutation
                _ = matrix[:, inv_perm]  # Direct inverse permutation

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            times.append(end_time - start_time)

        return np.mean(times)

    def _time_realistic_encrypted_inference(self, encrypted_layers: List[int], acm_key: List[int], 
                                           num_runs: int = None, warmup_runs: int = 3) -> Dict[str, float]:
        """Time realistic encrypted inference with actual model weight manipulation.
        
        This follows the Simulation.ipynb methodology where we actually encrypt/decrypt
        model weights in-place during inference, providing realistic timing measurements.
        
        Returns separate timing for ACM and FFN components.
        """
        from collections import defaultdict
        
        # Set number of runs based on device if not specified
        if num_runs is None:
            num_runs = NUM_RUNS_GPU if self.device.type == 'cuda' else NUM_RUNS_CPU
        
        # Store original model state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Generate permutation indices for each layer DURING timing (not pre-computed)
        # This ensures fair benchmarking where everything is computed from scratch
        def generate_permutation_indices(layer_idx: int):
            """Generate permutation indices for a specific layer during timing."""
            layer = self.model.vit.encoder.layer[layer_idx]
            # Get FFN dimensions
            intermediate_weight = layer.intermediate.dense.weight
            output_weight = layer.output.dense.weight
            
            # Generate permutation indices (simulating PUF-derived keys)
            torch.manual_seed(42 + layer_idx)  # Deterministic for reproducibility
            return {
                'in': torch.randperm(intermediate_weight.shape[1], device=self.device),
                'hidden': torch.randperm(output_weight.shape[1], device=self.device)
            }

        # Get sample input for current batch size
        sample_input = self._get_sample_input()

        # Warmup runs - general model inference
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model(sample_input)
        
        # ACM-specific warmup to avoid compilation overhead
        sample_weight = torch.randn(768, 768, device=self.device)
        # Warm up ACM operations
        for _ in range(3):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            _ = self.arnold_encrypt(sample_weight, acm_key)
            _ = self.arnold_decrypt(sample_weight, acm_key)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Additional GPU warmup to ensure consistent performance
        if self.device.type == 'cuda':
            # Force CUDA compilation and memory allocation
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Additional warmup runs to ensure stable performance
            for _ in range(2):
                _ = self.arnold_encrypt(sample_weight, acm_key)
                _ = self.arnold_decrypt(sample_weight, acm_key)
                torch.cuda.synchronize()
            
            # Warm up FFN operations to avoid cold start penalty
            sample_ffn = torch.randn(768, 768, device=self.device)
            sample_perms = torch.randperm(768, device=self.device)
            
            # Warm up fast inverse permutation and column permutation operations
            for _ in range(3):
                # Fast O(n) inverse permutation
                inv_perms = torch.zeros_like(sample_perms)
                inv_perms[sample_perms] = torch.arange(len(sample_perms), device=sample_perms.device, dtype=sample_perms.dtype)
                _ = sample_ffn[:, inv_perms]  # Inverse permutation
                _ = sample_ffn[:, sample_perms]  # Forward permutation
                torch.cuda.synchronize()
            
        


        # Measure baseline inference (no encryption)
        baseline_times = []
        for _ in range(num_runs):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model(sample_input)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            baseline_times.append(time.perf_counter() - start_time)

        # Measure encrypted inference with realistic sequential processing
        encrypted_times = []
        acm_decrypt_times = []
        acm_encrypt_times = []
        ffn_decrypt_times = []
        ffn_encrypt_times = []
        
        for run_idx in range(num_runs):
            # Reset model to original state
            self.model.load_state_dict(original_state)
            
            run_acm_decrypt_time = 0
            run_acm_encrypt_time = 0
            run_ffn_decrypt_time = 0
            run_ffn_encrypt_time = 0
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            inference_start = time.perf_counter()
            
            # Decryption phase - decrypt all layers before inference
            for layer_idx in encrypted_layers:
                layer = self.model.vit.encoder.layer[layer_idx]
                perms = generate_permutation_indices(layer_idx)
                
                # Time attention decryption (ACM)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                attention_weights = [
                    layer.attention.attention.query.weight.data,
                    layer.attention.attention.key.weight.data,
                    layer.attention.attention.value.weight.data,
                    layer.attention.output.dense.weight.data
                ]
                for w in attention_weights:
                    if w.shape[0] == w.shape[1]:  # Only square matrices for ACM
                        w.copy_(self.arnold_decrypt(w, acm_key))
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                run_acm_decrypt_time += time.perf_counter() - start
                
                # Time FFN decryption
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                # Fast O(n) inverse permutation instead of O(n log n) argsort
                inv_perm_in = torch.zeros_like(perms['in'])
                inv_perm_hidden = torch.zeros_like(perms['hidden'])
                inv_perm_in[perms['in']] = torch.arange(len(perms['in']), device=perms['in'].device, dtype=perms['in'].dtype)
                inv_perm_hidden[perms['hidden']] = torch.arange(len(perms['hidden']), device=perms['hidden'].device, dtype=perms['hidden'].dtype)
                
                layer.intermediate.dense.weight.data = layer.intermediate.dense.weight.data[:, inv_perm_in]
                layer.output.dense.weight.data = layer.output.dense.weight.data[:, inv_perm_hidden]
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                run_ffn_decrypt_time += time.perf_counter() - start

            # Perform inference on decrypted model
            with torch.no_grad():
                _ = self.model(sample_input)
            
            # Re-encryption phase - encrypt all layers after inference
            for layer_idx in encrypted_layers:
                layer = self.model.vit.encoder.layer[layer_idx]
                perms = generate_permutation_indices(layer_idx)
                
                # Time attention re-encryption (ACM)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                attention_weights = [
                    layer.attention.attention.query.weight.data,
                    layer.attention.attention.key.weight.data,
                    layer.attention.attention.value.weight.data,
                    layer.attention.output.dense.weight.data
                ]
                for w in attention_weights:
                    if w.shape[0] == w.shape[1]:  # Only square matrices for ACM
                        w.copy_(self.arnold_encrypt(w, acm_key))
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                run_acm_encrypt_time += time.perf_counter() - start
                
                # Time FFN re-encryption
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                # Direct permutation
                layer.intermediate.dense.weight.data = layer.intermediate.dense.weight.data[:, perms['in']]
                layer.output.dense.weight.data = layer.output.dense.weight.data[:, perms['hidden']]
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                run_ffn_encrypt_time += time.perf_counter() - start
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            total_inference_time = time.perf_counter() - inference_start
            
            encrypted_times.append(total_inference_time)
            acm_decrypt_times.append(run_acm_decrypt_time)
            acm_encrypt_times.append(run_acm_encrypt_time)
            ffn_decrypt_times.append(run_ffn_decrypt_time)
            ffn_encrypt_times.append(run_ffn_encrypt_time)

        # Restore original model state
        self.model.load_state_dict(original_state)
        

        
        return {
            'baseline_time': np.mean(baseline_times),
            'encrypted_time': np.mean(encrypted_times),
            'acm_decrypt_time': np.mean(acm_decrypt_times),
            'acm_encrypt_time': np.mean(acm_encrypt_times),
            'ffn_decrypt_time': np.mean(ffn_decrypt_times),
            'ffn_encrypt_time': np.mean(ffn_encrypt_times),
            'total_overhead': np.mean(acm_decrypt_times) + np.mean(acm_encrypt_times) + np.mean(ffn_decrypt_times) + np.mean(ffn_encrypt_times)
        }

    def _apply_acm_encryption(self, matrix: torch.Tensor, key: List[int]) -> torch.Tensor:
        """Apply ACM encryption (single operation, not timed with warmup)."""
        return arnold(matrix, key)

    def _apply_acm_decryption(self, matrix: torch.Tensor, key: List[int]) -> torch.Tensor:
        """Apply ACM decryption (single operation, not timed with warmup)."""
        return iarnold(matrix, key)

    def _apply_ffn_encryption(self, matrix: torch.Tensor, perm_indices: torch.Tensor) -> torch.Tensor:
        """Apply FFN encryption (single operation, not timed with warmup)."""
        return matrix[:, perm_indices]

    def _apply_ffn_decryption(self, matrix: torch.Tensor, perm_indices: torch.Tensor) -> torch.Tensor:
        """Apply FFN decryption (single operation, not timed with warmup)."""
        # Fast O(n) inverse permutation instead of O(n log n) argsort
        inv_perm = torch.zeros_like(perm_indices)
        inv_perm[perm_indices] = torch.arange(len(perm_indices), device=perm_indices.device, dtype=perm_indices.dtype)
        return matrix[:, inv_perm]



    def analyze_acm_iterations_overhead(self, max_iterations: int = 20) -> List[TimingBreakdown]:
        """Analyze overhead vs number of ACM iterations using realistic methodology."""
        print(f"\nAnalyzing ACM iterations overhead (max_iterations={max_iterations})...")

        results = []

        # Test specific ACM iterations: 1, 3, 5, 8, 10, 12
        acm_iterations = [1, 3, 5, 8, 10, 12]
        
        # Fixed number of layers for ACM iterations analysis (matches figure (b))
        num_layers = 6
        encrypted_layers = list(range(num_layers))
        
        for n_iter in tqdm(acm_iterations):
            key = [n_iter, 1, 1, 1, 2]  # Arnold key

            # Use realistic timing methodology from Simulation.ipynb
            timing_result = self._time_realistic_encrypted_inference(encrypted_layers, key, num_runs=None)
            
            # Use actual measured component times (no artificial splitting!)
            acm_enc_time = timing_result['acm_encrypt_time']
            acm_dec_time = timing_result['acm_decrypt_time']
            ffn_enc_time = timing_result['ffn_encrypt_time']
            ffn_dec_time = timing_result['ffn_decrypt_time']
            
            baseline_time = timing_result['baseline_time']
            total_time = timing_result['encrypted_time']
            overhead_ratio = total_time / baseline_time

            results.append(TimingBreakdown(
                total_time=total_time,
                forward_time=baseline_time,
                acm_enc_time=acm_enc_time,
                acm_dec_time=acm_dec_time,
                ffn_enc_time=ffn_enc_time,
                ffn_dec_time=ffn_dec_time,
                overhead_ratio=overhead_ratio
            ))

        return results

    def analyze_layers_overhead(self, max_layers: int = 12) -> List[TimingBreakdown]:
        """Analyze overhead vs number of encrypted layers using realistic methodology."""
        print(f"\nAnalyzing layers overhead (max_layers={max_layers})...")

        results = []

        # Fixed ACM key (n=3 as mentioned in paper)
        # This matches the ACM iteration=3 point in figure (a)
        key = [3, 1, 1, 1, 2]

        # Test specific layer counts: 2, 4, 6, 8, 10, 12
        layer_counts = [2, 4, 6, 8, 10, 12]
        for num_layers in tqdm(layer_counts):
            encrypted_layers = list(range(num_layers))
            
            # Use realistic timing methodology from Simulation.ipynb
            timing_result = self._time_realistic_encrypted_inference(encrypted_layers, key, num_runs=None)
            
            # Use actual measured component times (no artificial splitting!)
            acm_enc_time = timing_result['acm_encrypt_time']
            acm_dec_time = timing_result['acm_decrypt_time']
            ffn_enc_time = timing_result['ffn_encrypt_time']
            ffn_dec_time = timing_result['ffn_decrypt_time']
            
            baseline_time = timing_result['baseline_time']
            total_time = timing_result['encrypted_time']
            overhead_ratio = total_time / baseline_time

            results.append(TimingBreakdown(
                total_time=total_time,
                forward_time=baseline_time,
                acm_enc_time=acm_enc_time,
                acm_dec_time=acm_dec_time,
                ffn_enc_time=ffn_enc_time,
                ffn_dec_time=ffn_dec_time,
                overhead_ratio=overhead_ratio
            ))

        return results

    def create_overhead_breakdown_plot(self,
                                     acm_results: List[TimingBreakdown],
                                     layers_results: List[TimingBreakdown],
                                     acm_results_batch64: List[TimingBreakdown],
                                     layers_results_batch64: List[TimingBreakdown],
                                     device_name: str = "GPU",
                                     output_dir: str = "results/analysis"):
        """Create comprehensive overhead breakdown plot with batch size comparison."""
        print("\nCreating overhead breakdown plot...")

        # Use class-level font size configuration
        FONT_SIZES = self.FONT_SIZES

        # Set up the plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # Create figure with subplots (2x2 grid) - wider for better bar visibility
        fig, axes = plt.subplots(2, 2, figsize=(36, 16))
        fig.suptitle(f'Dual Encryption Inference Overhead Analysis ({device_name})',
                    fontsize=FONT_SIZES['main_title'], fontweight='bold', y=0.98)

        # Add more spacing between title and subplots
        plt.subplots_adjust(top=1, hspace=0.3, wspace=0.2)

        # Colors for different components
        colors = {
            'Forward Pass': '#2ecc71',
            'ACM Encryption': '#e74c3c',      # Bright red
            'ACM Decryption': '#c0392b',      # Darker red
            'FFN Encryption': '#3498db',      # Bright blue
            'FFN Decryption': '#2980b9'       # Darker blue
        }

        # Plot 1: Overhead vs ACM iterations (timing breakdown) - Batch Size 1
        ax1 = axes[0, 0]
        n_iterations = [1, 3, 5, 8, 10, 12]  # Specific ACM iterations

        forward_times = [r.forward_time * 1000 for r in acm_results]
        acm_enc_times = [r.acm_enc_time * 1000 for r in acm_results]
        acm_dec_times = [r.acm_dec_time * 1000 for r in acm_results]
        ffn_enc_times = [r.ffn_enc_time * 1000 for r in acm_results]
        ffn_dec_times = [r.ffn_dec_time * 1000 for r in acm_results]

        # Use integer positions for bars
        bar_width = 0.8
        x_positions = np.arange(len(n_iterations))

        ax1.bar(x_positions, forward_times, width=bar_width, label='Forward Pass', color=colors['Forward Pass'], alpha=0.7)
        ax1.bar(x_positions, acm_enc_times, width=bar_width, bottom=forward_times, label='ACM Encryption',
                color=colors['ACM Encryption'], alpha=0.7)
        ax1.bar(x_positions, acm_dec_times, width=bar_width,
                bottom=[f + e for f, e in zip(forward_times, acm_enc_times)],
                label='ACM Decryption', color=colors['ACM Decryption'], alpha=0.7)
        ax1.bar(x_positions, ffn_enc_times, width=bar_width,
                bottom=[f + e + d for f, e, d in zip(forward_times, acm_enc_times, acm_dec_times)],
                label='FFN Encryption', color=colors['FFN Encryption'], alpha=0.7)
        ax1.bar(x_positions, ffn_dec_times, width=bar_width,
                bottom=[f + e + d + fe for f, e, d, fe in zip(forward_times, acm_enc_times, acm_dec_times, ffn_enc_times)],
                label='FFN Decryption', color=colors['FFN Decryption'], alpha=0.7)

        # Add overhead ratios above bars
        for i, (total_time, forward_time) in enumerate(zip(
            [f + ae + ad + fe + fd for f, ae, ad, fe, fd in zip(
                forward_times, acm_enc_times, acm_dec_times, ffn_enc_times, ffn_dec_times)],
            forward_times)):
            if forward_time > 0:
                overhead_ratio = total_time / forward_time
                ax1.text(i, total_time + 0.05, f'{overhead_ratio:.3f}×', ha='center', va='bottom',
                        fontsize=FONT_SIZES['ratio_text'], fontweight='bold')
            else:
                ax1.text(i, total_time + 0.05, 'N/A', ha='center', va='bottom',
                        fontsize=FONT_SIZES['ratio_text'], fontweight='bold')

        ax1.set_xlabel('Number of ACM Iterations', fontsize=FONT_SIZES['axis_labels'])
        ax1.set_ylabel('Time (ms)', fontsize=FONT_SIZES['axis_labels'])
        # ax1.set_title('(a) Performance vs ACM Iterations (Batch Size = 1)', fontsize=FONT_SIZES['subtitles'], pad=30)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(n_iterations, fontsize=FONT_SIZES['tick_labels'])
        ax1.tick_params(axis='y', labelsize=FONT_SIZES['tick_labels'])

        # Increase y-axis range for better spacing at the top
        y_max = max([f + ae + ad + fe + fd for f, ae, ad, fe, fd in zip(
            forward_times, acm_enc_times, acm_dec_times, ffn_enc_times, ffn_dec_times)])
        ax1.set_ylim(0, y_max * 1.15)  # Add 15% extra space at the top

        legend = ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=FONT_SIZES['legend'])
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_edgecolor('lightgray')
        ax1.grid(True, alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_alpha(0.3)
        ax1.spines['bottom'].set_alpha(0.3)

        # Plot 2: Overhead vs number of layers (timing breakdown) - Batch Size 1
        ax2 = axes[0, 1]
        n_layers = [2, 4, 6, 8, 10, 12]  # Specific layer counts

        forward_times = [r.forward_time * 1000 for r in layers_results]
        acm_enc_times = [r.acm_enc_time * 1000 for r in layers_results]
        acm_dec_times = [r.acm_dec_time * 1000 for r in layers_results]
        ffn_enc_times = [r.ffn_enc_time * 1000 for r in layers_results]
        ffn_dec_times = [r.ffn_dec_time * 1000 for r in layers_results]

        # Use integer positions for bars
        bar_width = 0.8
        x_positions = np.arange(len(n_layers))

        ax2.bar(x_positions, forward_times, width=bar_width, label='Forward Pass', color=colors['Forward Pass'], alpha=0.7)
        ax2.bar(x_positions, acm_enc_times, width=bar_width, bottom=forward_times, label='ACM Encryption',
                color=colors['ACM Encryption'], alpha=0.7)
        ax2.bar(x_positions, acm_dec_times, width=bar_width,
                bottom=[f + e for f, e in zip(forward_times, acm_enc_times)],
                label='ACM Decryption', color=colors['ACM Decryption'], alpha=0.7)
        ax2.bar(x_positions, ffn_enc_times, width=bar_width,
                bottom=[f + e + d for f, e, d in zip(forward_times, acm_enc_times, acm_dec_times)],
                label='FFN Encryption', color=colors['FFN Encryption'], alpha=0.7)
        ax2.bar(x_positions, ffn_dec_times, width=bar_width,
                bottom=[f + e + d + fe for f, e, d, fe in zip(forward_times, acm_enc_times, acm_dec_times, ffn_enc_times)],
                label='FFN Decryption', color=colors['FFN Decryption'], alpha=0.7)

        # Add overhead ratios above bars
        for i, (total_time, forward_time) in enumerate(zip(
            [f + ae + ad + fe + fd for f, ae, ad, fe, fd in zip(
                forward_times, acm_enc_times, acm_dec_times, ffn_enc_times, ffn_dec_times)],
            forward_times)):
            if forward_time > 0:
                overhead_ratio = total_time / forward_time
                ax2.text(i, total_time + 0.05, f'{overhead_ratio:.3f}×', ha='center', va='bottom',
                        fontsize=FONT_SIZES['ratio_text'], fontweight='bold')
            else:
                ax2.text(i, total_time + 0.05, 'N/A', ha='center', va='bottom',
                        fontsize=FONT_SIZES['ratio_text'], fontweight='bold')

        ax2.set_xlabel('Number of Encrypted Layers', fontsize=FONT_SIZES['axis_labels'])
        ax2.set_ylabel('Time (ms)', fontsize=FONT_SIZES['axis_labels'])
        # ax2.set_title('(b) Performance vs Number of Encrypted Layers (Batch Size = 1)', fontsize=FONT_SIZES['subtitles'], pad=30)
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(n_layers, fontsize=FONT_SIZES['tick_labels'])
        ax2.tick_params(axis='y', labelsize=FONT_SIZES['tick_labels'])

        # Increase y-axis range for better spacing at the top
        y_max = max([f + ae + ad + fe + fd for f, ae, ad, fe, fd in zip(
            forward_times, acm_enc_times, acm_dec_times, ffn_enc_times, ffn_dec_times)])
        ax2.set_ylim(0, y_max * 1.15)  # Add 15% extra space at the top

        # No legend for second subfigure to save vertical space
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_alpha(0.3)
        ax2.spines['bottom'].set_alpha(0.3)

        # Plot 3: Overhead vs ACM iterations (timing breakdown) - Batch Size 64
        ax3 = axes[1, 0]

        # Process batch size 64 data
        forward_times_b64 = [r.forward_time * 1000 for r in acm_results_batch64]
        acm_enc_times_b64 = [r.acm_enc_time * 1000 for r in acm_results_batch64]
        acm_dec_times_b64 = [r.acm_dec_time * 1000 for r in acm_results_batch64]
        ffn_enc_times_b64 = [r.ffn_enc_time * 1000 for r in acm_results_batch64]
        ffn_dec_times_b64 = [r.ffn_dec_time * 1000 for r in acm_results_batch64]

        # Use integer positions for bars
        x_positions = np.arange(len(n_iterations))

        ax3.bar(x_positions, forward_times_b64, width=bar_width, label='Forward Pass', color=colors['Forward Pass'], alpha=0.7)
        ax3.bar(x_positions, acm_enc_times_b64, width=bar_width, bottom=forward_times_b64, label='ACM Encryption',
                color=colors['ACM Encryption'], alpha=0.7)
        ax3.bar(x_positions, acm_dec_times_b64, width=bar_width,
                bottom=[f + e for f, e in zip(forward_times_b64, acm_enc_times_b64)],
                label='ACM Decryption', color=colors['ACM Decryption'], alpha=0.7)
        ax3.bar(x_positions, ffn_enc_times_b64, width=bar_width,
                bottom=[f + e + d for f, e, d in zip(forward_times_b64, acm_enc_times_b64, acm_dec_times_b64)],
                label='FFN Encryption', color=colors['FFN Encryption'], alpha=0.7)
        ax3.bar(x_positions, ffn_dec_times_b64, width=bar_width,
                bottom=[f + e + d + fe for f, e, d, fe in zip(forward_times_b64, acm_enc_times_b64, acm_dec_times_b64, ffn_enc_times_b64)],
                label='FFN Decryption', color=colors['FFN Decryption'], alpha=0.7)

        # Add overhead ratios above bars
        for i, (total_time, forward_time) in enumerate(zip(
            [f + ae + ad + fe + fd for f, ae, ad, fe, fd in zip(
                forward_times_b64, acm_enc_times_b64, acm_dec_times_b64, ffn_enc_times_b64, ffn_dec_times_b64)],
            forward_times_b64)):
            if forward_time > 0:
                overhead_ratio = total_time / forward_time
                ax3.text(i, total_time + 0.05, f'{overhead_ratio:.3f}×', ha='center', va='bottom',
                        fontsize=FONT_SIZES['ratio_text'], fontweight='bold')
            else:
                ax3.text(i, total_time + 0.05, 'N/A', ha='center', va='bottom',
                        fontsize=FONT_SIZES['ratio_text'], fontweight='bold')

        ax3.set_xlabel('Number of ACM Iterations', fontsize=FONT_SIZES['axis_labels'])
        ax3.set_ylabel('Time (ms)', fontsize=FONT_SIZES['axis_labels'])
        # ax3.set_title('(c) Performance vs ACM Iterations (Batch Size = 64)', fontsize=FONT_SIZES['subtitles'], pad=30)
        ax3.set_xticks(x_positions)
        ax3.set_xticklabels(n_iterations, fontsize=FONT_SIZES['tick_labels'])
        ax3.tick_params(axis='y', labelsize=FONT_SIZES['tick_labels'])

        # Increase y-axis range for better spacing at the top
        y_max = max([f + ae + ad + fe + fd for f, ae, ad, fe, fd in zip(
            forward_times_b64, acm_enc_times_b64, acm_dec_times_b64, ffn_enc_times_b64, ffn_dec_times_b64)])
        ax3.set_ylim(0, y_max * 1.15)  # Add 15% extra space at the top

        legend = ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=FONT_SIZES['legend'])
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_edgecolor('lightgray')
        ax3.grid(True, alpha=0.3)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_alpha(0.3)
        ax3.spines['bottom'].set_alpha(0.3)

        # Plot 4: Overhead vs number of layers (timing breakdown) - Batch Size 64
        ax4 = axes[1, 1]

        # Process batch size 64 data for layers
        forward_times_b64 = [r.forward_time * 1000 for r in layers_results_batch64]
        acm_enc_times_b64 = [r.acm_enc_time * 1000 for r in layers_results_batch64]
        acm_dec_times_b64 = [r.acm_dec_time * 1000 for r in layers_results_batch64]
        ffn_enc_times_b64 = [r.ffn_enc_time * 1000 for r in layers_results_batch64]
        ffn_dec_times_b64 = [r.ffn_dec_time * 1000 for r in layers_results_batch64]

        # Use integer positions for bars
        x_positions = np.arange(len(n_layers))
        ax4.bar(x_positions, forward_times_b64, width=bar_width, label='Forward Pass', color=colors['Forward Pass'], alpha=0.7)
        ax4.bar(x_positions, acm_enc_times_b64, width=bar_width, bottom=forward_times_b64, label='ACM Encryption',
                color=colors['ACM Encryption'], alpha=0.7)
        ax4.bar(x_positions, acm_dec_times_b64, width=bar_width,
                bottom=[f + e for f, e in zip(forward_times_b64, acm_enc_times_b64)],
                label='ACM Decryption', color=colors['ACM Decryption'], alpha=0.7)
        ax4.bar(x_positions, ffn_enc_times_b64, width=bar_width,
                bottom=[f + e + d for f, e, d in zip(forward_times_b64, acm_enc_times_b64, acm_dec_times_b64)],
                label='FFN Encryption', color=colors['FFN Encryption'], alpha=0.7)
        ax4.bar(x_positions, ffn_dec_times_b64, width=bar_width,
                bottom=[f + e + d + fe for f, e, d, fe in zip(forward_times_b64, acm_enc_times_b64, acm_dec_times_b64, ffn_enc_times_b64)],
                label='FFN Decryption', color=colors['FFN Decryption'], alpha=0.7)

        # Add overhead ratios above bars
        for i, (total_time, forward_time) in enumerate(zip(
            [f + ae + ad + fe + fd for f, ae, ad, fe, fd in zip(
                forward_times_b64, acm_enc_times_b64, acm_dec_times_b64, ffn_enc_times_b64, ffn_dec_times_b64)],
            forward_times_b64)):
            if forward_time > 0:
                overhead_ratio = total_time / forward_time
                ax4.text(i, total_time + 0.05, f'{overhead_ratio:.3f}×', ha='center', va='bottom',
                        fontsize=FONT_SIZES['ratio_text'], fontweight='bold')
            else:
                ax4.text(i, total_time + 0.05, 'N/A', ha='center', va='bottom',
                        fontsize=FONT_SIZES['ratio_text'], fontweight='bold')

        ax4.set_xlabel('Number of Encrypted Layers', fontsize=FONT_SIZES['axis_labels'])
        ax4.set_ylabel('Time (ms)', fontsize=FONT_SIZES['axis_labels'])
        # ax4.set_title('(d) Performance vs Number of Encrypted Layers (Batch Size = 64)', fontsize=FONT_SIZES['subtitles'], pad=30)
        ax4.set_xticks(x_positions)
        ax4.set_xticklabels(n_layers, fontsize=FONT_SIZES['tick_labels'])
        ax4.tick_params(axis='y', labelsize=FONT_SIZES['tick_labels'])

        # Increase y-axis range for better spacing at the top
        y_max = max([f + ae + ad + fe + fd for f, ae, ad, fe, fd in zip(
            forward_times_b64, acm_enc_times_b64, acm_dec_times_b64, ffn_enc_times_b64, ffn_dec_times_b64)])
        ax4.set_ylim(0, y_max * 1.15)  # Add 15% extra space at the top

        # No legend for second subfigure to save vertical space
        ax4.grid(True, alpha=0.3)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_alpha(0.3)
        ax4.spines['bottom'].set_alpha(0.3)

        plt.tight_layout()

        # Save both PNG and PDF versions
        png_path = f'{output_dir}/inference_overhead_breakdown_{device_name.lower()}.png'
        pdf_path = f'{output_dir}/inference_overhead_breakdown_{device_name.lower()}.pdf'

        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')

        print(f"✓ Saved: {png_path}")
        print(f"✓ Saved: {pdf_path}")

        # Save data as well
        self.save_overhead_data(acm_results, layers_results, output_dir, device_name)

    def save_overhead_data(self, acm_results: List[TimingBreakdown],
                          layers_results: List[TimingBreakdown],
                          output_dir: str,
                          device_name: str = "GPU"):
        """Save the overhead analysis data."""
        import pandas as pd

        # ACM iterations data
        acm_data = []
        acm_iterations = [1, 3, 5, 8, 10, 12]  # Specific iterations used
        for i, result in enumerate(acm_results):
            n_iter = acm_iterations[i]
            acm_data.append({
                'n_iterations': n_iter,
                'forward_time_ms': result.forward_time * 1000,
                'acm_enc_time_ms': result.acm_enc_time * 1000,
                'acm_dec_time_ms': result.acm_dec_time * 1000,
                'ffn_enc_time_ms': result.ffn_enc_time * 1000,
                'ffn_dec_time_ms': result.ffn_dec_time * 1000,
                'total_time_ms': result.total_time * 1000,
                'overhead_ratio': result.overhead_ratio
            })

        # Layers data
        layers_data = []
        layer_counts = [2, 4, 6, 8, 10, 12]  # Specific layer counts used
        for i, result in enumerate(layers_results):
            n_layers = layer_counts[i]
            layers_data.append({
                'n_layers': n_layers,
                'forward_time_ms': result.forward_time * 1000,
                'acm_enc_time_ms': result.acm_enc_time * 1000,
                'acm_dec_time_ms': result.acm_dec_time * 1000,
                'ffn_enc_time_ms': result.ffn_enc_time * 1000,
                'ffn_dec_time_ms': result.ffn_dec_time * 1000,
                'total_time_ms': result.total_time * 1000,
                'overhead_ratio': result.overhead_ratio
            })

        # Save to CSV with device suffix
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        acm_filename = f'acm_iterations_overhead_{device_name.lower()}.csv'
        layers_filename = f'layers_overhead_{device_name.lower()}.csv'
        
        pd.DataFrame(acm_data).to_csv(f'{output_dir}/{acm_filename}', index=False)
        pd.DataFrame(layers_data).to_csv(f'{output_dir}/{layers_filename}', index=False)

        print(f"✓ Saved data: {output_dir}/{acm_filename}")
        print(f"✓ Saved data: {output_dir}/{layers_filename}")

    def create_batch64_only_plot(self,
                                acm_results_batch64: List[TimingBreakdown],
                                layers_results_batch64: List[TimingBreakdown],
                                device_name: str = "GPU",
                                output_dir: str = "results/analysis"):
        """Create a separate figure with only batch size 64 plots (c) and (d)."""
        print("\nCreating batch size 64 only plot...")

        # Use class-level font size configuration
        FONT_SIZES = self.FONT_SIZES

        # Set up the plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # Create figure with subplots (1x2 grid) - wider for better bar visibility
        fig, axes = plt.subplots(1, 2, figsize=(36, 8))
        fig.suptitle(f'Dual Encryption Inference Overhead Analysis - Batch Size 64 ({device_name})',
                    fontsize=FONT_SIZES['main_title'], fontweight='bold', y=0.95)

        # Add more spacing between title and subplots
        plt.subplots_adjust(top=1, hspace=0.3, wspace=0.2)

        # Colors for different components
        colors = {
            'Forward Pass': '#2ecc71',
            'ACM Encryption': '#e74c3c',      # Bright red
            'ACM Decryption': '#c0392b',      # Darker red
            'FFN Encryption': '#3498db',      # Bright blue
            'FFN Decryption': '#2980b9'       # Darker blue
        }

        # Plot 1: Overhead vs ACM iterations (timing breakdown) - Batch Size 64
        ax1 = axes[0]
        n_iterations = [1, 3, 5, 8, 10, 12]  # Specific ACM iterations

        # Process batch size 64 data
        forward_times_b64 = [r.forward_time * 1000 for r in acm_results_batch64]
        acm_enc_times_b64 = [r.acm_enc_time * 1000 for r in acm_results_batch64]
        acm_dec_times_b64 = [r.acm_dec_time * 1000 for r in acm_results_batch64]
        ffn_enc_times_b64 = [r.ffn_enc_time * 1000 for r in acm_results_batch64]
        ffn_dec_times_b64 = [r.ffn_dec_time * 1000 for r in acm_results_batch64]

        # Use integer positions for bars
        bar_width = 0.8
        x_positions = np.arange(len(n_iterations))

        ax1.bar(x_positions, forward_times_b64, width=bar_width, label='Forward Pass', color=colors['Forward Pass'], alpha=0.7)
        ax1.bar(x_positions, acm_enc_times_b64, width=bar_width, bottom=forward_times_b64, label='ACM Encryption',
                color=colors['ACM Encryption'], alpha=0.7)
        ax1.bar(x_positions, acm_dec_times_b64, width=bar_width,
                bottom=[f + e for f, e in zip(forward_times_b64, acm_enc_times_b64)],
                label='ACM Decryption', color=colors['ACM Decryption'], alpha=0.7)
        ax1.bar(x_positions, ffn_enc_times_b64, width=bar_width,
                bottom=[f + e + d for f, e, d in zip(forward_times_b64, acm_enc_times_b64, acm_dec_times_b64)],
                label='FFN Encryption', color=colors['FFN Encryption'], alpha=0.7)
        ax1.bar(x_positions, ffn_dec_times_b64, width=bar_width,
                bottom=[f + e + d + fe for f, e, d, fe in zip(forward_times_b64, acm_enc_times_b64, acm_dec_times_b64, ffn_enc_times_b64)],
                label='FFN Decryption', color=colors['FFN Decryption'], alpha=0.7)

        # Add overhead ratios above bars
        for i, (total_time, forward_time) in enumerate(zip(
            [f + ae + ad + fe + fd for f, ae, ad, fe, fd in zip(
                forward_times_b64, acm_enc_times_b64, acm_dec_times_b64, ffn_enc_times_b64, ffn_dec_times_b64)],
            forward_times_b64)):
            if forward_time > 0:
                overhead_ratio = total_time / forward_time
                ax1.text(i, total_time + 0.05, f'{overhead_ratio:.3f}×', ha='center', va='bottom',
                        fontsize=FONT_SIZES['ratio_text'], fontweight='bold')
            else:
                ax1.text(i, total_time + 0.05, 'N/A', ha='center', va='bottom',
                        fontsize=FONT_SIZES['ratio_text'], fontweight='bold')

        ax1.set_xlabel('Number of ACM Iterations', fontsize=FONT_SIZES['axis_labels'])
        ax1.set_ylabel('Time (ms)', fontsize=FONT_SIZES['axis_labels'])
        # ax1.set_title('(a) Performance vs ACM Iterations (Batch Size = 64)', fontsize=FONT_SIZES['subtitles'], pad=30)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(n_iterations, fontsize=FONT_SIZES['tick_labels'])
        ax1.tick_params(axis='y', labelsize=FONT_SIZES['tick_labels'])

        # Increase y-axis range for better spacing at the top
        y_max = max([f + ae + ad + fe + fd for f, ae, ad, fe, fd in zip(
            forward_times_b64, acm_enc_times_b64, acm_dec_times_b64, ffn_enc_times_b64, ffn_dec_times_b64)])
        ax1.set_ylim(0, y_max * 1.15)  # Add 15% extra space at the top

        legend = ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=FONT_SIZES['legend'])
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_edgecolor('lightgray')
        ax1.grid(True, alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_alpha(0.3)
        ax1.spines['bottom'].set_alpha(0.3)

        # Plot 2: Overhead vs number of layers (timing breakdown) - Batch Size 64
        ax2 = axes[1]
        n_layers = [2, 4, 6, 8, 10, 12]  # Specific layer counts

        # Process batch size 64 data for layers
        forward_times_b64 = [r.forward_time * 1000 for r in layers_results_batch64]
        acm_enc_times_b64 = [r.acm_enc_time * 1000 for r in layers_results_batch64]
        acm_dec_times_b64 = [r.acm_dec_time * 1000 for r in layers_results_batch64]
        ffn_enc_times_b64 = [r.ffn_enc_time * 1000 for r in layers_results_batch64]
        ffn_dec_times_b64 = [r.ffn_dec_time * 1000 for r in layers_results_batch64]

        # Use integer positions for bars
        x_positions = np.arange(len(n_layers))
        ax2.bar(x_positions, forward_times_b64, width=bar_width, label='Forward Pass', color=colors['Forward Pass'], alpha=0.7)
        ax2.bar(x_positions, acm_enc_times_b64, width=bar_width, bottom=forward_times_b64, label='ACM Encryption',
                color=colors['ACM Encryption'], alpha=0.7)
        ax2.bar(x_positions, acm_dec_times_b64, width=bar_width,
                bottom=[f + e for f, e in zip(forward_times_b64, acm_enc_times_b64)],
                label='ACM Decryption', color=colors['ACM Decryption'], alpha=0.7)
        ax2.bar(x_positions, ffn_enc_times_b64, width=bar_width,
                bottom=[f + e + d for f, e, d in zip(forward_times_b64, acm_enc_times_b64, acm_dec_times_b64)],
                label='FFN Encryption', color=colors['FFN Encryption'], alpha=0.7)
        ax2.bar(x_positions, ffn_dec_times_b64, width=bar_width,
                bottom=[f + e + d + fe for f, e, d, fe in zip(forward_times_b64, acm_enc_times_b64, acm_dec_times_b64, ffn_enc_times_b64)],
                label='FFN Decryption', color=colors['FFN Decryption'], alpha=0.7)

        # Add overhead ratios above bars
        for i, (total_time, forward_time) in enumerate(zip(
            [f + ae + ad + fe + fd for f, ae, ad, fe, fd in zip(
                forward_times_b64, acm_enc_times_b64, acm_dec_times_b64, ffn_enc_times_b64, ffn_dec_times_b64)],
            forward_times_b64)):
            if forward_time > 0:
                overhead_ratio = total_time / forward_time
                ax2.text(i, total_time + 0.05, f'{overhead_ratio:.3f}×', ha='center', va='bottom',
                        fontsize=FONT_SIZES['ratio_text'], fontweight='bold')
            else:
                ax2.text(i, total_time + 0.05, 'N/A', ha='center', va='bottom',
                        fontsize=FONT_SIZES['ratio_text'], fontweight='bold')

        ax2.set_xlabel('Number of Encrypted Layers', fontsize=FONT_SIZES['axis_labels'])
        ax2.set_ylabel('Time (ms)', fontsize=FONT_SIZES['axis_labels'])
        # ax2.set_title('(b) Performance vs Number of Encrypted Layers (Batch Size = 64)', fontsize=FONT_SIZES['subtitles'], pad=30)
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(n_layers, fontsize=FONT_SIZES['tick_labels'])
        ax2.tick_params(axis='y', labelsize=FONT_SIZES['tick_labels'])

        # Increase y-axis range for better spacing at the top
        y_max = max([f + ae + ad + fe + fd for f, ae, ad, fe, fd in zip(
            forward_times_b64, acm_enc_times_b64, acm_dec_times_b64, ffn_enc_times_b64, ffn_dec_times_b64)])
        ax2.set_ylim(0, y_max * 1.15)  # Add 15% extra space at the top

        # No legend for second subfigure to save vertical space
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_alpha(0.3)
        ax2.spines['bottom'].set_alpha(0.3)

        plt.tight_layout()

        # Save both PNG and PDF versions
        png_path = f'{output_dir}/inference_overhead_batch64_only_{device_name.lower()}.png'
        pdf_path = f'{output_dir}/inference_overhead_batch64_only_{device_name.lower()}.pdf'

        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')

        print(f"✓ Saved: {png_path}")
        print(f"✓ Saved: {pdf_path}")

    def regenerate_plots_from_csv(self, device_name: str = "gpu", output_dir: str = "results/analysis", include_batch64: bool = True):
        """Regenerate plots from existing CSV files without rerunning the experiments.

        Args:
            device_name: Device name used in CSV filenames (e.g., 'gpu', 'cpu')
            output_dir: Directory where CSV files are located
            include_batch64: Whether to include batch size 64 data if available
        """
        import pandas as pd
        from pathlib import Path

        print(f"\nRegenerating plots from CSV files for {device_name.upper()}...")

        # Check if CSV files exist
        acm_csv_path = Path(output_dir) / f"acm_iterations_overhead_{device_name.lower()}.csv"
        layers_csv_path = Path(output_dir) / f"layers_overhead_{device_name.lower()}.csv"

        if not acm_csv_path.exists() or not layers_csv_path.exists():
            print(f"Error: CSV files not found in {output_dir}")
            print(f"Expected files: {acm_csv_path.name} and {layers_csv_path.name}")
            return

        # Read CSV data
        acm_df = pd.read_csv(acm_csv_path)
        layers_df = pd.read_csv(layers_csv_path)

        # Convert CSV data back to TimingBreakdown objects
        acm_results = []
        for _, row in acm_df.iterrows():
            forward_time_ms = row['forward_time_ms']
            total_time_ms = row['total_time_ms']

            # Ensure forward_time is never zero (use a small epsilon if needed)
            forward_time = forward_time_ms / 1000.0
            if forward_time < 1e-8:  # Very small threshold
                forward_time = 1e-8  # Prevent division by zero

            acm_results.append(TimingBreakdown(
                total_time=total_time_ms / 1000.0,
                forward_time=forward_time,
                acm_enc_time=row['acm_enc_time_ms'] / 1000.0,
                acm_dec_time=row['acm_dec_time_ms'] / 1000.0,
                ffn_enc_time=row['ffn_enc_time_ms'] / 1000.0,
                ffn_dec_time=row['ffn_dec_time_ms'] / 1000.0,
                overhead_ratio=row['overhead_ratio']
            ))

        layers_results = []
        for _, row in layers_df.iterrows():
            forward_time_ms = row['forward_time_ms']
            total_time_ms = row['total_time_ms']

            # Ensure forward_time is never zero (use a small epsilon if needed)
            forward_time = forward_time_ms / 1000.0
            if forward_time < 1e-8:  # Very small threshold
                forward_time = 1e-8  # Prevent division by zero

            layers_results.append(TimingBreakdown(
                total_time=total_time_ms / 1000.0,
                forward_time=forward_time,
                acm_enc_time=row['acm_enc_time_ms'] / 1000.0,
                acm_dec_time=row['acm_dec_time_ms'] / 1000.0,
                ffn_enc_time=row['ffn_enc_time_ms'] / 1000.0,
                ffn_dec_time=row['ffn_dec_time_ms'] / 1000.0,
                overhead_ratio=row['overhead_ratio']
            ))

        # Check if batch size 64 data is available
        acm_results_batch64 = None
        layers_results_batch64 = None

        if include_batch64:
            acm_csv_path_b64 = Path(output_dir) / f"acm_iterations_overhead_{device_name.lower()}_batch64.csv"
            layers_csv_path_b64 = Path(output_dir) / f"layers_overhead_{device_name.lower()}_batch64.csv"

            if acm_csv_path_b64.exists() and layers_csv_path_b64.exists():
                acm_df_b64 = pd.read_csv(acm_csv_path_b64)
                layers_df_b64 = pd.read_csv(layers_csv_path_b64)

                acm_results_batch64 = []
                for _, row in acm_df_b64.iterrows():
                    forward_time_ms = row['forward_time_ms']
                    total_time_ms = row['total_time_ms']

                    # Ensure forward_time is never zero (use a small epsilon if needed)
                    forward_time = forward_time_ms / 1000.0
                    if forward_time < 1e-8:  # Very small threshold
                        forward_time = 1e-8  # Prevent division by zero

                    acm_results_batch64.append(TimingBreakdown(
                        total_time=total_time_ms / 1000.0,
                        forward_time=forward_time,
                        acm_enc_time=row['acm_enc_time_ms'] / 1000.0,
                        acm_dec_time=row['acm_dec_time_ms'] / 1000.0,
                        ffn_enc_time=row['ffn_enc_time_ms'] / 1000.0,
                        ffn_dec_time=row['ffn_dec_time_ms'] / 1000.0,
                        overhead_ratio=row['overhead_ratio']
                    ))

                layers_results_batch64 = []
                for _, row in layers_df_b64.iterrows():
                    forward_time_ms = row['forward_time_ms']
                    total_time_ms = row['total_time_ms']

                    # Ensure forward_time is never zero (use a small epsilon if needed)
                    forward_time = forward_time_ms / 1000.0
                    if forward_time < 1e-8:  # Very small threshold
                        forward_time = 1e-8  # Prevent division by zero

                    layers_results_batch64.append(TimingBreakdown(
                        total_time=total_time_ms / 1000.0,
                        forward_time=forward_time,
                        acm_enc_time=row['acm_enc_time_ms'] / 1000.0,
                        acm_dec_time=row['acm_dec_time_ms'] / 1000.0,
                        ffn_enc_time=row['ffn_enc_time_ms'] / 1000.0,
                        ffn_dec_time=row['ffn_dec_time_ms'] / 1000.0,
                        overhead_ratio=row['overhead_ratio']
                    ))
                print("✓ Found batch size 64 data")
            else:
                print("⚠ Batch size 64 CSV files not found, using only batch size 1 data")
                include_batch64 = False

        # Create plots
        if include_batch64 and acm_results_batch64 and layers_results_batch64:
            self.create_overhead_breakdown_plot(
                acm_results, layers_results,
                acm_results_batch64, layers_results_batch64,
                device_name.upper(), output_dir
            )
            # Also create the separate batch64-only plot
            self.create_batch64_only_plot(
                acm_results_batch64, layers_results_batch64,
                device_name.upper(), output_dir
            )
        else:
            # Create dummy batch64 results for plotting (this won't be used but needed for the function signature)
            dummy_batch64 = [TimingBreakdown(0, 0) for _ in acm_results]
            self.create_overhead_breakdown_plot(
                acm_results, layers_results,
                dummy_batch64, dummy_batch64,
                device_name.upper(), output_dir
            )

        print(f"✓ Successfully regenerated plots for {device_name.upper()}")

    def run_complete_analysis(self, output_dir: str = "results/analysis"):
        """Run the complete overhead analysis with batch size comparison."""
        print("Starting complete inference overhead analysis...")

        # Store current batch size
        original_batch_size = self.batch_size
        
        # Analyze with batch size 1 (single inference)
        self.batch_size = 1
        print(f"\nRunning analysis with batch_size={self.batch_size}...")

        # Sample input will be handled by _get_sample_input() method
        
        # Analyze ACM iterations overhead
        acm_results = self.analyze_acm_iterations_overhead(max_iterations=19)

        # Analyze layers overhead
        layers_results = self.analyze_layers_overhead(max_layers=12)

        # Save batch size 1 results to CSV
        device_name = "GPU" if self.device.type == 'cuda' else "CPU"
        self.save_overhead_data(acm_results, layers_results, output_dir, device_name)

        # Now analyze with batch size 64
        self.batch_size = 64
        print(f"\nRunning analysis with batch_size={self.batch_size}...")

        # Sample input will be handled by _get_sample_input() method

        # Analyze ACM iterations overhead with batch size 64
        acm_results_batch64 = self.analyze_acm_iterations_overhead(max_iterations=19)

        # Analyze layers overhead with batch size 64
        layers_results_batch64 = self.analyze_layers_overhead(max_layers=12)

        # Save batch size 64 results to CSV with suffix
        self.save_overhead_data(acm_results_batch64, layers_results_batch64, output_dir, f"{device_name}_batch64")
        
        # Restore original batch size
        self.batch_size = original_batch_size

        # Create comprehensive plot with both batch sizes
        device_name = "GPU" if self.device.type == 'cuda' else "CPU"
        self.create_overhead_breakdown_plot(
            acm_results, layers_results,
            acm_results_batch64, layers_results_batch64,
            device_name, output_dir
        )

        # Create separate batch size 64 only plot
        self.create_batch64_only_plot(
            acm_results_batch64, layers_results_batch64,
            device_name, output_dir
        )

        print("\nInference overhead analysis complete!")
        print(f"Results saved to: {output_dir}/")


def run_device_analysis(device: str):
    """Run analysis for a specific device."""
    print(f"\n{'='*60}")
    print(f"Running analysis on {device.upper()}")
    print(f"{'='*60}")
    
    # Initialize analyzer with batch_size=1 (single sequential inference)
    # This simulates the real-world scenario where each inference processes
    # one input and requires sequential encryption/decryption of all layers
    print(f"Initializing analyzer with batch_size=1 (sequential inference) on {device.upper()}...")
    analyzer = InferenceOverheadAnalyzer(device=device, batch_size=1)
    
    # Run complete analysis
    analyzer.run_complete_analysis()


def regenerate_plots(device_name: str = "gpu", output_dir: str = "results/analysis", include_batch64: bool = True):
    """Function to regenerate plots from existing CSV files.

    Args:
        device_name: Device name used in CSV filenames (e.g., 'gpu', 'cpu')
        output_dir: Directory where CSV files are located
        include_batch64: Whether to include batch size 64 data if available
    """
    # Create a minimal analyzer instance just for plotting (skip ImageNet loading)
    analyzer = InferenceOverheadAnalyzer(device="cpu", batch_size=1, skip_imagenet=True)  # Device doesn't matter for CSV regeneration
    analyzer.regenerate_plots_from_csv(device_name, output_dir, include_batch64)


def regenerate_batch64_only_plot(device_name: str = "gpu", output_dir: str = "results/analysis"):
    """Function to regenerate only the batch size 64 plot from existing CSV files.

    Args:
        device_name: Device name used in CSV filenames (e.g., 'gpu', 'cpu')
        output_dir: Directory where CSV files are located
    """
    import pandas as pd
    from pathlib import Path

    print(f"\nRegenerating batch size 64 only plot from CSV files for {device_name.upper()}...")

    # Check if batch size 64 CSV files exist
    acm_csv_path_b64 = Path(output_dir) / f"acm_iterations_overhead_{device_name.lower()}_batch64.csv"
    layers_csv_path_b64 = Path(output_dir) / f"layers_overhead_{device_name.lower()}_batch64.csv"

    if not acm_csv_path_b64.exists() or not layers_csv_path_b64.exists():
        print(f"Error: Batch size 64 CSV files not found in {output_dir}")
        print(f"Expected files: {acm_csv_path_b64.name} and {layers_csv_path_b64.name}")
        return

    # Read CSV data
    acm_df_b64 = pd.read_csv(acm_csv_path_b64)
    layers_df_b64 = pd.read_csv(layers_csv_path_b64)

    # Convert CSV data back to TimingBreakdown objects
    acm_results_batch64 = []
    for _, row in acm_df_b64.iterrows():
        forward_time_ms = row['forward_time_ms']
        total_time_ms = row['total_time_ms']

        # Ensure forward_time is never zero (use a small epsilon if needed)
        forward_time = forward_time_ms / 1000.0
        if forward_time < 1e-8:  # Very small threshold
            forward_time = 1e-8  # Prevent division by zero

        acm_results_batch64.append(TimingBreakdown(
            total_time=total_time_ms / 1000.0,
            forward_time=forward_time,
            acm_enc_time=row['acm_enc_time_ms'] / 1000.0,
            acm_dec_time=row['acm_dec_time_ms'] / 1000.0,
            ffn_enc_time=row['ffn_enc_time_ms'] / 1000.0,
            ffn_dec_time=row['ffn_dec_time_ms'] / 1000.0,
            overhead_ratio=row['overhead_ratio']
        ))

    layers_results_batch64 = []
    for _, row in layers_df_b64.iterrows():
        forward_time_ms = row['forward_time_ms']
        total_time_ms = row['total_time_ms']

        # Ensure forward_time is never zero (use a small epsilon if needed)
        forward_time = forward_time_ms / 1000.0
        if forward_time < 1e-8:  # Very small threshold
            forward_time = 1e-8  # Prevent division by zero

        layers_results_batch64.append(TimingBreakdown(
            total_time=total_time_ms / 1000.0,
            forward_time=forward_time,
            acm_enc_time=row['acm_enc_time_ms'] / 1000.0,
            acm_dec_time=row['acm_dec_time_ms'] / 1000.0,
            ffn_enc_time=row['ffn_enc_time_ms'] / 1000.0,
            ffn_dec_time=row['ffn_dec_time_ms'] / 1000.0,
            overhead_ratio=row['overhead_ratio']
        ))

    # Create a minimal analyzer instance just for plotting
    analyzer = InferenceOverheadAnalyzer(device="cpu", batch_size=1)  # Device doesn't matter for CSV regeneration

    # Create the batch64-only plot
    analyzer.create_batch64_only_plot(
        acm_results_batch64, layers_results_batch64,
        device_name.upper(), output_dir
    )

    print(f"✓ Successfully regenerated batch size 64 only plot for {device_name.upper()}")

def main():
    """Main function to run the inference overhead analysis on both GPU and CPU.

    To adjust the number of evaluation runs:
    - Modify NUM_RUNS_CPU for CPU evaluation
    - Modify NUM_RUNS_GPU for GPU evaluation

    To regenerate plots from existing CSV files, use:
    regenerate_plots(device_name="gpu", output_dir="results/analysis")

    Command line usage:
    python inference_overhead_analysis.py --replot --device gpu --output-dir results/analysis
    python inference_overhead_analysis.py --replot --device cuda:1 --output-dir results/analysis
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Inference Overhead Analysis for Dual Encryption')
    parser.add_argument('--replot', action='store_true',
                       help='Regenerate plots from existing CSV files instead of running new experiments')
    parser.add_argument('--device', type=str, default='gpu',
                       help='Device for computation (gpu, cpu, cuda, cuda:0, cuda:1, etc.)')
    parser.add_argument('--output-dir', type=str, default='results/analysis',
                       help='Directory containing CSV files or where to save results (default: results/analysis)')
    parser.add_argument('--include-batch64', action='store_true', default=True,
                       help='Include batch size 64 data if available (default: True)')

    args = parser.parse_args()

    # Handle CSV regeneration mode
    if args.replot:
        print("Regenerating plots from CSV files...")
        # For CSV regeneration, normalize device name for file lookup
        if args.device.startswith('cuda') or args.device == 'gpu':
            device_name = 'gpu'
        else:
            device_name = 'cpu'
        regenerate_plots(
            device_name=device_name,
            output_dir=args.output_dir,
            include_batch64=args.include_batch64
        )
        return

    # Check available devices and validate requested device
    available_devices = []
    requested_device = args.device

    if torch.cuda.is_available():
        available_devices.append("cuda")
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            print(f"✓ Multiple CUDA devices available: {torch.cuda.device_count()} devices")

    available_devices.append("cpu")
    print("✓ CPU available")

    # Validate and normalize the requested device
    if requested_device == 'gpu':
        if torch.cuda.is_available():
            device_to_run = 'cuda'
        else:
            print("Warning: GPU requested but CUDA not available, using CPU")
            device_to_run = 'cpu'
    elif requested_device.startswith('cuda'):
        if torch.cuda.is_available():
            # Validate specific CUDA device if specified
            if ':' in requested_device:
                device_id = int(requested_device.split(':')[1])
                if device_id >= torch.cuda.device_count():
                    print(f"Error: CUDA device {device_id} not available. Available devices: 0-{torch.cuda.device_count()-1}")
                    return
            device_to_run = requested_device
        else:
            print("Error: CUDA requested but not available")
            return
    elif requested_device == 'cpu':
        device_to_run = 'cpu'
    else:
        print(f"Error: Invalid device '{requested_device}'. Use 'gpu', 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.")
        return

    print(f"\nRunning analysis on device: {device_to_run.upper()}")
    print(f"Configuration: CPU runs={NUM_RUNS_CPU}, GPU runs={NUM_RUNS_GPU}")

    # Run analysis on the specified device
    try:
        run_device_analysis(device_to_run)
    except Exception as e:
        print(f"Error running analysis on {device_to_run.upper()}: {str(e)}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()

# Example usage:
#
# 1. Run the full analysis (generates CSV files and plots):
#    python inference_overhead_analysis.py
#
# 2. Regenerate plots from existing CSV files:
#    python inference_overhead_analysis.py --replot --device gpu
#
# 3. Regenerate plots from CSV files for CPU:
#    python inference_overhead_analysis.py --replot --device cpu
#
# 4. Use Python API directly:
#    from inference_overhead_analysis import regenerate_plots
#    regenerate_plots(device_name="gpu", output_dir="results/analysis")
#
# Quick test (run this to verify the fix works):
# python -c "from src.analysis.inference_overhead_analysis import regenerate_plots; regenerate_plots('gpu', 'results/analysis')"
