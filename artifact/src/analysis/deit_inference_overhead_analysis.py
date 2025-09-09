#!/usr/bin/env python3
"""
Inference Overhead Analysis for DeiT Dual Encryption on CIFAR-100

This module analyzes the performance overhead of the dual encryption scheme
during inference for DeiT models on CIFAR-100, providing detailed timing breakdowns.

Usage:
    python src/analysis/deit_inference_overhead_analysis.py --device cuda
"""

import torch
import numpy as np
import time
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
import matplotlib.pyplot as plt
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.cifar100_eval import CIFAR100ValidationDataset
from src.encryption.dual_encryption import DualEncryption
from src.encryption.arnold_transform import arnold, iarnold

# Constants for timing measurements
NUM_RUNS_GPU = 100
NUM_RUNS_CPU = 20
WARMUP_RUNS = 10


class DeiTInferenceOverheadAnalyzer:
    """Analyzes inference overhead for DeiT models with dual encryption on CIFAR-100."""
    
    def __init__(self, model_name="facebook/deit-tiny-patch16-224", device="cuda", 
                 cifar100_path="dataset/cifar100"):
        """
        Initialize the analyzer.

        Args:
            model_name: HuggingFace model identifier
            device: Device for computation ('cuda', 'cuda:0', 'cuda:1', 'cpu')
            cifar100_path: Path to CIFAR-100 dataset
        """
        # Handle device specification with validation
        if device.startswith('cuda') and torch.cuda.is_available():
            self.device = torch.device(device)
        elif device == 'cpu':
            self.device = torch.device('cpu')
        else:
            print(f"Warning: CUDA not available or invalid device '{device}', falling back to CPU")
            self.device = torch.device('cpu')
        self.model_name = model_name
        self.cifar100_path = cifar100_path
        
        # Load model and processor
        print(f"Loading {model_name}...")

        # Use fine-tuned CIFAR-100 model if available, otherwise try the provided model
        finetuned_model_path = Path("../model/deit-tiny-cifar100-finetuned/model")

        if finetuned_model_path.exists():
            print("Using fine-tuned CIFAR-100 model...")
            config = ViTConfig.from_pretrained(finetuned_model_path)
            config.num_labels = 100  # CIFAR-100 has 100 classes
            self.model = ViTForImageClassification.from_pretrained(finetuned_model_path, config=config)
            self.processor = ViTImageProcessor.from_pretrained(finetuned_model_path)
        else:
            # Fallback to provided model path
            model_path = Path(model_name)
            if model_path.exists() and (model_path / "model").exists():
                # Local model with config
                config = ViTConfig.from_pretrained(model_path / "model")
                config.num_labels = 100  # CIFAR-100 has 100 classes
                self.model = ViTForImageClassification.from_pretrained(model_path / "model", config=config)
                self.processor = ViTImageProcessor.from_pretrained(model_path / "model")
            else:
                # HuggingFace model or simple path
                self.model = ViTForImageClassification.from_pretrained(model_name)
                self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")
        
        # Create test dataset
        self.test_dataset = CIFAR100ValidationDataset(
            cifar100_path, self.processor, split='test'
        )
        
        # Create test batch
        self.test_batch = self._create_test_batch()
        
        # Initialize dual encryptor with model dimensions
        self.dual_encryptor = DualEncryption.from_model(
            model=self.model,
            arnold_key=[3, 1, 1, 1, 2],
            device=device
        )
        
        print(f"Analyzer initialized on {self.device}")
        print(f"Model has {len(self._get_encoder_layers())} layers")

    def _get_encoder_layers(self):
        """Get encoder layers from ViT model."""
        return self.model.vit.encoder.layer
    
    def _create_test_batch(self, batch_size=64):
        """Create a test batch for timing measurements."""
        indices = torch.randperm(len(self.test_dataset))[:batch_size]
        batch_data = []
        
        for idx in indices:
            pixel_values, _ = self.test_dataset[idx]
            batch_data.append(pixel_values)
        
        batch_tensor = torch.stack(batch_data).to(self.device)
        return batch_tensor
    
    def _get_layer_weights(self, layer_idx):
        """Extract attention and FFN weights from a specific layer."""
        layer = self._get_encoder_layers()[layer_idx]
        
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
    
    def _apply_encrypted_weights(self, layer_idx, encrypted_attention, encrypted_ffn):
        """Apply encrypted weights to a specific layer."""
        layer = self._get_encoder_layers()[layer_idx]
        
        # Apply encrypted attention weights
        layer.attention.attention.query.weight.data = encrypted_attention['query']
        layer.attention.attention.key.weight.data = encrypted_attention['key']
        layer.attention.attention.value.weight.data = encrypted_attention['value']
        layer.attention.output.dense.weight.data = encrypted_attention['output']
        
        # Apply encrypted FFN weights
        layer.intermediate.dense.weight.data = encrypted_ffn['intermediate']
        layer.output.dense.weight.data = encrypted_ffn['output']
    
    def _time_baseline_inference(self, num_runs=None, warmup_runs=WARMUP_RUNS):
        """Time baseline inference without encryption."""
        if num_runs is None:
            num_runs = NUM_RUNS_GPU if self.device.type == 'cuda' else NUM_RUNS_CPU
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model(self.test_batch)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Actual timing
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = self.model(self.test_batch)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return np.mean(times)
    
    def _time_encrypted_inference(self, encrypted_layers, acm_key, num_runs=None, warmup_runs=WARMUP_RUNS):
        """Time encrypted inference with actual weight manipulation."""
        if num_runs is None:
            num_runs = NUM_RUNS_GPU if self.device.type == 'cuda' else NUM_RUNS_CPU
        
        # Store original model state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Encrypt specified layers
        for layer_idx in encrypted_layers:
            attention_weights, ffn_weights = self._get_layer_weights(layer_idx)
            encryption_result = self.dual_encryptor.encrypt_layer_weights(
                attention_weights, ffn_weights, permutation_matrix_idx=0
            )
            self._apply_encrypted_weights(
                layer_idx, 
                encryption_result.encrypted_attention,
                encryption_result.encrypted_ffn
            )
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model(self.test_batch)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timing measurements
        baseline_times = []
        encrypted_times = []
        acm_decrypt_times = []
        acm_encrypt_times = []
        ffn_decrypt_times = []
        ffn_encrypt_times = []
        
        for _ in range(num_runs):
            # Time baseline forward pass (with encrypted weights)
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = self.model(self.test_batch)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            baseline_time = time.perf_counter() - start_time
            baseline_times.append(baseline_time)
            
            # Time decryption + forward pass + encryption
            run_acm_decrypt_time = 0
            run_acm_encrypt_time = 0
            run_ffn_decrypt_time = 0
            run_ffn_encrypt_time = 0
            
            start_total = time.perf_counter()
            
            # Decrypt layers
            for layer_idx in encrypted_layers:
                attention_weights, ffn_weights = self._get_layer_weights(layer_idx)
                
                # Time ACM decryption
                start_acm = time.perf_counter()
                decrypted_attention = {}
                for key, weight in attention_weights.items():
                    decrypted_attention[key] = iarnold(weight, acm_key)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                run_acm_decrypt_time += time.perf_counter() - start_acm
                
                # Time FFN decryption
                start_ffn = time.perf_counter()
                decrypted_ffn = self.dual_encryptor.decrypt_ffn_weights(ffn_weights, 0)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                run_ffn_decrypt_time += time.perf_counter() - start_ffn
                
                # Apply decrypted weights
                self._apply_encrypted_weights(layer_idx, decrypted_attention, decrypted_ffn)
            
            # Forward pass with decrypted weights
            with torch.no_grad():
                _ = self.model(self.test_batch)
            
            # Re-encrypt layers
            for layer_idx in encrypted_layers:
                attention_weights, ffn_weights = self._get_layer_weights(layer_idx)
                
                # Time ACM encryption
                start_acm = time.perf_counter()
                encrypted_attention = {}
                for key, weight in attention_weights.items():
                    encrypted_attention[key] = arnold(weight, acm_key)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                run_acm_encrypt_time += time.perf_counter() - start_acm
                
                # Time FFN encryption
                start_ffn = time.perf_counter()
                encrypted_ffn = self.dual_encryptor.encrypt_ffn_weights(ffn_weights, 0)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                run_ffn_encrypt_time += time.perf_counter() - start_ffn
                
                # Apply encrypted weights
                self._apply_encrypted_weights(layer_idx, encrypted_attention, encrypted_ffn)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            total_time = time.perf_counter() - start_total
            
            encrypted_times.append(total_time)
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
            'total_overhead': np.mean(acm_decrypt_times) + np.mean(acm_encrypt_times) + 
                           np.mean(ffn_decrypt_times) + np.mean(ffn_encrypt_times)
        }
    
    def analyze_acm_iterations_overhead(self, encrypted_layers=[0, 1, 2, 3], acm_iterations=[1, 2, 3, 5, 7, 10]):
        """Analyze overhead as a function of ACM iterations."""
        print(f"Analyzing ACM iterations overhead for layers {encrypted_layers}...")
        
        results = []
        baseline_time = self._time_baseline_inference()
        
        for n_iter in tqdm(acm_iterations):
            key = [n_iter, 1, 1, 1, 2]  # Arnold key
            
            timing_result = self._time_encrypted_inference(encrypted_layers, key)
            
            overhead_ratio = timing_result['encrypted_time'] / baseline_time
            
            results.append({
                'acm_iterations': n_iter,
                'baseline_time': baseline_time,
                'encrypted_time': timing_result['encrypted_time'],
                'overhead_ratio': overhead_ratio,
                'acm_decrypt_time': timing_result['acm_decrypt_time'],
                'acm_encrypt_time': timing_result['acm_encrypt_time'],
                'ffn_decrypt_time': timing_result['ffn_decrypt_time'],
                'ffn_encrypt_time': timing_result['ffn_encrypt_time'],
                'total_overhead': timing_result['total_overhead']
            })
        
        return results
    
    def analyze_layer_count_overhead(self, max_layers=None, acm_key=[3, 1, 1, 1, 2]):
        """Analyze overhead as a function of number of encrypted layers."""
        if max_layers is None:
            max_layers = min(6, len(self._get_encoder_layers()))  # DeiT-tiny has 12 layers
        
        print(f"Analyzing layer count overhead up to {max_layers} layers...")
        
        results = []
        baseline_time = self._time_baseline_inference()
        
        # Test specific layer counts: 1, 2, 3, 4, 5, 6
        layer_counts = list(range(1, max_layers + 1))
        for num_layers in tqdm(layer_counts):
            encrypted_layers = list(range(num_layers))
            
            timing_result = self._time_encrypted_inference(encrypted_layers, acm_key)
            
            overhead_ratio = timing_result['encrypted_time'] / baseline_time
            
            results.append({
                'num_encrypted_layers': num_layers,
                'baseline_time': baseline_time,
                'encrypted_time': timing_result['encrypted_time'],
                'overhead_ratio': overhead_ratio,
                'acm_decrypt_time': timing_result['acm_decrypt_time'],
                'acm_encrypt_time': timing_result['acm_encrypt_time'],
                'ffn_decrypt_time': timing_result['ffn_decrypt_time'],
                'ffn_encrypt_time': timing_result['ffn_encrypt_time'],
                'total_overhead': timing_result['total_overhead']
            })
        
        return results
    
    def save_results(self, results, output_path):
        """Save analysis results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def plot_results(self, results, output_dir="results/analysis"):
        """Plot analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if 'acm_iterations' in results[0]:
            # Plot ACM iterations overhead
            iterations = [r['acm_iterations'] for r in results]
            overhead_ratios = [r['overhead_ratio'] for r in results]
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, overhead_ratios, 'b-o', linewidth=2, markersize=8)
            plt.xlabel('ACM Iterations')
            plt.ylabel('Overhead Ratio')
            plt.title('DeiT-tiny Inference Overhead vs ACM Iterations (CIFAR-100)')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'deit_acm_iterations_overhead.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        elif 'num_encrypted_layers' in results[0]:
            # Plot layer count overhead
            layer_counts = [r['num_encrypted_layers'] for r in results]
            overhead_ratios = [r['overhead_ratio'] for r in results]
            
            plt.figure(figsize=(10, 6))
            plt.plot(layer_counts, overhead_ratios, 'r-s', linewidth=2, markersize=8)
            plt.xlabel('Number of Encrypted Layers')
            plt.ylabel('Overhead Ratio')
            plt.title('DeiT-tiny Inference Overhead vs Number of Encrypted Layers (CIFAR-100)')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'deit_layer_count_overhead.png', dpi=300, bbox_inches='tight')
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='DeiT Inference Overhead Analysis')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for computation (e.g., cuda, cuda:0, cuda:1, cpu)')
    parser.add_argument('--model', type=str, default='facebook/deit-tiny-patch16-224',
                       help='HuggingFace model identifier')
    parser.add_argument('--cifar100-path', type=str, default='dataset/cifar100',
                       help='Path to CIFAR-100 dataset')
    parser.add_argument('--output-dir', type=str, default='results/analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DeiTInferenceOverheadAnalyzer(
        model_name=args.model,
        device=args.device,
        cifar100_path=args.cifar100_path
    )
    
    # Run analyses
    print("Running ACM iterations analysis...")
    acm_results = analyzer.analyze_acm_iterations_overhead()
    analyzer.save_results(acm_results, f"{args.output_dir}/deit_acm_iterations_overhead.json")
    analyzer.plot_results(acm_results, args.output_dir)
    
    print("Running layer count analysis...")
    layer_results = analyzer.analyze_layer_count_overhead()
    analyzer.save_results(layer_results, f"{args.output_dir}/deit_layer_count_overhead.json")
    analyzer.plot_results(layer_results, args.output_dir)
    
    print(f"Analysis completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
