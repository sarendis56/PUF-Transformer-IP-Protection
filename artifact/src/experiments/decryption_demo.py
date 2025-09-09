#!/usr/bin/env python3
"""
Decryption demonstration script for Claim 2.
Shows that decryption with correct keys restores original performance.
"""

import torch
import sys
from pathlib import Path
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.cifar100_eval import CIFAR100ModelEvaluator
from src.encryption.dual_encryption import DualEncryption

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "results/claim2/deit_encryption_*/checkpoints/final"

    print("=== Decryption Demonstration ===")
    print(f"Loading encrypted model from: {model_path}")

    # Load encrypted model (saved as ViT with CIFAR-100 config)
    config = ViTConfig.from_pretrained(Path(model_path) / "model")
    config.num_labels = 100  # CIFAR-100 has 100 classes
    model = ViTForImageClassification.from_pretrained(Path(model_path) / "model", config=config)
    processor = ViTImageProcessor.from_pretrained(Path(model_path) / "model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Initialize evaluator
    evaluator = CIFAR100ModelEvaluator(
        cifar100_path="../dataset/cifar100",  # Correct path from artifact directory
        batch_size=64,
        num_workers=8,
        device=device.type
    )
    
    # Evaluate encrypted model
    print("\n1. Evaluating encrypted model performance...")
    encrypted_metrics = evaluator.evaluate_model(model, processor)
    print(f"Encrypted model accuracy: {encrypted_metrics.top1_accuracy:.2f}%")
    
    # Initialize decryptor with correct keys and model dimensions
    dual_encryptor = DualEncryption.from_model(
        model=model,
        arnold_key=[3, 1, 1, 1, 2],  # Correct key from config
        device=device.type
    )
    
    # Load encryption metadata to know which layers were encrypted
    import json
    metadata_path = Path(model_path) / "performance.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Get the original accuracy before encryption
    initial_accuracy = metadata['initial_accuracy']
    print(f"Original model accuracy (before encryption): {initial_accuracy:.2f}%")

    encrypted_layers = []
    for step in metadata['encryption_steps']:
        encrypted_layers.append(step['layer_idx'])

    print(f"\n2. Decrypting {len(encrypted_layers)} encrypted layers: {encrypted_layers}")
    
    # Decrypt each layer
    for layer_idx in encrypted_layers:
        layer = model.vit.encoder.layer[layer_idx]

        # Get current (encrypted) weights
        attention_weights = {
            'query': layer.attention.attention.query.weight.data,
            'key': layer.attention.attention.key.weight.data,
            'value': layer.attention.attention.value.weight.data,
            'output': layer.attention.output.dense.weight.data
        }

        ffn_weights = {
            'intermediate': layer.intermediate.dense.weight.data,
            'output': layer.output.dense.weight.data
        }

        # Find the permutation matrix index used for this layer from metadata
        perm_matrix_idx = 0  # Default to first permutation matrix
        for step in metadata['encryption_steps']:
            if step['layer_idx'] == layer_idx:
                perm_matrix_idx = step.get('permutation_matrix_idx', 0)
                break

        # Decrypt attention weights using the DualEncryption method
        decrypted_attention = dual_encryptor.decrypt_attention_weights(attention_weights)

        # Decrypt FFN weights using permutation inverse
        decrypted_ffn = dual_encryptor.decrypt_ffn_weights(ffn_weights, perm_matrix_idx)

        # Apply decrypted weights
        layer.attention.attention.query.weight.data = decrypted_attention['query']
        layer.attention.attention.key.weight.data = decrypted_attention['key']
        layer.attention.attention.value.weight.data = decrypted_attention['value']
        layer.attention.output.dense.weight.data = decrypted_attention['output']

        layer.intermediate.dense.weight.data = decrypted_ffn['intermediate']
        layer.output.dense.weight.data = decrypted_ffn['output']

        print(f"   Decrypted layer {layer_idx} (perm matrix {perm_matrix_idx})")
    
    # Evaluate decrypted model
    print("\n3. Evaluating decrypted model performance...")
    decrypted_metrics = evaluator.evaluate_model(model, processor)
    print(f"Decrypted model accuracy: {decrypted_metrics.top1_accuracy:.2f}%")
    
    # Calculate restoration
    if encrypted_metrics.top1_accuracy > 0:
        restoration_rate = decrypted_metrics.top1_accuracy / encrypted_metrics.top1_accuracy
    else:
        restoration_rate = float('inf') if decrypted_metrics.top1_accuracy > 0 else 1.0

    # Calculate restoration metrics
    accuracy_recovery = (decrypted_metrics.top1_accuracy / initial_accuracy)

    print(f"\n=== Performance Restoration Results ===")
    print(f"Original accuracy (before encryption): {initial_accuracy:.2f}%")
    print(f"Encrypted accuracy: {encrypted_metrics.top1_accuracy:.2f}%")
    print(f"Decrypted accuracy: {decrypted_metrics.top1_accuracy:.2f}%")
    print(f"Restoration factor: {restoration_rate:.1f}x")
    print(f"Accuracy recovery: {accuracy_recovery:.1f}% of original performance")

    # Save results to file
    results = {
        'original_accuracy': initial_accuracy,
        'encrypted_accuracy': encrypted_metrics.top1_accuracy,
        'decrypted_accuracy': decrypted_metrics.top1_accuracy,
        'restoration_factor': restoration_rate,
        'accuracy_recovery_percent': accuracy_recovery,
        'num_decrypted_layers': len(encrypted_layers)
    }

    import json
    results_dir = Path('results/claim2')
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'decryption_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Success criteria: recovered at least 95% of original performance
    if accuracy_recovery >= 95.0:
        print("✓ SUCCESS: Full performance restoration achieved!")
        print(f"  Recovered {accuracy_recovery:.1f}% of original performance ({decrypted_metrics.top1_accuracy:.2f}% vs {initial_accuracy:.2f}%)")
        print(f"  Restoration factor: {restoration_rate:.1f}x improvement from encrypted state")
    elif accuracy_recovery >= 90.0:
        print("✓ GOOD: Near-complete performance restoration achieved!")
        print(f"  Recovered {accuracy_recovery:.1f}% of original performance")
    else:
        print("✗ WARNING: Performance not fully restored. Check decryption keys.")
        print(f"  Only recovered {accuracy_recovery:.1f}% of original performance")

    return decrypted_metrics.top1_accuracy

if __name__ == "__main__":
    main()
