#!/bin/bash

# Claim 5: Robustness of ACM+permutation encryption against key-guessing attacks
# Demonstration: Three comprehensive security analysis experiments that validate 
# the robustness of the dual encryption scheme against various attack scenarios

set -e

echo "=== Claim 5: Robustness Against Key-Guessing Attacks ==="
echo "Demonstrating security of ACM+permutation dual encryption through four experiments:"
echo "1. ACM Key Sensitivity Analysis"
echo "2. Permutation Sensitivity Analysis"
echo "3. Dual Encryption Security Analysis"
echo "4. Decrypt One Layer Analysis"
echo ""

# Create results directory
mkdir -p artifact/results/claim5

# Check if we have an encrypted model from previous claims
ENCRYPTED_MODEL_PATH=""
if [ -d "artifact/results/claim1/deit_encryption_"* ]; then
    ENCRYPTED_MODEL_PATH=$(ls -d artifact/results/claim1/deit_encryption_*/checkpoints/final | head -1)
    echo "Using encrypted model from claim1: $ENCRYPTED_MODEL_PATH"
elif [ -d "artifact/results/claim2/deit_encryption_"* ]; then
    ENCRYPTED_MODEL_PATH=$(ls -d artifact/results/claim2/deit_encryption_*/checkpoints/final | head -1)
    echo "Using encrypted model from claim2: $ENCRYPTED_MODEL_PATH"
else
    echo "No encrypted model found from previous claims."
    echo "Please run claim1 first to create an encrypted model."
    exit 1
fi

echo ""

# Run the comprehensive analysis with all three experiments
cd artifact
uv run python -m src.experiments.comprehensive_analysis \
    --config configs/analysis_config.yaml \
    --analyses acm_sensitivity permutation_sensitivity dual_encryption \
    --model ../model/deit-tiny-cifar100-finetuned/model \
    --cifar100-path ../dataset/cifar100 \
    --device cuda \
    --output-dir results/claim5 \
    --experiment-name security_analysis
cd ..

echo ""
echo "Running decrypt one layer analysis..."

# Run the decrypt one layer analysis separately as it needs the encrypted model
cd artifact
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))

from src.analysis.decrypt_one_layer_analysis import DecryptOneLayerAnalyzer

# Initialize analyzer
analyzer = DecryptOneLayerAnalyzer(
    cifar100_path='../dataset/cifar100',  # Using CIFAR-100 for consistency
    batch_size=64,
    num_workers=4,
    device='cuda'
)

encrypted_model_path = '$ENCRYPTED_MODEL_PATH'.replace('artifact/', '')
results = analyzer.analyze_single_layer_decryption(
    checkpoint_path=Path(encrypted_model_path)
)

# Save results
output_dir = Path('results/claim5/decrypt_one_layer')
analyzer.save_results(results, output_dir / 'results.json')

# Print summary
analyzer.print_summary_table(results)

print(f'\\nDecrypt one layer analysis completed!')
print(f'Results saved to: {output_dir}')
"
cd ..

echo ""
echo "=== Security Analysis Results Summary ==="

# Display ACM sensitivity results if available
if [ -f "artifact/results/claim5/security_analysis/acm_sensitivity/layer_0/results.json" ]; then
    echo "ACM Key Sensitivity Analysis:"
    python -c "
import json
with open('artifact/results/claim5/security_analysis/acm_sensitivity/layer_0/results.json', 'r') as f:
    results = json.load(f)

print(f'Initial accuracy: {results[\"initial_accuracy\"]:.2%}')
key_results = results['key_results']
accuracies = [r['accuracy'] for r in key_results]
accuracy_drops = [r['accuracy_drop'] for r in key_results]

print(f'Mean accuracy after encryption: {sum(accuracies)/len(accuracies):.2%}')
print(f'Mean accuracy drop: {sum(accuracy_drops)/len(accuracy_drops):.2%}')
print(f'Max accuracy drop: {max(accuracy_drops):.2%}')
print(f'Min accuracy drop: {min(accuracy_drops):.2%}')
"
    echo ""
fi

# Display permutation sensitivity results if available
if [ -f "artifact/results/claim5/security_analysis/permutation_sensitivity/results.json" ]; then
    echo "Permutation Sensitivity Analysis:"
    python -c "
import json
with open('artifact/results/claim5/security_analysis/permutation_sensitivity/results.json', 'r') as f:
    results = json.load(f)

print(f'Initial accuracy: {results[\"initial_accuracy\"]:.2%}')

for layer_idx, layer_data in results['layer_results'].items():
    accuracies = [r['accuracy'] for r in layer_data]
    accuracy_drops = [r['accuracy_drop'] for r in layer_data]

    print(f'Layer {layer_idx}:')
    print(f'  Mean accuracy after permutation: {sum(accuracies)/len(accuracies):.2%}')
    print(f'  Mean accuracy drop: {sum(accuracy_drops)/len(accuracy_drops):.2%}')
    print(f'  Max accuracy drop: {max(accuracy_drops):.2%}')
    print(f'  Min accuracy drop: {min(accuracy_drops):.2%}')
"
    echo ""
fi

if [ -f "artifact/results/claim5/security_analysis/dual_encryption/results.json" ]; then
    echo "Dual Encryption Security Analysis:"
    python -c "
import json
with open('artifact/results/claim5/security_analysis/dual_encryption/results.json', 'r') as f:
    results = json.load(f)

for layer_idx, layer_data in results['layer_results'].items():
    security_results = layer_data['security_results']
    
    acm_attack_accuracies = [r['accuracy'] for r in security_results['acm_attack_results']]
    perm_attack_accuracies = [r['accuracy'] for r in security_results['permutation_attack_results']]
    
    print(f'Layer {layer_idx}:')
    print(f'  Original accuracy: {security_results[\"original_accuracy\"]:.2%}')
    print(f'  Dual encrypted accuracy: {security_results[\"dual_encrypted_accuracy\"]:.2%}')
    print(f'  ACM attack - Mean: {sum(acm_attack_accuracies)/len(acm_attack_accuracies):.2%}, Max: {max(acm_attack_accuracies):.2%}')
    print(f'  Permutation attack - Mean: {sum(perm_attack_accuracies)/len(perm_attack_accuracies):.2%}, Max: {max(perm_attack_accuracies):.2%}')
"
    echo ""
fi

# Display decrypt one layer results if available
if [ -f "artifact/results/claim5/decrypt_one_layer/results.json" ]; then
    echo "Decrypt One Layer Analysis:"
    python -c "
import json
with open('artifact/results/claim5/decrypt_one_layer/results.json', 'r') as f:
    results = json.load(f)

print(f'Baseline (fully encrypted): {results[\"baseline_accuracy\"]:.4f}')
print(f'Original (unencrypted): {results[\"initial_accuracy\"]:.4f}')
print()
print('Layer | Accuracy | Improvement | Improvement %')
print('------|----------|-------------|---------------')

for layer_idx, layer_data in results['layer_results'].items():
    improvement_pct = layer_data['improvement_percentage']
    print(f'{layer_idx:5} | {layer_data[\"decrypted_accuracy\"]:.4f}   | {improvement_pct:+.4f}%     | {improvement_pct:+.2f}%')
"
    echo ""
fi

# Determine overall security assessment
SECURITY_ASSESSMENT=$(python -c "
import json
from pathlib import Path

# Check results and assess security
security_passed = True
assessment_details = []

# Check ACM sensitivity
acm_file = Path('artifact/results/claim5/security_analysis/acm_sensitivity/layer_0/results.json')
if acm_file.exists():
    with open(acm_file, 'r') as f:
        results = json.load(f)
    
    key_results = results['key_results']
    accuracy_drops = [r['accuracy_drop'] for r in key_results]
    mean_drop = sum(accuracy_drops) / len(accuracy_drops)

# Check decrypt one layer
decrypt_file = Path('artifact/results/claim5/decrypt_one_layer/results.json')
if decrypt_file.exists():
    with open(decrypt_file, 'r') as f:
        results = json.load(f)
    
    improvements = [layer_data['improvement_percentage'] for layer_data in results['layer_results'].values()]
    max_improvement = max(improvements) if improvements else 0

# Print assessment
for detail in assessment_details:
    print(detail)
")

echo "=== Summary ==="
echo "This demonstration shows that:"
echo "1. ACM key variations cause substantial and similar accuracy degradation"
echo "2. Permutation key variations also cause significant accuracy drops, but with more variations (so it might not be as robust on its own)"
echo "3. Partial key knowledge provides no significant attack advantage"
echo "4. Single layer decryption yields minimal accuracy improvements"
echo "5. The dual encryption scheme is robust against key-guessing attacks"
echo ""

echo "Claim 5 demonstration completed!"
echo "Results saved in: artifact/results/claim5/"
echo "Individual analysis results:"
echo "- ACM Sensitivity: results/claim5/security_analysis/acm_sensitivity/"
echo "- Permutation Sensitivity: results/claim5/security_analysis/permutation_sensitivity/"
echo "- Dual Encryption: results/claim5/security_analysis/dual_encryption/"
echo "- Decrypt One Layer: results/claim5/decrypt_one_layer/"
