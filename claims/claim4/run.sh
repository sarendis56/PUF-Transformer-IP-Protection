#!/bin/bash

# Claim 4: Low Inference Overhead
# Demonstration: Measure the baseline inference time for the DeiT-tiny model on a batch of images. 
# The relative increase in time is low and corresponds to the findings in Figure 7 of the paper 

set -e

echo "=== Claim 4: Low Inference Overhead ==="
echo "Demonstrating that dual encryption has low inference overhead"
echo ""

# Change to artifact directory
cd artifact

# Create results directory
mkdir -p results/claim4

echo "Running inference overhead analysis for DeiT-tiny on CIFAR-100..."
echo "This will measure:"
echo "1. Baseline inference time (no encryption)"
echo "2. Encrypted inference time (with decryption/encryption during inference)"
echo "3. Overhead as a function of number of encrypted layers"
echo "4. Overhead as a function of ACM iterations"
echo ""

# Run the inference overhead analysis
uv run python src/analysis/deit_inference_overhead_analysis.py \
    --device cuda \
    --model facebook/deit-tiny-patch16-224 \
    --cifar100-path ../dataset/cifar100 \
    --output-dir results/claim4 \
    --device cuda \

echo ""
echo "=== Inference Overhead Results ==="

# Display ACM iterations results
if [ -f "results/claim4/deit_acm_iterations_overhead.json" ]; then
    echo "ACM Iterations Overhead Analysis:"
    python -c "
import json
with open('results/claim4/deit_acm_iterations_overhead.json', 'r') as f:
    results = json.load(f)

print('ACM Iterations | Baseline Time | Full Time | Overhead Ratio')
print('---------------|---------------|-----------|---------------')
for r in results:
    print(f'{r[\"acm_iterations\"]:13d} | {r[\"baseline_time\"]*1000:11.2f}ms | {r[\"encrypted_time\"]*1000:12.2f}ms | {r[\"overhead_ratio\"]:12.2f}x')
"
    echo ""
fi

# Display layer count results
if [ -f "results/claim4/deit_layer_count_overhead.json" ]; then
    echo "Layer Count Overhead Analysis:"
    python -c "
import json
with open('results/claim4/deit_layer_count_overhead.json', 'r') as f:
    results = json.load(f)

print('Encrypted Layers | Baseline Time | Full Time | Overhead Ratio')
print('-----------------|---------------|-----------|---------------')
for r in results:
    print(f'{r[\"num_encrypted_layers\"]:15d} | {r[\"baseline_time\"]*1000:11.2f}ms | {r[\"encrypted_time\"]*1000:12.2f}ms | {r[\"overhead_ratio\"]:12.2f}x')
"
    echo ""
fi

# Calculate and display key metrics
echo "=== Key Performance Metrics ==="

# Get overhead for typical configuration (4 layers, 3 ACM iterations)
TYPICAL_OVERHEAD=$(python -c "
import json

# Get layer count overhead for 4 layers
with open('results/claim4/deit_layer_count_overhead.json', 'r') as f:
    layer_results = json.load(f)

# Find result for 4 layers
for r in layer_results:
    if r['num_encrypted_layers'] == 4:
        print(f'{r[\"overhead_ratio\"]:.2f}')
        break
")

echo "Typical overhead (4 encrypted layers, 3 ACM iterations): ${TYPICAL_OVERHEAD}x"

echo "$OVERHEAD_CHECK"
echo ""

# Display component breakdown
echo "=== Overhead Component Breakdown ==="
python -c "
import json
with open('results/claim4/deit_layer_count_overhead.json', 'r') as f:
    results = json.load(f)

# Get breakdown for 4 layers
for r in results:
    if r['num_encrypted_layers'] == 4:
        total_overhead = r['total_overhead'] * 1000  # Convert to ms
        acm_decrypt = r['acm_decrypt_time'] * 1000
        acm_encrypt = r['acm_encrypt_time'] * 1000
        ffn_decrypt = r['ffn_decrypt_time'] * 1000
        ffn_encrypt = r['ffn_encrypt_time'] * 1000
        
        print(f'For {r[\"num_encrypted_layers\"]} encrypted layers:')
        print(f'  Total overhead: {total_overhead:.2f}ms')
        print(f'  ACM decryption: {acm_decrypt:.2f}ms ({acm_decrypt/total_overhead*100:.1f}%)')
        print(f'  ACM encryption: {acm_encrypt:.2f}ms ({acm_encrypt/total_overhead*100:.1f}%)')
        print(f'  FFN decryption: {ffn_decrypt:.2f}ms ({ffn_decrypt/total_overhead*100:.1f}%)')
        print(f'  FFN encryption: {ffn_encrypt:.2f}ms ({ffn_encrypt/total_overhead*100:.1f}%)')
        break
"

echo ""
echo "=== Visualization ==="
echo "Overhead analysis plots have been generated:"
echo "- results/claim4/deit_acm_iterations_overhead.png"
echo "- results/claim4/deit_layer_count_overhead.png"

echo "Claim 4 demonstration completed!"
echo "Results saved in: artifact/results/claim4/"
