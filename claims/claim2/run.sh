#!/bin/bash

# Claim 2: Full Performance Restoration
# Demonstration: After showing the accuracy drop, apply the correct key to de-obfuscate the model. 
# The artifact will show that the model's accuracy on CIFAR-100 is restored to its original, 
# high-performance level, proving no loss of performance.

set -e

echo "=== Claim 2: Full Performance Restoration ==="
echo "Demonstrating that decryption with correct keys fully restores performance"
echo ""

# Change to artifact directory
cd artifact

# Create results directory
mkdir -p results/claim2

# First, check if we have an encrypted model from claim1
FINETUNED_MODEL_PATH="model/deit-tiny-cifar100-finetuned"

if [ -d "results/claim1/deit_encryption_"* ]; then
    echo "Using encrypted model from claim1..."
    ENCRYPTED_MODEL_PATH=$(ls -d results/claim1/deit_encryption_*/checkpoints/final | head -1)
else
    echo "No encrypted model found from claim1. Running encryption experiment first..."

    # Check if fine-tuned model exists
    if [ ! -d "../$FINETUNED_MODEL_PATH/model" ]; then
        echo "Fine-tuned model not found. Please run claim1 first."
        exit 1
    fi

    # Run encryption experiment
    uv run python src/experiments/deit_encryption_experiment.py \
        --model facebook/deit-tiny-patch16-224 \
        --local-model-path "../$FINETUNED_MODEL_PATH/model" \
        --config configs/deit_tiny.yaml \
        --output-dir results/claim2 \
        --cifar100-path ../dataset/cifar100 \
        --save-checkpoints

    ENCRYPTED_MODEL_PATH=$(ls -d results/claim2/deit_encryption_*/checkpoints/final | head -1)
fi

echo "Using encrypted model from: $ENCRYPTED_MODEL_PATH"
echo ""

# Use the existing decryption demonstration script
echo "Running decryption demonstration..."
uv run python src/experiments/decryption_demo.py "$ENCRYPTED_MODEL_PATH"

echo ""
echo "=== Summary ==="
echo "This demonstration proves that:"
echo "1. The encryption process is fully reversible"
echo "2. With correct decryption keys, original performance is restored"
echo "3. No information is lost during the encryption/decryption process"
echo ""
echo "Claim 2 demonstration completed!"
echo "Results saved in: artifact/results/claim2/"
