#!/bin/bash

# Claim 1: Drastic Accuracy Drop
# Demonstration: Encrypt a few layers of the fine-tuned DeiT-tiny model. 
# The accuracy on the CIFAR-100 test set drops from a high baseline (e.g., >90%) 
# to approximately 1% (random guess), just as it did on ImageNet.

set -e

echo "=== Claim 1: Drastic Accuracy Drop ==="
echo "Demonstrating that encrypting a few layers causes drastic accuracy drop"
echo ""

# Change to artifact directory
cd artifact

# Create results directory
mkdir -p results/claim1

# Step 1: Fine-tune DeiT model on CIFAR-100
echo "Step 1: Fine-tuning DeiT-tiny on CIFAR-100..."
echo ""

FINETUNED_MODEL_PATH="model/deit-tiny-cifar100-finetuned"

# Check if fine-tuned model already exists
if [ -d "../$FINETUNED_MODEL_PATH/model" ]; then
    echo "Fine-tuned model already exists at $FINETUNED_MODEL_PATH"
    echo "Skipping fine-tuning step..."
else
    echo "Fine-tuning DeiT-tiny on CIFAR-100..."
    uv run python src/experiments/deit_cifar100_finetune.py \
        --model facebook/deit-tiny-patch16-224 \
        --output-dir "../$FINETUNED_MODEL_PATH" \
        --epochs 20 \
        --batch-size 64 \
        --lr 1e-4 \
        --training-rate 1.0 \
        --cifar100-path ../dataset/cifar100

    if [ $? -ne 0 ]; then
        echo "Fine-tuning failed!"
        exit 1
    fi
fi

echo ""
echo "Step 2: Running DeiT encryption experiment on CIFAR-100..."
echo "This will:"
echo "1. Load fine-tuned DeiT-tiny model"
echo "2. Evaluate initial accuracy on CIFAR-100"
echo "3. Progressively encrypt layers"
echo "4. Show accuracy drop to ~1% (random guess level)"
echo ""

# Update config to use fine-tuned model
uv run python src/experiments/deit_encryption_experiment.py \
    --model facebook/deit-tiny-patch16-224 \
    --local-model-path "../$FINETUNED_MODEL_PATH/model" \
    --config configs/deit_tiny.yaml \
    --output-dir results/claim1 \
    --cifar100-path ../dataset/cifar100 \
    --save-checkpoints

echo ""

# Extract and display key results
if [ -f "results/claim1/deit_encryption_*/evaluation_report.txt" ]; then
    echo "=== Evaluation Report ==="
    cat results/claim1/deit_encryption_*/evaluation_report.txt
else
    echo "Evaluation report not found. Check results/claim1/ directory for detailed results."
fi

echo ""
echo "Claim 1 demonstration completed!"
echo "Results saved in: artifact/results/claim1/"
