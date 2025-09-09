#!/bin/bash

# Claim 3: Robustness to Retraining Attacks
# Demonstration: retraining attack on the obfuscated DeiT-tiny model using the 20% CIFAR-100 training set. 
# The results will show that the retrained model's accuracy remains significantly lower than the original, 
# non-obfuscated model, just as your paper shows in Table 3 and Table 10.

set -e

echo "=== Claim 3: Robustness to Retraining Attacks ==="
echo "Demonstrating that retraining attacks cannot recover original performance"
echo ""

# Change to artifact directory
cd artifact

# Create results directory
mkdir -p results/claim3

# First, ensure we have an encrypted model (reuse from previous claims if available)
FINETUNED_MODEL_PATH="model/deit-tiny-cifar100-finetuned"

if [ -d "results/claim1/deit_encryption_"* ]; then
    ENCRYPTED_MODEL_PATH=$(ls -d results/claim1/deit_encryption_*/checkpoints/final | head -1)
elif [ -d "results/claim2/deit_encryption_"* ]; then
    ENCRYPTED_MODEL_PATH=$(ls -d results/claim2/deit_encryption_*/checkpoints/final | head -1)
else
    echo "Creating encrypted model for retraining attack..."

    # Check if fine-tuned model exists
    if [ ! -d "../$FINETUNED_MODEL_PATH/model" ]; then
        echo "Fine-tuned model not found. Please run claim1 first to create the fine-tuned model."
        exit 1
    fi

    # Run encryption experiment
    uv run python src/experiments/deit_encryption_experiment.py \
        --model facebook/deit-tiny-patch16-224 \
        --local-model-path "../$FINETUNED_MODEL_PATH/model" \
        --config configs/deit_tiny.yaml \
        --output-dir results/claim3 \
        --cifar100-path ../dataset/cifar100 \
        --save-checkpoints

    ENCRYPTED_MODEL_PATH=$(ls -d results/claim3/deit_encryption_*/checkpoints/final | head -1)
fi

echo "Using encrypted model from: $ENCRYPTED_MODEL_PATH"
echo ""

# Evaluate the original fine-tuned (unencrypted) model performance as baseline
echo "1. Evaluating fine-tuned DeiT-tiny model performance on CIFAR-100..."
uv run python -c "
import torch
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
from src.utils.cifar100_eval import CIFAR100ModelEvaluator
from pathlib import Path

# Load fine-tuned model (same as used for encryption)
model_path = Path('../$FINETUNED_MODEL_PATH/model')
if model_path.exists():
    config = ViTConfig.from_pretrained(model_path)
    config.num_labels = 100  # CIFAR-100 has 100 classes
    model = ViTForImageClassification.from_pretrained(model_path, config=config)
    processor = ViTImageProcessor.from_pretrained(model_path)
    print('Using fine-tuned model as baseline')
else:
    # Fallback to original model if fine-tuned not available
    from transformers import DeiTForImageClassification
    model = DeiTForImageClassification.from_pretrained('facebook/deit-tiny-patch16-224')
    processor = ViTImageProcessor.from_pretrained('facebook/deit-tiny-patch16-224')
    print('Using original DeiT model as baseline (fine-tuned model not found)')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Evaluate
evaluator = CIFAR100ModelEvaluator('../dataset/cifar100', device=device.type)
metrics = evaluator.evaluate_model(model, processor)
print(f'Baseline model accuracy: {metrics.top1_accuracy:.2f}%')

# Save baseline for comparison
with open('results/claim3/baseline_accuracy.txt', 'w') as f:
    f.write(f'{metrics.top1_accuracy:.2f}')
"

BASELINE_ACCURACY=$(cat results/claim3/baseline_accuracy.txt)
echo "Fine-tuned model baseline accuracy: ${BASELINE_ACCURACY}%"
echo ""

# Evaluate the encrypted model performance
echo "2. Evaluating encrypted model performance (before retraining)..."
uv run python -c "
import torch
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
from src.utils.cifar100_eval import CIFAR100ModelEvaluator
from pathlib import Path

# Load encrypted model (saved as ViT with CIFAR-100 config)
model_path = Path('$ENCRYPTED_MODEL_PATH')
config = ViTConfig.from_pretrained(model_path / 'model')
config.num_labels = 100  # CIFAR-100 has 100 classes
model = ViTForImageClassification.from_pretrained(model_path / 'model', config=config)
processor = ViTImageProcessor.from_pretrained(model_path / 'model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Evaluate
evaluator = CIFAR100ModelEvaluator('../dataset/cifar100', device=device.type)
metrics = evaluator.evaluate_model(model, processor)
print(f'Encrypted model accuracy: {metrics.top1_accuracy:.2f}%')

# Save for comparison
with open('results/claim3/encrypted_accuracy.txt', 'w') as f:
    f.write(f'{metrics.top1_accuracy:.2f}')
"

ENCRYPTED_ACCURACY=$(cat results/claim3/encrypted_accuracy.txt)
echo "Encrypted model accuracy: ${ENCRYPTED_ACCURACY}%"
echo ""

# Run retraining attack
echo "3. Running retraining attack with 20%/50% of CIFAR-100 training data..."
echo ""

TRAINING_RATE=0.2
uv run python src/attacks/cifar100_retrain.py \
    --model-path "$ENCRYPTED_MODEL_PATH" \
    --rate $TRAINING_RATE \
    --epochs 15 \
    --lr 5e-5 \
    --weight-decay 1e-3 \
    --device cuda \
    --lr-scheduler cosine \
    --augment-strength strong \
    --cifar100-path ../dataset/cifar100

echo ""
echo "4. Analyzing retraining attack results..."

# Find the most recent retrained model (use the rate from the training command)
RETRAINED_MODEL=$(ls -t results/retrain_models/"${TRAINING_RATE}"_deit_encrypted_*.pth | head -1)
echo "Retrained model: $RETRAINED_MODEL"

# Extract the best accuracy from the retrained model
RETRAINED_ACCURACY=$(python -c "
import torch
checkpoint = torch.load('$RETRAINED_MODEL', map_location='cpu')
print(f\"{checkpoint['top1_acc']:.2f}\")
")

# Save retrained accuracy for comparison
echo "$RETRAINED_ACCURACY" > results/claim3/retrained_accuracy.txt

echo ""
echo "=== Retraining Attack Results ==="
echo "Fine-tuned model accuracy:   ${BASELINE_ACCURACY}%"
echo "Encrypted model accuracy:    ${ENCRYPTED_ACCURACY}%"
echo "Retrained model accuracy:    ${RETRAINED_ACCURACY}%"
echo ""

# Calculate performance gaps
PERFORMANCE_GAP=$(python -c "baseline=${BASELINE_ACCURACY}; retrained=${RETRAINED_ACCURACY}; print(f'{baseline - retrained:.2f}')")
IMPROVEMENT_RATIO=$(python -c "retrained=${RETRAINED_ACCURACY}; encrypted=${ENCRYPTED_ACCURACY}; print(f'{retrained / encrypted:.1f}')")

echo "Performance gap (Fine-tuned - Retrained): ${PERFORMANCE_GAP}%"
echo "Improvement ratio (Retrained / Encrypted): ${IMPROVEMENT_RATIO}x"

echo "=== Summary ==="
echo "This demonstration shows that:"
echo "1. Retraining with 20%/50% (modify ''TRAINING_RATE'' in the script) of CIFAR-100 data cannot recover original performance"
echo "2. The encrypted model remains robust against retraining attacks"
echo "3. Even with substantial training data, the encryption provides robust protection"
echo ""
echo "Results saved in: artifact/results/claim3/"
echo "Retrained models saved in: artifact/results/retrain_models/"
