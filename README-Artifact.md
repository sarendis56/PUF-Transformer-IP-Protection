# DeiT IP Protection Artifact

This artifact provides a scaled-down, reproducible demonstration of the dual encryption IP protection method using DeiT-tiny on CIFAR-100, adapted from the full-scale ViT-base + ImageNet implementation.

This version has been tested on a single RTX 4090, and A100 on Google Colab.

## Overview

This artifact demonstrates four key claims about the dual encryption IP protection method:

1. **Drastic Accuracy Drop**: Encrypting a few layers causes accuracy to drop to random guess level (~1%)
2. **Full Performance Restoration**: Decryption with correct keys fully restores original performance
3. **Robustness to Retraining Attacks**: Retraining with 20% of data cannot recover original performance
4. **Low Inference Overhead**: Encryption adds minimal computational overhead (~1.2x)
5. **Robustness to Key-Guessing Attacks**: Guessing the correct key is infeasible

## Quick Start

### One-Click Setup

```bash
cd artifact

# Install uv manually
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment
uv sync

# Download model manually
uv run python -c "
from transformers import ViTForImageClassification, ViTImageProcessor
model = ViTForImageClassification.from_pretrained('facebook/deit-tiny-patch16-224')
processor = ViTImageProcessor.from_pretrained('facebook/deit-tiny-patch16-224')
model.save_pretrained('model/deit-tiny-patch16-224')
processor.save_pretrained('model/deit-tiny-patch16-224')
"
```

### Running Demonstrations

Each claim can be demonstrated independently:

```bash
cd ..
# Claim 1: Drastic Accuracy Drop
bash claims/claim1/run.sh

# Claim 2: Full Performance Restoration
bash claims/claim2/run.sh

# Claim 3: Robustness to Retraining Attacks
bash claims/claim3/run.sh

# Claim 4: Low Inference Overhead
bash claims/claim4/run.sh

# Claim 5: Robustness to Key-Guessing Attacks
bash claims/claim5/run.sh
```

### Configuration

Modify `artifact/configs/deit_tiny.yaml` to adjust:
- Batch sizes
- Number of encrypted layers
- Arnold Cat Map keys
- Training parameters

There are also tunable parameters in the `run.sh` for each claim, for example, the training parameters for the retraining attack in Claim 3.

## Validation

Each claim includes expected output in `claims/claimX/expected/output.txt`. Compare your results with these expected outputs to validate correct execution. Unfortunately due to the randomness in the training process and underlying hardware, the results may not be exactly the same as the expected output. However, they should be very close.

## Contact and Support

This artifact demonstrates the core concepts from the paper using a smaller, faster setup suitable for conference evaluation. The full-scale experiments (ViT-base + ImageNet) are available in the main repository.

For issues or questions about this artifact, please refer to the paper or contact the authors.
