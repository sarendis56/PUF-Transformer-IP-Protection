# On-device Transformer Model IP Protection with PUF (ACSAC 2025)

The official code for "Securing On-device Transformers with Reversible Obfuscation and Hardware Binding", accepted by ACSAC 2025. This paper introduces a novel hardware-bound encryption framework for protecting intellectual property in Vision Transformer models. The approach leverages Physical Unclonable Functions (PUF) to generate device-specific cryptographic keys, which are used to encrypt model weights through a dual encryption scheme: Arnold Cat Map (ACM) encryption for attention weights and permutation-based encryption for Feed-Forward Network weights. This combination of hardware binding and heterogeneous cryptographic methods ensures that models can only execute correctly on authorized devices while providing enhanced security against various attack vectors.

## Overview

This repository implements a novel dual encryption approach for Vision Transformer (ViT) models that combines:

- **Arnold Cat Map (ACM)** encryption for attention weights
- **Permutation-based** encryption for Feed-Forward Network (FFN) weights

The dual approach provides enhanced security by using different encryption methods for different neural network components, making it significantly more difficult for attackers to reverse-engineer the protection scheme.

In order for the artifact to be [reviewed](https://www.acsac.org/2025/submissions/papers/artifacts/) by artifact reviewers who do not have access to high-end GPUs, we provide a scaled-down, reproducible demonstration of the method using DeiT-tiny on CIFAR-100, adapted from the full-scale ViT-base + ImageNet implementation. The artifact is available in the `artifact/` directory. Please refer to `README.md-Artifact` for instructions.

## Installation

### Prerequisites

- CUDA-capable GPU (RTX 4090 and A100 tested)
- Around 200GB Storage (~160GB for ImageNet dataset)

### Method 1: Setup with uv (Recommended)

[uv](https://docs.astral.sh/uv/) provides fast, reliable dependency management and automatic virtual environment handling.

1. **Install uv** (if not already installed):
```bash
# On macOS and Linux, recommended
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

2. **Clone and setup the project**:
```bash
git clone <repository-url>
cd transformer-ip-protection

# Create virtual environment and install all dependencies
uv sync
```

3. **Run scripts using uv**:
```bash
# Run any Python script
uv run python your_script.py

# Run experiments
uv run python src/experiments/vit_encryption_experiment.py --config configs/vit_base.yaml
```

### Method 2: Setup with venv + pip

If you prefer traditional Python virtual environments:

1. **Clone the repository**:
```bash
git clone <repository-url>
cd transformer-ip-protection
```

2. **Create and activate virtual environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run scripts** (with activated environment):
```bash
python src/experiments/vit_encryption_experiment.py --config configs/vit_base.yaml
```

### Dataset Setup

Download ImageNet dataset and place it under `dataset/imagenet`. The validation set should be organized as a flat directory structure under `dataset/imagenet/val_nolabel/` with images named sequentially (e.g., `ILSVRC2012_val_00000001.JPEG`, `ILSVRC2012_val_00000002.JPEG`, etc.). This differs from the official ImageNet validation structure which organizes images in synset subdirectories. The ground truth labels are provided through separate mapping files (`src/attacks/synset_words.txt` and `src/attacks/imagenet_2012_validation_synset_labels.txt`). Update the `imagenet_path` in configuration files as needed. Models will be automatically downloaded from HuggingFace Hub on first use.

## Quick Start

### Main Encryption Scheme

Run a ViT-base encryption experiment:

**With uv:**
```bash
uv run python src/experiments/vit_encryption_experiment.py \
    --model google/vit-base-patch16-224 \
    --config configs/vit_base.yaml \
    --imagenet-path /path/to/imagenet/val \
    --output-dir results/vit_base_experiment \
    --batch-size 64 \
    --num-workers 8
```

**With activated venv:**
```bash
python src/experiments/vit_encryption_experiment.py \
    --model google/vit-base-patch16-224 \
    --config configs/vit_base.yaml \
    --imagenet-path /path/to/imagenet/val \
    --output-dir results/vit_base_experiment \
    --batch-size 64 \
    --num-workers 8
```

## Encryption Methods

### Arnold Cat Map (ACM)

The Arnold Cat Map is a chaotic transformation that scrambles 2D matrices:

```
[x']   [a b] [x]
[y'] = [c d] [y]  (mod N)
```

Where `(ad - bc) ≡ ±1 (mod N)` ensures invertibility.

**Applied to**: Attention weights (Query, Key, Value, Output)

### Permutation Encryption

Row/column permutation using cryptographically secure permutation matrices:

```
Encrypted = P × Original × P^T
```

**Applied to**: FFN weights (Intermediate and Output layers)

## Analysis Experiments

The framework includes comprehensive analysis tools for studying encryption effectiveness:

### ACM Key Sensitivity Analysis

Analyze how different Arnold Cat Map keys affect model performance:

```bash
# With uv
uv run python src/experiments/acm_sensitivity_experiment.py --layer 0 --num-variants 30
uv run python src/experiments/acm_sensitivity_experiment.py --config configs/analysis_config.yaml

# With activated venv
python src/experiments/acm_sensitivity_experiment.py --layer 0 --num-variants 30
python src/experiments/acm_sensitivity_experiment.py --config configs/analysis_config.yaml
```

### Dual Encryption Security Analysis

Analyze the security of dual encryption against key guessing attacks:

```bash
# With uv
uv run python src/experiments/dual_encryption_analysis_experiment.py --layers 0 1 2 --num-attack-variants 20
uv run python src/experiments/dual_encryption_analysis_experiment.py --layers 0 --num-attack-variants 30

# With activated venv
python src/experiments/dual_encryption_analysis_experiment.py --layers 0 1 2 --num-attack-variants 20
python src/experiments/dual_encryption_analysis_experiment.py --layers 0 --num-attack-variants 30
```

### Permutation Sensitivity Analysis

Analyze how different permutation matrices affect FFN weight encryption effectiveness:

```bash
# With uv
uv run python src/experiments/permutation_sensitivity_experiment.py --layers 0 --num-permutations 20
uv run python src/experiments/permutation_sensitivity_experiment.py --layers 0 1 2 --num-permutations 50
uv run python src/experiments/permutation_sensitivity_experiment.py --config configs/analysis_config.yaml

# With activated venv
python src/experiments/permutation_sensitivity_experiment.py --layers 0 --num-permutations 20
python src/experiments/permutation_sensitivity_experiment.py --layers 0 1 2 --num-permutations 50
python src/experiments/permutation_sensitivity_experiment.py --config configs/analysis_config.yaml
```

### Decrypt One Layer Security Analysis

Test the security impact of (fortuitously) decrypting one individual layer from fully encrypted models:

```bash
# With uv
uv run python src/experiments/decrypt_one_layer_experiment.py --checkpoint results/vit_encryption_timestamp_checkpoints/final
uv run python src/experiments/decrypt_one_layer_experiment.py --checkpoint-dir results/ --analyze-all

# With activated venv
python src/experiments/decrypt_one_layer_experiment.py --checkpoint results/vit_encryption_timestamp_checkpoints/final --layers 0 1 4
python src/experiments/decrypt_one_layer_experiment.py --checkpoint-dir results/ --analyze-all
```

### Comprehensive Analysis

Run all three analysis experiments in sequence. Update the `analysis_config.yaml` file to specify the desired analyses and their types:

```bash
# With uv
uv run python src/experiments/comprehensive_analysis.py --config configs/analysis_config.yaml

# With activated venv
python src/experiments/comprehensive_analysis.py --config configs/analysis_config.yaml
```

### ACM Performance Benchmarking

Comprehensive CPU vs GPU performance analysis for Arnold Cat Map implementations:

```bash
# With uv
uv run python src/analysis/acm_performance_cpu_gpu.py

# With activated venv
python src/analysis/acm_performance_cpu_gpu.py
```

**The script benchmarks:**
- CPU Efficient (1-Scatter) implementation
- GPU Efficient (1-Scatter) implementation
- GPU Inefficient (N-Scatter) implementation
- Matrix sizes: 16, 32, 64, 128, 256, 512, 768, 1024
- N_iterations: 1, 2, 3, 5, 7, 10, 15, 20, 25, 30

**Generated Outputs:**
- Performance comparison plots saved to `results/analysis/`
- Detailed benchmark data with 240 test combinations
- CPU vs GPU scaling analysis and efficiency comparison between implementations

### Inference Overhead Analysis

Detailed analysis of dual encryption inference overhead during model execution:

```bash
# With uv
uv run python src/analysis/inference_overhead_analysis.py

# With activated venv
python src/analysis/inference_overhead_analysis.py
```

**Analyzes overhead from:**
- ACM (Arnold Cat Map) encryption/decryption
- FFN (Feed-Forward Network) permutation encryption/decryption
- Forward pass execution time

Because we cannot ship the PUF to the reviewers or other researchers, we cannot demonstrate the performance of our hardware-related perforamnce. However, we note that the time required for PUF evaluation and key generation is microsecond-level, while the network inference is millisecond-level. Therefore, the overhead of our PUF-based key generation is negligible compared to the inference time.


## Contact

For issues or questions about this artifact, please refer to the paper or contact the authors by email (peichunhua@link.cuhk.edu.cn).

## Citation
