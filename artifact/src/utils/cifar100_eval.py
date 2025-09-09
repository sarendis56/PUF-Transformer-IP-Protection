"""
CIFAR-100 Dataset Evaluation Module

This module provides evaluation functionality for Vision Transformer models on CIFAR-100 dataset,
adapted from the ImageNet evaluation module.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor, DeiTForImageClassification
from collections import defaultdict
import numpy as np


class CIFAR100ValidationDataset(Dataset):
    """CIFAR-100 validation dataset for Vision Transformer models."""
    
    def __init__(self, cifar100_path, processor, split='test'):
        """
        Initialize CIFAR-100 dataset.
        
        Args:
            cifar100_path: Path to CIFAR-100 dataset directory
            processor: HuggingFace image processor
            split: Dataset split ('test' for validation, 'train' for training)
        """
        self.processor = processor
        self.split = split
        
        # Load CIFAR-100 dataset
        # Check if dataset exists, only download if needed
        import os
        # Check for the actual dataset files that torchvision expects
        dataset_exists = (
            os.path.exists(os.path.join(cifar100_path, 'cifar-100-python', 'train')) and
            os.path.exists(os.path.join(cifar100_path, 'cifar-100-python', 'test')) and
            os.path.exists(os.path.join(cifar100_path, 'cifar-100-python', 'meta'))
        )

        self.cifar100_dataset = datasets.CIFAR100(
            root=cifar100_path,
            train=(split == 'train'),
            download=not dataset_exists,
            transform=None  # We'll handle transforms via processor
        )
        
        # CIFAR-100 class names
        self.class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
        
        print(f"Loaded CIFAR-100 {split} set with {len(self.cifar100_dataset)} images")

    def __len__(self):
        return len(self.cifar100_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar100_dataset[idx]
        
        # Convert PIL image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image using the model's processor
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return pixel_values, label


def validate_model_cifar100(model, dataset, device, class_names, batch_size=64, num_workers=8):
    """
    Validate model on CIFAR-100 dataset.
    
    Args:
        model: The Vision Transformer model to evaluate
        dataset: CIFAR-100 validation dataset
        device: torch device to use
        class_names: List of CIFAR-100 class names
        batch_size: Batch size for validation
        num_workers: Number of data loading workers
    
    Returns:
        tuple: (top1_accuracy, top5_accuracy, class_accuracies)
    """
    model.eval()
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    correct = 0
    total = 0
    top5_correct = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    print()
    print("Starting CIFAR-100 validation...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixel_values, labels = batch
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values)
            # Get top predictions
            _, predicted = torch.max(outputs.logits, 1)
            _, top5_pred = torch.topk(outputs.logits, 5, dim=1)

            # For each image in the batch
            for pred, top5, true_label in zip(predicted.cpu(), top5_pred.cpu(), labels.cpu()):
                total += 1

                # Check if prediction is correct
                if pred.item() == true_label.item():
                    correct += 1
                if true_label.item() in top5:
                    top5_correct += 1

                # Track per-class accuracy
                class_name = class_names[true_label.item()]
                class_total[class_name] += 1
                if pred.item() == true_label.item():
                    class_correct[class_name] += 1
    
    # Calculate accuracies
    top1_accuracy = 100 * correct / total
    top5_accuracy = 100 * top5_correct / total
    
    # Calculate per-class accuracies
    class_accuracies = {
        class_name: (100 * correct_count / class_total[class_name])
        for class_name, correct_count in class_correct.items()
    }
    
    # Sort classes by accuracy
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    print("\nCIFAR-100 Validation Results:")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
    print(f"Total samples evaluated: {total}")
    
    print("\nTop 5 Best Performing Classes:")
    for class_name, acc in sorted_classes[:5]:
        print(f"{class_name:20} | Samples: {class_total[class_name]:4d} | Accuracy: {acc:.2f}%")
    
    print("\nTop 5 Worst Performing Classes:")
    for class_name, acc in sorted_classes[-5:]:
        print(f"{class_name:20} | Samples: {class_total[class_name]:4d} | Accuracy: {acc:.2f}%")
    
    return top1_accuracy, top5_accuracy, class_accuracies


class CIFAR100ModelEvaluator:
    """
    Comprehensive model evaluator for Vision Transformer models on CIFAR-100.
    
    This class provides a unified interface for evaluating ViT models on CIFAR-100
    dataset with comprehensive metrics and progress tracking.
    """

    def __init__(self,
                 cifar100_path: str,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 device: str = "cuda"):
        """
        Initialize the CIFAR-100 model evaluator.

        Args:
            cifar100_path: Path to CIFAR-100 dataset
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            device: Device for evaluation ('cuda' or 'cpu')
        """
        self.cifar100_path = cifar100_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        # CIFAR-100 class names
        self.class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]

    def evaluate_model(self, model, processor):
        """
        Evaluate a model and return comprehensive metrics.

        Args:
            model: Vision Transformer model to evaluate
            processor: Image processor for the model

        Returns:
            AccuracyMetrics: Comprehensive accuracy metrics
        """
        from .metrics import AccuracyMetrics

        # Create dataset
        dataset = CIFAR100ValidationDataset(
            self.cifar100_path,
            processor,
            split='test'
        )

        # Run validation
        top1_acc, top5_acc, class_accs = validate_model_cifar100(
            model=model,
            dataset=dataset,
            device=self.device,
            class_names=self.class_names,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        return AccuracyMetrics(
            top1_accuracy=top1_acc,
            top5_accuracy=top5_acc,
            per_class_accuracies=class_accs,
            total_samples=len(dataset.cifar100_dataset)
        )


# Sample Usage
if __name__ == "__main__":
    # Load the model and processor
    print("Loading DeiT-tiny model and processor...")
    model_name = "facebook/deit-tiny-patch16-224"
    model = DeiTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create evaluator
    evaluator = CIFAR100ModelEvaluator(
        cifar100_path="dataset/cifar100",
        batch_size=64,
        num_workers=8,
        device=device.type
    )
    
    # Run evaluation
    metrics = evaluator.evaluate_model(model, processor)
    print(f"\nFinal Results:")
    print(f"Top-1 Accuracy: {metrics.top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {metrics.top5_accuracy:.2f}%")
