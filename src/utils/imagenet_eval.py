"""
ImageNet Evaluation Utilities for Vision Transformer Models
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import numpy as np

def load_synset_mapping():
    """Load synset list to map predicted indices to class names"""
    from pathlib import Path

    # Look for synset files in data directory first, then current directory
    synset_words_paths = [
        Path("data/synset_words.txt"),
        Path("./synset_words.txt")
    ]

    validation_labels_paths = [
        Path("data/imagenet_2012_validation_synset_labels.txt"),
        Path("./imagenet_2012_validation_synset_labels.txt")
    ]

    # Load synset to name mapping
    synset_to_name = {}
    synset_words_file = None
    for path in synset_words_paths:
        if path.exists():
            synset_words_file = path
            break

    if synset_words_file is None:
        raise FileNotFoundError("synset_words.txt not found. Please run create_synset_mapping.py first.")

    with open(synset_words_file, 'r') as f:
        for idx, line in enumerate(f):
            synset, name = line.strip().split(' ', 1)
            synset_to_name[synset] = name

    # Load validation synsets
    validation_labels_file = None
    for path in validation_labels_paths:
        if path.exists():
            validation_labels_file = path
            break

    if validation_labels_file is None:
        raise FileNotFoundError("imagenet_2012_validation_synset_labels.txt not found. Please run create_synset_mapping.py first.")

    with open(validation_labels_file, 'r') as f:
        validation_synsets = [line.strip() for line in f]

    return synset_to_name, validation_synsets

class ImageNetValidationDataset(Dataset):
    def __init__(self, image_dir, validation_synsets, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.validation_synsets = validation_synsets

        # Build list of all images from synset directories
        self.image_files = []
        self.image_labels = []

        for synset_idx, synset in enumerate(validation_synsets):
            synset_dir = os.path.join(image_dir, synset)
            if os.path.exists(synset_dir):
                synset_images = [f for f in os.listdir(synset_dir) if f.endswith('.JPEG')]
                for img_file in synset_images:
                    self.image_files.append(os.path.join(synset, img_file))
                    self.image_labels.append(synset_idx)

        print(f"Found {len(self.image_files)} images across {len(validation_synsets)} classes")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)

        # Get the label for this image
        label = self.image_labels[idx]

        return pixel_values, label

def validate_model(model, dataset, device, synset_to_name, validation_synsets, batch_size=64, num_workers=8):
    """
    Validate model on ImageNet validation set
    
    Args:
        model: The ViT model to evaluate
        dataset: ImageNet validation dataset
        device: torch device to use
        synset_to_name: Mapping from synset IDs to class names
        batch_size: Batch size for validation
        num_workers: Number of data loading workers
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
    print("Starting full validation...")
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
                synset = validation_synsets[true_label.item()]
                class_name = synset_to_name.get(synset, synset)
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
    
    print("\nValidation Results:")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
    print(f"Total samples evaluated: {total}")
    
    print("\nTop 5 Best Performing Classes:")
    for class_name, acc in sorted_classes[:5]:
        print(f"{class_name[:50]:50} | Samples: {class_total[class_name]:4d} | Accuracy: {acc:.2f}%")
    
    print("\nTop 5 Worst Performing Classes:")
    for class_name, acc in sorted_classes[-5:]:
        print(f"{class_name[:50]:50} | Samples: {class_total[class_name]:4d} | Accuracy: {acc:.2f}%")
    
    return top1_accuracy, top5_accuracy, class_accuracies


class ModelEvaluator:
    """
    Comprehensive model evaluator for Vision Transformer models.

    This class provides a unified interface for evaluating ViT models on ImageNet
    validation dataset with comprehensive metrics and progress tracking.
    """

    def __init__(self,
                 imagenet_path: str,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 device: str = "cuda"):
        """
        Initialize the model evaluator.

        Args:
            imagenet_path: Path to ImageNet validation dataset
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            device: Device for evaluation ('cuda' or 'cpu')
        """
        self.imagenet_path = imagenet_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        # Load synset mappings
        self.synset_to_name, self.validation_synsets = load_synset_mapping()

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
        dataset = ImageNetValidationDataset(
            self.imagenet_path,
            self.validation_synsets,
            processor
        )

        # Run validation
        top1_acc, top5_acc, class_accs = validate_model(
            model=model,
            dataset=dataset,
            device=self.device,
            synset_to_name=self.synset_to_name,
            validation_synsets=self.validation_synsets,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        return AccuracyMetrics(
            top1_accuracy=top1_acc,
            top5_accuracy=top5_acc,
            per_class_accuracies=class_accs,
            total_samples=len(dataset)
        )


# Sample Usage
if __name__ == "__main__":
    # Load the model and processor
    print("Loading model and processor...")
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load mappings
    print("Loading synset mappings...")
    synset_to_name, validation_synsets = load_synset_mapping()

    # Create dataset
    print("Creating dataset...")
    val_dir = "dataset/imagenet/val"
    dataset = ImageNetValidationDataset(val_dir, validation_synsets, processor)
    
    # Run validation
    top1_acc, top5_acc, class_accuracies = validate_model(model, dataset, device, synset_to_name)
