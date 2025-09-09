#!/usr/bin/env python3
"""
DeiT Fine-tuning for CIFAR-100

This script fine-tunes a pre-trained DeiT model on CIFAR-100 dataset.
The fine-tuned model can then be used for encryption experiments.

Usage:
    python src/experiments/deit_cifar100_finetune.py --model facebook/deit-tiny-patch16-224 --epochs 10
"""

import argparse
import logging
import sys
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.cifar100_eval import CIFAR100ValidationDataset, validate_model_cifar100


class CIFAR100TrainingDataset:
    """CIFAR-100 training dataset with augmentation."""
    
    def __init__(self, cifar100_path, processor, augment=True, augment_strength='medium'):
        self.processor = processor
        self.augment = augment
        
        # Load CIFAR-100 training dataset
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
            train=True,
            download=not dataset_exists,
            transform=None
        )
        
        # Setup augmentation transforms
        if augment:
            if augment_strength == 'light':
                self.augment_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(5),
                ])
            elif augment_strength == 'medium':
                self.augment_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                ])
            elif augment_strength == 'strong':
                self.augment_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                ])
            else:
                self.augment_transform = None
        else:
            self.augment_transform = None
    
    def __len__(self):
        return len(self.cifar100_dataset)
    
    def __getitem__(self, idx):
        image, label = self.cifar100_dataset[idx]
        
        # Convert PIL image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply augmentation if enabled
        if self.augment and self.augment_transform:
            image = self.augment_transform(image)
        
        # Process image using the model's processor
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return pixel_values, label


def finetune_deit_cifar100(model_name, output_dir, epochs=10, learning_rate=5e-5, 
                          weight_decay=1e-3, batch_size=256, num_workers=16,
                          device='cuda', cifar100_path='dataset/cifar100',
                          use_augmentation=True, augment_strength='medium',
                          lr_scheduler='cosine', training_rate=1.0):
    """
    Fine-tune DeiT model on CIFAR-100.
    
    Args:
        model_name: HuggingFace model identifier
        output_dir: Directory to save the fine-tuned model
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        batch_size: Batch size
        num_workers: Number of data loading workers
        device: Device for training
        cifar100_path: Path to CIFAR-100 dataset
        use_augmentation: Whether to use data augmentation
        augment_strength: Strength of data augmentation
        lr_scheduler: Learning rate scheduler type
        training_rate: Proportion of training data to use
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Fine-tuning {model_name} on CIFAR-100")
    logger.info(f"Output directory: {output_path}")
    
    # Load model and processor
    logger.info("Loading pre-trained model and processor...")
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # Modify classifier head for CIFAR-100 (100 classes)
    model.classifier = nn.Linear(model.config.hidden_size, 100)
    
    # Move to device
    device_obj = torch.device(device)
    model.to(device_obj)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = CIFAR100TrainingDataset(
        cifar100_path, processor, use_augmentation, augment_strength
    )
    val_dataset = CIFAR100ValidationDataset(cifar100_path, processor, split='test')
    
    # Use subset of training data if specified
    if training_rate < 1.0:
        num_train_samples = int(len(train_dataset) * training_rate)
        indices = np.random.choice(len(train_dataset), num_train_samples, replace=False).tolist()
        train_dataset = Subset(train_dataset, indices)
        logger.info(f"Using {len(train_dataset)} training samples ({training_rate*100:.1f}%)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    # Setup training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Setup learning rate scheduler
    scheduler = None
    if lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.01
        )
    elif lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
    
    # Get CIFAR-100 class names
    class_names = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    
    # Training loop
    logger.info("Starting training...")
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device_obj), labels.to(device_obj)
            
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Evaluation phase
        model.eval()
        top1_acc, top5_acc, _ = validate_model_cifar100(
            model, val_dataset, device_obj, class_names, batch_size=batch_size
        )
        
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, "
                   f"Top1={top1_acc:.2f}%, Top5={top5_acc:.2f}%")
        
        # Save best model
        if top1_acc > best_accuracy:
            best_accuracy = top1_acc
            logger.info(f"New best accuracy: {best_accuracy:.2f}%")
            
            # Save model and processor
            model_save_path = output_path / "model"
            model.save_pretrained(model_save_path)
            processor.save_pretrained(model_save_path)
            
            # Save training info
            training_info = {
                'model_name': model_name,
                'best_accuracy': best_accuracy,
                'epoch': epoch + 1,
                'training_params': {
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'batch_size': batch_size,
                    'use_augmentation': use_augmentation,
                    'augment_strength': augment_strength,
                    'lr_scheduler': lr_scheduler,
                    'training_rate': training_rate
                }
            }
            
            import json
            with open(output_path / "training_info.json", 'w') as f:
                json.dump(training_info, f, indent=2)
    
    logger.info(f"Training completed! Best accuracy: {best_accuracy:.2f}%")
    logger.info(f"Fine-tuned model saved to: {output_path / 'model'}")
    
    return best_accuracy


def main():
    parser = argparse.ArgumentParser(description='Fine-tune DeiT model on CIFAR-100')
    parser.add_argument('--model', type=str, default='facebook/deit-tiny-patch16-224',
                       help='HuggingFace model identifier')
    parser.add_argument('--output-dir', type=str, default='model/deit-tiny-cifar100-finetuned',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                       help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=16,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for training')
    parser.add_argument('--cifar100-path', type=str, default='dataset/cifar100',
                       help='Path to CIFAR-100 dataset')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--augment-strength', type=str, choices=['light', 'medium', 'strong'],
                       default='medium', help='Strength of data augmentation')
    parser.add_argument('--lr-scheduler', type=str, choices=['none', 'cosine', 'step'],
                       default='cosine', help='Learning rate scheduler')
    parser.add_argument('--training-rate', type=float, default=1.0,
                       help='Proportion of training data to use')
    
    args = parser.parse_args()
    
    finetune_deit_cifar100(
        model_name=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        cifar100_path=args.cifar100_path,
        use_augmentation=not args.no_augmentation,
        augment_strength=args.augment_strength,
        lr_scheduler=args.lr_scheduler,
        training_rate=args.training_rate
    )


if __name__ == '__main__':
    main()
