#!/usr/bin/env python3
"""
CIFAR-100 Retraining Attack for DeiT Models

This script implements retraining attacks on encrypted DeiT models using CIFAR-100 dataset.
It demonstrates the robustness of the encryption scheme against retraining attacks.

Usage:
    python src/attacks/cifar100_retrain.py --model-path results/deit_encryption_timestamp/checkpoints/final --rate 0.2 --epochs 10
"""

import torch
import numpy as np
import os
import argparse
import json
import sys
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from collections import defaultdict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.cifar100_eval import CIFAR100ValidationDataset, validate_model_cifar100


class AugmentedCIFAR100Dataset(Dataset):
    """CIFAR-100 dataset with efficient augmentation for Vision Transformers."""

    def __init__(self, cifar100_path, processor, split='train', augment=True, augment_strength='medium'):
        """
        Args:
            cifar100_path: Path to CIFAR-100 dataset directory
            processor: HuggingFace image processor
            split: Dataset split ('train' or 'test')
            augment: Whether to apply data augmentation
            augment_strength: Strength of augmentation ('light', 'medium', 'strong')
        """
        self.processor = processor
        self.augment = augment and (split == 'train')
        
        # Load CIFAR-100 dataset
        self.cifar100_dataset = datasets.CIFAR100(
            root=cifar100_path,
            train=(split == 'train'),
            download=True,
            transform=None  # We'll handle transforms via processor
        )
        
        # Define augmentation transforms based on strength
        if self.augment:
            if augment_strength == 'light':
                self.augment_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                ])
            elif augment_strength == 'medium':
                self.augment_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ])
            elif augment_strength == 'strong':
                self.augment_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=20),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                ])
            else:
                self.augment_transform = None
        else:
            self.augment_transform = None
        
        print(f"Loaded CIFAR-100 {split} set with {len(self.cifar100_dataset)} images")
        if self.augment:
            print(f"Using {augment_strength} augmentation")

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


def train_encrypted_model(model_path, model_name, rate, epoch_num, lr_rate, weight_decay, device,
                         use_augmentation=True, augment_strength='medium', lr_scheduler='cosine',
                         lr_warmup_epochs=0, lr_min_factor=0.01, cifar100_path='dataset/cifar100'):
    """
    Train an encrypted DeiT model on a subset of CIFAR-100.
    
    Args:
        model_path: Path to the encrypted model checkpoint
        model_name: HuggingFace model identifier
        rate: Proportion of the dataset to use for training
        epoch_num: Number of training epochs
        lr_rate: Learning rate
        weight_decay: Weight decay
        device: Device to use for training
        use_augmentation: Whether to use data augmentation
        augment_strength: Strength of data augmentation
        lr_scheduler: Learning rate scheduler type
        lr_warmup_epochs: Number of warmup epochs
        lr_min_factor: Minimum learning rate factor
        cifar100_path: Path to CIFAR-100 dataset
    """
    print(f"=== RETRAINING ATTACK CONFIGURATION ===")
    print(f"Model path: {model_path}")
    print(f"Training rate: {rate} ({rate*100:.1f}% of dataset)")
    print(f"Epochs: {epoch_num}")
    print(f"Learning rate: {lr_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Device: {device}")
    print(f"Augmentation: {use_augmentation} ({augment_strength} strength)")
    print(f"LR scheduler: {lr_scheduler}")
    print(f"CIFAR-100 path: {cifar100_path}")
    print("=" * 50)

    # Load the encrypted model and processor (saved as ViT with CIFAR-100 config)
    model_dir = Path(model_path)
    config = ViTConfig.from_pretrained(model_dir / "model")
    config.num_labels = 100  # CIFAR-100 has 100 classes
    model = ViTForImageClassification.from_pretrained(model_dir / "model", config=config)
    processor = ViTImageProcessor.from_pretrained(model_dir / "model")
    
    device_obj = torch.device(device)
    model.to(device_obj)

    # Create validation dataset
    print("Creating CIFAR-100 validation dataset...")
    val_dataset = CIFAR100ValidationDataset(cifar100_path, processor, split='test')
    
    # CIFAR-100 class names
    class_names = val_dataset.class_names

    # Create augmented training dataset
    if use_augmentation:
        print(f"Creating augmented training dataset with {augment_strength} augmentation...")
    else:
        print("Creating training dataset without augmentation...")

    train_full_dataset = AugmentedCIFAR100Dataset(
        cifar100_path=cifar100_path,
        processor=processor,
        split='train',
        augment=use_augmentation,
        augment_strength=augment_strength
    )

    # Sample part of the dataset
    full_dataset_size = len(train_full_dataset)
    sample_size = int(rate * full_dataset_size)

    print(f"Full dataset size: {full_dataset_size}")
    print(f"Requested rate: {rate} ({rate*100:.1f}%)")
    print(f"Calculated sample size: {sample_size}")

    # Set random seed for reproducible sampling
    np.random.seed(42)
    indices = np.random.choice(full_dataset_size, sample_size, replace=False).tolist()
    train_dataset = Subset(train_full_dataset, indices)

    print(f"Using {len(train_dataset)} training samples ({rate*100:.1f}% of full dataset)")
    print(f"Actual percentage used: {len(train_dataset)/full_dataset_size*100:.1f}%")
    
    batch_size = 128  # Suitable for DeiT-tiny
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    # Load encryption metadata
    metadata_path = model_dir / "performance.json"
    encryption_metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            encryption_metadata = json.load(f)
        print(f"Loaded encryption metadata: {len(encryption_metadata.get('encryption_steps', []))} encrypted layers")
    else:
        print("Warning: No encryption metadata found")

    # Setup training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Setup learning rate scheduler
    scheduler = None
    if lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epoch_num, eta_min=lr_rate * lr_min_factor
        )
    elif lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch_num//3, gamma=0.1)
    elif lr_scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Training tracking
    epoch_losses = []
    epoch_accs_top1 = []
    epoch_accs_top5 = []

    train_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create necessary directories
    os.makedirs('./results/retrain_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Include augmentation and scheduler info in model save name
    aug_suffix = f"_aug_{augment_strength}" if use_augmentation else "_no_aug"
    lr_suffix = f"_lr_{lr_scheduler}" if lr_scheduler != 'none' else "_lr_none"
    model_save_name = f'deit_encrypted_{model_path.split("/")[-2]}{aug_suffix}{lr_suffix}'

    best_model_path = f'./results/retrain_models/{rate}_{model_save_name}_{train_time}.pth'
    best_top1_acc = 0.0
    log_file = f'logs/{train_time}_{model_save_name}_training.log'

    # Create log file
    with open(log_file, 'w') as f:
        f.write(f"Training Time: {train_time}\n")
        f.write(f"=== TRAINING CONFIGURATION ===\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Dataset: CIFAR-100\n")
        f.write(f"Training Rate: {rate}\n")
        f.write(f"Epochs: {epoch_num}\n")
        f.write(f"Learning Rate: {lr_rate}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Augmentation: {use_augmentation} ({augment_strength})\n")
        f.write(f"LR Scheduler: {lr_scheduler}\n")
        f.write(f"Encrypted Layers: {len(encryption_metadata.get('encryption_steps', []))}\n")
        f.write(f"=== TRAINING LOG ===\n")

    print(f"Starting training for {epoch_num} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr_rate}")

    for epoch in range(epoch_num):
        # Training
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epoch_num}"):
            images, labels = images.to(device_obj), labels.to(device_obj)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Evaluation
        model.eval()
        top1_acc, top5_acc, _ = validate_model_cifar100(
            model, val_dataset, device_obj, class_names, batch_size=batch_size
        )
        epoch_accs_top1.append(top1_acc)
        epoch_accs_top5.append(top5_acc)

        # Save best model
        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'top1_acc': top1_acc,
                'top5_acc': top5_acc,
                'loss': epoch_loss,
                'encryption_metadata': encryption_metadata
            }, best_model_path)

        print(f"Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Top-1: {top1_acc:.2f}%, Top-5: {top5_acc:.2f}%")
        
        # Log results
        log_entry = {
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'best_top1_acc': best_top1_acc
        }
        
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}: {log_entry}\n")

    print(f"\nTraining completed!")
    print(f"Best Top-1 Accuracy: {best_top1_acc:.2f}%")
    print(f"Model saved to: {best_model_path}")
    print(f"Log saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(description='Retrain an encrypted DeiT model on CIFAR-100.')
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to encrypted model checkpoint')
    parser.add_argument('--model-name', type=str, default='facebook/deit-tiny-patch16-224',
                       help='HuggingFace model identifier')
    parser.add_argument('--rate', type=float, default=0.2, 
                       help='Proportion of the dataset to use for training')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, 
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, 
                       help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use for training')
    parser.add_argument('--cifar100-path', type=str, default='dataset/cifar100',
                       help='Path to CIFAR-100 dataset')
    parser.add_argument('--no-augmentation', action='store_true', 
                       help='Disable data augmentation')
    parser.add_argument('--augment-strength', type=str, choices=['light', 'medium', 'strong'],
                       default='medium', help='Strength of data augmentation')
    parser.add_argument('--lr-scheduler', type=str,
                       choices=['none', 'cosine', 'step', 'exponential'],
                       default='cosine', help='Learning rate scheduler type')
    parser.add_argument('--lr-warmup-epochs', type=int, default=0,
                       help='Number of warmup epochs')
    parser.add_argument('--lr-min-factor', type=float, default=0.01,
                       help='Minimum learning rate as a factor of initial LR')
    
    args = parser.parse_args()

    # Determine augmentation settings
    use_augmentation = not args.no_augmentation

    train_encrypted_model(
        model_path=args.model_path,
        model_name=args.model_name,
        rate=args.rate,
        epoch_num=args.epochs,
        lr_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        use_augmentation=use_augmentation,
        augment_strength=args.augment_strength,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_min_factor=args.lr_min_factor,
        cifar100_path=args.cifar100_path
    )


if __name__ == '__main__':
    main()
