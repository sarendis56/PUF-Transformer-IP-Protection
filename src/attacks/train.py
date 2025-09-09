import torch
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
from transformers import AutoImageProcessor
from transformers import ViTForImageClassification
from transformers import SwinForImageClassification
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from ImageNetEval import load_synset_mapping, ImageNetValidationDataset, validate_model
from pathlib import Path
import argparse
import json


class AugmentedImageNetDataset(Dataset):
    """ImageNet dataset with efficient augmentation for Vision Transformers."""

    def __init__(self, root_dir, processor, augment=True, augment_strength='medium'):
        """
        Args:
            root_dir: Path to ImageNet training directory
            processor: HuggingFace image processor
            augment: Whether to apply data augmentation
            augment_strength: Strength of augmentation ('light', 'medium', 'strong')
        """
        self.root_dir = root_dir
        self.processor = processor
        self.augment = augment

        # Load ImageNet training data
        self.imagenet_dataset = ImageFolder(root=root_dir)

        # Define augmentation transforms based on strength
        if augment:
            if augment_strength == 'light':
                self.augment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
                ])
            elif augment_strength == 'medium':
                self.augment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                    transforms.RandomRotation(degrees=5),
                    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                ])
            elif augment_strength == 'strong':
                self.augment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                ])
            else:
                raise ValueError(f"Unknown augment_strength: {augment_strength}")
        else:
            self.augment_transform = None

    def __len__(self):
        return len(self.imagenet_dataset)

    def __getitem__(self, idx):
        image, label = self.imagenet_dataset[idx]

        # Apply augmentation if enabled
        if self.augment and self.augment_transform is not None:
            image = self.augment_transform(image)

        # Process with HuggingFace processor
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)

        return pixel_values, label


def train(model_path, model_name, rate, epoch_num, lr_rate, weight_decay, device,
          use_augmentation=True, augment_strength='medium', lr_scheduler='cosine',
          lr_warmup_epochs=0, lr_min_factor=0.01):
    # Load the model and processor
    model_dir = Path(model_path)
    # config = ViTConfig.from_pretrained(model_dir / "model")
    # print(config)
    if 'swin' in model_name:
        model = SwinForImageClassification.from_pretrained(model_dir / "model")
    else:
        model = ViTForImageClassification.from_pretrained(model_dir / "model")


    processor = AutoImageProcessor.from_pretrained(model_name)
    device_obj = torch.device(device)
    model.to(device_obj)

    # Load mappings
    print("Loading synset mappings...")
    synset_to_name, validation_synsets = load_synset_mapping()

    # Create validation dataset
    print("Creating dataset...")
    val_dir = './dataset/imagenet/val_nolabel'
    val_dataset = ImageNetValidationDataset(val_dir, validation_synsets, processor)

    # Create augmented training dataset
    if use_augmentation:
        print(f"Creating augmented training dataset with {augment_strength} augmentation...")
    else:
        print("Creating training dataset without augmentation...")

    imgnet_train = AugmentedImageNetDataset(
        root_dir='./dataset/imagenet/train',
        processor=processor,
        augment=use_augmentation,
        augment_strength=augment_strength
    )

    # Sample part of the dataset
    sample_size = int(rate * len(imgnet_train))
    indices = np.random.choice(len(imgnet_train), sample_size, replace=False).tolist()
    train_dataset = torch.utils.data.Subset(imgnet_train, indices)
    
    batch_size = 128 if '224' in model_name else 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    
    # Load encryption metadata
    metadata_path = model_dir / "encryption_metadata.json"
    with open(metadata_path, 'r') as f:
        encryption_metadata = json.load(f)

    # Extract encrypted layers from encryption history
    encrypted_layers = []
    if "encryption_history" in encryption_metadata:
        encrypted_layers = [entry["layer_idx"] for entry in encryption_metadata["encryption_history"]]
        encrypted_layers = sorted(list(set(encrypted_layers)))  # Remove duplicates and sort

    print(f"Encrypted Layers: {encrypted_layers}")
    print(f"Number of Encrypted Layers: {len(encrypted_layers)}")
    print(f"Data Augmentation: {'Enabled' if use_augmentation else 'Disabled'}")
    if use_augmentation:
        print(f"Augmentation Strength: {augment_strength}")
    print(f"Learning Rate Scheduler: {lr_scheduler}")
    if lr_scheduler == 'cosine_warmup':
        print(f"Warmup Epochs: {lr_warmup_epochs}")
    if lr_scheduler != 'none':
        print(f"Min LR Factor: {lr_min_factor}")

    # Include augmentation and scheduler info in model save name
    aug_suffix = f"_aug_{augment_strength}" if use_augmentation else "_no_aug"
    lr_suffix = f"_lr_{lr_scheduler}" if lr_scheduler != 'none' else "_lr_none"
    model_save_name = f'vit_encrypted_{model_path.split("/")[-1]}{aug_suffix}{lr_suffix}'

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Setup learning rate scheduler
    scheduler = None
    if lr_scheduler == 'cosine':
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epoch_num, eta_min=lr_rate * lr_min_factor
        )
    elif lr_scheduler == 'cosine_warmup':
        # Cosine annealing with warmup
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(epoch):
            if epoch < lr_warmup_epochs:
                return epoch / lr_warmup_epochs
            else:
                progress = (epoch - lr_warmup_epochs) / (epoch_num - lr_warmup_epochs)
                return lr_min_factor + (1 - lr_min_factor) * 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = LambdaLR(optimizer, lr_lambda)
    elif lr_scheduler == 'step':
        # Step decay every 1/3 of total epochs
        step_size = max(1, epoch_num // 3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    elif lr_scheduler == 'exponential':
        # Exponential decay
        gamma = (lr_min_factor) ** (1.0 / epoch_num)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif lr_scheduler == 'plateau':
        # Reduce on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
    elif lr_scheduler == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")

    train_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create necessary directories if they don't exist
    os.makedirs('./results/retrain_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    best_model_path = f'./results/retrain_models/{rate}_{model_save_name}_{train_time}.pth'
    best_top1_acc = 0.0
    log_file = f'logs/{train_time}_{model_save_name}_training.log'
    # Create a log file

    with open(log_file, 'w') as f:
        f.write(f"Training Time: {train_time}\n")
        f.write(f"=== TRAINING CONFIGURATION ===\n")

    with open(log_file, 'a') as f:
        f.write(f"model_path: {model_path}\n")
        f.write(f"dataset_amount: {rate}\n")
        f.write(f"epoch_num: {epoch_num}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"learning_rate: {lr_rate}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"use_augmentation: {use_augmentation}\n")
        f.write(f"augment_strength: {augment_strength if use_augmentation else 'N/A'}\n")
        f.write(f"lr_scheduler: {lr_scheduler}\n")
        f.write(f"lr_warmup_epochs: {lr_warmup_epochs if lr_scheduler == 'cosine_warmup' else 'N/A'}\n")
        f.write(f"lr_min_factor: {lr_min_factor if lr_scheduler != 'none' else 'N/A'}\n")
        f.write(f"model_save_name: {model_save_name}\n")
        f.write(f"=== TRAINING PROGRESS ===\n")

    # Train the model
    epoch_losses = []
    epoch_accs_top1 = []
    epoch_accs_top5= []

    for epoch in range(epoch_num):
        
        # Training
        running_loss = 0.0

        epoch_loss = 0.0
        model.train()
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device_obj), labels.to(device_obj)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss/len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

        # Evaluation
        eval_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        model.eval()
        top1_acc, top5_acc, _ = validate_model(model, val_dataset, device_obj, synset_to_name)
        epoch_accs_top1.append(top1_acc)
        epoch_accs_top5.append(top5_acc)

        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(top1_acc)  # Use validation accuracy for plateau scheduler
            else:
                scheduler.step()  # Step-based schedulers

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.2e}")

        # Save the model with the best top1 accuracy
        if epoch == 0 or top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            torch.save(model.state_dict(), best_model_path)
        
        log_entry = {
            'eval_time': eval_time,
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'learning_rate': current_lr,
            'best_top1_acc': best_top1_acc,
            'top1_acc': top1_acc,
            'top5_acc': top5_acc
        }
        for entries in log_entry:
            with open(log_file, 'a') as f:
                f.write(f"{entries}: {log_entry[entries]}\n")

if __name__ == '__main__':

    model_path_list = [
        # Replace with the correct timestamp
        "results/vit_encryption_20250905_232539/checkpoints/final",
        "results/vit_encryption_20250906_203206/checkpoints/final"
    ]
    model_name_list = ["google/vit-base-patch16-224",
                       "google/vit-large-patch16-224"]

    parser = argparse.ArgumentParser(description='Train an Encrypted model on a subset of ImageNet.')
    parser.add_argument('--model_index', type=int, required=False, default=0, help='Index of chosen model.')
    parser.add_argument('--rate', type=float, required=False, default=0.2, help='Proportion of the dataset to use for training.')
    parser.add_argument('--time', type=int, required=False, default=5, help='Number of training times.')
    parser.add_argument('--epoch', type=int, required=False, default=10, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, required=False, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-3, help='Weight decay')
    parser.add_argument('--device', type=str, required=False, default='cuda', help='Device to use for training (e.g., cuda, cuda:0, cuda:1, cpu)')
    parser.add_argument('--no_augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--augment_strength', type=str, choices=['light', 'medium', 'strong'],
                        default='medium', help='Strength of data augmentation (light/medium/strong)')
    parser.add_argument('--lr_scheduler', type=str,
                        choices=['none', 'cosine', 'cosine_warmup', 'step', 'exponential', 'plateau'],
                        default='cosine', help='Learning rate scheduler type')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs for cosine_warmup scheduler')
    parser.add_argument('--lr_min_factor', type=float, default=0.01,
                        help='Minimum learning rate as a factor of initial LR')
    
    args = parser.parse_args()

    model_path = model_path_list[args.model_index]
    model_name = model_name_list[args.model_index]

    # Determine augmentation settings
    use_augmentation = not args.no_augmentation
    augment_strength = args.augment_strength

    for i in range(args.time):
        train(model_path, model_name, args.rate, args.epoch, args.lr, args.weight_decay,
              args.device, use_augmentation, augment_strength, args.lr_scheduler,
              args.lr_warmup_epochs, args.lr_min_factor)
