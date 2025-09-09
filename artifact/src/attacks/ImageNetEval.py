import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor
from collections import defaultdict
import matplotlib.pyplot as plt

def load_synset_mapping():
    """Load synset list to map predicted indices to class names"""
    synset_to_name = {}
    # Load the ordered list of synsets and their names
    with open('src/attacks/synset_words.txt', 'r') as f:
        for idx, line in enumerate(f):
            synset, name = line.strip().split(' ', 1)
            synset_to_name[synset] = name
    
    # Load validation ground truth
    with open('src/attacks/imagenet_2012_validation_synset_labels.txt', 'r') as f:
        validation_synsets = [line.strip() for line in f]
    
    return synset_to_name, validation_synsets

class ImageNetValidationDataset(Dataset):
    def __init__(self, image_dir, validation_synsets, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.validation_synsets = validation_synsets
        
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith('.JPEG')],
            key=lambda x: int(x.split('_')[2].split('.')[0])
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        # Get synset for this validation image
        synset = self.validation_synsets[idx]
        
        return {
            'pixel_values': pixel_values,
            'synset': synset,
            'image_path': image_path
        }

def validate_model(model, dataset, device, synset_to_name, batch_size=64, num_workers=8):
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
            pixel_values = batch['pixel_values'].to(device)
            synsets = batch['synset']
            
            outputs = model(pixel_values)
            # Get top predictions
            _, predicted = torch.max(outputs.logits, 1)
            _, top5_pred = torch.topk(outputs.logits, 5, dim=1)
            
            # For each image in the batch
            for pred, top5, synset in zip(predicted.cpu(), top5_pred.cpu(), synsets):
                # Get the true label (position in synset list)
                with open('src/attacks/synset_words.txt', 'r') as f:
                    true_idx = sum(1 for line in f if line.split()[0] < synset)
                
                # Check if prediction is correct
                if pred.item() == true_idx:
                    correct += 1
                if true_idx in top5:
                    top5_correct += 1
                
                # Track per-class accuracy
                class_name = synset_to_name[synset]
                class_total[class_name] += 1
                if pred.item() == true_idx:
                    class_correct[class_name] += 1
                
            total += len(synsets)
    
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
    val_dir = "/root/autodl-tmp/imagenet/val"
    dataset = ImageNetValidationDataset(val_dir, validation_synsets, processor)
    
    # Run validation
    top1_acc, top5_acc, class_accuracies = validate_model(model, dataset, device, synset_to_name)
