"""
Metrics and Utilities for Evaluating IP Protection Effectiveness
"""

import torch
import numpy as np
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


class AccuracyMetrics(NamedTuple):
    """Container for accuracy metrics."""
    top1_accuracy: float
    top5_accuracy: float
    per_class_accuracies: Dict[str, float]
    total_samples: int


@dataclass
class EncryptionStep:
    """Single step in the encryption process."""
    layer_idx: int
    accuracy: float
    accuracy_drop: float
    permutation_matrix_idx: int
    arnold_key: List[int]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PerformanceTracker:
    """
    Tracks performance metrics throughout the encryption process.
    
    This class maintains a history of accuracy changes as layers are
    progressively encrypted, enabling analysis of encryption impact.
    """
    initial_accuracy: float
    encryption_steps: List[EncryptionStep] = field(default_factory=list)
    model_name: str = ""
    experiment_config: Dict = field(default_factory=dict)
    
    def add_encryption_step(self, 
                          layer_idx: int,
                          accuracy: float,
                          permutation_matrix_idx: int,
                          arnold_key: List[int]) -> None:
        """
        Add a new encryption step to the tracking history.
        
        Args:
            layer_idx: Index of the encrypted layer
            accuracy: Model accuracy after encrypting this layer
            permutation_matrix_idx: Index of permutation matrix used
            arnold_key: Arnold key parameters used
        """
        accuracy_drop = self.initial_accuracy - accuracy
        
        step = EncryptionStep(
            layer_idx=layer_idx,
            accuracy=accuracy,
            accuracy_drop=accuracy_drop,
            permutation_matrix_idx=permutation_matrix_idx,
            arnold_key=arnold_key.copy()
        )
        
        self.encryption_steps.append(step)
    
    def get_current_accuracy(self) -> float:
        """Get the most recent accuracy measurement."""
        if not self.encryption_steps:
            return self.initial_accuracy
        return self.encryption_steps[-1].accuracy
    
    def get_total_accuracy_drop(self) -> float:
        """Get the total accuracy drop from initial to current state."""
        current_accuracy = self.get_current_accuracy()
        return self.initial_accuracy - current_accuracy
    
    def get_encrypted_layers(self) -> List[int]:
        """Get list of all encrypted layer indices."""
        return [step.layer_idx for step in self.encryption_steps]
    
    def get_accuracy_history(self) -> List[float]:
        """Get chronological list of accuracy values."""
        accuracies = [self.initial_accuracy]
        accuracies.extend([step.accuracy for step in self.encryption_steps])
        return accuracies
    
    def save_to_file(self, filepath: Path) -> None:
        """
        Save tracking data to JSON file.
        
        Args:
            filepath: Path where to save the tracking data
        """
        data = {
            'model_name': self.model_name,
            'initial_accuracy': self.initial_accuracy,
            'final_accuracy': self.get_current_accuracy(),
            'total_accuracy_drop': self.get_total_accuracy_drop(),
            'num_encrypted_layers': len(self.encryption_steps),
            'experiment_config': self.experiment_config,
            'encryption_steps': [
                {
                    'layer_idx': step.layer_idx,
                    'accuracy': step.accuracy,
                    'accuracy_drop': step.accuracy_drop,
                    'permutation_matrix_idx': step.permutation_matrix_idx,
                    'arnold_key': step.arnold_key,
                    'timestamp': step.timestamp
                }
                for step in self.encryption_steps
            ],
            'export_timestamp': datetime.now().isoformat()
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'PerformanceTracker':
        """
        Load tracking data from JSON file.
        
        Args:
            filepath: Path to the saved tracking data
            
        Returns:
            PerformanceTracker: Loaded tracker instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tracker = cls(
            initial_accuracy=data['initial_accuracy'],
            model_name=data.get('model_name', ''),
            experiment_config=data.get('experiment_config', {})
        )
        
        for step_data in data['encryption_steps']:
            step = EncryptionStep(
                layer_idx=step_data['layer_idx'],
                accuracy=step_data['accuracy'],
                accuracy_drop=step_data['accuracy_drop'],
                permutation_matrix_idx=step_data['permutation_matrix_idx'],
                arnold_key=step_data['arnold_key'],
                timestamp=step_data['timestamp']
            )
            tracker.encryption_steps.append(step)
        
        return tracker


def calculate_accuracy_drop(initial_accuracy: float, 
                          current_accuracy: float) -> float:
    """
    Calculate the accuracy drop percentage.
    
    Args:
        initial_accuracy: Original model accuracy (0-1 scale)
        current_accuracy: Current model accuracy (0-1 scale)
        
    Returns:
        float: Accuracy drop as percentage (0-100 scale)
    """
    return (initial_accuracy - current_accuracy) * 100


def calculate_top_k_accuracy(predictions: torch.Tensor, 
                           targets: torch.Tensor, 
                           k: int = 5) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        predictions: Model predictions (batch_size, num_classes)
        targets: Ground truth labels (batch_size,)
        k: Number of top predictions to consider
        
    Returns:
        float: Top-k accuracy as percentage (0-100 scale)
    """
    batch_size = targets.size(0)
    _, top_k_pred = predictions.topk(k, dim=1, largest=True, sorted=True)
    
    # Expand targets to match top_k_pred shape
    targets_expanded = targets.view(-1, 1).expand_as(top_k_pred)
    
    # Check if true label is in top-k predictions
    correct = top_k_pred.eq(targets_expanded).any(dim=1)
    
    return correct.float().sum().item() / batch_size * 100


def calculate_per_class_accuracy(predictions: torch.Tensor,
                               targets: torch.Tensor,
                               num_classes: int) -> Dict[int, float]:
    """
    Calculate per-class accuracy.
    
    Args:
        predictions: Model predictions (batch_size, num_classes)
        targets: Ground truth labels (batch_size,)
        num_classes: Total number of classes
        
    Returns:
        Dict[int, float]: Per-class accuracy percentages
    """
    pred_labels = predictions.argmax(dim=1)
    per_class_correct = {}
    per_class_total = {}
    
    for class_idx in range(num_classes):
        class_mask = (targets == class_idx)
        if class_mask.sum() > 0:
            class_predictions = pred_labels[class_mask]
            class_targets = targets[class_mask]
            correct = (class_predictions == class_targets).sum().item()
            total = class_mask.sum().item()
            per_class_correct[class_idx] = correct
            per_class_total[class_idx] = total
        else:
            per_class_correct[class_idx] = 0
            per_class_total[class_idx] = 0
    
    per_class_accuracy = {}
    for class_idx in range(num_classes):
        if per_class_total[class_idx] > 0:
            per_class_accuracy[class_idx] = (
                per_class_correct[class_idx] / per_class_total[class_idx] * 100
            )
        else:
            per_class_accuracy[class_idx] = 0.0
    
    return per_class_accuracy


def generate_evaluation_report(tracker: PerformanceTracker,
                             output_path: Optional[Path] = None) -> str:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        tracker: Performance tracker with experiment data
        output_path: Optional path to save the report
        
    Returns:
        str: Formatted evaluation report
    """
    report_lines = [
        "=" * 60,
        "IP PROTECTION EXPERIMENT EVALUATION REPORT",
        "=" * 60,
        "",
        f"Model: {tracker.model_name}",
        f"Initial Accuracy: {tracker.initial_accuracy:.2%}",
        f"Final Accuracy: {tracker.get_current_accuracy():.2%}",
        f"Total Accuracy Drop: {tracker.get_total_accuracy_drop():.2%}",
        f"Number of Encrypted Layers: {len(tracker.encryption_steps)}",
        "",
        "ENCRYPTION PROGRESSION:",
        "-" * 40,
    ]
    
    for i, step in enumerate(tracker.encryption_steps):
        report_lines.extend([
            f"Step {i+1}: Layer {step.layer_idx}",
            f"  Accuracy: {step.accuracy:.2%}",
            f"  Drop: {step.accuracy_drop:.2%}",
            f"  Permutation Matrix: {step.permutation_matrix_idx}",
            f"  Arnold Key: {step.arnold_key}",
            f"  Timestamp: {step.timestamp}",
            ""
        ])
    
    
    report = "\n".join(report_lines)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report
