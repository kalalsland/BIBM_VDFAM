"""Helper utilities for stroke prognosis model.

This module provides various utility functions including:
- Random seed setting for reproducibility
- Model checkpoint saving/loading
- Learning rate scheduling
- Early stopping
- Logging utilities
"""

import os
import random
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    This function sets seeds for:
    - Python's random module
    - NumPy random number generator
    - PyTorch CPU and CUDA operations
    - PyTorch backend (cuDNN) for deterministic behavior
    
    Args:
        seed: Random seed value (default: 42)
    
    Example:
        >>> set_seed(42)
        >>> # All random operations will now be reproducible
    
    Note:
        Setting deterministic mode may impact performance.
        For full reproducibility, also ensure:
        - Same hardware configuration
        - Same PyTorch/CUDA versions
        - Same number of workers in DataLoader
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    scheduler: Optional[_LRScheduler] = None,
    **kwargs: Any,
) -> None:
    """Save model checkpoint with training state.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer with current state
        epoch: Current training epoch
        metrics: Dictionary of current metrics (e.g., loss, accuracy)
        save_path: Path to save checkpoint file
        scheduler: Optional learning rate scheduler
        **kwargs: Additional items to save in checkpoint
    
    Example:
        >>> save_checkpoint(
        ...     model=my_model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     metrics={'val_loss': 0.5, 'val_auc': 0.85},
        ...     save_path='checkpoints/best_model.pth',
        ...     scheduler=scheduler,
        ... )
    
    Checkpoint Structure:
        {
            'epoch': int,
            'model_state_dict': OrderedDict,
            'optimizer_state_dict': dict,
            'scheduler_state_dict': dict (optional),
            'metrics': dict,
            **kwargs
        }
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add any additional items
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Load model checkpoint and restore training state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to load checkpoint to ('cpu' or 'cuda')
    
    Returns:
        Dictionary containing checkpoint information:
            - 'epoch': Epoch number
            - 'metrics': Saved metrics
            - Additional saved items
    
    Example:
        >>> info = load_checkpoint(
        ...     checkpoint_path='checkpoints/best_model.pth',
        ...     model=my_model,
        ...     optimizer=optimizer,
        ...     device='cuda',
        ... )
        >>> print(f"Resuming from epoch {info['epoch']}")
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {checkpoint['epoch']}")
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {}),
        **{k: v for k, v in checkpoint.items() 
           if k not in ['model_state_dict', 'optimizer_state_dict', 
                       'scheduler_state_dict', 'epoch', 'metrics']}
    }


class EarlyStopping:
    """Early stopping to terminate training when validation metric stops improving.
    
    This class monitors a validation metric and stops training if the metric
    doesn't improve for a specified number of epochs (patience).
    
    Attributes:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for metrics to minimize (loss), 'max' for metrics to maximize (accuracy)
        counter: Current count of epochs without improvement
        best_score: Best metric value seen so far
        early_stop: Whether to stop training
        
    Example:
        >>> early_stopping = EarlyStopping(patience=10, mode='max')
        >>> for epoch in range(100):
        ...     val_auc = train_and_validate()
        ...     early_stopping(val_auc)
        ...     if early_stopping.early_stop:
        ...         print(f"Early stopping triggered at epoch {epoch}")
        ...         break
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' depending on metric to monitor
        
        Raises:
            ValueError: If mode is not 'min' or 'max'
        """
        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric: float) -> bool:
        """Check if training should stop.
        
        Args:
            metric: Current validation metric value
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
            return False
        
        # Check if there's improvement
        if self.mode == 'max':
            improved = metric > self.best_score + self.min_delta
        else:  # mode == 'min'
            improved = metric < self.best_score - self.min_delta
        
        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
        
        return False
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class AverageMeter:
    """Compute and store the average and current value.
    
    Useful for tracking metrics during training, such as loss or accuracy.
    
    Example:
        >>> loss_meter = AverageMeter()
        >>> for batch in dataloader:
        ...     loss = compute_loss(batch)
        ...     loss_meter.update(loss.item(), batch_size)
        >>> print(f"Average loss: {loss_meter.avg:.4f}")
    """
    
    def __init__(self):
        """Initialize meter."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """Update statistics with new value.
        
        Args:
            val: New value to add
            n: Number of samples this value represents (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Number of parameters
    
    Example:
        >>> model = MyModel()
        >>> total_params = count_parameters(model, trainable_only=False)
        >>> trainable_params = count_parameters(model, trainable_only=True)
        >>> print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for PyTorch operations.
    
    Args:
        device: Specific device string ('cpu', 'cuda', 'cuda:0', etc.)
                If None, automatically selects GPU if available
    
    Returns:
        PyTorch device object
    
    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
        Using device: cuda:0
    """
    if device is not None:
        return torch.device(device)
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    
    Example:
        >>> print(format_time(5000))
        1h 23m 20s
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_output_dir(base_dir: str, experiment_name: Optional[str] = None) -> Path:
    """Create output directory for experiment results.
    
    Args:
        base_dir: Base directory path
        experiment_name: Optional experiment name (if None, uses timestamp)
    
    Returns:
        Path object for created directory
    
    Example:
        >>> output_dir = create_output_dir('experiments', 'my_experiment')
        >>> print(f"Results will be saved to: {output_dir}")
    """
    if experiment_name is None:
        from datetime import datetime
        experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_dir = Path(base_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def print_model_summary(model: nn.Module, input_shapes: Optional[Dict[str, tuple]] = None) -> None:
    """Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shapes: Optional dictionary of input names and their shapes
    
    Example:
        >>> model = StrokePrognosisModel()
        >>> print_model_summary(model, {
        ...     'image': (1, 1, 128, 128, 128),
        ...     'text': (1, 512),
        ...     'table': (1, 20)
        ... })
    """
    print("=" * 80)
    print("Model Architecture Summary")
    print("=" * 80)
    print(model)
    print("=" * 80)
    
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    if input_shapes:
        print("\nExpected Input Shapes:")
        for name, shape in input_shapes.items():
            print(f"  {name}: {shape}")
    
    print("=" * 80)
