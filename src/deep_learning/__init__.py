"""
Deep Learning Module for EEG Motor Imagery Classification.
Provides EEGNet architecture, PyTorch datasets, trainer, and evaluator.
"""
from src.deep_learning.eegnet import EEGNet
from src.deep_learning.dataset import EEGDataset, create_leakage_safe_dataloaders
from src.deep_learning.trainer import EEGNetTrainer
from src.deep_learning.evaluator import EEGNetEvaluator

__all__ = [
    "EEGNet",
    "EEGDataset",
    "create_leakage_safe_dataloaders",
    "EEGNetTrainer",
    "EEGNetEvaluator"
]
