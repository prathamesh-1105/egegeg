import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional

from src.deep_learning.eeg_conformer import EEGConformer
from src.deep_learning.eegnet import EEGNet

# Core Speech-Relevant Electrodes (Broca's area, Wernicke's area, Motor Mouth region)
SPEECH_ROI_CHANNELS = [
    'F7', 'F8', 'FC5', 'FC6', 'T7', 'T8', 
    'C3', 'C4', 'CP5', 'CP6', 'F3', 'F4', 'P7', 'P8', 'Cz'
]

WORD_CLASSES = ["Hello", "Hi", "Yes", "No", "Rest"]


def generate_synthetic_speech_eeg(
    n_trials_per_class: int = 40,
    n_channels: int = 15,
    n_samples: int = 640,
    sfreq: float = 160.0,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generates synthetic multi-channel EEG data representing imagined speech trials
    for 5 target classes ("Hello", "Hi", "Yes", "No", "Rest").
    
    Includes speech-band power modulation (Theta 4-8Hz, Gamma 30-40Hz)
    in Speech ROIs (Broca's & Wernicke's regions) with realistic trial noise.
    """
    np.random.seed(random_seed)
    n_classes = len(WORD_CLASSES)
    total_trials = n_trials_per_class * n_classes
    
    X = np.zeros((total_trials, n_channels, n_samples), dtype=np.float32)
    y = np.zeros(total_trials, dtype=np.int64)
    
    t = np.linspace(0, n_samples / sfreq, n_samples)
    
    trial_idx = 0
    for class_id, word in enumerate(WORD_CLASSES):
        for _ in range(n_trials_per_class):
            # Base background EEG noise (1/f pink-ish noise approximation)
            noise = np.random.normal(0, 1.5, size=(n_channels, n_samples))
            
            # Alpha rhythm background (8-12 Hz)
            alpha_wave = np.sin(2 * np.pi * 10 * t) * np.random.uniform(0.5, 1.2, size=(n_channels, 1))
            trial_data = noise + alpha_wave
            
            if word != "Rest":
                # Speech-specific temporal envelope (word utterance duration ~1.5s - 2.5s)
                onset = int(sfreq * 1.0)
                duration = int(sfreq * 2.0)
                env = np.zeros(n_samples)
                env[onset:onset+duration] = np.hanning(duration)
                
                # Class-specific spectral & spatial signatures
                if word == "Hello":
                    # Strong Theta (6 Hz) in Left Frontal (F7 / Broca) + Gamma (35 Hz)
                    target_ch = [0, 2, 4]  # F7, FC5, T7
                    signal = np.sin(2 * np.pi * 6 * t) * 3.5 + np.sin(2 * np.pi * 35 * t) * 1.8
                elif word == "Hi":
                    # Fast transient Gamma burst (38 Hz) + Beta (20 Hz)
                    target_ch = [1, 3, 5]  # F8, FC6, T8
                    signal = np.sin(2 * np.pi * 38 * t) * 4.0 + np.sin(2 * np.pi * 20 * t) * 2.0
                elif word == "Yes":
                    # Sensorimotor Mouth region (C3/C4) Mu desynchronization + Beta burst
                    target_ch = [6, 7, 8]  # C3, C4, CP5
                    signal = np.sin(2 * np.pi * 18 * t) * 3.8
                elif word == "No":
                    # Bilateral Frontal Theta (5 Hz) + Auditory Wernicke (T7/T8)
                    target_ch = [0, 1, 4, 5]  # F7, F8, T7, T8
                    signal = np.sin(2 * np.pi * 5 * t) * 4.2 + np.sin(2 * np.pi * 12 * t) * 2.2
                
                for ch in target_ch:
                    if ch < n_channels:
                        trial_data[ch] += signal * env
            
            X[trial_idx] = trial_data
            y[trial_idx] = class_id
            trial_idx += 1
            
    # Shuffle dataset
    perm = np.random.permutation(total_trials)
    X = X[perm]
    y = y[perm]
    
    return X, y, WORD_CLASSES


class ImaginedSpeechDecoder:
    """
    High-level Imagined Speech & Word Decoding Engine using EEGConformer.
    """
    def __init__(self, n_channels: int = 15, n_samples: int = 640, device: Optional[str] = None):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.classes = WORD_CLASSES
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = EEGConformer(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=len(self.classes),
            emb_size=64,
            n_heads=4,
            n_layers=2
        ).to(self.device)
        
        self.is_trained = False

    def train_on_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 15,
        batch_size: int = 16,
        lr: float = 1e-3
    ) -> Dict[str, List[float]]:
        """
        Trains the EEGConformer model on provided EEG trial tensor X (Trials, Channels, Samples).
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Train / Validation split (80/20)
        n_val = int(len(X) * 0.2)
        X_train, X_val = X_tensor[:-n_val], X_tensor[-n_val:]
        y_train, y_val = y_tensor[:-n_val], y_tensor[-n_val:]
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * len(batch_y)
                
            scheduler.step()
            train_loss = running_loss / len(X_train)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    logits = self.model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * len(batch_y)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == batch_y).sum().item()
                    
            val_acc = correct / len(X_val)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss / len(X_val))
            history["val_acc"].append(val_acc)
            self.model.train()
            
        self.model.eval()
        self.is_trained = True
        return history

    def predict(self, single_trial: np.ndarray) -> Dict:
        """
        Predicts the imagined word for a single multi-channel EEG trial array (Channels, Samples).
        Returns predicted word label, probability distribution, and channel ROI activity weights.
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(single_trial, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
        pred_idx = int(np.argmax(probs))
        predicted_word = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        
        # Calculate channel power RMS across speech ROI electrodes
        channel_power = np.sqrt(np.mean(single_trial ** 2, axis=1))
        channel_power_norm = (channel_power / (np.max(channel_power) + 1e-6)).tolist()
        
        return {
            "predicted_word": predicted_word,
            "confidence": confidence,
            "probabilities": {word: float(prob) for word, prob in zip(self.classes, probs)},
            "roi_channel_power": dict(zip(SPEECH_ROI_CHANNELS[:self.n_channels], channel_power_norm))
        }
