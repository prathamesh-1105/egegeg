import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, Tuple, List, Optional

from src.deep_learning.eegnet import EEGNet


class EEGNetTrainer:
    """
    Reproducible EEGNet Trainer.
    Supports class weighting, early stopping on validation Macro F1, and training history logging.
    """

    def __init__(
        self,
        model: EEGNet,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cpu",
        random_seed: int = 42
    ):
        self.seed = random_seed
        self._set_seed(self.seed)

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_macro_f1": []
        }
        self.best_model_weights = None
        self.best_val_macro_f1 = 0.0

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())

        avg_loss = total_loss / max(1, len(dataloader.dataset))
        acc = float(accuracy_score(all_targets, all_preds) * 100.0) if all_targets else 0.0
        return avg_loss, acc

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.criterion(logits, y)

                total_loss += loss.item() * len(y)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.cpu().numpy())

        if not all_targets:
            return 0.0, 0.0, 0.0

        avg_loss = total_loss / len(dataloader.dataset)
        acc = float(accuracy_score(all_targets, all_preds) * 100.0)
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro") * 100.0)
        return avg_loss, acc, macro_f1

    def fit(
        self,
        dl_train: DataLoader,
        dl_val: DataLoader,
        epochs: int = 150,
        patience: int = 25
    ) -> Dict:
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self.train_epoch(dl_train)
            val_loss, val_acc, val_f1 = self.evaluate(dl_val)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_macro_f1"].append(val_f1)

            # Early stopping based ONLY on validation Macro F1
            if val_f1 > self.best_val_macro_f1:
                self.best_val_macro_f1 = val_f1
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                # Early stop
                break

        # Load best weights
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)

        return self.history
