import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithConstraint(nn.Conv2d):
    """
    Conv2d layer with optional max-norm constraint on weights (useful for depthwise layers in EEGNet).
    """
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    """
    EEGNet Architecture (Lawhern et al., 2018) optimized for 15-channel, 160 Hz (480-sample window) EEG Motor Imagery.
    Input Tensor Shape: (Batch, 1, n_channels=15, n_samples=480)
    Output Tensor Shape: (Batch, n_classes=3)
    """

    def __init__(
        self,
        n_channels: int = 15,
        n_samples: int = 480,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 32,
        dropout_rate: float = 0.25
    ):
        super(EEGNet, self).__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout_rate = dropout_rate

        # Block 1: Temporal Conv -> Depthwise Spatial Conv
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.bn_temporal = nn.BatchNorm2d(F1)

        self.conv_spatial = Conv2dWithConstraint(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
            max_norm=1.0
        )
        self.bn_spatial = nn.BatchNorm2d(F1 * D)
        self.act_spatial = nn.ELU()
        self.pool_spatial = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop_spatial = nn.Dropout(p=dropout_rate)

        # Block 2: Separable Conv (Depthwise + Pointwise)
        self.conv_separable_depth = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 16 // 2),
            groups=F1 * D,
            bias=False
        )
        self.conv_separable_point = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn_separable = nn.BatchNorm2d(F2)
        self.act_separable = nn.ELU()
        self.pool_separable = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop_separable = nn.Dropout(p=dropout_rate)

        # Calculate feature map size dynamically
        dummy_input = torch.zeros(1, 1, n_channels, n_samples)
        out = self._forward_features(dummy_input)
        self.flatten_dim = out.view(1, -1).size(1)

        # Dense Classifier
        self.classifier = nn.Linear(self.flatten_dim, n_classes)

    def _forward_features(self, x):
        # Block 1
        x = self.conv_temporal(x)
        x = self.bn_temporal(x)
        x = self.conv_spatial(x)
        x = self.bn_spatial(x)
        x = self.act_spatial(x)
        x = self.pool_spatial(x)
        x = self.drop_spatial(x)

        # Block 2
        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bn_separable(x)
        x = self.act_separable(x)
        x = self.pool_separable(x)
        x = self.drop_separable(x)
        return x

    def forward(self, x):
        """
        Forward pass.
        x: (Batch, 1, 15, 480) or (Batch, 15, 480)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1) # Add channel dimension -> (Batch, 1, 15, 480)

        feats = self._forward_features(x)
        flattened = feats.view(feats.size(0), -1)
        logits = self.classifier(flattened)
        return logits
