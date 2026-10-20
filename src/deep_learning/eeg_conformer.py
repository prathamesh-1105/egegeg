import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dWithConstraint(nn.Conv2d):
    """
    Conv2d layer with optional max-norm weight constraint.
    """
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None and self.training:
            with torch.no_grad():
                self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGConformer(nn.Module):
    """
    SOTA EEGConformer Architecture (Spatial-Temporal Transformer)
    for Multi-Class Imagined Speech & Word Decoding from Scalp EEG.
    
    Combines:
    1. Filter Bank Temporal Convolutions (Frequency Feature Extraction)
    2. Depthwise Spatial Convolutions (Speech ROI Electrode Feature Extraction)
    3. Multi-Head Self-Attention Transformer Encoder (Long-range Temporal Context)
    4. Dense Softmax Classifier Head
    """
    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 640,
        n_classes: int = 5,
        emb_size: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout_rate: float = 0.25
    ):
        super(EEGConformer, self).__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.emb_size = emb_size

        # 1. Temporal Convolutional Block (Extracts spectral representations)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU()
        )

        # 2. Depthwise Spatial Convolutional Block (Speech ROI spatial filtering)
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(16, 32, kernel_size=(n_channels, 1), groups=16, bias=False, max_norm=1.0),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # Downsample temporal dimension by 4
            nn.Dropout(dropout_rate)
        )

        # 3. Separable Convolutional Block (Refining temporal-spatial features)
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), padding=(0, 7), groups=32, bias=False),
            nn.Conv2d(32, emb_size, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # Downsample temporal dimension by another 4
            nn.Dropout(dropout_rate)
        )

        # Calculate downsampled time steps
        self.downsampled_time = n_samples // 16

        # Position Embedding for Transformer
        self.pos_embedding = nn.Parameter(torch.randn(1, self.downsampled_time, emb_size))

        # 4. Multi-Head Self-Attention Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            dim_feedforward=emb_size * 2,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 5. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, 32),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input x: Tensor of shape (Batch, n_channels, n_samples) or (Batch, 1, n_channels, n_samples)
        Output: Logits tensor of shape (Batch, n_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension: (B, 1, C, T)

        # Convolutional Feature Extractor
        x = self.temporal_conv(x)   # -> (B, 16, C, T)
        x = self.spatial_conv(x)    # -> (B, 32, 1, T/4)
        x = self.separable_conv(x)  # -> (B, emb_size, 1, T/16)

        # Reshape to Sequence format for Transformer: (B, T_seq, emb_size)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T_seq, emb_size)

        # Add positional embedding
        if x.size(1) == self.pos_embedding.size(1):
            x = x + self.pos_embedding
        else:
            # Dynamic positional interpolation if sample length varies
            pos_emb = F.interpolate(
                self.pos_embedding.permute(0, 2, 1),
                size=x.size(1),
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)
            x = x + pos_emb

        # Transformer Temporal Self-Attention
        x = self.transformer(x)

        # Global average pooling across time sequence
        feat = x.mean(dim=1)

        # Final Classification
        logits = self.classifier(feat)
        return logits
