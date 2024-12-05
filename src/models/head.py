import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        """
        Simple classifier head that takes sequence output and predicts classes.

        Args:
            d_model: Hidden dimension size from the Mamba model
            num_classes: Number of classification categories
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, num_classes)
        """
        # Average pooling over sequence length
        x = x.mean(dim=1)  # (batch_size, d_model)

        # Apply layer norm and dropout
        x = self.norm(x)
        x = self.dropout(x)

        # Final classification layer
        return self.fc(x)
