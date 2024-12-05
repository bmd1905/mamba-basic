from typing import Optional

import torch
import torch.nn as nn

from src.models.backbone import Mamba, MambaConfig
from src.models.head import ClassifierHead


class MambaClassifier(nn.Module):
    def __init__(
        self,
        d_model: int = 16,
        n_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        """
        End-to-end Mamba model for classification tasks.

        Args:
            d_model: Hidden dimension size
            n_layers: Number of Mamba layers
            num_classes: Number of classification categories
            dropout: Dropout rate for classifier head
        """
        super().__init__()

        # Initialize Mamba backbone
        config = MambaConfig(d_model=d_model, n_layers=n_layers)
        self.backbone = Mamba(config)

        # Initialize classifier head
        self.classifier = ClassifierHead(
            d_model=d_model, num_classes=num_classes, dropout=dropout
        )

        # Initialize loss function
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            labels: Optional tensor of shape (batch_size,) for computing loss

        Returns:
            dict containing:
                logits: Classification logits
                loss: Classification loss (if labels provided)
        """
        # Get sequence representations from backbone
        sequence_output = self.backbone(x)

        # Get classification logits
        logits = self.classifier(sequence_output)

        output = {"logits": logits}

        # Compute loss if labels are provided
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            output["loss"] = loss

        return output
