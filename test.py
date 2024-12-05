# Example usage
import torch

from src.models.model import MambaClassifier

model = MambaClassifier(d_model=16, n_layers=2, num_classes=10, dropout=0.1, vocab_size=1000)

# Forward pass without labels (inference)
B, L = 2, 64  # batch_size, sequence_length
x = torch.randint(0, 1000, (B, L, 1)).float()  # Shape: (batch_size, seq_len, 1)
outputs = model(x)
logits = outputs["logits"]  # Shape: (B, num_classes)

# Forward pass with labels (training)
labels = torch.randint(0, 10, (B,))
outputs = model(x, labels)
loss = outputs["loss"]  # Cross entropy loss
logits = outputs["logits"]

print(f"loss: {loss}")
print(f"logits.shape: {logits.shape}")