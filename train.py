import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb
from src.models.model import MambaClassifier


def tokenize_function(examples, tokenizer, max_length=512):
    return {
        **tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ),
        "labels": examples["label"],
    }


def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = torch.tensor([item["labels"] for item in batch])
    input_ids = pad_sequence(input_ids, batch_first=True)
    input_ids = input_ids.unsqueeze(-1)
    return {"input_ids": input_ids, "labels": labels}


def train():
    # Initialize wandb
    wandb.init(project="mamba-classification")

    # Hyperparameters
    config = {
        "d_model": 512,
        "n_layers": 4,
        "num_classes": 2,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "num_epochs": 3,
        "max_length": 512,
    }

    # Load dataset
    dataset = load_dataset("imdb")

    dataset["train"] = dataset["train"].select(range(100))
    dataset["test"] = dataset["test"].select(range(100))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize datasets
    tokenized_train = dataset["train"].map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenized_test = dataset["test"].map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=dataset["test"].column_names,
    )

    # Create dataloaders with collate_fn
    train_loader = DataLoader(
        tokenized_train,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        tokenized_test, batch_size=config["batch_size"], collate_fn=collate_fn
    )

    # Initialize model with input projection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaClassifier(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        vocab_size=tokenizer.vocab_size,
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            # Convert input_ids and labels to tensors - use clone().detach() to avoid warnings
            input_ids = batch["input_ids"].clone().detach().to(device).float()
            labels = batch["labels"].clone().detach().to(device)

            # Forward pass
            outputs = model(input_ids, labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            train_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": loss.item(), "acc": train_correct / train_total}
            )

        # Evaluation
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device).float()
                labels = batch["labels"].to(device)

                outputs = model(input_ids, labels)
                test_loss += outputs["loss"].item()
                predictions = torch.argmax(outputs["logits"], dim=1)
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)

        # Log metrics
        metrics = {
            "train_loss": train_loss / len(train_loader),
            "train_acc": train_correct / train_total,
            "test_loss": test_loss / len(test_loader),
            "test_acc": test_correct / test_total,
        }
        wandb.log(metrics)
        print(f"Epoch {epoch+1} metrics:", metrics)


if __name__ == "__main__":
    train()
