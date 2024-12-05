import torch
from datasets import load_dataset
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


def get_default_config():
    return {
        "d_model": 256,
        "n_layers": 4,
        "num_classes": 2,
        "dropout": 0.1,
        "learning_rate": 2e-4,
        "batch_size": 32,
        "num_epochs": 3,
        "max_length": 512,
    }


def prepare_data(config):
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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

    train_loader = DataLoader(
        tokenized_train,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        tokenized_test, batch_size=config["batch_size"], collate_fn=collate_fn
    )

    return train_loader, test_loader, tokenizer


def setup_model(config, tokenizer):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = MambaClassifier(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        vocab_size=tokenizer.vocab_size,
    ).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    return model, device


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].clone().detach().to(device).float()
        labels = batch["labels"].clone().detach().to(device)

        outputs = model(input_ids, labels)
        loss = outputs["loss"]
        logits = outputs["logits"]

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)
        train_loss += loss.item()

        progress_bar.set_postfix(
            {"loss": loss.item(), "acc": train_correct / train_total}
        )

    return train_loss / len(train_loader), train_correct / train_total


def evaluate(model, test_loader, device):
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

    return test_loss / len(test_loader), test_correct / test_total


def save_checkpoint(model, optimizer, epoch, test_acc):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_acc": test_acc,
        },
        "best_model.pt",
    )


def train():
    wandb.init(project="mamba-classification")
    config = get_default_config()

    train_loader, test_loader, tokenizer = prepare_data(config)
    model, device = setup_model(config, tokenizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    best_test_acc = 0
    patience_counter = 0
    max_patience = 5

    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        wandb.log(metrics)
        print(f"Epoch {epoch+1} metrics:", metrics)

        scheduler.step(test_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(model, optimizer, epoch, test_acc)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break


if __name__ == "__main__":
    train()
