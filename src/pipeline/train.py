# train.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from Dataset.dataset import IMDBDataset
from src.pipeline.model import MiniTransformer
from src.utils import train_one_epoch, evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 32
    EPOCHS = 25
    MAX_VOCAB = 1000
    MAX_LEN = 100

    # Build vocab ONLY from training data
    train_dataset = IMDBDataset(
        "imdb_train.csv",
        max_len=MAX_LEN,
        max_vocab=MAX_VOCAB,
        build_vocab=True
    )

    test_dataset = IMDBDataset(
        "imdb_test.csv",
        vocab=train_dataset.vocab,
        max_len=MAX_LEN,
        max_vocab=MAX_VOCAB
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MiniTransformer(
        vocab_size=len(train_dataset.vocab)
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Test  Acc  : {test_acc:.4f}")
        print("-" * 40)
    torch.save(model.state_dict(), "mini_transformer.pth")
    print("Model saved as mini_transformer.pth")


if __name__ == "__main__":
    main()
