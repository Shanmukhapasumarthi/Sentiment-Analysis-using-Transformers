# input.py
import torch
import torch.nn.functional as F
import re

from Dataset.dataset import IMDBDataset
from src.pipeline.model import MiniTransformer


# ------------------ CONFIG ------------------
MODEL_PATH = "mini_transformer.pth"
MAX_LEN = 100
MAX_VOCAB = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.split()


def encode_text(text, vocab):
    tokens = clean_text(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]

    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    return torch.tensor(ids, dtype=torch.long)


def predict_sentiment(text, model, vocab):
    encoded = encode_text(text, vocab).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(encoded)
        probs = F.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    return label, confidence


def main():
    # 🔹 Load training dataset ONLY to get vocab
    train_dataset = IMDBDataset(
        "imdb_train.csv",
        max_len=MAX_LEN,
        max_vocab=MAX_VOCAB,
        build_vocab=True
    )

    vocab = train_dataset.vocab

    # 🔹 Load trained model
    model = MiniTransformer(
        vocab_size=len(vocab)
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    print("\nMiniTransformer Sentiment Analyzer")
    print("Type a review (or 'exit' to quit)\n")

    while True:
        text = input("Review: ")
        if text.lower() == "exit":
            break

        label, confidence = predict_sentiment(text, model, vocab)
        print(f"Prediction: {label} | Confidence: {confidence:.2f}\n")


if __name__ == "__main__":
    main()
