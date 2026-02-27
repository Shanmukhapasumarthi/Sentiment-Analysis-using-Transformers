from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
import re
import os

from Dataset.dataset import IMDBDataset
from src.pipeline.model import MiniTransformer

app = Flask(__name__)

# ------------------ CONFIG ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "imdb_train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "mini_transformer.pth")

MAX_LEN = 100
MAX_VOCAB = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ TEXT PROCESSING ------------------

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

# ------------------ LOAD MODEL ONCE ------------------

print("Loading model...")

train_dataset = IMDBDataset(
    DATASET_PATH,   # ✅ Correct full path
    max_len=MAX_LEN,
    max_vocab=MAX_VOCAB,
    build_vocab=True
)

vocab = train_dataset.vocab

model = MiniTransformer(
    vocab_size=len(vocab)
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully!")

# ------------------ PREDICTION FUNCTION ------------------

def predict_sentiment(text):
    encoded = encode_text(text, vocab).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(encoded)
        probs = F.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    return label, round(confidence * 100, 2)

# ------------------ ROUTES ------------------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        review = request.form["review"]
        prediction, confidence = predict_sentiment(review)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)