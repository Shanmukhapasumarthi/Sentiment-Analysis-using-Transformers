# dataset.py
from collections import Counter
import torch
from torch.utils.data import Dataset
import pandas as pd
import re
import os


class IMDBDataset(Dataset):
    def __init__(self, csv_filename, vocab=None, max_len=100, max_vocab=1000, build_vocab=False):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, csv_filename)

        self.data = pd.read_csv(csv_path)
        self.max_len = max_len
        self.max_vocab = max_vocab

        self.texts = self.data["review"].values
        self.labels = self.data["sentiment"].map(
            {"negative": 0, "positive": 1}
        ).values

        if build_vocab:
            self.vocab = self.build_vocab(self.texts)
        else:
            self.vocab = vocab

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text.split()

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(self.clean_text(text))

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for i, (word, _) in enumerate(
            counter.most_common(self.max_vocab - 2), start=2
        ):
            vocab[word] = i

        return vocab

    def encode(self, text):
        tokens = self.clean_text(text)
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]

        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            self.encode(self.texts[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
