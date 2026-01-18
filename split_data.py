import pandas as pd
from sklearn.model_selection import train_test_split

import os
print("CURRENT WORKING DIRECTORY:", os.getcwd())


# Load dataset
df = pd.read_csv("Dataset/IMDB Dataset.csv")

# Features and labels
X = df["review"]
y = df["sentiment"]

# 70% Train, 30% Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Combine back into DataFrames
train_df = pd.DataFrame({"review": X_train, "sentiment": y_train})
test_df = pd.DataFrame({"review": X_test, "sentiment": y_test})

# Save to CSV
train_df.to_csv("imdb_train.csv", index=False)
test_df.to_csv("imdb_test.csv", index=False)

print("Train size:", len(train_df))
print("Test size:", len(test_df))
