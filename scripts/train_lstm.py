import argparse
import pathlib
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "cyber_detect_backend-master"
sys.path.append(str(BACKEND_DIR))

# Reuse preprocessing/tokenizer and prepared arrays from lstmtest3
import lstmtest3 as m


def build_model(vocab_size: int, maxlen: int, embedding_dim: int = 64, lstm_units: int = 64) -> keras.Model:
    model = keras.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
        layers.LSTM(lstm_units),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=5000, help="Limit training samples for quick build (None for full)")
    args = parser.parse_args()

    # Determine vocab size compatible with m.tokenizer
    vocab_size = min(getattr(m, "num_words", 50000), len(m.tokenizer.word_index) + 1)
    maxlen = getattr(m, "maxlen", 50)

    x_train = m.x_train
    y_train = m.train_labels
    x_valid = m.x_valid
    y_valid = m.valid_labels

    if args.limit and args.limit < x_train.shape[0]:
        x_train = x_train[: args.limit]
        y_train = y_train[: args.limit]

    model = build_model(vocab_size=vocab_size, maxlen=maxlen)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    out_path = BACKEND_DIR / "model3.h5"
    model.save(out_path)
    print(f"Saved LSTM model to {out_path}")


if __name__ == "__main__":
    main()

