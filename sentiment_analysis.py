"""Entrenamiento y evaluacion reproducible de sentimiento sobre IMDB."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
LABELS = ["negative", "positive"]


def clean_review(text: str) -> str:
    """Elimina HTML y caracteres que no aportan al modelo."""
    plain_text = BeautifulSoup(text, "html.parser").get_text(" ")
    return re.sub(r"[^A-Za-z\s']", " ", plain_text).lower()


def load_data(path: Path, samples_per_class: int | None) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"review", "sentiment"}
    if not required.issubset(data.columns):
        raise ValueError(f"El CSV debe contener las columnas: {sorted(required)}")
    if data[list(required)].isna().any().any():
        raise ValueError("El conjunto de datos contiene valores nulos")

    if samples_per_class is not None:
        data = (
            data.groupby("sentiment", group_keys=False)
            .sample(n=samples_per_class, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
    return data


def build_model(max_features: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_review,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=max_features,
                    min_df=2,
                ),
            ),
            ("classifier", LinearSVC(random_state=RANDOM_STATE)),
        ]
    )


def save_confusion_matrix(y_true: pd.Series, y_pred: list[str], path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.xlabel("Prediccion")
    plt.ylabel("Valor real")
    plt.title("Matriz de confusion - IMDB")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/IMDB Dataset.csv"))
    parser.add_argument("--samples-per-class", type=int, default=5000)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.data, args.samples_per_class)
    x_train, x_test, y_train, y_test = train_test_split(
        data["review"],
        data["sentiment"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["sentiment"],
    )

    model = build_model(args.max_features)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "dataset_size": len(data),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
        "classification_report": classification_report(
            y_test, predictions, labels=LABELS, output_dict=True
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    save_confusion_matrix(y_test, predictions, args.output / "confusion_matrix.png")

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")
    print(classification_report(y_test, predictions, labels=LABELS))


if __name__ == "__main__":
    main()
