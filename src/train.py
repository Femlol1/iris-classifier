#!/usr/bin/env python3
"""Run the Iris notebook workflow without Jupyter.

Install: python -m pip install scikit-learn matplotlib
Run:     python src/train.py --test-size 0.2 --random-state 42
Paths are relative to the current working directory.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a GUI.
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data used for testing (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Seed for the split and decision tree (default: 42)")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/confusion/_matrix.png"),
                        help="Decision tree confusion matrix PNG path")
    parser.add_argument("--tree-output", type=Path, default=None,
                        help="Optionally save a decision tree PNG")
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1 (exclusive)")
    if not 0 <= args.random_state <= 2**32 - 1:
        parser.error("--random-state must be between 0 and 4294967295")

    iris = load_iris()
    # k-NN requires at least five training samples.
    import math
    if len(iris.target) - math.ceil(len(iris.target) * args.test_size) < 5:
        parser.error("--test-size must leave at least five training samples")
    # Preserve the notebook's exact split (no stratification).
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=args.test_size, random_state=args.random_state
    )
    model = DecisionTreeClassifier(random_state=args.random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Decision tree accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nDecision tree classification report:")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2],
                                target_names=iris.target_names, zero_division=0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=[0, 1, 2], display_labels=iris.target_names,
        cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title("Iris decision tree: test confusion matrix")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160, format="png")
    plt.close(fig)
    print(f"Confusion matrix saved to: {args.output.resolve()}")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    print(f"k-NN accuracy: {accuracy_score(y_test, knn.predict(X_test)):.4f}")

    if args.tree_output is not None:
        args.tree_output.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(20, 12))
        plot_tree(model, feature_names=iris.feature_names,
                  class_names=list(iris.target_names), filled=True, ax=ax)
        fig.tight_layout()
        fig.savefig(args.tree_output, dpi=160, format="png")
        plt.close(fig)
        print(f"Decision tree saved to: {args.tree_output.resolve()}")


if __name__ == "__main__":
    main()