# Iris Classifier  (Decision Tree)

A Python CLI project that trains and compares a decision tree and a k-nearest neighbours (k-NN) classifier on the Iris dataset. It prints accuracy scores and a classification report, saves a confusion matrix, and can export a decision tree diagram.

## Dataset and workflow

The script loads the Iris dataset directly from scikit-learn, so no separate dataset download is needed. It contains 150 flowers, with four measurements: sepal length, sepal width, petal length and petal width. The target classes are setosa, versicolor and virginica.

The workflow:

1. Split the data into training and test sets.
2. Train a decision tree and predict the test labels.
3. Print decision tree accuracy, precision, recall and F1-score.
4. Save the decision tree confusion matrix as a PNG.
5. Train a k-NN classifier with five neighbours and print its test accuracy.
6. Optionally save a diagram of the trained decision tree.

Both models use the same split. The script preserves the original notebook workflow: it does not stratify the split or scale the features.

## Setup

Run these commands from the project root, the folder containing `src/train.py`. Python and pip must be installed. Jupyter and TensorFlow are not required.

Create a virtual environment if you do not already have one:

```bash
python -m venv venv
```

Activate it using the command for your terminal:

| Terminal | Command |
| --- | --- |
| Windows Git Bash | `source venv/Scripts/activate` |
| Windows Command Prompt | `venv\Scripts\activate.bat` |
| Windows PowerShell | `.\venv\Scripts\Activate.ps1` |
| macOS / Linux | `source venv/bin/activate` |

Install the dependencies:

```bash
python -m pip install scikit-learn matplotlib
```

## Run training

```bash
python src/train.py --test-size 0.2 --random-state 42
```

This uses 120 samples for training and 30 for testing. The random seed controls the split and the decision tree's randomness.

The default values are the same, so you can also run:

```bash
python src/train.py
```

## CLI options

| Option | Default | Purpose |
| --- | --- | --- |
| `--test-size` | `0.2` | Fraction used for testing; must be between 0 and 1 and leave at least five training samples. |
| `--random-state` | `42` | Integer seed from 0 to 4294967295. |
| `--output` | `outputs/confusion/_matrix.png` | Path for the decision tree confusion matrix PNG. |
| `--tree-output` | None | Optional path for a decision tree diagram PNG. |
| `--help` | — | Show usage and available options. |

Change the split:

```bash
python src/train.py --test-size 0.3 --random-state 7
```

Save the decision tree diagram as well:

```bash
python src/train.py --test-size 0.2 --random-state 42 --tree-output outputs/decision_tree.png
```

Choose a different confusion matrix path:

```bash
python src/train.py --output outputs/confusion/custom_matrix.png
```

## Outputs

| File | Description |
| --- | --- |
| `src/train.py` | Training and evaluation entry point. |
| `outputs/confusion/_matrix.png` | Generated decision tree confusion matrix. |
| `outputs/decision_tree.png` | Generated only when requested with the example `--tree-output` command above. |

Output paths are relative to the directory where you run the command. Parent folders are created automatically. Running again with the same output path replaces the previous image. Plots are saved without opening a GUI window. Trained models are not saved to disk.

In the verified run with `--test-size 0.2 --random-state 42`, both models achieved accuracy of `1.0000` (100%). The decision tree correctly classified all 30 test samples: 10 setosa, 9 versicolor and 11 virginica.

These results describe one small test split; they do not guarantee perfect accuracy on other splits or new data.

## Troubleshooting

- **`ModuleNotFoundError`:** activate your virtual environment and rerun the dependency installation command.
- **`venv/bin/activate` not found in Windows Git Bash:** use `source venv/Scripts/activate`.
- **`src/train.py` not found:** run the command from the project root and check that the script is saved inside `src`.
- **`KeyboardInterrupt`:** the process was interrupted. Run it again and allow the imports and training to finish.

To leave the virtual environment:

```bash
deactivate
```
