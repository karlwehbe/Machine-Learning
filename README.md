# COMP 551 — Applied Machine Learning

Coursework from McGill COMP 551: implementations of classical ML and neural models in Python/NumPy, with evaluation on public tabular and vision/text datasets.

## Repository contents

| File | Assignment | Topic |
|------|------------|--------|
| `Assignment_1.ipynb` | A1 | k-NN & decision trees |
| `Assignment_2.ipynb` | A2 | Linear & logistic regression, classification |
| `Assignment_3.ipynb` | A3 | Multilayer perceptrons on KMNIST |
| `Assignment_4.ipynb` | A4 | BERT fine-tuning & embedding probes |

Each assignment includes a companion report PDF (`Assignment_*-Report.pdf`).

## Assignments

### A1 — Classification from scratch

- **What:** Weighted k-NN and decision trees for species and medical diagnosis.
- **Architecture:** Custom distance metrics (Euclidean, Manhattan, Gower), tree splits with Gini, entropy, and misclassification costs.
- **Datasets:** Penguin morphology (`penguins_size.csv`), UCI heart disease.
- **Results:** Hyperparameter search over *K* and tree depth; metrics include accuracy and AUROC on train/validation/test splits.

### A2 — Linear regression & GLMs

- **What:** Linear, binary logistic, and multinomial logistic regression implemented from scratch.
- **Architecture:** NumPy gradient descent with validation-based early stopping; closed-form least squares where applicable.
- **Datasets:** UCI breast cancer (binary), penguin species (multiclass).
- **Results:** Breast cancer — **97.7%** test accuracy, **0.97** test AUROC; penguin multiclass — up to **99.7%** accuracy on held-out data.

### A3 — Neural networks (MLP)

- **What:** Feed-forward classifier for handwritten character recognition.
- **Architecture:** Configurable MLP (ReLU hidden layers, softmax output), cross-entropy loss, backpropagation, optional L2; `GridSearchCV` over depth, width, learning rate, and batch size.
- **Dataset:** KMNIST (60k train / 10k test, 28×28 grayscale).
- **Results:** Baseline 2×64 MLP ~**83%** test accuracy; tuned **256–128** hidden layers → **89.1%** test accuracy; gradients verified with numerical checks.

### A4 — Transformers & representation learning

- **What:** News topic classification with BERT on AG News.
- **Architecture:** Hugging Face `BertForSequenceClassification` (full fine-tune) vs. frozen BERT embeddings (CLS / mean / max) + KNN or logistic regression probes.
- **Results:** Best embedding probe (mean pooling + KNN, *k*=10) — **91.9%** test; logistic probe — **91.3%**; fine-tuned BERT — **93.1%** test.

## Tech stack

- Python 3, NumPy, pandas, matplotlib, seaborn
- scikit-learn (splits, metrics, `GridSearchCV`)
- PyTorch / Hugging Face Transformers (A4)
- Google Colab for notebook execution

## Running the notebooks

1. Clone the repo and open a notebook in Jupyter or Colab.
2. Install dependencies as needed in the first cells (`ucimlrepo`, `gower`, `transformers`, etc.).
3. **A1:** Penguin CSV and heart-disease data paths may need updating if not using Google Drive.
4. **A2:** Breast cancer data is fetched via `ucimlrepo`; penguin CSV may need a local path update if not on Drive.
5. **A3:** Place KMNIST `.npz` files where the notebook expects them, or update `dataset_path` / load paths in the data cell.
6. **A4:** AG News is loaded via Hugging Face `datasets`; GPU recommended for BERT fine-tuning.

## Notes

- Notebooks were developed in Colab; paths like `/content/drive/...` should be adjusted for local runs.
- Reported metrics come from the executed notebook outputs at submission time; slight variance is normal if data splits or seeds differ.
