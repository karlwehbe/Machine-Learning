# COMP 551 — Applied Machine Learning

McGill **COMP 551** coursework: end-to-end ML pipelines in Python spanning **instance-based learners**, **generalized linear models**, **feed-forward and convolutional neural networks**, and **pretrained transformer fine-tuning**. Core learners are implemented **from scratch** in NumPy where required; evaluation follows a consistent **build → validate → test → compare** workflow across assignments.

## Repository layout

| Artifact | Scope |
|----------|--------|
| `a1_knn_decision_trees.ipynb` | k-NN, decision trees, heterogeneous feature distances |
| `a2_glm_regression_classification.ipynb` | Linear / logistic / softmax regression (GLMs) |
| `a3_mlp_cnn_kmnist.ipynb` | MLP from scratch; CNN baseline; regularization & activation ablations |
| `a4_bert_probing_finetuning_ag_news.ipynb` | BERT representation probing and full fine-tuning |
| `a*_..._report.pdf` | Written reports per assignment |

---

## A1 — Instance-based & tree-based classification

**Objective.** Implement non-parametric and recursive-partitioning classifiers without relying on `sklearn` for the core prediction logic, then benchmark them on multiclass morphology data and binary clinical data.

### Build

| Component | Details |
|-----------|---------|
| **Distance functions** | Euclidean, Manhattan, and **Gower** (mixed-type) metrics for k-NN. |
| **Weighted k-NN** | Class votes weighted by inverse distance; supports dense features or **precomputed distance submatrices** (train/val/test blocks sliced from a full Gower matrix). |
| **Decision tree** | Recursive binary splits via **greedy threshold search** on sorted feature values; leaf nodes store class histograms for probability estimates. |
| **Impurity functions** | **Gini index**, **entropy**, and **misclassification rate** implemented as interchangeable split criteria. |
| **Tree traversal** | Inference walks the learned tree to produce hard labels and per-class probabilities. |
| **Preprocessing** | Missing-value removal, label encoding, **standardization**, invalid-category filtering, and (for heart data) **one-hot encoding** of categorical fields. |

**Data pipeline (both datasets).** Load → clean → standardize → exploratory group-mean analysis → stratified **train / validation / test** splits.

### Test

- **k-NN:** Sweep neighborhood size `K`; tune on **validation accuracy** and **AUROC**, then refit on train+val and evaluate on the **held-out test** set.
- **Decision trees:** Sweep **maximum depth** with the same validation → test protocol.
- **Multiclass:** One-vs-rest **AUROC** curves where applicable.
- **Binary (heart):** ROC curves and AUROC for positive-class detection.
- **Diagnostics:** `confusion_matrix` heatmaps via helper utilities.

### Compare

| Comparison | What was contrasted |
|------------|---------------------|
| **Distance metrics (penguins)** | Euclidean vs. Manhattan k-NN on the same feature space — overlay plots of test accuracy vs. `K`. |
| **Split criteria (trees)** | Gini vs. entropy vs. misclassification — validation/test accuracy and AUROC vs. depth on both datasets. |
| **k-NN vs. decision trees** | Unified plots of test accuracy vs. `K` (k-NN) and vs. depth (trees) on penguin data. |
| **Heart disease — cost functions** | Side-by-side test accuracy and AUROC curves for all three impurity functions. |
| **Heart disease — k-NN vs. DT** | Gower k-NN AUROC vs. tree AUROC across hyperparameters. |
| **Feature importance** | Top split features per tree (Gini / entropy / misclassification) on penguin and heart models. |
| **Correlation analysis (heart)** | Pearson heatmap to study redundant clinical features before modeling. |

---

## A2 — Generalized linear models & gradient-based optimization

**Objective.** Derive and optimize **linear**, **binary logistic**, and **multinomial softmax** models from first principles, then compare their inductive biases on binary and multiclass tabular tasks.

### Build

| Model | Implementation |
|-------|----------------|
| **Linear regression** | Bias-augmented design matrix; weights via **`np.linalg.lstsq`** (closed form). |
| **Binary logistic regression** | Sigmoid link, **log-loss**, hand-derived gradient, **mini-batch gradient descent** with convergence on gradient norm. |
| **Multinomial logistic** | Softmax over \(K\) classes; **cross-entropy** loss; matrix-valued weights \(W \in \mathbb{R}^{D \times K}\). |
| **Training utilities** | Per-iteration **validation accuracy tracking**; **best-weight checkpointing** (`use_best_w=True` at inference). |
| **Sanity checks** | **Analytical vs. numerical gradient** comparison (finite differences, relative error reporting). |

**Data work.** Feature standardization; removal of features with negative linear-regression coefficients (breast cancer); one-hot multiclass targets for penguins.

### Test

**Breast cancer (binary)**
1. Fit SLR / MLR / BLR on full cleaned data — report in-sample accuracy.
2. **Train / validation / test** split (70% → 50/50 train-val, 30% test).
3. Logistic model trained up to 10k iterations; select **best validation-accuracy** iterate.
4. Report **test accuracy**, **test AUROC**, and **CE loss curves** (train vs. validation) with best-iteration marker.

**Penguins (multiclass)**
1. One-hot encode species labels; fit multinomial logistic with gradient descent.
2. Gradient check on softmax regression.
3. Extended experiments: train/val/test splits, CE loss monitoring, held-out accuracy.

### Compare

| Analysis | Models / views |
|----------|----------------|
| **Feature ranking (Task 4.1)** | Horizontal barplots of SLR coefficients — identify weak or negatively correlated predictors. |
| **In-sample baselines (4.2–4.3)** | Multiple linear regression vs. binary logistic on all data. |
| **Held-out head-to-head (4.5–4.6)** | **SLR vs. MLR vs. BLR** — accuracy, AUROC, and **overlaid ROC curves** on the same test fold. |
| **Weight visualization (4.8)** | **Heatmap** of multinomial logistic weight matrix across features × classes. |
| **Penguin experiments** | Multivariate linear vs. multinomial logistic; training/validation CE curves; final test accuracy (reported up to **100%** on some experiment splits in-notebook). |

**Reported test metrics (primary held-out runs).**

| Task | Metric |
|------|--------|
| Binary logistic (breast cancer) | **97.7%** accuracy, **0.97** AUROC |
| Multinomial logistic (penguins) | up to **99.7%** accuracy |

---

## A3 — Deep learning: MLP & CNN on KMNIST

**Objective.** Implement a fully differentiable **MLP** in NumPy, validate optimization correctness, tune hyperparameters systematically, and contrast against a **convolutional baseline** on 28×28 character recognition (**KMNIST**: 60k train / 10k test).

### Build

| Layer | Details |
|-------|---------|
| **Activations** | Modular `Relu`, `LeakyRelu`, `Sigmoid`, `Softmax` with forward and derivative (`prime`) methods. |
| **Loss** | `Cross_Entropy` coupled to the output activation. |
| **MLP** | Arbitrary depth/width; layer-wise `(W, b)`; **mini-batch SGD**; optional **L2 penalty** on weights in the backward pass. |
| **Training loop** | Per-epoch train/val accuracy traces; **checkpoint best weights** by validation accuracy (`best_W`, `best_iteration`). |
| **sklearn bridge** | `MLPClassifierWrapper` exposing the custom MLP to **`GridSearchCV`**. |
| **CNN (baseline)** | TensorFlow/Keras `Sequential` model on reshaped `(28, 28, 1)` inputs — not from scratch. |

**Data ingestion.** Load `.npz` tensors → flatten to 784-D (MLP) or preserve spatial layout (CNN) → **per-pixel standardization** (train statistics applied to test) → train/val split from the 60k training pool.

### Test

1. **Smoke test:** 2×64 ReLU MLP, 50 epochs — plot train/val accuracy; mark best epoch on test (**~83.4%**).
2. **Gradient checking:** Finite-difference vs. backprop on a small batch — per-layer error **~1e-7**.
3. **Grid search (2 hidden layers):** 3-fold CV over hidden sizes `(64,64)`, `(128,128)`, `(128,64)`, `(256,128)` × learning rates × batch sizes → deploy best config on full test set (**89.1%**).
4. **Grid search (1 hidden layer):** Second search over `(32,)`, `(64,)`, `(128,)`, `(256,)` topologies.
5. **Task 3.1 depth ablation:** 0 / 1 / 2 hidden layers — train/val curves + test accuracy (e.g. no-hidden-layer generalizes poorly to test despite high train acc).
6. **Task 3.2 activations:** Compare **ReLU vs. LeakyReLU vs. Sigmoid** hidden units under fixed architecture.
7. **Task 3.3 L2 regularization:** Sweep `λ` — study bias–variance and overfitting.
8. **Task 3.4 CNN:** Train Keras CNN; record test performance.
9. **Extra:** **K-fold cross-validation** on MLP configurations.

### Compare

| Experiment | Finding |
|------------|---------|
| **Depth (3.1)** | Deeper MLPs increase train accuracy sharply; 2-hidden-layer model best balances val/test generalization among scratch MLPs. |
| **Activations (3.2)** | ReLU family outperforms sigmoid on convergence and final accuracy. |
| **L2 (3.3)** | Regularization mitigates overfitting when train ≫ val gap. |
| **MLP vs. CNN (3.4)** | Direct comparison of test accuracy — CNN leverages spatial locality; MLP operates on flattened pixels. |
| **GridSearchCV** | CV mean scores visualized by hidden topology, learning rate, and batch size (seaborn line plots). |

---

## A4 — Transformer representations & transfer learning

**Objective.** Classify news headlines into **four topics** (AG News) by (1) **linear probing** frozen BERT embeddings and (2) **end-to-end fine-tuning**, then analyze whether full adaptation beats shallow probes.

### Build

| Stage | Details |
|-------|---------|
| **Data** | `load_dataset("ag_news")` — label distribution plots for train/val/test. |
| **Tokenizer & encoder** | `bert-base-uncased` loaded on GPU; tokenized inputs for batch inference. |
| **Feature extraction** | `extract_features()` with strategies **`cls`**, **`mean`**, **`max`** pooling over last hidden states; optional `max_samples` subsampling for probing runs. |
| **Caching** | `save_features` / `load_features` — persist `(X, y)` matrices per strategy to avoid recomputation. |
| **Probing classifiers** | **k-NN** (sweep `K`) and **logistic regression** on frozen embeddings — tuned on validation accuracy per strategy. |
| **Fine-tuning** | `BertForSequenceClassification` + Hugging Face **`Trainer`** / `TrainingArguments`; subset training (20k/2k) for manageable runtime; **`EarlyStoppingCallback`**; full-weight updates including classification head. |
| **Interpretability** | `bertviz` attention maps on high-confidence correct vs. incorrect test examples. |

### Test

1. **Probing loop:** For each pooling strategy, extract train/val features → fit k-NN and logistic probes → record **validation accuracy** tables.
2. **Strategy selection:** Pick **best embedding strategy** (mean pooling) from validation probes.
3. **Test-set evaluation:** Load or extract test features → k-NN (`k=10`) and logistic test accuracy.
4. **Fine-tune:** Train BERT end-to-end → **validation accuracy** after training (~**92.4%**).
5. **Final test:** Evaluate fine-tuned checkpoint on held-out AG News test split (**93.1%**).
6. **Bar chart:** Side-by-side test accuracies — **KNN probe vs. logistic probe vs. fine-tuned BERT**.

### Compare

| Comparison | Result (test) |
|------------|---------------|
| **Pooling strategies × probes** | Tabulated validation accuracy for CLS / mean / max × (logistic, k-NN); bar plots and **K vs. accuracy** curves for k-NN. |
| **Best probe vs. fine-tune** | Mean + k-NN **91.9%**; mean + logistic **91.3%**; fine-tuned BERT **93.1%** — full fine-tuning wins by ~1–2 pts. |
| **Attention analysis** | Qualitative comparison of attention patterns on correct high-confidence vs. misclassified examples. |

---

## Stack & tooling

| Layer | Libraries |
|-------|-----------|
| Numerics | NumPy, pandas, SciPy |
| Classical ML | scikit-learn (`train_test_split`, `KFold`, metrics, `GridSearchCV`) |
| Deep learning | TensorFlow/Keras (A3 CNN); Hugging Face `transformers`, `datasets`, `Trainer` (A4) |
| Visualization | matplotlib, seaborn, bertviz |
| Environment | Jupyter / Google Colab (GPU for BERT) |

Additional A1 deps: `ucimlrepo`, `gower`.

---

## Reproducibility

1. Open the target `.ipynb` in Jupyter Lab or Colab.
2. Run install cells (`pip install …`) for assignment-specific packages.
3. **Rewrite filesystem paths** — notebooks default to Colab Drive paths (`/content/drive/...`).
4. **A1:** Update local paths for penguin CSV; heart data via `ucimlrepo`.
5. **A3:** Provide KMNIST `.npz` archives or update load paths in the data ingestion cell.
6. **A4:** `load_dataset("ag_news")` from Hugging Face; CUDA strongly recommended for fine-tuning.

Metrics in this README match executed notebook outputs at submission time; reruns may differ slightly under new seeds or library versions.
