# Darknet Traffic Classification — CIC-Darknet2020

**CMP7200 Individual Master's Project**

Classification of darknet (Tor / VPN) network traffic using the CIC-Darknet2020 dataset,
comparing tree-based baselines, deep tabular models, adapted traffic-classification
architectures, and a proposed model (**BFT**).

---

## Contents

| Path | Description |
| --- | --- |
| `Darknet_Traffic_Classification.ipynb` | Full notebook, executed end to end (Stage 1 + Stage 2) |
| `processed_data/` | Preprocessing artefacts |
| `models/` | All trained / fine-tuned models (pickle) |
| `results/` | Comparison tables and run config |

---

## Two Tasks

Every model is trained once per task, and the filename suffix says which:

| Suffix | Task | Target | Classes |
| --- | --- | --- | --- |
| `*_binary.pkl` | **Task A** | Darknet (Tor/VPN) vs. benign | 2 |
| `*_classification.pkl` | **Task B** | Application type | 8 |

---

## `processed_data/`

| File | Contents |
| --- | --- |
| `model_ready.npz` | `X_train`, `X_test`, `y_train`, `y_test`, `app_train`, `app_test` |
| `scaler.pkl` | `StandardScaler`, fitted on the training partition only |
| `feature_metadata.json` | 52 feature names, engineered list, `log1p` list, label encodings |
| `darknet_cleaned.csv.gz` | Cleaned + engineered dataset (values are `log1p`-transformed) |

---

## `models/`

**Tree baselines**
`random_forest` · `xgboost` · `lightgbm` · `catboost`

**Deep tabular**
`tabnet` · `ft_transformer`

**Adapted architectures** *(not faithful reimplementations)*
`etbert_inspired` · `tfe_gnn_inspired`
`etbert_inspired_pretrained.pkl` — MLM-pretrained encoder, before fine-tuning

**Proposed model**
`bft_proposal` — the proposal
`bft_leaf_forest` — Random Forest supplying BFT's leaf tokens (required to reuse BFT)
`ablation_*` — three BFT ablation variants (8-class task)

---

## Loading

```python
import pickle
import numpy as np

d = np.load("processed_data/model_ready.npz")
X_test = d["X_test"].astype(np.float32)

model = pickle.load(open("models/xgboost_binary.pkl", "rb"))
pred = model.predict(X_test)
```

> **Note**
> Torch-based models (`ft_transformer`, `etbert_inspired`, `tfe_gnn_inspired`,
> `bft_proposal`, `ablation_*`) were pickled on a CUDA device. They need a GPU to
> unpickle, plus their class definitions from the notebook in scope.
> The sklearn / boosting models load anywhere.

---

## Headline Results

Test set, macro-F1:

| Task | Best model | Score | BFT |
| --- | --- | --- | --- |
| **A** — binary | LightGBM | **0.9756** | 0.9710 |
| **B** — application | XGBoost | **0.8506** | 0.8354 |
