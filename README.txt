Darknet Traffic Classification - CIC-Darknet2020
CMP7200 Individual Master's Project

CONTENTS
  Darknet_Traffic_Classification.ipynb   Full notebook, executed end to end (Stage 1 + Stage 2)
  processed_data/                        Preprocessing artefacts
  models/                                All trained / fine-tuned models (pickle)
  results/                               Comparison tables and run config

TWO TASKS  (every model is trained once per task)
  *_binary.pkl           Task A - darknet (Tor/VPN) vs benign        2 classes
  *_classification.pkl   Task B - application type                   8 classes

processed_data/
  model_ready.npz        X_train, X_test, y_train, y_test, app_train, app_test
  scaler.pkl             StandardScaler, fitted on the training partition only
  feature_metadata.json  52 feature names, engineered list, log1p list, label encodings
  darknet_cleaned.csv.gz Cleaned + engineered dataset (values are log1p-transformed)

models/
  random_forest, xgboost, lightgbm, catboost          tree baselines
  tabnet, ft_transformer                              deep tabular
  etbert_inspired, tfe_gnn_inspired                   adapted architectures (* not faithful reimplementations)
  etbert_inspired_pretrained.pkl                      MLM-pretrained encoder, before fine-tuning
  bft_proposal                                        the proposal
  bft_leaf_forest                                     Random Forest supplying BFT's leaf tokens (needed to reuse BFT)
  ablation_*                                          three BFT ablation variants (8-class task)

LOADING
  import pickle, numpy as np
  d = np.load("processed_data/model_ready.npz")
  X_test = d["X_test"].astype(np.float32)
  model = pickle.load(open("models/xgboost_binary.pkl", "rb"))
  pred = model.predict(X_test)

NOTE  Torch-based models (ft_transformer, etbert_inspired, tfe_gnn_inspired, bft_proposal,
      ablation_*) were pickled on a CUDA device and need a GPU to unpickle, plus their class
      definitions from the notebook in scope. The sklearn/boosting models load anywhere.

HEADLINE RESULTS (test set, macro-F1)
  Task A binary          LightGBM 0.9756  |  BFT 0.9710
  Task B application     XGBoost  0.8506  |  BFT 0.8354
