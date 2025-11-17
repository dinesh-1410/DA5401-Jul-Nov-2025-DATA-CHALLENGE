
## DA5401 Data Challenge - JUL - NOV 2025

### Student Information

- **Name:** Saggurthi Dinesh  
- **Roll Number:** BE21B032  
- **Email:** be21b032@smail.iitm.ac.in  
- **Course:** DA5401 - Data Analytics Laboratory  
- **Project:** DA5401 Data Challenge 

---

### Project Structure

```text
DA5401-DATA-CHALLENGE/
├── DA5401_Data_Challenge_Training_Saggurthi_Dinesh_BE21B032.ipynb   # Model training & analysis
├── DA5401_Data_Challenge_Inference_Saggurthi_Dinesh_BE21B032ipynb.ipynb  # Inference & submission
├── DA5401-Data-Challenge-Report-Saggurthi-Dinesh-BE21B032.pdf       # Written report
└── README.md                                                        # This file
```

---

### Problem Description

- **Task:** Predict a **discrete evaluation score in \([0, 10]\)** for a large language model (LLM) response, given:
  - An **evaluation metric** (e.g., helpfulness, correctness)
  - The **system prompt**
  - The **user prompt** / `prompt`
  - The **LLM response** / `expected_response`
- **Input format (train):** Records in `train_data.json` with fields such as:
  - `metric_name`
  - `system_prompt` (optional)
  - `user_prompt` or `prompt`
  - `response` or `expected_response`
  - `score` (integer in \([0, 10]\))
- **Goal:** Learn a robust model that maps **(metric, prompt–response pair)** to an integer score, and generate a **`submission.csv`** on the hidden test set.

---

### Dataset

- **Source:** DA5401 2025 Data Challenge Kaggle dataset
- **Key files:**
  - `train_data.json` – labeled training samples with scores
  - `test_data.json` – unlabeled test samples
  - `metric_names.json` – list of all evaluation metrics
  - `metric_name_embeddings.npy` – precomputed vector embeddings for metrics
- **Text fields used (combined):**
  - `system_prompt`
  - `user_prompt` or `prompt`
  - `response` or `expected_response`
- **Target variable:**
  - `score` ∈ {0, 1, …, 10} (integer quality rating)

---

### Requirements

- **Python:** 3.8+
- **Core libraries:**
  ```text
  numpy
  pandas
  torch
  scikit-learn
  sentence-transformers
  transformers
  tqdm
  ```
- **Optional (for notebooks / Kaggle environment):**
  ```text
  matplotlib   # for any visualizations
  seaborn      # for nicer plots (if used)
  ```

---

### Installation

You can install the required packages with:

```bash
pip install numpy pandas torch scikit-learn sentence-transformers transformers tqdm
```

If you also want plotting utilities:

```bash
pip install matplotlib seaborn
```

---

### Implementation Details

#### Part A: Data Loading and Problem Setup

- **Data ingestion:**
  - Load `train_data.json`, `metric_names.json`, and `metric_name_embeddings.npy` from `CONFIG['data_dir']`.
  - Build a `metric_names_map` to index into `metric_name_embeddings` by `metric_name`.
- **Text preprocessing:**
  - For each record, construct a single input text via:
    - `system_prompt`  
    - `user_prompt` / `prompt`  
    - `response` / `expected_response`  
  - Concatenate these fields into one sequence using a helper `combine_text_fields(...)`.
- **Score distribution analysis:**
  - Compute counts and percentages for each score in \([0, 10]\).
  - Observe class imbalance across different score bins (e.g., more high scores, fewer extreme low scores).

#### Part B: Representation Learning and Model Architecture

##### 1. Text & Metric Embeddings

- **Sentence encoder:**
  - Use `SentenceTransformer` with **`paraphrase-multilingual-mpnet-base-v2`**.
  - Encode combined text fields into dense vectors.
- **Metric embeddings:**
  - Use precomputed `metric_name_embeddings.npy`.
  - Map each `metric_name` to its corresponding embedding via `metric_names_map`.
- **Combined representation:**
  - For each sample, obtain:
    - `metric_emb` – from metric embeddings
    - `text_emb` – from SentenceTransformer
  - Feed both into a **bilinear + MLP** architecture.

##### 2. Model Variants

- **Standard classification model (`StandardMetricMatchingModel`):**
  - Bilinear interaction layer between `metric_emb` and `text_emb`.
  - Stacked fully-connected layers with `LayerNorm`, ReLU, and dropout.
  - Output logits over 11 classes (scores 0–10).
  - Prediction via expected value of softmax probabilities.

- **Heteroscedastic regression model (`HeteroscedasticMatchingModel`):**
  - Same bilinear + MLP backbone.
  - Two heads:
    - **Mean head** predicts score \(\mu\).
    - **Log-variance head** predicts log-variance \(\log \sigma^2\).
  - Uses a **heteroscedastic loss** that penalizes both errors and overconfident uncertainty:
    \[
    \mathcal{L} \propto \frac{(y - \mu)^2}{\sigma^2} + \log \sigma^2
    \]
  - Predictions are clamped to \([0, 10]\).

- **Ordinal model (`OrdinalMetricMatchingModel`):**
  - Outputs cumulative logits for \(P(\text{score} \leq k)\) across thresholds.
  - Converts cumulative probabilities to class probabilities and expected score.
  - Paired with a custom **ordinal regression loss**.

##### 3. Loss Functions and Weighting

- Implemented advanced loss functions:
  - **OrdinalRegressionLoss** – for ordered classes via cumulative probabilities.
  - **HybridLoss** – combination of CrossEntropy and MSE on expected score.
  - **FocalLoss / WeightedFocalLoss** – for handling class imbalance.
  - **HeteroscedasticLoss** – models both mean and uncertainty for scores.
- **Sample weighting:**
  - Compute **sqrt inverse-frequency** weights per score:
    - Rarer scores receive higher weights.
    - High scores (9–10) can be explicitly **boosted** to combat under-prediction.
  - A unified helper `compute_weighted_loss(...)` applies per-sample weights consistently across loss types.

##### 4. Data Augmentation: Synthetic Negatives

- **Motivation:** Some low-score regions are under-represented.
- **Strategy (Option 2 – Synthetic Negatives):**
  - Create synthetic training samples by **misaligning** metric–text pairs:
    - Sample a record.
    - Replace its `metric_name` with a **different random metric**.
    - Assign a **low score** in a configurable range (e.g., 1–6).
  - Extend the original training set with these synthetic “low-fitness” pairs.
  - Recompute score distributions before and after augmentation to verify increased coverage of low scores.

##### 5. Encoder Fine-Tuning (SimCSE-Style)

- Optionally fine-tune the SentenceTransformer encoder using a **SimCSE-style** contrastive learning setup:
  - Create positive pairs by repeating each sentence (`[t, t]`) with dropout-based augmentation.
  - Use `MultipleNegativesRankingLoss` (or `ContrastiveLoss` as fallback).
  - Train for a small number of epochs on the training corpus.
- This step aims to produce **task-specific embeddings** adapted to the data challenge.

---

### Part C: Training Loop and Evaluation

- **Dataset wrapper (`WeightedDataset`):**
  - Returns metric embeddings, text embeddings, target scores, and precomputed weights.
- **Cross-validation:**
  - Use **StratifiedKFold** to create folds preserving score distribution.
  - Save folds for reproducibility (`cv_folds.pkl`).
- **Single-model training mode:**
  - Train a heteroscedastic (or standard / ordinal) model using:
    - AdamW optimizer
    - Learning rate scheduling via `ReduceLROnPlateau`
    - Gradient clipping and optional AMP (`use_amp`) for stability and speed.
  - Monitor **validation RMSE** and **MAE**, perform **early stopping** with a patience parameter.
  - Save:
    - Best model checkpoint (`best_model_synth.pth`)
    - Final checkpoint and metrics (`final_model.pth`, `training_metrics.csv`)

- **Ensemble pipeline (optional):**
  - Train multiple model variants across folds and collect **out-of-fold (OOF) predictions**.
  - Compute **engineered features** (cosine similarity, distances, statistics).
  - Train a **Ridge** meta-learner on OOF predictions + engineered features.
  - Optionally calibrate outputs using **isotonic regression**.
  - Save models, meta-learner, scaler, calibration objects, and diagnostics.

- **Evaluation focus:**
  - Overall RMSE and MAE on validation set.
  - Per-score-bin RMSE to understand performance across rare vs common scores.
  - Prediction distribution vs true distribution to check calibration and bias.
  - Detailed diagnostics are printed in the notebook; exact best RMSE values are reported there.

---

### Part D: Inference and Submission

- **Inference notebook:** `DA5401_Data_Challenge_Inference_Saggurthi_Dinesh_BE21B032ipynb.ipynb`
- **Core steps:**
  1. **Configuration:**
     - Set `CONFIG['data_dir']` to the data challenge directory containing:
       - `metric_names.json`
       - `metric_name_embeddings.npy`
       - `test_data.json`
     - Set `CONFIG['model_path']` to the trained model checkpoint (e.g., `best_model_synth.pth`).
  2. **Load data & embeddings:**
     - Load metric names and embeddings.
     - Load `test_data.json` and build `metric_names_map`.
  3. **Encode test texts:**
     - Load the same SentenceTransformer encoder as used in training.
     - Encode combined text fields using `combine_text_fields(...)`.
  4. **Create model & load weights:**
     - Instantiate `HeteroscedasticMatchingModel` (or `StandardMetricMatchingModel`).
     - Load weights from `CONFIG['model_path']`.
  5. **Generate predictions:**
     - Run batched inference with `DataLoader`.
     - Clamp predictions to \([0, 10]\) and **round to the nearest integer**.
  6. **Create submission file:**
     - Construct a `submission.csv` with:
       - `ID` – 1-based indices (1, 2, …, N)
       - `score` – predicted scores (floats like 7.0, 8.0, …)
     - Save to `CONFIG['output_dir']`.

---

### Key Results and Insights

- **Improved handling of rare scores:**
  - Synthetic negatives and inverse-frequency weighting help the model better learn low-score regions.
- **Uncertainty-aware modeling:**
  - Heteroscedastic regression provides more stable training and better handling of noisy labels by modeling per-sample variance.
- **Calibration and distribution:**
  - Per-score-bin RMSE and prediction distributions indicate a more balanced and calibrated model compared to a naive baseline without augmentation or weighting.
- **Reproducibility:**
  - All folds, configurations, and metrics are saved to disk; exact numerical results are documented within the training notebook and PDF report.

---

### Running the Notebooks

#### 1. Training Notebook

1. **Prepare data:**
   - Place the challenge dataset files in a directory and update:
     - `CONFIG['data_dir']` in `DA5401_Data_Challenge_Training_Saggurthi_Dinesh_BE21B032.ipynb`.
2. **Environment:**
   - Install requirements (see **Installation**).
   - GPU is recommended for training (uses PyTorch + SentenceTransformers).
3. **Execute:**
   - Run all cells **top-to-bottom**.
   - Training time depends on hardware and settings (epochs, ensemble usage, etc.).
4. **Outputs:**
   - Best model checkpoint: `best_model_synth.pth`
   - Checkpoints directory: `checkpoints/`
   - Training metrics: `training_metrics.csv`
   - (Optional) ensemble artifacts: meta-learner, calibrator, engineered features.

#### 2. Inference Notebook

1. **Configure paths:**
   - Set `CONFIG['data_dir']` to the directory containing `metric_names.json`, `metric_name_embeddings.npy`, and `test_data.json`.
   - Set `CONFIG['model_path']` to the trained model file.
2. **Run cells:**
   - Execute all cells sequentially.
   - The notebook will print progress for encoding and prediction.
3. **Submission:**
   - Final output: `submission.csv` in `CONFIG['output_dir']`.
   - Upload this file to Kaggle for evaluation.
