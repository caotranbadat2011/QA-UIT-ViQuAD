# 🇻🇳 QA-UIT-ViQuAD: Extractive Question Answering for Vietnamese

This repository provides an end-to-end pipeline for fine-tuning **Extractive Question Answering** models on the **UIT-ViQuAD** Vietnamese dataset using Transformer architectures (`xlm-roberta-base`) and the Hugging Face `Trainer` API. The project supports unanswerable question detection and features an interactive Web Demo built with **Streamlit**.

---

## 📋 Table of Contents
1. [Overview & Methodology Report](#-overview--methodology-report)
2. [Directory Structure](#-directory-structure)
3. [Environment Setup](#-environment-setup)
4. [Configuration Parameters](#-configuration-parameters)
5. [Usage & Pipeline Execution](#-usage--pipeline-execution)
   - [Step 1: Data Preprocessing & Cleaning](#step-1-data-preprocessing--cleaning)
   - [Step 2: Model Training](#step-2-model-training)
   - [Step 3: Model Evaluation](#step-3-model-evaluation)
   - [Step 4: Launch Web App Demo](#step-4-launch-web-app-demo)
6. [Evaluation & Benchmark Results](#-evaluation--benchmark-results)
7. [License](#-license)

---

## Overview & Methodology Report

### 1. Data Challenges & Technical Solutions:
* **Offset Relocation (`ViQuADCleaner`):** Raw extractive QA datasets frequently contain misaligned `answer_start` character positions due to encoding shifts or text formatting issues. The `ViQuADCleaner` class automatically searches within a sliding window (`search_window`, default 20 characters) to relocate and correct the exact answer position in the context paragraph.
* **Data Leakage Prevention:** Standard random splitting at the question level can cause the same context paragraph to appear across Train, Validation, and Test sets. We apply `GroupShuffleSplit` grouped by `context_id`, guaranteeing that all questions associated with a given context belong exclusively to a single split (80% Train / 10% Val / 10% Test).
* **Long Context Handling (Overlapping Windows):** Context paragraphs exceeding token limits are handled via a sliding window mechanism (`max_length=384`, `doc_stride=128`). This breaks long contexts into overlapping features without truncating critical context.
* **Unanswerable Questions Detection:** Questions without answers are mapped to the first special token (`[CLS]` or `<s>`). During inference, if the `null_score` exceeds a configured threshold, the model predicts that the question cannot be answered from the provided text.

---

## Directory Structure

```text
QA-UIT-ViQuAD/
├── app/
│   └── app.py                      # Interactive Streamlit Web Application Demo
├── configs/
│   └── config.yaml                 # Centralized configuration YAML file
├── data/
│   ├── raw/
│   │   └── train_formatted.json    # Raw ViQuAD dataset in SQuAD JSON format
│   └── processed/                  # Cleaned Parquet datasets (train, val, test)
├── models/
│   └── best_checkpoint/            # Saved model checkpoint & tokenizer after training
├── notebooks/
│   └── 01_eda_and_flatten.ipynb    # Exploratory Data Analysis (EDA) notebook
├── src/
│   ├── data_preprocessing.py       # Preprocessing, offset cleaning & dataset splitting
│   ├── train.py                    # Training pipeline using Hugging Face Trainer
│   └── evaluate.py                 # Evaluation script (Exact Match & F1 calculation)
├── predictions.json                # Detailed test set predictions output file
├── requirement.txt                 # Dependencies list
├── requirements.txt                # Standard dependencies list
├── LICENSE                         # MIT License
└── README.md                       # Comprehensive project documentation
```

---

## Environment Setup

### Prerequisites:
* Python 3.8 or higher
* PyTorch (GPU with CUDA support recommended)

### Install Dependencies:
```bash
pip install -r requirements.txt
```

---

## Configuration Parameters

All system configurations are managed centrally in [`configs/config.yaml`](configs/config.yaml):

```yaml
# 1. Data Configuration
data:
  raw_path: "data/raw/train_formatted.json"
  processed_dir: "data/processed"
  train_file: "data/processed/train.parquet"
  val_file: "data/processed/val.parquet"
  test_file: "data/processed/test.parquet"
  split_ratios:
    train: 0.8
    val: 0.1
    test: 0.1
  seed: 42

# 2. Model & Preprocessing Configuration
model:
  name: "xlm-roberta-base"
  output_dir: "models/best_checkpoint"
  max_length: 384
  doc_stride: 128
  search_window: 20

# 3. Training Hyperparameters
training:
  num_train_epochs: 3
  batch_size: 16
  learning_rate: 3.0e-5
  weight_decay: 0.01
  eval_strategy: "epoch"
  save_strategy: "epoch"
  logging_steps: 50
  fp16: true
  load_best_model_at_end: true
  metric_for_best_model: "loss"

# 4. Evaluation Settings
evaluation:
  n_best: 20
  max_answer_length: 64
  predictions_file: "predictions.json"

# 5. Web App Configuration
app:
  no_answer_threshold: 0.0
  page_title: "Vietnamese QA Demo"
  page_icon: "🔎"
```

---

## Usage & Pipeline Execution

### Step 1: Data Preprocessing & Cleaning
Load raw JSON data, automatically relocate offset positions, split into Train/Val/Test datasets by context, and export to `data/processed/`:
```bash
python src/data_preprocessing.py
```

### Step 2: Model Training
Fine-tune the Transformer model on the cleaned dataset:
```bash
python src/train.py
```
*Alternatively, override parameters directly via CLI:*
```bash
python src/train.py --model_name xlm-roberta-base --num_train_epochs 3 --batch_size 16 --learning_rate 3e-5
```

### Step 3: Model Evaluation
Evaluate model accuracy on the test split (`data/processed/test.parquet`) using standard SQuAD metrics:
```bash
python src/evaluate.py
```
Detailed prediction outputs will be saved to `predictions.json`.

### Step 4: Launch Web App Demo
Start the interactive Streamlit web interface:
```bash
streamlit run app/app.py
```
The web app will automatically open in your browser at `http://localhost:8501`.

---

## Evaluation & Benchmark Results

Evaluated on 2,882 test samples from the UIT-ViQuAD dataset using standard Extractive QA metrics (SQuAD Benchmark):

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Overall EM** | **45.80%** | Exact match rate across all test samples |
| **Overall F1** | **58.76%** | Word-level F1 score across all test samples |
| **HasAns EM** | **48.50%** | Exact match rate on answerable questions |
| **HasAns F1** | **67.77%** | Word-level F1 score on answerable questions |
| **NoAns Accuracy** | **40.25%** | Accuracy in detecting unanswerable questions |

---

## License
Distributed under the [MIT License](LICENSE). Developed for research and educational purposes using the UIT-ViQuAD dataset.
