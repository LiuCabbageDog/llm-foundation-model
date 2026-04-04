# llm-foundation-model
Training a small-scale transformer from scratch by implementing a mini-GPT and performing next-token prediction.

---

## Overview

This project implements an end-to-end pipeline for **foundation model pretraining**, including:

* Large-scale text data collection and preprocessing
* Tokenization and custom dataset construction
* Training a small-scale transformer (mini-GPT) from scratch
* Visualization of training metrics (loss and perplexity)

The project follows a **hybrid workflow**:

* Local development (Jupyter + VSCode)
* Cloud training using Amazon SageMaker

---

## Project Structure

```text
foundation-model-assignment/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_tokenizer_debug.ipynb
│   └── 03_training_results.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── configs/
│   └── train_config.yaml
│
├── outputs/
│   ├── sample_dataset.pt
│   ├── logs/
│   └── figures/
│
├── requirements.txt
├── README.md
```

---

## Workflow

### 1. Data Exploration (Notebook)

* Inspect raw dataset
* Analyze text quality and distribution
* Validate cleaning strategy

### 2. Preprocessing Pipeline (Assignment 1)

Implemented in `src/preprocess.py`:

* Data collection (≥1GB dataset)
* Text cleaning and normalization
* Deduplication and noise removal
* Tokenization using transformer-compatible tokenizer
* Chunking long sequences into fixed-length blocks
* Saving processed dataset

This step satisfies the preprocessing requirements including:

* cleaning, normalization, tokenization, batching 

---

### 3. Custom Dataset & DataLoader

Implemented in `src/dataset.py`:

* PyTorch Dataset
* DataLoader with batching and shuffling
* Padding / truncation handling
* Efficient memory usage

---

### 4. Model Implementation (Assignment 2)

Implemented in `src/model.py`:

* Mini-GPT (transformer-based model)
* Embedding + positional encoding
* Multi-head self-attention
* Layer normalization

Model design follows Assignment 2 requirements 

---

### 5. Training Pipeline

Implemented in `src/train.py`:

* Forward pass
* Cross-entropy loss
* Backpropagation
* Optimizer step
* Checkpoint saving
* Logging training metrics

Tracks:

* Loss
* Perplexity

---

### 6. Visualization (Notebook)

In `03_training_results.ipynb`:

* Loss curves
* Perplexity curves
* Hyperparameter comparison

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 1: Run Preprocessing

```bash
python src/preprocess.py
```

Output:

* `sample_dataset.pt`

---

### Step 2: Train Model (Local)

```bash
python src/train.py
```

---

### Step 3: Train on SageMaker (Optional but Recommended)

You can submit training jobs from local environment:

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    role="<your-sagemaker-role>"
)

estimator.fit({
    "training": "s3://your-bucket/data/"
})
```

---

## Configuration

Edit `configs/train_config.yaml`:

```yaml
batch_size: 32
block_size: 64
learning_rate: 0.001
epochs: 5
n_layers: 2
n_heads: 4
embed_dim: 128
```

---

## Key Design Decisions

### Data Processing

* Removed short/low-quality text (<50 words)
* Applied normalization and deduplication
* Chunked sequences to fixed block size

### Tokenization

* Transformer-compatible tokenizer (BPE / WordPiece / GPT-style)
* Efficient storage format (tensor-based)

### Model

* Small-scale transformer (mini-GPT)
* Designed for educational clarity and efficiency

---

## Results

* Training loss decreases over epochs
* Perplexity improves with tuning
* Model learns basic language patterns

---

## Assignment Deliverables

### Assignment 1

* Preprocessing pipeline
* Sample tokenized dataset (`.pt`)
* Report (dataset, cleaning, tokenization, challenges) 

---

### Assignment 2

* Model implementation
* Training loop
* Checkpoints
* Loss & perplexity visualization
* Report (architecture, experiments, observations) 

---

## Tech Stack

* Python 3.x
* PyTorch
* Hugging Face Transformers
* NumPy / Pandas
* Matplotlib
* Amazon SageMaker

---

## Tools
- Code Repo: Github
- AI: Codex + Copilot
- Environment: Miniconda
- IDE
  - Jupyter Notebook(.ipynb): experimentation & visualization
  - VS Code(.py): production-ready pipeline

---

## Future Improvements

* Distributed training
* Larger datasets (multi-domain scaling)
* Better tokenizer (custom BPE)
* Evaluation on downstream tasks
