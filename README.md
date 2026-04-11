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

## Working Principle
1. **Data**: Origin Corpus (String)
2. **Action**: Preprocess
3. **Data**: Preprocessed Data (String)
4. **Action**: Tokenization (Tool: Tokenizer)
  * tokenizer will generate a dictionary
  * we need to get token_ids from the dic
5. **Data**: token_ids (Integer List)
6. **Action**: chunk
7. **Data**: chuncked token_ids (List of Integer List)
8. **Action**: Convert data type to Tensor (Tool: PyTorch)
9. **Data**: .pt file(tensor) [batch * sequence]
10. **Action**: Use DataLoader to train model by batch

### Model
1. **Input Data**: batch (tensor) [ sequence ]
2. **Action**: token embedding + position embedding + embedding dropout(avoid overfitting)
3. **Data**: embeddings (original embedding) [sequence * dimension] (dimension number in middle layer)
4. **Action**: Transform (Tool: Transformer)
  * Target: Recalculate embeddings to embeddings with context info.
  * Transformer contains multiple layers of TransformerBlock
  * Transformer Block:
    * Attention Layer: LayerNorm + SelfAttention + ResidualConnection
      * Target: Get relationship between tokens.
      * SelFAttention: 
        * Linear projection to obtain Q/K/V
        * Split into multiple heads
        * Compute ( QK^T ) (relationship score)
        * Apply causal mask (next-token prodiction)
        * Apply softmax
        * Weighted sum of V
        * Concatenate multiple heads
        * Output projection
    * MLP Layer: LayerNorm + MLP + ResidualConnection
      * Target: Every token process information get by attention layer to increase understanding.
      * MLP: expand dimensions + nonlinear transformation + project back
5. **Data**: Hidden States (embedding with context info) [sequence * dimension] (dimension number in middle layer)
6. **Action**: Final LayerNorm + Linear Head
7. **Data**: logits (tensor) [sequence * dimension] (dimension number is vocab_size)
  * Every token getss scores of every token in the vocabulary to predict next token.
8. **Action**: Use target and logits to calculate cross entropy and refine the model.
  * Use `optimizer` to refine model.

---

## Tech Stack

* Python 3.x
* PyTorch (Dataset)
* Hugging Face Transformers
* Amazon SageMaker
* NumPy / Pandas
* Matplotlib

---

## Tools
- Code Repo: Github
- AI: Codex + Copilot
- Environment: Miniconda
- IDE
  - Jupyter Notebook(.ipynb): experimentation & visualization
  - VS Code(.py): production-ready pipeline

---

## Project Structure

```text
llm-foundation-model/
├── README.md
├── configs/
│   └── train_config.yaml
├── data_collection/
│   ├── README.md
│   ├── logs/
│   │   └── collection_summary.json
│   ├── output/
│   │   ├── corpus_merged_1300mb.jsonl
│   │   ├── news_390mb.jsonl
│   │   ├── web_390mb.jsonl
│   │   └── wikipedia_520mb.jsonl
│   └── scripts/
│       ├── collect_and_merge.py
│       ├── config.py
│       └── inspect_collection.py
├── experiment/
│   ├── raw_data_exploration.py
│   ├── sample_dataset_exploration.ipynb
│   ├── tokenizer_debug.py
│   └── training_results.py
├── outputs/
│   ├── logs/
│   │   └── preprocess_summary.json
│   ├── sample_dataset.pt
│   ├── sample_dataset_small.pt
│   └── tokenizer_hf/
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── requirements.txt
└── src/
    ├── dataset.py
    ├── model.py
    ├── preprocess.py
    ├── train.py
    └── utils.py
```

---

## Workflow

### 1. Data Exploration

Implemented in `experiment/raw_data_exploration.py`:

* Inspect raw dataset
* Analyze text quality and distribution
* Validate cleaning strategy

### 2. Explore Tokenizer

Implemented in `experiment/tokenizer_debug.py`:

### 3. Preprocessing Pipeline

Implemented in `src/preprocess.py`:

* Data collection (≥1GB dataset)
* Text cleaning and normalization
* Deduplication and noise removal
* Tokenization using transformer-compatible tokenizer
* Chunking long sequences into fixed-length blocks
* Saving processed dataset

---

### 4. Explore Sample Dataset

Implemented in `experiment/sample_dataset_exploration`:

* Inspect preprocessed sample dataset
* Extract first five sample data into `sample_dataset_small.pt`

---

### 5. Custom Dataset & DataLoader

Implemented in `src/dataset.py`:

* Use PyTorch Dataset / Dataloader
* Define what does a Dataset look like
* Define DataLoader about how to feed model with datasets
  * DataLoader with batching and shuffling
  * Efficient memory usage

---

### 6. Model Implementation

Implemented in `src/model.py`:

* Build Mini-GPT (Decoder-style language model)
* Embedding + positional encoding
* Casual attention mask (next-token prediction)
* Use **PyTorch** TransformerEncoder
* Linear Head
* Calculate cross entropy

---

### 7. Training Pipeline

Implemented in `src/train.py`:

* Train iteration in nested epoc and dataloader loop
* Optimizer
* Save checkpoint + metrics

Tracks:

* Loss
* Perplexity

---

### 8. Visualization

Implemented in `experiment/training_results_visualization`:

* Loss curves
* Perplexity curves
* Hyperparameter comparison

---

### 7. Hyperparameter tuning

In `train_config.yaml`

* Tweak one hyperparameter at a time and record the metric changes.

In `03_training_results.ipynb`:

* Loss curves
* Perplexity curves
* Hyperparameter comparison

---

## Running the Project

### Before start

Follow the instruction in `data_collection/README.md` to collect data.

---

### Step 1: Run Preprocessing

```bash
python src/preprocess.py
```

Output:

* `sample_dataset.pt`

---

### Step 2A: Train Locally

```bash
python -m src.train
```

Training writes:

* `outputs/mini_gpt.pt`
* `outputs/logs/train_metrics.json`

---

### Step 2B: Train on SageMaker

Keep using the same project code, but package your local `src/` as `source_dir` and run `train.py` as the SageMaker entrypoint.

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

> If you launch SageMaker training from this repo, make sure your training data path and IAM role match your AWS account setup.

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

## Using a Trained Model to Predict the Next Token

During inference, there is no target label to compare against, so the model must generate the next token step by step based on the current context.

### Basic process
1. Feed the current token sequence into the model.
2. The model outputs logits for every position in the sequence.
3. Take the logits from the last position, which represent the prediction for the next token.
4. Choose the next token from that distribution.
  * You can use argmax to select the highest-scoring token (greedy decoding).
  * Or you can sample from the probability distribution.
5. Append the predicted token to the original input sequence.
6. Repeat the process to generate more tokens until you reach the maximum length or an end token.

---

## Future Improvements

* Distributed training
* Larger datasets (multi-domain scaling)
* Better tokenizer (custom BPE)
* Evaluation on downstream tasks
