# Data Collection Stage

This folder contains a data-collection pipeline for the **data collection phase**. The pipeline streams three public text datasets from Hugging Face, collects them in a **4:3:3 ratio**, stops when the total raw-text budget reaches **about 1.3 GB**, and merges them into one JSONL corpus.

- **Wikipedia**: 520 MB (40%)
- **News**: 390 MB (30%)
- **Web text**: 390 MB (30%)
- **Merged total**: about 1300 MB

All sizes are measured as **UTF-8 bytes written to JSONL**, which makes the size accounting easy to explain in the report.

## Chosen sources

- **Wikipedia** → encyclopedic domain
- **CNN/DailyMail** → news domain
- **OpenWebText** → general web-text domain

The Hugging Face dataset cards show that the English Wikipedia dump can be loaded with `load_dataset("wikipedia", "20220301.en")`, CNN/DailyMail is available as `abisee/cnn_dailymail` with version `3.0.0`, and OpenWebText is available as `Skylion007/openwebtext`. Dataset streaming in Hugging Face is enabled by passing `streaming=True` to `load_dataset`, which avoids downloading the full dataset up front. 

## What each file does

### `scripts/config.py`
Stores all target sizes, output paths, and collection-stage filters.

### `scripts/collect_and_merge.py`
Main script.

It does four things:
1. Streams each dataset from Hugging Face.
2. Applies light collection-stage filtering.
3. Writes one JSONL file per source until the byte budget is reached.
4. Merges the three source files into one combined JSONL corpus.

### `scripts/inspect_collection.py`
Prints file sizes and a few record previews so you can quickly verify the outputs.

## Prerequisites

- Install compatible dependencies from `requirements.txt` before running the scripts.
- Hugging Face dataset streaming (`streaming=True`) requires active network access while collecting data.

## How to run

Working directory: `data_collection`

```bash
cd data_collection
python scripts/collect_and_merge.py
python scripts/inspect_collection.py
```

Alternative from repository root (also supported):

```bash
python data_collection/scripts/collect_and_merge.py
python data_collection/scripts/inspect_collection.py
```

## Collection-stage filtering

This stage intentionally keeps filtering light because deeper cleaning, deduplication, normalization, tokenization, and chunking belong to later preprocessing steps.

Current filters:
- remove empty documents
- remove documents shorter than **50 words**
- remove extremely long documents above **200,000 characters**
- normalize repeated whitespace


## Expected outputs

After a successful run, you should have:

- `output/wikipedia_520mb.jsonl`
- `output/news_390mb.jsonl`
- `output/web_390mb.jsonl`
- `output/corpus_merged_1300mb.jsonl`
- `logs/collection_summary.json`

The summary file records:
- target size per source
- actual written size
- number of kept documents
- number of skipped documents by reason
- merged corpus size and document count
