![Terms of Service; Didn't Read](./tosdr.png)

# 🤖 Docbot 🤖

This repo contains research and production tooling for using Machine Learning to automate document annotation
on [ToS;DR](https://tosdr.org/)

The process before Docbot:

- We crawl documents
- We wait for volunteers to submit Points, which are privacy policy quotations that highlight evidence that a Case
  (privacy-related statement) is true.
- We wait for curators, trusted volunteers, to approve or reject Points
- We score the service A-F based on the approved Points

The process after is the same, but now Docbot does the initial document analysis and submits Points, along with a
confidence score, to curators.
We achieve this by fine-tuning large language models into binary classifiers.

Each week an automated job runs 123 case models in total, analyzing any new documents that were added to ToS;DR's
database since the last run.

# Contributing / Using the models

If you would like to have some privacy policies or T&Cs analyzed with docbot, the best way is to add them to the ToS;DR
platform on [edit.tosdr.org](https://edit.tosdr.org)
and wait until our automated system picks them up.

If you have a particular need to run the models on standalone documents, or have any other questions, please get in
touch with us at [team@tosdr.org](mailto:team@tosdr.org).

We welcome contributions to the engineering or research to improve our models.

We also plan to release our datasets used for training/evaluation, which could be of value to NLP researchers.

# Dev Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. With uv installed, run:

```
uv sync
uv run spacy download en_core_web_md
```

## Incorporating new training data

We create training corpora from database dumps. The first step is to convert from sql to pandas.

- Start postgresql (on a mac: `brew services restart postgresql`)
- Run `createdb phoenix`
- In the interactive REPL (`psql -d phoenix`), run
  `create user phoenix with superuser; ALTER ROLE "phoenix" WITH LOGIN;`
- `pg_restore -d phoenix -cC path/to/dump.sql`
- Log into psql with `psql -d phoenix -U phoenix`, confirm all tables exist with `\dt`. Tables can be inspected with
  i.e. `\d+ topics`
- Run `sql_to_pandas.py` to load the tables and save them as pickled pandas DataFrames in `data/db_dumps/`

### Creating datasets

Run `src/clean_data.py` on the output of `sql_to_pandas.py` to clean the data (or step through the same cleaning
interactively in `notebooks/explore.py`), and then see `make_classification_datasets.py` to turn that into a
classification dataset apt for training or eval.

## Training models

First, `src/tfidf.py` trains high-recall TF-IDF prefilter models (one per case)
that we use to skip obviously irrelevant sentences, cutting deep-model inference by ~80-95%. It takes no
arguments; models are written to `data/models/{MODEL_VERSION}` and baked into both the training and inference Docker
images.

As the second stage, `src/sent_spans/train.py` fine-tunes the case models (BERT + LoRA); run `train.py --help` to see
options. CUDA will be
used if available. Producing a new model version (e.g. `v4`) is a three-step process:

1. **Cross-validation**: `train.py --all --model_version v4 --upload` trains every case across all pre-assigned CV
   folds and uploads adapters and test-fold predictions to S3 under `v4/cv/`.
2. **Threshold selection**: `thresholds.py --model_version v4` computes per-case threshold menus from the CV document
   predictions and writes `thresholds.json` to `data/models/v4/{case_id}/` (`--upload` also pushes them to S3).
3. **Final models**: `train.py --all --model_version v4 --train_final --upload` trains production models on all data
   (minus a small early-stopping slice) and uploads each adapter alongside its case's `thresholds.json`.

Steps 1 and 3 each have two modes: serial on a single host, or with `--parallel` to spread cases across several
containers using AWS SQS (run `train_push.py` first with the same `--model_version` to enqueue the case IDs).

### Training in Docker

`Dockerfile.train` packages the environment for GPU cloud instances. Build for x86_64 — an arm64 build
silently gets CPU-only torch:

```
docker build --platform linux/amd64 -f Dockerfile.train -t tosdr-train:latest .
```

# Notebooks

### `explore.py`

A [marimo](https://marimo.io) notebook for exploratory data analysis and working out our data cleaning (copied over to
`clean_data.py`)

Data that was removed:

- Services, Documents and Points marked as `deleted`, and rows cascading from deleted Services/Documents
- Services that lack any Documents, and associated Points
- Points whose status isn't one of `[approved, declined, pending]` (e.g. `changes-requested`, `*-not-found`, `draft`)
- Documents without text (~7k, roughly a third are uncrawled) and associated Points
- A handful of Points that have a `quote_start` but no `quoteText`
- Points with `quoteText` that no longer matches `document.text[point.quote_start:point.quote_end]`, likely due to
  re-crawled text that changed. About 2k,
  down from 3k (1k were saved by re-searching for the quote and updating `quote_start`/`quote_end`)
- [Service 502](https://edit.tosdr.org/services/502), [Case 235](https://edit.tosdr.org/cases/235), [Topic 53](https://edit.tosdr.org/topics/53),
  Document 1378, and all associated Points

Data that was kept for now in case they're useful:

- Points that don't refer to docs
- Docs without any points (there are many)
- Non-english docs and points (can be filtered by doing `documents = documents[documents.lang == 'en']`)

### `summarize.ipynb`

The highlights of EDA from `explore.py`, like graphs and dataset size

### `examine_cases.ipynb`

An early notebook used to look at points for brainstorming, and help decide whether sentence classification vs sentence
spans is the right paradigm.
Finds that a lot of points span multiple sentences, so that's ideal, but over 5 sentences is rare.

### `prediction_lengths.ipynb`

Tests whether positive predictions using our inference strategy of sentence expansion (
`inference.apply_sent_span_model()`) yields spans about the
same length as human submitted points.

### `pr_curves.ipynb`

Plots ROC curves, precision recall curves
