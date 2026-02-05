![Terms of Service; Didn't Read](./tosdr.png)

# 🤖 Docbot 🤖

This repo contains research and production tooling for using Machine Learning to automate document annotation on [ToS;DR](https://tosdr.org/)

The process before Docbot:
- We crawl documents
- We wait for volunteers to submit Points, which are privacy policy quotations that highlight evidence that a Case (privacy-related statement) is true.
- We wait for curators, trusted volunteers, to approve or reject Points
- We score the service A-F based on the approved Points

The process after is the same, but now Docbot does the initial document analysis and submits Points, along with a confidence score, to curators.
We achieve this by fine-tuning large language models into binary classifiers.

Each week an automated job runs 123 case models in total, analyzing any new documents that were added to ToS;DR's database since the last run.

# Contributing / Using the models

If you would like to have some privacy policies or T&Cs analyzed with docbot, the best way is to add them to the ToS;DR platform on [edit.tosdr.org](https://edit.tosdr.org)
and wait until our automated system picks them up.

If you have a particular need to run the models on standalone documents, or have any other questions, please get in touch with us at [team@tosdr.org](mailto:team@tosdr.org).

We welcome contributions to the engineering or research to improve our models.

We also plan to release our datasets used for training/evaluation, which could be of value to NLP researchers.

# Dev Setup

In a pip or conda environment with python 3.12, run:
```
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

## Incorporating new training data

We create training corpora from database dumps. The first step is to convert from sql to pandas.

- Start postgresql (on a mac: `brew services restart postgresql`)
- Run `createdb phoenix`
- In the interactive REPL (`psql -d phoenix`), run `create user phoenix with superuser; ALTER ROLE "phoenix" WITH LOGIN;`
- `pg_restore -d phoenix -cC path/to/dump.sql`
- Log into psql with `psql -d phoenix -U phoenix`, confirm all tables exist with `\dt`. Tables can be inspected with i.e. `\d+ topics`
- Run `sql_to_pandas.py` to load the tables and save them as pickled pandas DataFrames in `data/db_dumps/`

### Creating datasets

Run `explore.ipynb` on the output of `sql_to_pandas.py` to clean the data, and then see `make_classification_datasets.py` to turn that into a classification 
dataset apt for training or eval.

## Training models

Run `train.py --help` to see options. `Dockerfile.train` is available for convenience, `train.py` args can be added to the end of `docker run`.
CUDA will be used if available.

There are two modes, one to train all case models serially on a single host, and with `--parallel_key` to parallelize across several containers using AWS SQS.

# Notebooks

### `explore.ipynb`

Exploratory data analysis and data cleaning, saves new versions as `data/{DATASET_VERSION}_clean.pkl`

Data that was removed:
- Services and Documents marked as `deleted`, and associated Points
- Services that lack any Documents, and associated Points
- Points marked as `[disputed, changes-requested, pending, draft]` (only ~60)
- Documents without text (~1.5k) and associated Points
- A handful of Points that have a `quoteStart` but no `quoteText`
- Points with `quoteText` that no longer matches `document.text[point.quoteStart:point.quoteEnd]`, likely due to re-crawled text that changed. About 2k, 
down from 3k (1k were saved by re-searching for the quote and updating `quoteStart`/`quoteEnd`)
- [Service 502](https://edit.tosdr.org/services/502), [Case 235](https://edit.tosdr.org/cases/235), [Topic 53](https://edit.tosdr.org/topics/53), Document 1378, and all associated Points 

Data that was kept for now in case they're useful:
- Points that don't refer to docs
- Docs without any points (there are many)
- Non-english docs and points (can be filtered by doing `documents = documents[documents.lang == 'en']`)

### `summarize.ipynb`

The highlights of EDA from `explore.ipynb`, like graphs and dataset size

### `examine_cases.ipynb`

An early notebook used to look at points for brainstorming, and help decide whether sentence classification vs sentence spans is the right paradigm. 
Finds that a lot of points span multiple sentences, so that's ideal, but over 5 sentences is rare.

### `prediction_lengths.ipynb`

Tests whether positive predictions using our inference strategy of sentence expansion (`inference.apply_sent_span_model()`) yields spans about the 
same length as human submitted points.

### `pr_curves.ipynb`

Plots ROC curves, precision recall curves, and helps find optimal thresholds.
