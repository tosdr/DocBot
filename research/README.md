# Setup

In a pip or conda environment, run:
```
pip install requirements.txt
python -m spacy download en_core_web_md
```

### Converting sql dumps to pandas
This is only necessary if working with a fresh export. Otherwise, get `pkl` files from someone.

- Start postgresql (on a mac: `brew services restart postgresql`)
- Run `createdb tosdr`
- In the interactive `psql` REPL, run `create user phoenix with superuser`
- Comment out references to `public.users`, found near the bottom of `documents.sql`, `points.sql`, `public.users`
- Log into psql with `psql -d tosdr -U phoenix`
- `\i services.sql`
- `\i topics.sql`
- `\i documents.sql`
- `\i cases.sql`
- `\i points.sql`
- Confirm all 5 tables exist with `\dt public.*`. Tables can be inspected with i.e. `\d+ public.topics`
- Run `sql_to_pandas.py` to load the tables and save them as pickled pandas DataFrames in `data`

# Notebooks

## `explore.ipynb`

Loads a pandas version of the datasets, does exploratory data analysis and data cleaning, 
saves new versions as `data/{DATASET}_{VERSION}_clean.pkl`

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

## `summarize.ipynb`

The highlights of EDA from `explore.ipynb`, like graphs and dataset size

## `examine_cases.ipynb`

Work in progress
