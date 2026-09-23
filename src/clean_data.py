import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

DB_DUMP_VERSION = "2026-07-09"
DUMP_DIR = here / f"../data/db_dumps/{DB_DUMP_VERSION}"

# Hardcoded `none`/spam rows that surfaced during EDA (see notebooks/explore.py). The `(NONE)` Service (502) collects
# google-group discussions, spam, and unplaceable deleted Points; Case 235, Topic 53 and Document 1378 are similar.
NONE_SERVICE_ID = 502
NONE_CASE_ID = 235
NONE_TOPIC_ID = 53
EXAMPLE_DOCUMENT_ID = 1378

# We keep Points with these statuses. `approved`/`declined` make datasets; `pending` may be useful for pending docbot
# points. Everything else (`deleted`, `changes-requested`, `*-not-found`, `draft`, `disputed`, ...) is dropped.
KEEP_POINT_STATUSES = {"approved", "declined", "pending"}

# Output schemas, kept stable so downstream consumers (make_classification_datasets.py, tfidf.py, ...) don't break.
DOCUMENTS_CLEAN_COLS = [
    "id",
    "name",
    "url",
    "selector",
    "text",
    "created_at",
    "updated_at",
    "service_id",
    "reviewed",
    "user_id",
    "crawler_server",
    "text_version",
    "document_type_id",
    "last_crawl_date",
    "lang",
    "doc_len",
    "id_service",
    "name_service",
]
POINTS_CLEAN_COLS = [
    "id",
    "user_id",
    "title",
    "source",
    "status",
    "analysis",
    "created_at",
    "updated_at",
    "service_id",
    "quote_text",
    "case_id",
    "old_id",
    "point_change",
    "quote_start",
    "quote_end",
    "service_needs_rating_update",
    "document_id",
    "annotation_ref",
    "ml_score",
    "docbot_version",
    "lang",
]

DetectorFactory.seed = 0  # Make langdetect deterministic


def detect_lang(text):
    if text is None or pd.isnull(text) or text == "":
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def load_raw():
    frames = {}
    for name in ["cases", "documents", "points", "services", "topics"]:
        frames[name] = pickle.load(open(DUMP_DIR / f"{name}.pkl", "rb"))
        logger.info(f"Loaded {name}: {len(frames[name])} rows")
    return frames


def drop_deleted(services, documents, points):
    """Exclude every `status == "deleted"` row from Services, Documents and Points, cascading to related rows.

    All three entities carry a `status` column (the column was only added to Documents/Points more recently). We remove
    deleted rows up front so they're absent from the entire pipeline, plus the Documents/Points of deleted Services and
    the Points of deleted Documents. Services/Documents no longer need their `status` column afterwards, but Points do
    (it distinguishes approved/declined/pending), so we keep it there.
    """
    deleted_service_ids = services[services.status == "deleted"].id
    deleted_document_ids = documents[documents.status == "deleted"].id
    logger.info(
        f"Dropping deleted: {len(deleted_service_ids)} Services, {len(deleted_document_ids)} Documents, "
        f"{(points.status == 'deleted').sum()} Points (plus rows cascading from deleted Services/Documents)"
    )

    services = services[services.status != "deleted"].drop("status", axis=1)
    documents = documents[(documents.status != "deleted") & ~documents.service_id.isin(deleted_service_ids)].drop(
        "status", axis=1
    )
    points = points[
        (points.status != "deleted")
        & ~points.service_id.isin(deleted_service_ids)
        & ~points.document_id.isin(deleted_document_ids)
    ]
    return services, documents, points


def clean():
    frames = load_raw()
    cases, documents, points = frames["cases"], frames["documents"], frames["points"]
    services, topics = frames["services"], frames["topics"]

    # --- Deletions (see drop_deleted) ---
    services, documents, points = drop_deleted(services, documents, points)

    # --- Drop the `none`/spam Service, Case, Topic and the example Document, plus their Points ---
    services = services.drop(NONE_SERVICE_ID, errors="ignore")
    cases = cases.drop(NONE_CASE_ID, errors="ignore")
    topics = topics.drop(NONE_TOPIC_ID, errors="ignore")
    documents = documents.drop(EXAMPLE_DOCUMENT_ID, errors="ignore")
    points = points[(points.service_id != NONE_SERVICE_ID) & (points.case_id != NONE_CASE_ID)]

    # --- Document text features ---
    documents["text"] = documents.text.replace("", np.nan)  # Treat empty text as missing
    logger.info("Detecting document languages (this takes a minute)...")
    documents["lang"] = documents.text.map(detect_lang)
    documents["doc_len"] = documents.text.str.len()

    # --- Drop text-less Documents (and the Points that reference them) ---
    textless_doc_ids = documents[documents.text.isna() | (documents.doc_len == 0)].index
    logger.info(f"Dropping {len(textless_doc_ids)} text-less Documents")
    documents = documents.drop(textless_doc_ids)
    points = points[~points.document_id.isin(textless_doc_ids)]

    # --- Drop Services with no Documents (and their Points) -- only Points referencing Documents are useful ---
    no_doc_service_ids = set(services.id) - set(documents.service_id)
    logger.info(f"Dropping {len(no_doc_service_ids)} Services without Documents")
    services = services[~services.id.isin(no_doc_service_ids)]
    points = points[~points.service_id.isin(no_doc_service_ids)]

    # --- Point status ---
    # `deleted` is already gone; this keeps only the statuses useful for datasets.
    points = points[points.status.isin(KEEP_POINT_STATUSES)]
    points = points.drop("rank", axis=1)

    # --- Quotes ---
    points["quote_text"] = points.quote_text.replace("", np.nan)
    # A quote_start with no quote_text is anomalous; drop those Points.
    points = points.drop(points[points.quote_start.notna() & points.quote_text.isna()].index)

    # Attach the (possibly-updated) Document text so we can validate/realign quote offsets, plus the Document's detected
    # language (kept in the final schema). Both are NaN for the many Points that don't reference a Document.
    points["text"] = points.document_id.map(documents.text)
    points["lang"] = points.document_id.map(documents.lang)

    def extract_quote(point):
        if pd.isna(point.quote_start) or pd.isna(point.quote_end) or pd.isna(point.text):
            return np.nan
        return point.text[int(point.quote_start) : int(point.quote_end)]

    # Documents are often edited after a Point is made without resetting quote_start/quote_end, so the offsets no longer
    # line up with quote_text. Where the exact quote_text occurs exactly once in the current Document, realign to it.
    points["quote_text_extracted"] = points.apply(extract_quote, axis=1)
    mismatching = points[points.quote_start.notna() & (points.quote_text != points.quote_text_extracted)]
    logger.info(f"Realigning {len(mismatching)} Points whose quote offsets don't match quote_text")
    for i, point in mismatching.iterrows():
        if pd.isna(point.text) or pd.isna(point.quote_start):
            continue
        if point.text.count(point.quote_text) == 1:
            new_start = point.text.find(point.quote_text)
            points.at[i, "quote_start"] = new_start
            points.at[i, "quote_end"] = new_start + len(point.quote_text)

    # Recompute and drop anything that still doesn't match (0 or 2+ occurrences). Points without a quote_start are kept.
    points["quote_text_extracted"] = points.apply(extract_quote, axis=1)
    still_mismatching = points.quote_start.notna() & (points.quote_text != points.quote_text_extracted)
    logger.info(f"Dropping {still_mismatching.sum()} Points whose quotes still couldn't be placed")
    points = points[~still_mismatching]

    # --- Per-Service approved Point counts (a convenience column on Services) ---
    approved_per_service = points[points.status == "approved"].service_id.value_counts()
    services["approved_points"] = services.index.map(approved_per_service)

    # --- Attach Service id/name to Documents for readability (id_service is a copy of service_id) ---
    documents["id_service"] = documents.service_id
    documents["name_service"] = documents.service_id.map(services.name)

    # --- Select final columns ---
    documents = documents[DOCUMENTS_CLEAN_COLS]
    points = points[POINTS_CLEAN_COLS]

    # --- Sanity check: no dangling foreign keys ---
    assert len(set(documents.service_id) - set(services.index)) == 0
    assert len(set(cases.topic_id) - set(topics.index)) == 0
    assert len(set(points.service_id) - set(services.index)) == 0
    assert len(set(points.case_id) - set(cases.index)) == 0
    assert len(set(points.document_id.dropna()) - set(documents.index)) == 0

    return {"services": services, "documents": documents, "points": points, "cases": cases, "topics": topics}


def main():
    cleaned = clean()
    for name, df in cleaned.items():
        out_path = DUMP_DIR / f"{name}_clean.pkl"
        df.to_pickle(out_path)
        logger.info(f"Wrote {name}_clean.pkl: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
