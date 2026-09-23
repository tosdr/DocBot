import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploratory data analysis + data cleaning

    EDA for the ToS;DR database dumps, and the cleaning that produces the `*_clean.pkl` files consumed downstream.

    - **Service** — website, app, etc.
    - **Document** — crawled ToS/privacy policy for a Service (FK: service, user)
    - **Point** — a plain-text highlight summarizing an aspect of a Service's policy (FK: service, case, document, user)
    - **Case** — a Service-agnostic pattern that Points exhibit, e.g. *"No third-party analytics is used"* (FK: topic)
    - **Topic** — a high-level grouping of Cases, e.g. *"User Choice"*

    This is a [marimo](https://marimo.io) notebook (migrated from Jupyter). Because marimo requires each variable be
    defined in a single cell, the five entity frames are each bound once to their cleaned value; the raw dumps are
    `raw_*`. The same cleaning is available as a standalone script in `src/clean_data.py` — notebooks are intentionally
    kept out of data pipelines, but the cleaning lives here too since *what* to clean is itself an EDA outcome.
    """)
    return


@app.cell
def _():
    import pickle
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    sns.set_theme()
    DetectorFactory.seed = 0  # Make langdetect deterministic
    return LangDetectException, Path, detect, mo, np, pd, pickle, plt, sns


@app.cell
def _(Path):
    VERSION = "2026-07-09"
    try:
        _here = Path(__file__).parent
    except NameError:  # e.g. running cells interactively without a file
        _here = Path.cwd()
    DUMP_DIR = (_here / f"../data/db_dumps/{VERSION}").resolve()

    # Hardcoded `none`/spam rows surfaced during EDA. Service 502 `(NONE)` collects google-group discussions, spam and
    # unplaceable deleted Points; Case 235, Topic 53 and Document 1378 are analogous.
    NONE_SERVICE_ID = 502
    NONE_CASE_ID = 235
    NONE_TOPIC_ID = 53
    EXAMPLE_DOCUMENT_ID = 1378

    # Point statuses worth keeping. `approved`/`declined` make datasets; `pending` may be useful for pending docbot points.
    KEEP_POINT_STATUSES = {"approved", "declined", "pending"}

    # Final column schemas, kept stable so downstream consumers don't break.
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
    return (
        DOCUMENTS_CLEAN_COLS,
        DUMP_DIR,
        EXAMPLE_DOCUMENT_ID,
        KEEP_POINT_STATUSES,
        NONE_CASE_ID,
        NONE_SERVICE_ID,
        NONE_TOPIC_ID,
        POINTS_CLEAN_COLS,
    )


@app.cell
def _(LangDetectException, detect, pd):
    def detect_lang(text):
        if text is None or pd.isnull(text) or text == "":
            return None
        try:
            return detect(text)
        except LangDetectException:
            return None

    return (detect_lang,)


@app.cell
def _(DUMP_DIR, pickle):
    raw_cases = pickle.load(open(DUMP_DIR / "cases.pkl", "rb"))
    raw_documents = pickle.load(open(DUMP_DIR / "documents.pkl", "rb"))
    raw_points = pickle.load(open(DUMP_DIR / "points.pkl", "rb"))
    raw_services = pickle.load(open(DUMP_DIR / "services.pkl", "rb"))
    raw_topics = pickle.load(open(DUMP_DIR / "topics.pkl", "rb"))
    return raw_cases, raw_documents, raw_points, raw_services, raw_topics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Removing deleted rows

    Services, Documents and Points each carry a `status` that can be `deleted` (Documents and Points only gained the
    column in a recent schema revision). We exclude every deleted row from the entire analysis, cascading to the
    Documents/Points of deleted Services and the Points of deleted Documents.
    """)
    return


@app.cell
def _(pd, raw_documents, raw_points, raw_services):
    # How many deleted rows are there to exclude?
    pd.DataFrame(
        {
            "total": [len(raw_services), len(raw_documents), len(raw_points)],
            "deleted": [
                (raw_services.status == "deleted").sum(),
                (raw_documents.status == "deleted").sum(),
                (raw_points.status == "deleted").sum(),
            ],
        },
        index=["services", "documents", "points"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cleaning the Documents

    Drop deleted Documents (and those of deleted Services), the example Document, treat empty text as missing, detect
    languages, and drop text-less Documents (roughly a third are uncrawled). Finally attach the parent Service's id/name
    for readability.
    """)
    return


@app.cell
def _(DOCUMENTS_CLEAN_COLS, EXAMPLE_DOCUMENT_ID, detect_lang, np, raw_documents, raw_services):
    _deleted_service_ids = raw_services[raw_services.status == "deleted"].id

    documents = raw_documents[
        (raw_documents.status != "deleted") & ~raw_documents.service_id.isin(_deleted_service_ids)
    ].drop("status", axis=1)
    documents = documents.drop(EXAMPLE_DOCUMENT_ID, errors="ignore")

    documents["text"] = documents.text.replace("", np.nan)  # Treat empty text as missing
    documents["lang"] = documents.text.map(detect_lang)  # (slow: langdetect per Document)
    documents["doc_len"] = documents.text.str.len()

    # Drop text-less Documents, then attach Service id (a copy of service_id) and name.
    documents = documents[documents.text.notna() & (documents.doc_len > 0)]
    documents["id_service"] = documents.service_id
    documents["name_service"] = documents.service_id.map(raw_services.name)
    documents = documents[DOCUMENTS_CLEAN_COLS]
    return (documents,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cleaning the Points

    Keep only useful statuses (which excludes `deleted`), drop the `none`/spam Service & Case, drop Points of Services
    with no surviving Documents, and drop Points whose Document was removed (deleted/text-less). Then realign quotes:
    Documents are often edited after a Point is made without resetting `quote_start`/`quote_end`, so where the exact
    `quote_text` occurs once in the current Document we snap the offsets to it; anything still mismatched is dropped.
    """)
    return


@app.cell
def _(KEEP_POINT_STATUSES, NONE_CASE_ID, NONE_SERVICE_ID, POINTS_CLEAN_COLS, documents, np, pd, raw_points):
    _served_service_ids = set(documents.service_id)  # Services that still have at least one Document

    points = raw_points[
        raw_points.status.isin(KEEP_POINT_STATUSES)  # excludes `deleted` and other non-useful statuses
        & (raw_points.service_id != NONE_SERVICE_ID)
        & (raw_points.case_id != NONE_CASE_ID)
        & raw_points.service_id.isin(_served_service_ids)  # drops Points of deleted/doc-less Services
        & (
            raw_points.document_id.isna() | raw_points.document_id.isin(documents.index)
        )  # drops removed-Document Points
    ].drop("rank", axis=1)

    points["quote_text"] = points.quote_text.replace("", np.nan)
    # A quote_start with no quote_text is anomalous; drop those Points.
    points = points.drop(points[points.quote_start.notna() & points.quote_text.isna()].index)

    # Attach the (possibly-edited) Document text (for offset realignment) and language (kept in the final schema).
    points["text"] = points.document_id.map(documents.text)
    points["lang"] = points.document_id.map(documents.lang)

    def _extract_quote(point):
        if pd.isna(point.quote_start) or pd.isna(point.quote_end) or pd.isna(point.text):
            return np.nan
        return point.text[int(point.quote_start) : int(point.quote_end)]

    points["quote_text_extracted"] = points.apply(_extract_quote, axis=1)
    _mismatching = points[points.quote_start.notna() & (points.quote_text != points.quote_text_extracted)]
    for _i, _point in _mismatching.iterrows():
        if pd.isna(_point.text) or pd.isna(_point.quote_start):
            continue
        if _point.text.count(_point.quote_text) == 1:  # only realign unambiguous single matches
            _new_start = _point.text.find(_point.quote_text)
            points.at[_i, "quote_start"] = _new_start
            points.at[_i, "quote_end"] = _new_start + len(_point.quote_text)

    # Recompute and drop anything still mismatched (0 or 2+ occurrences). Points without a quote_start are kept.
    points["quote_text_extracted"] = points.apply(_extract_quote, axis=1)
    points = points[(points.quote_text == points.quote_text_extracted) | points.quote_start.isna()]
    points = points[POINTS_CLEAN_COLS]
    return (points,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cleaning Services, Cases and Topics

    Drop deleted Services, the `none` Service/Case/Topic, and Services left with no Documents. Attach each Service's count
    of approved Points as a convenience column.
    """)
    return


@app.cell
def _(NONE_SERVICE_ID, documents, points, raw_services):
    _served_service_ids = set(documents.service_id)
    _approved_per_service = points[points.status == "approved"].service_id.value_counts()

    services = raw_services[raw_services.status != "deleted"].drop("status", axis=1)
    services = services.drop(NONE_SERVICE_ID, errors="ignore")
    services = services[services.id.isin(_served_service_ids)]
    services["approved_points"] = services.index.map(_approved_per_service)
    return (services,)


@app.cell
def _(NONE_CASE_ID, NONE_TOPIC_ID, raw_cases, raw_topics):
    cases = raw_cases.drop(NONE_CASE_ID, errors="ignore")
    topics = raw_topics.drop(NONE_TOPIC_ID, errors="ignore")
    return cases, topics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sanity check — no dangling foreign keys
    """)
    return


@app.cell
def _(cases, documents, mo, points, services, topics):
    assert len(set(documents.service_id) - set(services.index)) == 0
    assert len(set(cases.topic_id) - set(topics.index)) == 0
    assert len(set(points.service_id) - set(services.index)) == 0
    assert len(set(points.case_id) - set(cases.index)) == 0
    assert len(set(points.document_id.dropna()) - set(documents.index)) == 0
    mo.md("No dangling foreign keys ✅")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EDA — Services
    """)
    return


@app.cell
def _(plt, services, sns):
    # How have the number of Services grown over time?
    plt.figure(figsize=(8, 6))
    sns.ecdfplot(services, x="created_at", stat="count")
    plt.title("Services added over time")
    plt.gca()
    return


@app.cell
def _(plt, services, sns):
    # How many Services are comprehensively reviewed, and how are they rated?
    plt.figure(figsize=(8, 6))
    sns.countplot(x=services.rating, order=["A", "B", "C", "D", "E", "N/A"])
    plt.title("Service ratings")
    plt.gca()
    return


@app.cell
def _(mo, services):
    # For what Services do we have the most approved Points?
    mo.vstack(
        [
            mo.md("**Approved Points per Service**"),
            services.approved_points.describe().round(1).to_frame().T,
            services.sort_values("approved_points", ascending=False).head(15)[["name", "approved_points"]],
        ]
    )
    return


@app.cell
def _(plt, services, sns):
    # Rated, comprehensively-reviewed Services tend to have more approved Points
    _fig, _ax = plt.subplots(figsize=(7, 5))
    sns.kdeplot(
        services[services.rating != "N/A"],
        x="approved_points",
        hue="is_comprehensively_reviewed",
        ax=_ax,
        bw_adjust=0.5,
    )
    _ax.set_xlim((0, 80))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EDA — Documents
    """)
    return


@app.cell
def _(documents):
    # Most common Document names
    documents.name.str.lower().str.strip().value_counts().head(25).to_frame()
    return


@app.cell
def _(documents):
    # In what languages are the Documents?
    documents.lang.value_counts().head(15).to_frame()
    return


@app.cell
def _(documents, plt, sns):
    # Document lengths (characters), outliers clipped so the x-axis isn't stretched
    _fig, _ax = plt.subplots(figsize=(14, 8))
    _ax.set_xlim((0, 100000))
    sns.kdeplot(documents.doc_len[documents.doc_len < 100000], ax=_ax)
    plt.title("Document length (num characters)")
    _fig
    return


@app.cell
def _(documents, mo):
    # Most Services have 1-2 Documents, but some have many more
    _counts = documents.service_id.value_counts()
    mo.vstack(
        [
            mo.md("**Documents per Service**"),
            _counts.describe().round(1).to_frame().T,
            documents.groupby(["service_id", "name_service"])
            .size()
            .sort_values(ascending=False)
            .head(15)
            .to_frame("n_docs"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EDA — Points
    """)
    return


@app.cell
def _(plt, points, sns):
    # How have Points grown over time, by status?
    plt.figure(figsize=(12, 8))
    sns.ecdfplot(points, x="created_at", hue="status", stat="count")
    plt.title("Points added over time")
    plt.gca()
    return


@app.cell
def _(cases, mo, points):
    # Distribution of Points per Case, and the Cases with the most approved Points
    _by_case = points.merge(cases[["title"]], left_on="case_id", right_index=True, suffixes=["_point", "_case"])
    _by_case = _by_case.groupby(["case_id", "title_case"]).size().sort_values(ascending=False)
    mo.vstack(
        [
            mo.md("**Points per Case**"),
            points.case_id.value_counts().describe().round(1).to_frame().T,
            _by_case.head(15).to_frame("n_points"),
        ]
    )
    return


@app.cell
def _(plt, points, sns):
    # Distribution of quote lengths (characters)
    _quote_len = (points.quote_end - points.quote_start).dropna()
    _fig, _ax = plt.subplots(figsize=(14, 8))
    _ax.set_xlim((0, 1000))
    sns.histplot(_quote_len[_quote_len < 1000], ax=_ax, kde=False)
    plt.title("Quote lengths (num characters)")
    _fig
    return


@app.cell
def _(cases, mo, points):
    # Is a Point's title usually the same as its Case's title? (true for roughly half of approved Points)
    _pts = points.merge(cases[["title"]], left_on="case_id", right_index=True, suffixes=["_point", "_case"])
    _approved = _pts[_pts.status == "approved"]
    mo.md(
        f"Point title equals Case title for **{(_approved.title_point == _approved.title_case).sum()} / "
        f"{len(_approved)}** approved Points"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EDA — Cases & Topics
    """)
    return


@app.cell
def _(cases, mo, topics):
    mo.vstack([mo.md("**Cases**"), cases.head(), mo.md("**Topics**"), topics.head()])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Re-serialize the cleaned tables

    Write the `*_clean.pkl` files. (`src/clean_data.py` does the same thing headlessly for pipelines.)
    """)
    return


@app.cell
def _(DUMP_DIR, cases, documents, mo, points, services, topics):
    services.to_pickle(DUMP_DIR / "services_clean.pkl")
    documents.to_pickle(DUMP_DIR / "documents_clean.pkl")
    points.to_pickle(DUMP_DIR / "points_clean.pkl")
    cases.to_pickle(DUMP_DIR / "cases_clean.pkl")
    topics.to_pickle(DUMP_DIR / "topics_clean.pkl")
    mo.md(
        f"Wrote clean pickles to `{DUMP_DIR}`: "
        f"{len(services)} services, {len(documents)} documents, {len(points)} points, "
        f"{len(cases)} cases, {len(topics)} topics"
    )
    return


if __name__ == "__main__":
    app.run()
