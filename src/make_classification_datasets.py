import bisect
import collections
import hashlib
import itertools
import logging
from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

from src import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

MIN_APPROVED = 40  # Focus on cases with enough approved points to fine-tune and evaluate
NUM_FOLDS = 5  # Pre-determined train-test splits for downstream ML

DB_DUMP_VERSION = "2026-01-28"
DB_DUMP_VERSIONS_VERSION = "2026-04-18"  # `versions` table was exported separately
LATEST_VERSION = "v1"

# Outputs are saved here
SENT_SPAN_LOC = here / f"../data/db_dumps/{DB_DUMP_VERSION}/sent_span_classification_{LATEST_VERSION}.pkl"
DOC_LOC = here / f"../data/db_dumps/{DB_DUMP_VERSION}/doc_classification_{LATEST_VERSION}.pkl"

"""
This module takes database dumps (cleaned up by explore.ipynb) and forms two binary text classification datasets
for each case: 1) sentence spans that look similar to points, 2) documents
We create both at the same time to ensure that we use the same pre-split folds for cross validation. This allows us 
to compare full document classification approaches (e.g. sequence tagging) with those that work on sentence spans.

For sentence span datasets, instances are similar to points, but with [sent_idx_start, sent_idx_end, sent_text, label,
source, fold] fields, and with boundaries stretched to start/end on sentence boundaries (to match the input format 
during inference).
    `fold` is an int from 0-4
    `label` is `positive` or `negative`
    `source` is one of [
        'approved', annotated by a human and approved by curator
        'declined', annotated by a human and declined by curator
        'surrounding', sentences on either side of approved points, which in theory should not contain evidence
        'topical', approved points of other cases in the same topic area
        'doc_random', random sentence segments from the docs of approved points
        'reviewed_random', random sentence segments from comprehensively reviewed docs
    ]
Negative instances were designed to allow models to learn the exact patterns showing evidence for cases, without 
just learning what topical words are.

Instances of the document datasets look very similar to the `documents` table exported from the DB, but with ['fold', 
'text', 'label', 'source']
`source` is one of [
        'approved', annotated by a human and approved by curator
        'declined', annotated by a human and declined by curator
        'reviewed', comprehensively reviewed docs that don't have an approved point for the case
    ] 
"""


# Utility functions, to load after this script has been run
def load_sent_span() -> dict[int, pd.DataFrame]:
    with open(SENT_SPAN_LOC, "rb") as f:
        return pickle.load(f)


def load_docs() -> dict[int, pd.DataFrame]:
    with open(DOC_LOC, "rb") as f:
        return pickle.load(f)


def _load_pending_not_found(versions: pd.DataFrame, points: pd.DataFrame) -> dict[int, set[int]]:
    """
    :param versions: db table tracking point changes
    :return: for each case, a set of point IDs that have ever been pending-not-found
    """
    point_ids = collections.defaultdict(set)
    total = 0

    for i, version in versions.iterrows():
        # We serialize the point attributes as yaml
        if version.item_type == "Point" and pd.notna(version.object) and "status: pending-not-found" in version.object:
            point_id = version.item_id
            if point_id in points.index:
                point_ids[points.loc[point_id, "case_id"]].add(point_id)
                total += 1

    logger.info(f"Found {total} pending-not-found points")
    return point_ids


def _get_sent_boundaries(documents) -> dict[int, list[int]]:
    """
    Sentence splitting strategy was worked out in sent_splitting_benchmarks.py
    :return: dict from doc id to list of char positions that start sentences
    """
    # Hash on (id_doc, text) so the cache invalidates if doc content changes
    hasher = hashlib.md5()
    for doc_id, text in sorted(zip(documents.id_doc, documents.text), key=lambda x: x[0]):
        hasher.update(str(doc_id).encode())
        hasher.update(text.encode("utf-8", errors="replace"))
    content_hash = hasher.hexdigest()[:10]
    # Since this can take ~20 minutes, cache
    cache_loc = here / f"../data/db_dumps/{DB_DUMP_VERSION}/doc_sentences_{content_hash}.pkl"
    try:
        with open(cache_loc, mode="rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.info(f"Cached sentence boundaries not found at {cache_loc}")
        nlp = spacy.load("en_core_web_md", disable=["attribute_ruler", "lemmatizer", "ner"])
        sent_boundaries = dict()
        # Spacy has a default limit of 1mil characters, we have at least one doc with more
        nlp.max_length = 1500000
        for doc_id, doc in tqdm(
            zip(documents.id_doc, nlp.pipe(documents.text, n_process=4, batch_size=10)),
            total=len(documents),
            desc="Splitting sentences",
        ):
            # Filter sentences down to those with actual content
            sents = list(filter(lambda sent: sent.text != "" and not sent.text.isspace(), doc.sents))
            sent_boundaries[doc_id] = list(sorted(map(lambda s: s.start_char, sents)))

        with open(cache_loc, "wb") as f:
            pickle.dump(sent_boundaries, f)
        return sent_boundaries


def _stretch_points(points: pd.DataFrame, sent_boundaries):
    """
    Expands the boundaries of points to start/end at sentence boundaries.

    :return: a copy of `points` with sent_idx_start, sent_idx_end, and num_sents set.
    Indices refer to sent_boundaries. sent_idx_end is exclusive right bound (to match python indexing), e.g.
    sent_idx_start and sent_idx_end of 3 & 5 means the instance spans two sentences
    """
    points = points.copy()
    for doc_id in sent_boundaries:
        for point_i, point in points[points.document_id == doc_id].iterrows():
            # Clamp to 0 in case the quote starts before the first detected sentence boundary
            sent_idx_start = max(0, bisect.bisect_right(sent_boundaries[doc_id], point.quote_start) - 1)
            sent_idx_end = bisect.bisect_left(sent_boundaries[doc_id], point.quote_end)

            points.at[point_i, "sent_idx_start"] = sent_idx_start
            points.at[point_i, "sent_idx_end"] = sent_idx_end
            assert sent_idx_end - sent_idx_start > 0
            points.at[point_i, "num_sents"] = sent_idx_end - sent_idx_start
    points.sent_idx_start = points.sent_idx_start.astype(int)
    points.sent_idx_end = points.sent_idx_end.astype(int)
    points.num_sents = points.num_sents.astype(int)
    return points


def _get_sample_window_size_fn(points):
    """
    Some negative instances will be randomly sampled from elsewhere in the doc or other random docs. These could be
    sentence windows of any length, but we'd like the distribution of windows lengths to be
    about the same between positive and negative instances so the model doesn't learn it as a feature.
    This distribution can vary by case. So create numpy arrays representing a multimodal probability distribution
    for different window lengths.
    :return: function that takes a case ID and return a realistic num sentences
    """
    case_size_proportions = dict()
    for case_id in points.case_id.unique():
        props = points[points.case_id == case_id].num_sents.value_counts(normalize=True)
        prop_array = np.array([0.0] * max(props.index))
        prop_array[[i - 1 for i in props.index]] = props
        case_size_proportions[case_id] = prop_array

    def sample_window_size(case_id):
        return np.random.choice(range(1, len(case_size_proportions[case_id]) + 1), p=case_size_proportions[case_id])

    return sample_window_size


def _make_surrounding(approved_df, sample_window_size_fn, sent_boundaries, target_case):
    surrounding = []
    for _, point in approved_df[approved_df.case_id == target_case.id].iterrows():
        if point.sent_idx_start >= 1:
            before_point = point.copy()
            before_point.sent_idx_end = point.sent_idx_start
            # Back the start up a realistic number of sentences
            before_point.sent_idx_start = max(point.sent_idx_start - sample_window_size_fn(target_case.id), 0)
            surrounding.append(before_point)
        if point.sent_idx_end < len(sent_boundaries[point.document_id]):
            after_point = point.copy()
            after_point.sent_idx_start = point.sent_idx_end
            # Advance the end a realistic number of sentences
            after_point.sent_idx_end = min(
                point.sent_idx_end + sample_window_size_fn(target_case.id), len(sent_boundaries[point.document_id])
            )
            surrounding.append(after_point)
    surrounding_df = pd.DataFrame(surrounding)
    surrounding_df = surrounding_df.assign(
        label="negative", source="surrounding", point_id=np.nan, status=np.nan, quote_start=np.nan, quote_end=np.nan
    )
    surrounding_df = surrounding_df.drop("id", axis=1)
    return surrounding_df


def _make_random_reviewed(
    approved, documents, points, sample_window_size_fn, sent_boundaries, target_case, comprehensive_threshold: int
):
    # There is an is_comprehensively_reviewed feature, but also make sure a min number of points were approved for
    # the doc. Avoid any docs that have a point for the case in question.
    eligible_docs = documents[
        documents.is_comprehensively_reviewed & (documents.num_approved >= comprehensive_threshold)
    ]
    # We'll avoid docs that already have a point for the case, except pending docbot points are ok (we don't want the
    # bias of the previously released model to leak)
    case_points = points[points.case_id == target_case.id]
    case_points_minus_pending_docbot = case_points[case_points.ml_score.isna() | (case_points.status != "pending")]
    documents_to_avoid = set(case_points_minus_pending_docbot.document_id.unique())
    eligible_docs = eligible_docs[~eligible_docs.id_doc.isin(documents_to_avoid)]

    reviewed_random = []
    # Of those eligible, choose 3*len(approved) random doc IDs, then for each of those choose a sent start/end
    desired_num = 3 * len(approved)
    for doc_id in np.random.choice(eligible_docs.index, size=desired_num, replace=True):
        n_sents = len(sent_boundaries[doc_id])
        instance_num_sents = min(sample_window_size_fn(target_case.id), n_sents)
        random_sent_start = n_sents - instance_num_sents
        sent_idx_start = np.random.randint(random_sent_start) if random_sent_start > 0 else 0
        reviewed_random.append(
            {
                "case_id": target_case.id,
                "document_id": doc_id,
                "service_id": documents.loc[doc_id].service_id,
                "lang": documents.loc[doc_id].lang,
                "sent_idx_start": int(sent_idx_start),
                "sent_idx_end": int(sent_idx_start) + instance_num_sents,
            }
        )
    return pd.DataFrame(reviewed_random).assign(
        label="negative", source="reviewed_random", point_id=np.nan, status=np.nan, quote_start=np.nan, quote_end=np.nan
    )


def _make_random_from_approved(approved, instance_df, sample_window_size_fn, sent_boundaries, target_case):
    doc_random = []
    for doc_id in approved.document_id.unique():
        doc_instances = instance_df[instance_df.document_id == doc_id]
        existing_ranges = list(doc_instances[["sent_idx_start", "sent_idx_end"]].values)
        n_sents = len(sent_boundaries[doc_id])
        template = doc_instances.iloc[0]
        # Higher is more diverse, but worse class imbalance
        for i in range(5):
            # We want to avoid sentence spans already represented by previous instances. It's totally possible to
            # write logic to do this perfectly, but the chances of collisions are small (around (5 / n_sentences))
            # so just try it randomly up to 5 times and then move on
            instance_num_sents = sample_window_size_fn(target_case.id)
            disallowed_sent_idx = set(
                itertools.chain.from_iterable(
                    [range(start - instance_num_sents, end) for start, end in existing_ranges]
                )
            )
            for sent_idx_start in np.random.randint(n_sents, size=5):
                if sent_idx_start not in disallowed_sent_idx:
                    sent_idx_end = min(int(sent_idx_start) + instance_num_sents, n_sents)
                    doc_random.append(
                        {
                            "case_id": template.case_id,
                            "document_id": doc_id,
                            "service_id": template.service_id,
                            "lang": template.lang,
                            "sent_idx_start": int(sent_idx_start),
                            "sent_idx_end": sent_idx_end,
                        }
                    )
                    existing_ranges.append([int(sent_idx_start), sent_idx_end])
                    break
    return pd.DataFrame(doc_random).assign(
        label="negative", source="doc_random", point_id=np.nan, status=np.nan, quote_start=np.nan, quote_end=np.nan
    )


def _make_topical(approved, cases, points, target_case):
    topical = points[
        (points.case_id != target_case.id)
        & (points.case_id.isin(cases[cases.topic_id == target_case.topic_id].id))
        & (points.status == "approved")
    ]
    # Remove those that overlap with approved points of the target case
    to_remove = set()
    for i, point in topical.iterrows():
        for i2, approved_point in approved[approved.document_id == point.document_id].iterrows():
            # If overlapping sentence indices
            if approved_point.sent_idx_start <= (point.sent_idx_end - 1) and point.sent_idx_start <= (
                approved_point.sent_idx_end - 1
            ):
                to_remove.add(i)
                break
    topical = topical.drop(list(to_remove))

    # to limit class imbalance
    desired_num = 3 * len(approved)
    if len(topical) > desired_num:
        topical = topical.sample(desired_num, random_state=0)
    topical = topical.assign(
        label="negative", source="topical", point_id=topical.id, status=np.nan, quote_start=np.nan, quote_end=np.nan
    )
    topical = topical.drop("id", axis=1)
    return topical


def make_sent_span_datasets(
    cases, documents, points, pending_not_found: dict[int, set[int]], comprehensive_threshold: int, en_only=True
) -> dict[int, pd.DataFrame]:
    """
    :param pending_not_found: for each case, a set of point IDs that have ever been in a "pending-not-found" state. We
        use this to avoid adding declined points that might be true negatives
    :param comprehensive_threshold: a doc needs to have this many approved points before we assume it's reviewed
        comprehensively enough for use as negative cases
    :return: dict from case id to DF of instances.
        An instance is similar to point, but with [sent_idx_start, sent_idx_end, sent_text, label, source] fields
            `label` is `positive` or `negative`
            `source` is one of ['approved', 'declined', 'surrounding', 'topical', 'random']
    """
    if en_only:
        documents = documents[documents.lang == "en"].copy()
        points = points[points.lang == "en"].copy()
    no_quote = points.quote_start.isna()
    logger.info(f"Dropping {no_quote.sum()} points with no quote start/end")
    points = points[~no_quote]

    point_counts = points[points.status == "approved"].case_id.value_counts()
    cases.loc[point_counts.index, "num_approved"] = point_counts
    point_counts = points[points.status == "approved"].document_id.value_counts()
    documents.loc[point_counts.index, "num_approved"] = point_counts

    deprecated_doc_ids = set(documents.loc[documents.is_deprecated].index)

    # Pre-compute sentence-splitting info (no need for docs without Points)
    # Maps doc id to list of char positions of sentence starts
    sent_boundaries: dict[int, list[int]] = _get_sent_boundaries(documents[documents.index.isin(points.document_id)])

    # Points with some quote_start and quote_end will act as a starting place
    # We want to stretch the boundaries to start/end on sentences (because we will perform inference on sentence boundaries)
    points = _stretch_points(points, sent_boundaries)

    sample_window_size_fn = _get_sample_window_size_fn(points)
    case_datasets = dict()
    for case_i, target_case in cases[cases.num_approved >= MIN_APPROVED].iterrows():
        case_datasets[target_case.id] = _build_sent_span_case_dataset(
            target_case,
            case_i,
            cases,
            documents,
            points,
            sent_boundaries,
            sample_window_size_fn,
            deprecated_doc_ids,
            pending_not_found,
            comprehensive_threshold,
        )
    return case_datasets


def _build_sent_span_case_dataset(
    target_case,
    case_i,
    cases,
    documents,
    points,
    sent_boundaries,
    sample_window_size_fn,
    deprecated_doc_ids,
    pending_not_found,
    comprehensive_threshold,
) -> pd.DataFrame:
    logger.info(f"{('=' * 20)} Case {case_i} {('=' * 20)}")

    # Use approved points as a starting point. Rename `id` -> `point_id` here so all sources share the same schema.
    approved = points[(points.case_id == target_case.id) & (points.status == "approved")]
    approved = approved.assign(label="positive", source="approved", point_id=approved.id).drop("id", axis=1)
    logger.info(f"{len(approved)} approved points")

    # Declined points, minus a few edge cases
    declined = points[(points.case_id == target_case.id) & (points.status == "declined")]
    # Avoid points whose doc already has an approved point (often declined only because there's a better one)
    has_approved = declined.document_id.isin(set(approved.document_id))
    # Avoid points that were ever in a pending-not-found status, in case they are true negatives
    was_pending_not_found = declined.id.isin(pending_not_found[case_i])
    # Avoid points from deprecated services or documents, because sometimes they were declined just for that
    was_deprecated = declined.document_id.isin(deprecated_doc_ids)
    declined = declined[~has_approved & ~was_pending_not_found & ~was_deprecated]
    logger.info(
        f"{len(declined)} declined points (excluded {has_approved.sum()} with approved doc,"
        f" {was_pending_not_found.sum()} pending-not-found, {was_deprecated.sum()} deprecated)"
    )
    declined = declined.assign(label="negative", source="declined", point_id=declined.id).drop("id", axis=1)

    # Segments just before and after positive examples — sharpens evidence vs. topic discrimination
    surrounding_df = _make_surrounding(approved, sample_window_size_fn, sent_boundaries, target_case)
    logger.info(f"{len(surrounding_df)} surrounding points")

    # Approved points of the same topic but a different case
    topical = _make_topical(approved, cases, points, target_case)
    logger.info(f"{len(topical)} topical points")

    # Random segments from the rest of approved docs. Pre-merge what we have so far so doc_random can
    # avoid sentence spans already represented. Pending points for the target case are included as
    # exclusion-only ranges — they may be true evidence (e.g. unreviewed docbot predictions) so we
    # don't want a doc_random window landing on them.
    pending_exclude = points[(points.case_id == target_case.id) & (points.status == "pending")]
    so_far = pd.concat([approved, declined, surrounding_df, topical, pending_exclude])
    doc_random_df = _make_random_from_approved(approved, so_far, sample_window_size_fn, sent_boundaries, target_case)
    logger.info(f"{len(doc_random_df)} random points from docs with approved")

    # Random segments of comprehensively reviewed docs
    reviewed_random_df = _make_random_reviewed(
        approved, documents, points, sample_window_size_fn, sent_boundaries, target_case, comprehensive_threshold
    )
    logger.info(f"{len(reviewed_random_df)} random points from reviewed docs")

    instance_df = pd.concat([approved, declined, surrounding_df, topical, doc_random_df, reviewed_random_df])

    # Reset num_sents — the sources above were not consistent about setting it
    instance_df["num_sents"] = instance_df.sent_idx_end - instance_df.sent_idx_start

    # Extract the string content for each instance based on sent positions
    instance_df["char_start"] = instance_df.apply(
        lambda i: int(sent_boundaries[i.document_id][i.sent_idx_start]), axis=1
    )

    def _char_end(i):
        # If an instance ends at the very end of the doc, the "end sentence" position doesn't exist
        # but we can just set end char to something past the end (we're just using this to slice a str)
        sent_pos = sent_boundaries[i.document_id]
        if i.sent_idx_end < len(sent_pos):
            return int(sent_pos[i.sent_idx_end])
        return len(documents.loc[i.document_id].text)

    instance_df["char_end"] = instance_df.apply(_char_end, axis=1)
    instance_df["text"] = instance_df.apply(
        lambda i: documents.loc[i.document_id].text[i.char_start : i.char_end], axis=1
    )

    return instance_df.drop(
        ["analysis", "created_at", "updated_at", "service_needs_rating_update", "user_id", "point_change"],
        axis=1,
        errors="ignore",
    ).reset_index(drop=True)


def make_doc_datasets(
    cases, documents, points, pending_not_found: dict[int, set[int]], comprehensive_threshold: int, en_only=True
) -> dict[int, pd.DataFrame]:
    """
    :param comprehensive_threshold: a doc needs to have this many approved points before we assume it's reviewed
        comprehensively enough for use as negative cases
    :return: dict from case id to DF of instances.
        An instance is a document, but with [label, source] fields
            `label` is `positive` or `negative`
            `source` is one of ['approved', 'declined', 'random']
    """
    if en_only:
        documents = documents[documents.lang == "en"].copy()
        points = points[points.lang == "en"].copy()

    # Attach num points
    point_counts = points[points.status == "approved"].document_id.value_counts()
    documents.loc[point_counts.index, "num_approved"] = point_counts

    deprecated_doc_ids = set(documents.loc[documents.is_deprecated].index)

    # Filter out docs without points
    documents = documents[documents.index.isin(points.document_id)]
    # For negative instances, restrict ourselves to comprehensively reviewed docs (this feature comes from the DB, but
    # we can also specify an additional threshold here)
    comprehensively_reviewed_docs = documents[
        (documents.is_comprehensively_reviewed) & (documents.num_approved >= comprehensive_threshold)
    ]

    case_datasets = dict()
    for case_i, target_case in cases[cases.num_approved >= MIN_APPROVED].iterrows():
        case_points = points[points.case_id == target_case.id]
        approved = case_points[case_points.status == "approved"]
        # `reindex` + dropna so an unexpected missing doc doesn't blow up the whole case
        approved_docs = documents.reindex(approved.document_id.unique()).dropna(subset=["id_doc"])
        approved_docs = approved_docs.assign(label="positive", source="approved")

        # Docs with declined points, minus a few edge cases
        declined_points = case_points[case_points.status == "declined"]
        declined_doc_ids = set(declined_points.document_id.unique()) - set(approved_docs.id_doc) - deprecated_doc_ids
        # Also filter out any docs with points that were ever pending-not-found, just in case it's a true negative
        pending_not_found_docs = set()
        for point_id in pending_not_found[case_i]:
            if point_id in case_points.index:
                pending_not_found_docs.add(case_points.loc[point_id, "document_id"])
        declined_doc_ids -= pending_not_found_docs
        declined_docs = documents.reindex(list(declined_doc_ids)).dropna(subset=["id_doc"])
        declined_docs = declined_docs.assign(label="negative", source="declined")

        # All comprehensively reviewed docs. Avoid any docs that have a point for the case, except pending
        # docbot points (we don't want the bias of the previously released model to leak)
        case_points_minus_pending_docbot = case_points[case_points.ml_score.isna() | (case_points.status != "pending")]
        documents_to_avoid = set(case_points_minus_pending_docbot.document_id.unique())
        reviewed_docs = comprehensively_reviewed_docs[~comprehensively_reviewed_docs.id_doc.isin(documents_to_avoid)]
        reviewed_docs = reviewed_docs.assign(label="negative", source="reviewed")

        instance_df = pd.concat([approved_docs, declined_docs, reviewed_docs])
        case_datasets[target_case.id] = instance_df[["id_doc", "text", "id_service", "label", "source"]].reset_index(
            drop=True
        )

    return case_datasets


def assign_folds(sent_span_datasets, doc_datasets):
    """
    Folds are split by document_id so a doc can't leak between the sent-span and doc datasets. Within
    that constraint, docs are bucketed into those that contribute any positive instance vs. those that
    don't, and each bucket is sliced across folds independently — this stratifies positives so each
    fold gets a representative share (positives are rare for many cases, and uniform doc-ID shuffling
    leaves cross-fold variance high).
    """

    def _gen_fold_slice(n, fold_i):
        return slice(round(fold_i * (n / NUM_FOLDS)), round((fold_i + 1) * (n / NUM_FOLDS)))

    rng = random.Random(0)
    for case_id in set(sent_span_datasets.keys()).union(set(doc_datasets.keys())):
        sent_df = sent_span_datasets[case_id]
        doc_df = doc_datasets[case_id]

        positive_doc_ids = set(sent_df.loc[sent_df.label == "positive", "document_id"]).union(
            set(doc_df.loc[doc_df.label == "positive", "id_doc"])
        )
        all_doc_ids = set(sent_df.document_id).union(set(doc_df.id_doc))
        negative_only_doc_ids = all_doc_ids - positive_doc_ids

        positive_doc_ids = list(positive_doc_ids)
        negative_only_doc_ids = list(negative_only_doc_ids)
        rng.shuffle(positive_doc_ids)
        rng.shuffle(negative_only_doc_ids)

        for fold_i in range(NUM_FOLDS):
            fold_doc_ids = set(positive_doc_ids[_gen_fold_slice(len(positive_doc_ids), fold_i)]) | set(
                negative_only_doc_ids[_gen_fold_slice(len(negative_only_doc_ids), fold_i)]
            )
            sent_df.loc[sent_df.document_id.isin(fold_doc_ids), "fold"] = fold_i
            doc_df.loc[doc_df.id_doc.isin(fold_doc_ids), "fold"] = fold_i
        sent_df.fold = sent_df.fold.astype(int)
        doc_df.fold = doc_df.fold.astype(int)
        for fold_i in range(NUM_FOLDS):
            sent_pos = ((sent_df.fold == fold_i) & (sent_df.label == "positive")).sum()
            doc_pos = ((doc_df.fold == fold_i) & (doc_df.label == "positive")).sum()
            if sent_pos == 0 or doc_pos == 0:
                logger.warning(
                    f"Case {case_id} fold {fold_i}: {sent_pos} sent-span positives, "
                    f"{doc_pos} doc positives — downstream metrics may be unreliable"
                )
        sent_span_datasets[case_id] = sent_df
        doc_datasets[case_id] = doc_df

    return sent_span_datasets, doc_datasets


def run():
    np.random.seed(0)

    cases = pd.read_pickle(here / f"../data/db_dumps/{DB_DUMP_VERSION}/cases_clean.pkl")
    documents = pd.read_pickle(here / f"../data/db_dumps/{DB_DUMP_VERSION}/documents_clean.pkl")
    points = pd.read_pickle(here / f"../data/db_dumps/{DB_DUMP_VERSION}/points_clean.pkl")
    services = pd.read_pickle(here / f"../data/db_dumps/{DB_DUMP_VERSION}/services_clean.pkl")
    versions = pd.read_pickle(here / f"../data/db_dumps/{DB_DUMP_VERSIONS_VERSION}/versions.pkl")

    # Drop points not associated with a doc
    points = points[points.document_id.notna()]

    # Join service info onto documents, avoiding duplicate columns
    documents = pd.merge(
        documents.drop(["id_service", "name_service"], axis=1),
        services,
        left_on="service_id",
        right_index=True,
        suffixes=["_doc", "_service"],
    )
    # Docs and services with a `deleted` status were dropped in explore.ipynb, but as of 5/26 we don't always use it,
    # instead adding 'deprecated' to the name somewhere
    documents["is_deprecated"] = documents.name_doc.str.lower().str.contains(
        "deprecated", na=False
    ) | documents.name_service.str.lower().str.contains("deprecated", na=False)
    # Clean up html, which is necessary for good sentence splitting. This should be done for inference as well.
    documents["text"] = documents.text.apply(utils.preprocess_doc_text)
    points["text"] = points.quote_text.apply(lambda text: None if pd.isna(text) else utils.preprocess_doc_text(text))
    # type coercion
    points["document_id"] = points.document_id.astype(np.int64)

    # Load points that have ever been in a pending-not-found state
    pending_not_found: dict[int, set[int]] = _load_pending_not_found(versions, points)

    comprehensive_threshold = 12
    sent_span_datasets = make_sent_span_datasets(cases, documents, points, pending_not_found, comprehensive_threshold)
    doc_datasets = make_doc_datasets(cases, documents, points, pending_not_found, comprehensive_threshold)
    sent_span_datasets, doc_datasets = assign_folds(sent_span_datasets, doc_datasets)

    logger.info(f"Saving sentence span classification datasets to {SENT_SPAN_LOC}")
    with open(SENT_SPAN_LOC, "wb") as f:
        pickle.dump(sent_span_datasets, f)
    logger.info(f"Saving document classification datasets to {DOC_LOC}")
    with open(DOC_LOC, "wb") as f:
        pickle.dump(doc_datasets, f)

    case_id = np.random.choice(list(sent_span_datasets.keys()))
    logger.info(f"Example structure for sent spans, case id {case_id}:")
    sent_span_datasets[case_id].info()
    logger.info(f"Negative case breakdown for sent spans, case {case_id}")
    logger.info(sent_span_datasets[case_id].source.value_counts())
    logger.info(f"Example structure for documents, case id {case_id}:")
    doc_datasets[case_id].info()

    for case_id in sent_span_datasets:
        logger.info(f"{'=' * 30} Case {case_id} {'=' * 30}")
        logger.info("Sentence spans:")
        logger.info(f"\t{sent_span_datasets[case_id].source.value_counts()}")
        logger.info("Documents:")
        logger.info(f"\t{doc_datasets[case_id].source.value_counts()}")
        logger.info(f"Sent span folds: {sent_span_datasets[case_id].fold.value_counts()}")
        logger.info(f"Doc folds: {doc_datasets[case_id].fold.value_counts()}")


if __name__ == "__main__":
    run()
