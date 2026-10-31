import logging
import pickle
import random
from argparse import ArgumentParser
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import spacy

# from sklearn.ensemble import RandomForestClassifier  # For the commented-out RF setups in gen_setups
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold
from textacy import extract
from textacy.representations.vectorizers import Vectorizer
from tqdm import tqdm

from src import make_classification_datasets, utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

DB_DUMP_VERSION = make_classification_datasets.DB_DUMP_VERSION
RANDOM_STATE = 0

# The version this script trains, and that train.py evaluates new case models with. Production loads
# inference.PREFILTER_VERSION instead, until the case models trained alongside this one ship. When that happens, be sure
# to modify .dockerignore so the models are copied into the inference Docker image
# tfidf_v3: trained on the train split only (make_classification_datasets.service_split), and thresholded by pass rate
# (see below) instead of recall100 thresholds
MODEL_VERSION = "tfidf_v3"

"""
This script trains high recall TF-IDF classification models that can be used as prefilters, to avoid unnecessary
inference with our deep learning models for completely irrelevant sentences.

Only train-split docs are used for training and choosing thresholds (make_classification_datasets.service_split): the
case models' thresholds are picked on test-split doc scores, which the prefilter shapes, so it mustn't have seen them
either.

Thresholds are chosen as pass rates (the share of irrelevant sentences let through to BERT) rather than scores, since a
pass rate means the same thing for every model. Per case:
1. A 5-fold CV within the train split (train_cv). Every sentence is scored by the fold model that didn't train on it,
   and gets a "pass rate needed": the fraction of its fold's held-out negatives scoring at least as high. Sorted, the
   positives' pass rates needed trace the high-recall edge of the PR curve ("the cliff"), which
   notebooks/tfidf_cliff.py plots.
2. choose_pass_rate picks a pass rate on the cliff (exchange_rule).
3. fit_full trains one model on every sentence, and finds the score threshold that lets through that pass rate. Its only
   negatives are ones it trained on, which it scores lower than unseen ones, so the CV measures by how much and
   corrects for it.

Each case's output goes to data/models/{MODEL_VERSION}/{case_id}: vectorizer.pkl, model.pkl and final_metrics.pkl, which
inference.load_prefilter_kwargs loads. For the notebook, and left out of Docker images, there's also cv.pkl with the CV
scores, and test.pkl with the shipped model's scores on test-split sentences. The test split is for information only:
choosing anything by it would let test-split labels shape the prefilter after all. The setup is fixed (SHIPPED_SETUP);
--compare_setups also cross-validates the rest of gen_setups for the notebook.
"""


def model_output_dir(case_id, version: str = MODEL_VERSION) -> Path:
    return here / f"../data/models/{version}/{case_id}"


def setups_output_path(case_id, version: str = MODEL_VERSION) -> Path:
    # Not deployable, so outside data/models
    return here / f"../data/tfidf/{version}/setups/{case_id}.pkl"


def make_vectorizer() -> Vectorizer:
    return Vectorizer(tf_type="sqrt", idf_type="smooth", norm="l1", min_df=4, max_df=0.8)


class Setup(TypedDict):
    name: str
    model_class: type[Any]
    model_kwargs: dict[str, Any]


# The gen_setups setup every case ships with. Over all cases, the 6 setups with a positive class weight >= 10 needed
# about the same pass rate for 95% recall (within 0.2 points on average), and this one was within a point of each case's
# best the most often. SGD cw=30 tied it
SHIPPED_SETUP = "SGD cw=balanced"
# exchange_rule's price of recall: one point of recall is worth MU points of pass rate
MU = 8


def gen_setups() -> Iterator[Setup]:
    # TODO also loop through the dataset labels (one with other topical positives, one without)
    # Trimmed from the old grid: elastic-net LR and depth-3 RF were rarely the best setup across 84 cases, and with
    # fewer setups, the best one is less likely to just be the luckiest
    # Upweighting positives replaces the old rebalance_to option, which subsampled negatives to 20 per positive (about a
    # 10x positive weight, plus sampling noise that alone swung held-out pass rates between 34% and 89% on case 297)
    for weight in [1, 3, 10, 30, "balanced"]:
        class_weight = weight if weight == "balanced" else {0: 1.0, 1: float(weight)}
        # lbfgs rather than saga: on these l1-normed features saga didn't converge in 5000 iterations for weights >= 10
        # (l1_ratio=0 is pure L2. penalty= was deprecated in sklearn 1.8 in favor of l1_ratio/C)
        yield {
            "name": f"LR cw={weight}",
            "model_class": LogisticRegression,
            "model_kwargs": dict(
                l1_ratio=0, solver="lbfgs", max_iter=5000, class_weight=class_weight, random_state=RANDOM_STATE
            ),
        }
        yield {
            "name": f"SGD cw={weight}",
            "model_class": SGDClassifier,
            "model_kwargs": dict(
                loss="log_loss",
                penalty="l2",
                alpha=0.0001,
                max_iter=5000,
                class_weight=class_weight,
                random_state=RANDOM_STATE,
            ),
        }
    # Random forests needed far more pass rate for 95% recall: a 22% mean over the first 15 cases, vs 9% for the best
    # linear setups
    # for max_depth in [5, 10]:
    #     yield {
    #         "name": f"RF depth={max_depth}",
    #         "model_class": RandomForestClassifier,
    #         "model_kwargs": dict(
    #             n_estimators=100,
    #             max_depth=max_depth,
    #             min_samples_split=8,
    #             min_samples_leaf=3,
    #             max_features="sqrt",
    #             class_weight="balanced_subsample",
    #             random_state=RANDOM_STATE,
    #         ),
    #     }
    # For some reason ComplementNB does very poorly. I don't think I have negative values so that's not it
    # TODO try GradientBoostingClassifier if it's not overkill


def fit_setup(setup: Setup, X, y):
    return setup["model_class"](**setup["model_kwargs"]).fit(X, y)


def pass_rates_needed(scores: np.ndarray, negative_scores: np.ndarray) -> np.ndarray:
    """For each score, the fraction of negative_scores at least as high: the pass rate it needs to get through"""
    negatives = np.sort(negative_scores)
    return 1 - np.searchsorted(negatives, scores) / len(negatives)


def threshold_for_pass_rate(negative_scores: np.ndarray, pass_rate: float) -> float:
    """
    The score threshold (sentences scoring at least this much get through) that lets through at most pass_rate of
    negative_scores. Consistent with pass_rates_needed: a score gets through iff the pass rate it needs is <= pass_rate
    """
    negatives = np.sort(negative_scores)
    n_through = int(np.floor(pass_rate * len(negatives) + 1e-9))
    if n_through >= len(negatives):
        return 0.0
    # Just above the highest negative that's kept out
    return float(np.nextafter(negatives[len(negatives) - n_through - 1], np.inf))


def train_cv(
    ngrams: list[list[str]], labels: np.ndarray, service_ids: np.ndarray, setups: list[Setup], cv_folds=5
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Scores every sentence with the fold model that didn't train on it, for each setup.

    Folds are grouped by service, like make_classification_datasets.service_split, so sentences from one service's docs
    (near-duplicate docs, or several sentences sampled from one doc) can't land on both sides of a fold and inflate
    held-out scores. Each fold gets its own vectorizer, fit on its training sentences only, like the full model's. One
    shared by all folds would also have seen the held-out sentences, so their rare words would stay in their vectors
    (with near-zero weight), instead of dropping out as an unseen sentence's do. In a nested simulation over all cases,
    that made held-out negatives score lower than unseen ones, and the pass rates came out 10% too low.

    Returns each sentence's held-out fold, and a long DataFrame with one row per setup x sentence:
        setup: gen_setups name, e.g. "LR cw=3"
        row: index into ngrams
        score: predicted probability from the fold model that held the sentence out
        pass_rate_needed: the fraction of that fold's held-out negatives scoring at least as high, i.e. the pass rate of
            the strictest threshold that still keeps this sentence. It's computed from negatives only, so for positives
            it's an honest held-out estimate
        in_sample_pass_rate_needed: the same, but against the fold model's own training negatives. It's needed to
            calibrate the full model, which has no other negatives (see choose_pass_rate)
    """
    sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    folds = np.empty(len(ngrams), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(np.zeros(len(ngrams)), labels, service_ids)):
        folds[val_idx] = fold_idx
    logger.info(f"Positives per fold: {np.bincount(folds[labels == 1], minlength=cv_folds).tolist()}")

    # Vectorize each fold once, for all setups
    fold_data = []
    for fold_idx in range(cv_folds):
        train_idx, val_idx = np.where(folds != fold_idx)[0], np.where(folds == fold_idx)[0]
        vectorizer = make_vectorizer()
        X_train = vectorizer.fit_transform([ngrams[i] for i in train_idx])
        X_val = vectorizer.transform([ngrams[i] for i in val_idx])
        fold_data.append((train_idx, val_idx, X_train, X_val))

    scores = []
    for setup in tqdm(setups, desc="Trying setups", disable=len(setups) == 1):
        score, held_out_needed, in_sample_needed = (np.empty(len(ngrams)) for _ in range(3))
        for train_idx, val_idx, X_train, X_val in fold_data:
            model = fit_setup(setup, X_train, labels[train_idx])
            val_scores = model.predict_proba(X_val)[:, 1]
            train_negative_scores = model.predict_proba(X_train[labels[train_idx] == 0])[:, 1]
            score[val_idx] = val_scores
            held_out_needed[val_idx] = pass_rates_needed(val_scores, val_scores[labels[val_idx] == 0])
            in_sample_needed[val_idx] = pass_rates_needed(val_scores, train_negative_scores)
        scores.append(
            pd.DataFrame(
                {
                    "setup": setup["name"],
                    "row": np.arange(len(ngrams)),
                    "score": score,
                    "pass_rate_needed": held_out_needed,
                    "in_sample_pass_rate_needed": in_sample_needed,
                }
            )
        )

    return folds, pd.concat(scores, ignore_index=True)


def summarize_cliff(scores: pd.DataFrame, labels: np.ndarray, pass_rates=(0.1, 0.25, 0.5)) -> pd.DataFrame:
    """Positives dropped per setup if every fold model let through a given share of its held-out negatives"""
    pass_needed = scores[labels[scores.row] == 1].groupby("setup", sort=False).pass_rate_needed
    return pd.DataFrame({f"dropped at {p:.0%}": pass_needed.apply(lambda q: (q > p).sum()) for p in pass_rates})


def exchange_rule(pass_needed: np.ndarray, mu: float = MU) -> float:
    """
    The pass rate p minimizing p + mu * (share of positives dropped), given the positives' held-out pass rates needed.

    BERT compute is linear in the pass rate, so this prices it linearly. Every case runs on every document, so total
    compute is the sum of the cases' pass rates, and minimizing that sum plus mu times the sum of the shares dropped
    splits into this same choice per case. The same mu everywhere means every case stops where its last point of pass
    rate buys the same recall. In a nested simulation this matched a pass-rate ceiling on mean recall (at the same mean
    pass rate), but its worst case kept 85% of positives where the ceiling's kept 70%.

    Only pass rates where a positive sits right on the edge are candidates, since anything in between costs more pass
    rate for the same recall
    """
    candidates = np.unique(np.append(pass_needed, 1.0))[::-1]  # Loosest first, so ties keep more positives
    dropped = (pass_needed[None, :] > candidates[:, None]).mean(axis=1)
    return float(candidates[np.argmin(candidates + mu * dropped)])


def expected_recall(pass_needed: np.ndarray, pass_rate: float) -> float:
    """
    The share of new positives kept at a pass rate that was picked from these positives' pass rates needed, e.g. by
    exchange_rule. The share of these positives kept is optimistic: the picked pass rate is always exactly one of
    theirs, so that positive is kept by construction. For a new positive, the chance of needing more is
    (d + 1) / (n + 1) on average, with d of the n dropped. In a nested simulation over all cases, this came within half
    a point of the realized mean recall, vs. 2 points for the share kept.
    """
    if pass_rate >= 1:
        return 1.0
    return float(1 - ((pass_needed > pass_rate).sum() + 1) / (len(pass_needed) + 1))


def choose_pass_rate(cv: pd.DataFrame, mu: float = MU) -> dict[str, float]:
    """
    cv: one setup's train_cv scores, with a label column. Returns:
        pass_rate: exchange_rule's pick, from the positives' held-out pass rates needed
        overshoot: the fold models' pass rate on their held-out negatives, when each one's threshold is set to let
            through pass_rate of its own training negatives, relative to pass_rate. Models score the negatives they
            trained on a little lower than unseen ones, so this is usually above 1
        in_sample_pass_rate: pass_rate / overshoot, what the full model should let through of its own training
            negatives to let through pass_rate of unseen ones. In a nested simulation over all cases, this landed at
            0.99x pass_rate on average (0.89-1.09x per case), vs. 1.12x without the correction
        n_pos, n_neg
        cv_dropped: positives whose held-out pass rate needed is above pass_rate
        expected_recall: see expected_recall
    """
    positives, negatives = cv[cv.label == 1], cv[cv.label == 0]
    pass_rate = exchange_rule(positives.pass_rate_needed.values, mu)
    overshoot = 1.0
    if 0 < pass_rate < 1:
        through = (negatives.in_sample_pass_rate_needed <= pass_rate).mean()
        if through > 0:
            overshoot = float(through / pass_rate)
        else:
            logger.warning(
                f"No held-out negatives get through at an in-sample pass rate of {pass_rate}, not correcting"
            )
    return {
        "pass_rate": pass_rate,
        "overshoot": overshoot,
        "in_sample_pass_rate": min(pass_rate / overshoot, 1.0),
        "n_pos": len(positives),
        "n_neg": len(negatives),
        "cv_dropped": int((positives.pass_rate_needed > pass_rate).sum()),
        "expected_recall": expected_recall(positives.pass_rate_needed.values, pass_rate),
    }


def fit_full(
    ngrams: list[list[str]], labels: np.ndarray, setup: Setup, in_sample_pass_rate: float
) -> tuple[Vectorizer, Any, float]:
    """
    One vectorizer and model on every sentence, plus the score threshold that lets through in_sample_pass_rate of the
    model's own training negatives
    """
    vectorizer = make_vectorizer()
    X = vectorizer.fit_transform(ngrams)
    logger.info(f"Vectorized shape: {X.shape}")
    model = fit_setup(setup, X, labels)
    threshold = threshold_for_pass_rate(model.predict_proba(X[labels == 0])[:, 1], in_sample_pass_rate)
    return vectorizer, model, threshold


def make_case_dataset(case_id, positives, negatives):
    # TODO try including topical points in the positives, or at least exclude from the negatives?

    positives = positives[case_id].copy()
    positives["label"] = 1

    to_drop = []
    for i, sent in negatives.iterrows():
        if "offlimits_cases" in sent and case_id in sent["offlimits_cases"]:
            to_drop.append(i)
    logger.info(f"Dropping {len(to_drop)} negative sentences due to points")
    negatives = negatives.drop(to_drop, axis=0)
    negatives["label"] = 0

    dataset = pd.concat([positives, negatives])
    logger.info(f"Case {case_id}: {len(positives)} positives, {len(negatives)} negatives")

    return dataset.reset_index(drop=True)


def prep_datasets(
    documents, points, services, case_ids, split="train", num_negative_docs=600, sents_per_doc=18
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """
    :param split: make_classification_datasets.service_split's "train" or "test"
    :return: make dataframes that can later be refined into datasets of sentences for any Case:
        1) dict from case_id to df of positive instances
        2) a huge dataframe of sentences that can be used as negative instances, with a `offlimits_cases` field
            so I can filter out sentences when they were positive points
    """
    # Points are assigned via their document's service, as in make_classification_datasets
    doc_split = documents.service_id.map(make_classification_datasets.service_split)
    documents = documents[doc_split == split]
    points = points[points.document_id.map(doc_split) == split]

    documents = documents[documents.lang == "en"]
    points = points[points.lang == "en"].copy()
    points = points[points.quote_start.notna()]

    # Join service info onto documents (we'll use is_comprehensively_reviewed), attach num points
    documents = pd.merge(documents, services, left_on="service_id", right_index=True, suffixes=["_doc", "_service"])
    points["document_id"] = points.document_id.astype(np.int64)

    # Filter out docs without enough points
    point_counts = points.document_id.value_counts()
    documents.loc[point_counts.index, "num_points"] = point_counts
    # documents = documents[documents.index.isin(points.document_id)]
    documents = documents[(documents.is_comprehensively_reviewed) & (documents.num_points >= 8)]

    # Clean up html, which is necessary for good sentence splitting. This should be done for inference as well.
    documents["text"] = documents.text.apply(utils.preprocess_doc_text)

    logger.info("Loading spacy model")
    spacy_model = spacy.load("en_core_web_md", disable=["attribute_ruler", "lemmatizer", "ner"])

    approved_points = points[points.status == "approved"]
    positives = dict()
    for case_id in case_ids:
        approved = approved_points[approved_points.case_id == case_id].copy()
        approved["text"] = approved.quote_text.apply(utils.preprocess_doc_text)
        approved["point_id"] = approved["id"]
        positives[case_id] = approved[["point_id", "case_id", "quote_start", "quote_end", "document_id", "text"]]

    # One split alone can have fewer eligible docs than num_negative_docs
    num_negative_docs = min(num_negative_docs, len(documents))
    # Its own generator, so each split's sentences don't depend on whether the other was prepped first
    rng = random.Random(RANDOM_STATE)
    negatives = []
    for i, (doc_id, doc) in tqdm(
        enumerate(documents.sample(num_negative_docs, random_state=RANDOM_STATE).iterrows()), total=num_negative_docs
    ):
        spacy_doc = spacy_model(doc.text)
        doc_sents = list(spacy_doc.sents)

        # Exclude any existing points in case they are positives
        # TODO if slow, add index document_id
        doc_points = points[(points.document_id == doc_id)]
        # Start/end positions for points, grouped by case for fast lookup
        case_offlimits: dict[int, list[tuple[int, int]]] = dict()
        for case_id in doc_points.case_id.unique():
            case_points = doc_points[doc_points.case_id == case_id]
            case_offlimits[case_id] = [(int(s), int(e)) for s, e in zip(case_points.quote_start, case_points.quote_end)]

        rng.shuffle(doc_sents)
        for sent in doc_sents[:sents_per_doc]:
            if sent.text != "" and not sent.text.isspace():
                offlimits_cases = []
                for case_id, boundaries in case_offlimits.items():
                    for start, end in boundaries:
                        if start < sent.end_char and end > sent.start_char:
                            offlimits_cases.append(case_id)
                negatives.append(
                    {
                        "document_id": doc_id,
                        "quote_start": sent.start_char,
                        "quote_end": sent.end_char,
                        "text": sent.text,
                        "offlimits_cases": offlimits_cases,
                    }
                )

    pos_lens = [len(pos) for pos in positives.values()]
    logger.info(f"Positive lengths: {pd.Series(pos_lens).describe()}")

    return positives, pd.DataFrame(negatives)


def extract_ngrams(case_dataset: pd.DataFrame) -> list[list[str]]:
    # Process text into ngrams
    logger.info("Loading spacy model and extracting ngrams")
    # Disable all components except tokenizer for faster ngram extraction
    nlp = spacy.load("en_core_web_md", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    ngrams = [
        [span.text.lower() for span in extract.ngrams(nlp(text), n=[1, 2])]
        for text in tqdm(case_dataset["text"], total=len(case_dataset))
    ]

    s = "\n".join(map(str, ngrams[:5]))
    logger.info(f"ngrams preview: {s}")
    s = "\n".join(map(str, [ngrams[i] for i in np.where(case_dataset.label.values)[0][:5]]))
    logger.info(f"ngrams preview (pos only): {s}")

    return ngrams


def run(case_ids: list[int] | None = None, skip_existing=False, compare_setups=False):
    # Same cases the deep learning models are trained on (see sent_spans/train.py)
    all_case_ids = sorted(make_classification_datasets.load_sent_span().keys())
    if case_ids is None:
        case_ids = all_case_ids
    elif unknown := set(case_ids) - set(all_case_ids):
        raise ValueError(f"No sent_span datasets for cases {sorted(unknown)}")

    documents = pickle.load(open(here / f"../data/db_dumps/{DB_DUMP_VERSION}/documents_clean.pkl", "rb"))
    points = pickle.load(open(here / f"../data/db_dumps/{DB_DUMP_VERSION}/points_clean.pkl", "rb"))
    services = pickle.load(open(here / f"../data/db_dumps/{DB_DUMP_VERSION}/services_clean.pkl", "rb"))

    # Try to load cached prep_datasets results, which cover all cases even when only some are run
    cache_dir = Path(here / "../data/tfidf")
    cache_dir.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for split in ["train", "test"]:
        prep_cache = cache_dir / f"{DB_DUMP_VERSION}_{MODEL_VERSION}_{split}_datasets_cache.pkl"
        try:
            datasets[split] = pickle.load(open(prep_cache, "rb"))
            logger.info(f"Loaded {split} prep_datasets from cache")
            missing = set(all_case_ids) - set(datasets[split][0].keys())
            if missing:
                logger.info(f"Cache is missing cases {sorted(missing)}, rebuilding")
                raise FileNotFoundError
        except FileNotFoundError:
            logger.info(f"Running {split} prep_datasets (this may take a while)")
            datasets[split] = prep_datasets(documents, points, services, all_case_ids, split)
            pickle.dump(datasets[split], open(prep_cache, "wb"))
    positives, negatives = datasets["train"]

    shipped_setup = next(setup for setup in gen_setups() if setup["name"] == SHIPPED_SETUP)
    setups = list(gen_setups()) if compare_setups else [shipped_setup]
    for case_id in case_ids:
        out_dir = model_output_dir(case_id)
        if skip_existing and (out_dir / "final_metrics.pkl").exists():
            logger.info(f"Skipping case {case_id}: {out_dir} already exists")
            continue
        logger.info(f"Processing case {case_id}")

        # Create dataset for this case, shuffled
        case_dataset: pd.DataFrame = make_case_dataset(case_id, positives, negatives)
        case_dataset = case_dataset.sample(n=len(case_dataset), random_state=RANDOM_STATE).reset_index(drop=True)
        # For grouping CV folds (see train_cv)
        case_dataset["service_id"] = case_dataset.document_id.map(documents.service_id)
        ngrams = extract_ngrams(case_dataset)
        labels = case_dataset.label.values

        folds, scores = train_cv(ngrams, labels, case_dataset.service_id.values, setups)
        if compare_setups:
            logger.info(
                f"Positives dropped at a given held-out pass rate, case {case_id}:\n{summarize_cliff(scores, labels)}"
            )
            # The notebook only plots positives
            setup_scores = scores[labels[scores.row] == 1].merge(
                case_dataset[["point_id"]].assign(fold=folds), left_on="row", right_index=True
            )
            setups_output_path(case_id).parent.mkdir(parents=True, exist_ok=True)
            setup_scores.to_pickle(setups_output_path(case_id))

        # Text is left out to keep this small: the notebook joins positives' text from points_clean by point_id
        cv = (
            case_dataset[["label", "point_id", "document_id", "service_id"]]
            .assign(fold=folds)
            .join(scores[scores.setup == SHIPPED_SETUP].drop(columns="setup").set_index("row"))
        )
        final_metrics = choose_pass_rate(cv)
        vectorizer, model, threshold = fit_full(ngrams, labels, shipped_setup, final_metrics["in_sample_pass_rate"])
        final_metrics |= {"threshold": threshold, "setup": SHIPPED_SETUP, "mu": MU}
        logger.info(
            f"Case {case_id}: target pass rate {final_metrics['pass_rate']:.2%}, "
            f"{final_metrics['cv_dropped']} of {final_metrics['n_pos']} positives dropped in CV, "
            f"expected recall {final_metrics['expected_recall']:.1%}, overshoot {final_metrics['overshoot']:.2f}, "
            f"threshold {threshold:.4f}"
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(vectorizer, open(out_dir / "vectorizer.pkl", "wb"))
        pickle.dump(model, open(out_dir / "model.pkl", "wb"))
        pickle.dump(final_metrics, open(out_dir / "final_metrics.pkl", "wb"))
        cv.to_pickle(out_dir / "cv.pkl")

        # For information only (see the docstring at the top)
        test_dataset = make_case_dataset(case_id, *datasets["test"])
        test_dataset["service_id"] = test_dataset.document_id.map(documents.service_id)
        test_scores = model.predict_proba(vectorizer.transform(extract_ngrams(test_dataset)))[:, 1]
        test = test_dataset[["label", "point_id", "document_id", "service_id"]].assign(score=test_scores)
        test.to_pickle(out_dir / "test.pkl")
        test_through = test.score >= threshold
        logger.info(
            f"Case {case_id} on the test split: pass rate {test_through[test.label == 0].mean():.2%}, "
            f"recall {test_through[test.label == 1].mean():.1%} of {(test.label == 1).sum()} positives"
        )
        logger.info(f"Saved case {case_id} to {out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a TF-IDF prefilter model for each case")
    parser.add_argument("--cases", type=int, nargs="+", help="Case ids to run (default: all)")
    parser.add_argument(
        "--skip_existing", action="store_true", help=f"Skip cases already in data/models/{MODEL_VERSION}"
    )
    parser.add_argument(
        "--compare_setups",
        action="store_true",
        help=f"Also cross-validate the rest of gen_setups for the notebook (to data/tfidf/{MODEL_VERSION}/setups)",
    )
    args = parser.parse_args()
    run(case_ids=args.cases, skip_existing=args.skip_existing, compare_setups=args.compare_setups)
