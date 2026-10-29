import json
import logging
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd

from src import inference, make_classification_datasets
from src.sent_spans import RESULTS_S3_BUCKET, train_push

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

"""
Selects decision thresholds for the sent-span case models from their test-split doc predictions (train.py output). The
model that scored the test split is the one that ships, so thresholds are chosen on the exact weights production runs.

For each case we compute a menu of operating points: thresholds targeting several precision levels, plus the
F-beta(1.5) optimum that production (apply_docbot.py) uses. Precision is prevalence-corrected so it estimates
precision against the production corpus of all documents rather than the constructed test set:
    P(t) = r(t)*pi / (r(t)*pi + fpr(t)*(1-pi))
where r(t) is recall on positive docs, fpr(t) is the false positive rate on production negatives, and pi is the
per-case prevalence estimated on the fly by compute_case_prevalence (below). fpr(t) is stratified by negative source
(production_fpr): `reviewed` negatives stand in for the production negatives no curator has flagged for the case, and
`declined` negatives for the share of them that a curator considered and declined, weighted by that share among
comprehensively reviewed docs. Declined docs are the hardest negatives and are enriched in the test set, so pooling them
unweighted would overstate production FPR, while leaving them out understates it.

Picking the lowest threshold whose estimated precision meets a target, on the same docs that precision is estimated
from, favors thresholds that got lucky. At high targets the estimate also rests on zero or one negatives above the
threshold, out of ~140, which says little about the true FPR. So precision targets are checked against a lower credible
bound instead (precision_lower_bound): a target counts as met only where the bound clears it, at confidence CONFIDENCE.
The F-beta operating point is the optimum of the point-estimate curve, with its bound reported alongside.

Labels come from the current doc datasets rather than the ones the model trained with (load_test_predictions), so after
test-split labels are audited (audit_comprehensive_negatives.py) and the datasets rebuilt, thresholds can be recomputed
without retraining. make_classification_datasets.service_split keeps the scored test docs in the test split.

Results are written as thresholds.json into each case's model version directory
(data/models/{model_version}/{case_id}/) — the same layout inference.get_threshold() loads from in production.
"""

TARGET_PRECISIONS = [0.7, 0.8, 0.9, 0.95]
FSCORE_BETA = 1.5
# One-sided confidence that a precision target's threshold really meets the target
CONFIDENCE = 0.8
# Posterior draws per threshold in precision_lower_bound
N_DRAWS = 4000
# Beta(PRIOR, PRIOR) is the Jeffreys prior for recall and each FPR stratum
PRIOR = 0.5


def _count_at_or_above(scores, thresholds) -> np.ndarray:
    """Number of `scores` at or above each threshold"""
    return len(scores) - np.searchsorted(np.sort(scores), thresholds, side="left")


def _frac_at_or_above(scores, thresholds):
    """Fraction of `scores` at or above each threshold (0 where there are no scores)"""
    if len(scores) == 0:
        return np.zeros(np.shape(thresholds))
    return _count_at_or_above(scores, thresholds) / len(scores)


def production_fpr(thresholds, reviewed_neg, declined_neg, declined_weight: float):
    """
    False positive rate against production negatives, stratified by negative source (see module docstring). Falls
    back to reviewed negatives alone when there are no declined negatives to estimate their stratum from.
    :param thresholds: a threshold or an array of them
    :param declined_weight: the declined stratum's share of production negatives (compute_case_prevalence)
    """
    fpr = _frac_at_or_above(reviewed_neg, thresholds)
    if len(declined_neg) == 0 or declined_weight == 0:
        return fpr
    return (1 - declined_weight) * fpr + declined_weight * _frac_at_or_above(declined_neg, thresholds)


def corrected_pr_curve(
    pos_scores, reviewed_neg, prevalence: float, declined_neg=None, declined_weight: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prevalence-corrected precision-recall curve.
    :param pos_scores: prediction scores of positive docs (recall comes only from these)
    :param reviewed_neg: prediction scores of `reviewed`-source negatives
    :param prevalence: estimated fraction of production docs that are positive for the case
    :param declined_neg: prediction scores of `declined`-source negatives, weighted by `declined_weight` in the FPR.
        Omitted, the FPR comes from reviewed negatives alone.
    :return: (thresholds ascending, recall, corrected precision) as np arrays; precision is nan where no doc scores
        at or above the threshold
    """
    declined_neg = np.array([]) if declined_neg is None else declined_neg
    thresholds = np.unique(np.concatenate([pos_scores, reviewed_neg, declined_neg]))
    recall = _frac_at_or_above(pos_scores, thresholds)
    fpr = production_fpr(thresholds, reviewed_neg, declined_neg, declined_weight)

    denom = recall * prevalence + fpr * (1 - prevalence)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(denom > 0, recall * prevalence / denom, np.nan)
    return thresholds, recall, precision


def _beta_draws(rng: np.random.Generator, k: np.ndarray, n: int, n_draws: int) -> np.ndarray:
    """Draws from the Jeffreys posterior of a rate observed as k out of n, for each element of k.
    :return: array of shape (n_draws, len(k))"""
    return rng.beta(k + PRIOR, n - k + PRIOR, size=(n_draws, len(k)))


def precision_lower_bound(
    thresholds,
    pos_scores,
    reviewed_neg,
    prevalence: float,
    declined_neg=None,
    declined_weight: float = 0.0,
    confidence: float = CONFIDENCE,
    n_draws: int = N_DRAWS,
    seed: int = 0,
) -> np.ndarray:
    """
    Lower credible bound on prevalence-corrected precision at each threshold. Recall and each negative stratum's FPR get
    independent Jeffreys posteriors from their counts at or above the threshold. Each posterior draw goes through the
    same formula as corrected_pr_curve, and the bound is the draws' (1 - confidence) quantile.

    Unlike a bootstrap, this stays uncertain where no negatives score above a threshold: 0 of 140 reviewed negatives is
    still consistent with an FPR around a percent, and at low prevalence that costs a lot of precision. Sparse strata
    (e.g. a handful of declined negatives) widen the bound in proportion to their weight.
    :param thresholds: a threshold or an array of them
    :return: the bound at each threshold
    """
    thresholds = np.atleast_1d(thresholds)
    declined_neg = np.array([]) if declined_neg is None else declined_neg
    rng = np.random.default_rng(seed)
    recall = _beta_draws(rng, _count_at_or_above(pos_scores, thresholds), len(pos_scores), n_draws)
    fpr = _beta_draws(rng, _count_at_or_above(reviewed_neg, thresholds), len(reviewed_neg), n_draws)
    if len(declined_neg) > 0 and declined_weight > 0:
        declined_fpr = _beta_draws(rng, _count_at_or_above(declined_neg, thresholds), len(declined_neg), n_draws)
        fpr = (1 - declined_weight) * fpr + declined_weight * declined_fpr
    # Beta draws are strictly positive, so the denominator is too
    precision = recall * prevalence / (recall * prevalence + fpr * (1 - prevalence))
    return np.quantile(precision, 1 - confidence, axis=0)


def _index_for_precision(recall, precision_lb, target: float) -> int | None:
    """The minimum threshold (i.e. maximum recall) whose precision lower bound meets the target, or None"""
    ok = (precision_lb >= target) & (recall > 0)
    if not ok.any():
        return None
    return int(np.argmax(ok))


def _index_for_fbeta(recall, precision, beta: float) -> int | None:
    """The threshold maximizing F-beta of corrected precision and recall, or None if the curve is degenerate"""
    precision = np.nan_to_num(precision)
    denom = np.square(beta) * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        fscore = np.where(denom > 0, (1 + np.square(beta)) * precision * recall / denom, 0.0)
    if fscore.max() <= 0:
        return None
    return int(np.argmax(fscore))


def _test_scores(test_doc_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Splits a case's test-split doc scores into positives, reviewed- and declined-source negatives, and all
    negatives"""
    negatives = test_doc_df[test_doc_df.label == "negative"]
    return {
        "pos": test_doc_df[test_doc_df.label == "positive"].pred_score.to_numpy(),
        "reviewed_neg": negatives[negatives.source == "reviewed"].pred_score.to_numpy(),
        "declined_neg": negatives[negatives.source == "declined"].pred_score.to_numpy(),
        "all_neg": negatives.pred_score.to_numpy(),
    }


def _operating_point(
    name: str, i: int | None, curve: tuple, scores: dict, target_precision: float | None = None
) -> dict[str, Any]:
    """
    An operating point's entry in thresholds.json, at index `i` of `curve` (None: not attainable)
    :param curve: (thresholds, recall, corrected precision, precision lower bound), aligned arrays
    """
    op: dict[str, Any] = {"name": name}
    if target_precision is not None:
        op["target_precision"] = target_precision
    if i is None:
        op.update({"attainable": False, "threshold": None})
        return op

    thresholds, recall, precision, precision_lb = curve
    threshold = float(thresholds[i])
    n_pos_above = int((scores["pos"] >= threshold).sum())
    n_neg_above = int((scores["all_neg"] >= threshold).sum())
    op.update(
        {
            "attainable": True,
            "threshold": threshold,
            "recall": float(recall[i]),
            "corrected_precision": float(precision[i]),
            "precision_lower_bound": float(precision_lb[i]),
            # Over all test negatives, uncorrected for prevalence
            "raw_precision": n_pos_above / (n_pos_above + n_neg_above),
            "n_pos_above": n_pos_above,
            "n_reviewed_neg_above": int((scores["reviewed_neg"] >= threshold).sum()),
            "n_declined_neg_above": int((scores["declined_neg"] >= threshold).sum()),
        }
    )
    return op


def compute_case_thresholds(test_doc_df: pd.DataFrame, prevalence_entry: dict, case_id: int) -> dict:
    """
    :param test_doc_df: the case's test-split docs with `pred_score` (load_test_predictions)
    :param prevalence_entry: this case's entry from compute_case_prevalence()
    :return: the thresholds.json artifact contents (see module docstring)
    """
    prevalence, declined_weight = prevalence_entry["prevalence"], prevalence_entry["declined_weight"]
    scored = test_doc_df[test_doc_df.pred_score.notna()]
    if len(scored) < len(test_doc_df):
        logger.warning(f"Case {case_id}: {len(test_doc_df) - len(scored)} test docs have no pred_score")
    scores = _test_scores(scored)

    targets = {f"precision{target * 100:.0f}": target for target in TARGET_PRECISIONS}
    if len(scores["pos"]) == 0 or len(scores["reviewed_neg"]) == 0:
        logger.warning(
            f"Case {case_id}: no test positives or no reviewed negatives, so no operating point is attainable"
        )
        operating_points = [_operating_point("fbeta15", None, (), scores)] + [
            _operating_point(name, None, (), scores, target_precision=target) for name, target in targets.items()
        ]
    else:
        args = (scores["pos"], scores["reviewed_neg"], prevalence, scores["declined_neg"], declined_weight)
        thresholds, recall, precision = corrected_pr_curve(*args)
        precision_lb = precision_lower_bound(thresholds, *args)
        curve = (thresholds, recall, precision, precision_lb)
        operating_points = [
            _operating_point("fbeta15", _index_for_fbeta(recall, precision, FSCORE_BETA), curve, scores)
        ]
        for name, target in targets.items():
            i = _index_for_precision(recall, precision_lb, target)
            operating_points.append(_operating_point(name, i, curve, scores, target_precision=target))

    return {
        "case_id": int(case_id),
        "prevalence": prevalence,
        "n_reviewed": prevalence_entry["n_reviewed"],
        "n_positive": prevalence_entry["n_positive"],
        "n_declined": prevalence_entry["n_declined"],
        "declined_weight": declined_weight,
        "confidence": CONFIDENCE,
        "n_test_pos": len(scores["pos"]),
        "n_test_reviewed_neg": len(scores["reviewed_neg"]),
        "n_test_declined_neg": len(scores["declined_neg"]),
        "dataset_version": make_classification_datasets.LATEST_VERSION,
        "operating_points": operating_points,
    }


def compute_case_prevalence(documents, points, comprehensive_threshold: int, en_only=True) -> dict[int, dict]:
    """
    Estimates per-case production prevalence: among comprehensively reviewed docs (same criteria as
    make_classification_datasets.make_doc_datasets), the fraction that have an approved point for the case. Used to
    convert test-set precision into an estimate of precision against the production corpus of all documents.
    Computed on the fly from the cleaned DB dumps rather than serialized, so it can't go stale.
    Also estimates the declined stratum's share of production negatives (declined_weight for production_fpr): among
    comprehensively reviewed docs without an approved point for the case, the fraction with a declined one.
    Caveat: comprehensively reviewed docs skew toward popular services, so these are proxies, not corrected for that
    sampling bias. They also count train-split docs, whose labels get less auditing than the test split's, so missed
    points there bias prevalence low.

    :return: dict from case id to
        {'prevalence': float, 'n_reviewed': int, 'n_positive': int, 'n_declined': int, 'declined_weight': float}
    """
    if en_only:
        documents = documents[documents.lang == "en"].copy()
        points = points[points.lang == "en"].copy()

    approved_points = points[points.status == "approved"]
    declined_points = points[points.status == "declined"]
    point_counts = approved_points.document_id.value_counts()
    documents.loc[point_counts.index, "num_approved"] = point_counts
    reviewed_doc_ids = set(
        documents[documents.is_comprehensively_reviewed & (documents.num_approved >= comprehensive_threshold)].index
    )

    prevalence = dict()
    # Select cases with the same quoted-approved count the dataset builders use, so prevalence covers exactly the
    # modeled cases. The prevalence numerator below still uses all approved points — a doc counts as positive if it
    # has any approved point for the case, quoted or not.
    case_num_approved = make_classification_datasets.quoted_approved_case_counts(points)
    for case_id in case_num_approved[case_num_approved >= make_classification_datasets.MIN_APPROVED].index:
        case_doc_ids = set(approved_points[approved_points.case_id == case_id].document_id)
        n_positive = len(case_doc_ids & reviewed_doc_ids)
        declined_doc_ids = set(declined_points[declined_points.case_id == case_id].document_id) - case_doc_ids
        n_declined = len(declined_doc_ids & reviewed_doc_ids)
        prevalence[int(case_id)] = {
            "prevalence": n_positive / len(reviewed_doc_ids),
            "n_reviewed": len(reviewed_doc_ids),
            "n_positive": n_positive,
            "n_declined": n_declined,
            "declined_weight": n_declined / (len(reviewed_doc_ids) - n_positive),
        }
    return prevalence


def load_test_predictions(model_version: str) -> dict[int, pd.DataFrame]:
    """
    Test-split doc scores by case (train.py output), with labels and sources from the current doc datasets rather than
    the ones the model trained with, so label fixes from auditing take effect without retraining. Current test docs
    the model never scored (e.g. new in a later DB dump) are left out, with a warning.
    :return: dict from case id to the case's current test docs [id_doc, id_service, label, source, pred_score]
    """
    doc_pred_path = here / f"../../data/training/{model_version}/doc_pred.pkl"
    logger.info(f"Loading doc predictions from {doc_pred_path}")
    with open(doc_pred_path, "rb") as f:
        doc_results: dict[int, pd.DataFrame] = pickle.load(f)
    scores = {
        case_id: df.loc[df.split == "test", ["id_doc", "label", "pred_score"]] for case_id, df in doc_results.items()
    }
    del doc_results  # the doc text makes this several GB

    logger.info("Loading the current doc datasets for labels")
    current = make_classification_datasets.load_docs()
    test_preds = dict()
    for case_id, case_scores in scores.items():
        if case_id not in current:
            logger.warning(f"Case {case_id} is no longer in the doc datasets, skipping")
            continue
        test_docs = current[case_id]
        test_docs = test_docs.loc[test_docs.split == "test", ["id_doc", "id_service", "label", "source"]]
        merged = test_docs.merge(
            case_scores.rename(columns={"label": "trained_label"}), on="id_doc", how="left", indicator=True
        )
        unscored = merged._merge == "left_only"
        if unscored.any():
            logger.warning(f"Case {case_id}: {unscored.sum()} current test docs were never scored, leaving them out")
        n_dropped = len(set(case_scores.id_doc) - set(test_docs.id_doc))
        if n_dropped:
            logger.info(f"Case {case_id}: {n_dropped} scored test docs are no longer in the dataset")
        merged = merged[~unscored]
        n_relabeled = (merged.label != merged.trained_label).sum()
        if n_relabeled:
            logger.info(f"Case {case_id}: {n_relabeled} test docs relabeled since training")
        test_preds[case_id] = merged.drop(columns=["trained_label", "_merge"])
    return test_preds


def run(model_version: str, upload: bool = False):
    test_preds = load_test_predictions(model_version)

    logger.info("Computing case prevalence from the cleaned DB dumps")
    _, documents, points = make_classification_datasets.load_clean_dumps()
    prevalence = compute_case_prevalence(documents, points, make_classification_datasets.COMPREHENSIVE_THRESHOLD)

    s3_client = boto3.client("s3", region_name=train_push.AWS_REGION) if upload else None

    for case_id, test_doc_df in test_preds.items():
        if case_id not in prevalence:
            logger.warning(f"Case {case_id} has no prevalence estimate, skipping")
            continue
        artifact = compute_case_thresholds(test_doc_df, prevalence[case_id], case_id)
        artifact["model_version"] = model_version

        out_dir = inference.model_dir(case_id, model_version)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "thresholds.json"
        logger.info(f"Writing {out_path}")
        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=2)

        if s3_client is not None:
            s3_key = f"{model_version}/{case_id}/thresholds.json"
            logger.info(f"Uploading to s3://{RESULTS_S3_BUCKET}/{s3_key}")
            s3_client.upload_file(
                out_path.as_posix(), RESULTS_S3_BUCKET, s3_key, ExtraArgs={"ContentType": "application/json"}
            )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compute per-case threshold menus (thresholds.json) from test-split predictions"
    )
    parser.add_argument(
        "--model_version",
        type=str,
        required=True,
        help="Model version being prepared (e.g. v4): reads data/training/{model_version}/doc_pred.pkl and "
        "writes thresholds.json to data/models/{model_version}/{case_id}/, the layout production inference reads",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Also upload each thresholds.json to S3 at {model_version}/{case_id}/, next to the adapter from "
        "train.py --upload. Once inference.MODEL_VERSION is {model_version}, this is what makes production serve "
        "the case",
    )
    args = parser.parse_args()
    run(args.model_version, args.upload)
