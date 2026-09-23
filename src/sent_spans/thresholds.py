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
Selects decision thresholds for the sent-span case models from cross-validation results (the out-of-fold doc-level
predictions produced by train.py with all folds). This replaces the notebook-based F-beta threshold picking that fed
inference.THRESHOLDS.

For each case we compute a menu of operating points: thresholds targeting several precision levels, plus the
F-beta(1.5) optimum that production (apply_docbot.py) uses. Precision is prevalence-corrected so it estimates
precision against the production corpus of all documents rather than the constructed eval set:
    P(t) = r(t)*pi / (r(t)*pi + fpr(t)*(1-pi))
where r(t) is recall on positive docs, fpr(t) is the false positive rate on `reviewed`-source negatives only
(declined negatives are enriched hard cases and would bias production FPR upward), and pi is the per-case prevalence
estimated on the fly by compute_case_prevalence (below).

Per-fold thresholds are aggregated in logit space because BERT softmax scores saturate near 0/1, making probability-
space statistics meaningless there. The cross-fold agreement (std of logits) doubles as a trust diagnostic: the
recommended threshold for the final full-data model is sigmoid(mean + CONSERVATIVE_K * std) when folds agree, and the
most conservative (max) fold threshold with `stable: false` flagged when they don't.

Results are written as thresholds.json into each case's model version directory
(data/models/{model_version}/{case_id}/) — the same layout inference.get_threshold() loads from in production, and
where the final model's adapter will land once trained (train.py --train_final) and pulled from S3.
"""

TARGET_PRECISIONS = [0.7, 0.8, 0.9, 0.95]
FSCORE_BETA = 1.5
# Conservative direction for precision targets is a *higher* threshold (opposite of tfidf.py's recall targets)
CONSERVATIVE_K = 0.5
# Initial stability gate on the std of per-fold threshold logits; to be recalibrated after the pilot runs
MAX_STD_LOGIT = 1.0
# An operating point must be attainable in all but one fold to be trusted
MAX_UNATTAINABLE_FOLDS = 1
# Fewer fold thresholds than this can't support a stability claim (std of 1-2 values means little)
MIN_FOLDS_FOR_STABILITY = 3

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def corrected_pr_curve(pos_scores, neg_scores, prevalence: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prevalence-corrected precision-recall curve.
    :param pos_scores: prediction scores of positive docs (recall comes only from these)
    :param neg_scores: prediction scores of negative docs representative of production negatives
        (use `reviewed`-source negatives, not the enriched `declined` ones)
    :param prevalence: estimated fraction of production docs that are positive for the case
    :return: (thresholds ascending, recall, corrected precision) as np arrays; precision is nan where no doc scores
        at or above the threshold
    """
    thresholds = np.unique(np.concatenate([pos_scores, neg_scores]))
    recall = np.array([(pos_scores >= t).mean() if len(pos_scores) else 0.0 for t in thresholds])
    fpr = np.array([(neg_scores >= t).mean() if len(neg_scores) else 0.0 for t in thresholds])

    denom = recall * prevalence + fpr * (1 - prevalence)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(denom > 0, recall * prevalence / denom, np.nan)
    return thresholds, recall, precision


def _threshold_for_precision(thresholds, recall, precision, target: float) -> float | None:
    """The minimum threshold (i.e. maximum recall) whose corrected precision meets the target, or None.
    Per-fold noise in these picks is handled by the cross-fold aggregation, not smoothed here."""
    ok = (precision >= target) & (recall > 0)
    if not ok.any():
        return None
    return float(thresholds[np.argmax(ok)])


def _threshold_for_fbeta(thresholds, recall, precision, beta: float) -> float | None:
    """The threshold maximizing F-beta of corrected precision and recall, or None if the curve is degenerate."""
    precision = np.nan_to_num(precision)
    denom = np.square(beta) * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        fscore = np.where(denom > 0, (1 + np.square(beta)) * precision * recall / denom, 0.0)
    if fscore.max() <= 0:
        return None
    return float(thresholds[np.argmax(fscore)])


def _metrics_at_threshold(threshold: float, fold_scores: list[dict], prevalence: float) -> dict:
    """Mean recall / corrected precision / raw (eval-set, all negatives) precision across folds at a fixed threshold"""
    recalls, corrected, raw = [], [], []
    for fold in fold_scores:
        pos, reviewed_neg, all_neg = fold["pos"], fold["reviewed_neg"], fold["all_neg"]
        if len(pos) == 0:
            continue
        r = (pos >= threshold).mean()
        recalls.append(r)
        if len(reviewed_neg) > 0:
            fpr = (reviewed_neg >= threshold).mean()
            denom = r * prevalence + fpr * (1 - prevalence)
            if denom > 0:
                corrected.append(r * prevalence / denom)
        tp = (pos >= threshold).sum()
        fp = (all_neg >= threshold).sum()
        if tp + fp > 0:
            raw.append(tp / (tp + fp))

    def _mean(values):
        return float(np.mean(values)) if values else None

    return {
        "achieved_recall_mean": _mean(recalls),
        "corrected_precision": _mean(corrected),
        "raw_precision": _mean(raw),
    }


def _aggregate_operating_point(
    name: str,
    per_fold_thresholds: list[float | None],
    fold_scores: list[dict],
    prevalence: float,
    target_precision: float | None = None,
) -> dict:
    valid = [t for t in per_fold_thresholds if t is not None]
    attainable = len(per_fold_thresholds) - len(valid) <= MAX_UNATTAINABLE_FOLDS and len(valid) > 0
    op: dict[str, Any] = {"name": name, "per_fold_thresholds": per_fold_thresholds, "attainable": attainable}
    if target_precision is not None:
        op["target_precision"] = target_precision
    if len(valid) == 0:
        op.update({"threshold": None, "mean_logit": None, "std_logit": None, "stable": False})
        return op

    logits = _logit(np.array(valid))
    mean_logit = float(np.mean(logits))
    std_logit = float(np.std(logits))
    stable = std_logit <= MAX_STD_LOGIT and len(valid) >= MIN_FOLDS_FOR_STABILITY
    if stable:
        threshold = float(_sigmoid(mean_logit + CONSERVATIVE_K * std_logit))
    else:
        # Folds disagree too much to trust an average; fall back to the most conservative fold threshold
        threshold = float(_sigmoid(logits.max()))

    op.update({"threshold": threshold, "mean_logit": mean_logit, "std_logit": std_logit, "stable": stable})
    op.update(_metrics_at_threshold(threshold, fold_scores, prevalence))
    return op


def compute_case_thresholds(doc_pred_df: pd.DataFrame, prevalence_entry: dict, case_id: int) -> dict:
    """
    :param doc_pred_df: the per-case doc dataset with out-of-fold `pred_score` for every fold (train.py output)
    :param prevalence_entry: this case's entry from compute_case_prevalence()
    :return: the thresholds.json artifact contents (see module docstring)
    """
    prevalence = prevalence_entry["prevalence"]
    df = doc_pred_df[doc_pred_df.pred_score.notna()]
    if len(df) < len(doc_pred_df):
        logger.warning(f"Case {case_id}: {len(doc_pred_df) - len(df)} docs are missing a pred_score")

    fold_scores: list[dict[str, Any]] = []
    for fold_i in sorted(df.fold.unique()):
        fold_df = df[df.fold == fold_i]
        negatives = fold_df[fold_df.label == "negative"]
        fold_scores.append(
            {
                "fold": int(fold_i),
                "pos": fold_df[fold_df.label == "positive"].pred_score.values,
                "reviewed_neg": negatives[negatives.source == "reviewed"].pred_score.values,
                "all_neg": negatives.pred_score.values,
            }
        )

    fold_curves: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
    for fold in fold_scores:
        if len(fold["pos"]) == 0 or len(fold["reviewed_neg"]) == 0:
            logger.warning(f"Case {case_id} fold {fold['fold']}: no positives or no reviewed negatives, skipping fold")
            fold_curves.append(None)
        else:
            fold_curves.append(corrected_pr_curve(fold["pos"], fold["reviewed_neg"], prevalence))

    operating_points = []
    fbeta_folds = [_threshold_for_fbeta(*curve, FSCORE_BETA) if curve is not None else None for curve in fold_curves]
    operating_points.append(_aggregate_operating_point("fbeta15", fbeta_folds, fold_scores, prevalence))
    for target in TARGET_PRECISIONS:
        per_fold = [_threshold_for_precision(*curve, target) if curve is not None else None for curve in fold_curves]
        operating_points.append(
            _aggregate_operating_point(
                f"precision{target * 100:.0f}", per_fold, fold_scores, prevalence, target_precision=target
            )
        )

    stable = all(op["stable"] for op in operating_points if op["attainable"])
    for op in operating_points:
        if op["attainable"] and not op["stable"]:
            logger.warning(
                f"Case {case_id} operating point {op['name']} has unstable fold thresholds "
                f"(std_logit {op['std_logit']:.2f} > {MAX_STD_LOGIT}); using most conservative fold threshold"
            )

    return {
        "case_id": int(case_id),
        "prevalence": prevalence,
        "n_reviewed": prevalence_entry["n_reviewed"],
        "n_positive": prevalence_entry["n_positive"],
        "stable": stable,
        "dataset_version": make_classification_datasets.LATEST_VERSION,
        "operating_points": operating_points,
    }


def compute_case_prevalence(documents, points, comprehensive_threshold: int, en_only=True) -> dict[int, dict]:
    """
    Estimates per-case production prevalence: among comprehensively reviewed docs (same criteria as
    make_classification_datasets.make_doc_datasets), the fraction that have an approved point for the case. Used to
    convert eval-set precision into an estimate of precision against the production corpus of all documents.
    Computed on the fly from the cleaned DB dumps rather than serialized, so it can't go stale.
    Caveat: comprehensively reviewed docs skew toward popular services, so this is a proxy, not corrected for that
    sampling bias.

    :return: dict from case id to {'prevalence': float, 'n_reviewed': int, 'n_positive': int}
    """
    if en_only:
        documents = documents[documents.lang == "en"].copy()
        points = points[points.lang == "en"].copy()

    approved_points = points[points.status == "approved"]
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
        prevalence[int(case_id)] = {
            "prevalence": n_positive / len(reviewed_doc_ids),
            "n_reviewed": len(reviewed_doc_ids),
            "n_positive": n_positive,
        }
    return prevalence


def run(model_version: str, upload: bool = False):
    doc_pred_path = here / f"../../data/training/{model_version}/cv/doc_pred.pkl"
    logger.info(f"Loading doc predictions from {doc_pred_path}")
    with open(doc_pred_path, "rb") as f:
        doc_results: dict[int, pd.DataFrame] = pickle.load(f)

    logger.info("Computing case prevalence from the cleaned DB dumps")
    _, documents, points = make_classification_datasets.load_clean_dumps()
    prevalence = compute_case_prevalence(documents, points, make_classification_datasets.COMPREHENSIVE_THRESHOLD)

    s3_client = boto3.client("s3", region_name=train_push.AWS_REGION) if upload else None

    for case_id, doc_pred_df in doc_results.items():
        if case_id not in prevalence:
            logger.warning(f"Case {case_id} has no prevalence estimate, skipping")
            continue
        artifact = compute_case_thresholds(doc_pred_df, prevalence[case_id], case_id)
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
        description="Compute per-case threshold menus (thresholds.json) from cross-validation doc predictions"
    )
    parser.add_argument(
        "--model_version",
        type=str,
        required=True,
        help="Model version being prepared (e.g. v4): reads data/training/{model_version}/cv/doc_pred.pkl and "
        "writes thresholds.json to data/models/{model_version}/{case_id}/, the layout production inference reads",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Also upload each thresholds.json to S3 at {model_version}/{case_id}/, the production path "
        "(train.py --train_final --upload re-uploads it there alongside the final adapter anyway)",
    )
    args = parser.parse_args()
    run(args.model_version, args.upload)
