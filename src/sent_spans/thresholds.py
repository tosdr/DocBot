import json
import logging
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
from scipy import stats

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
where r(t) is recall on positive docs, fpr(t) is the false positive rate on production negatives, and pi is the
per-case prevalence estimated on the fly by compute_case_prevalence (below). fpr(t) is stratified by negative source
(production_fpr): `reviewed` negatives stand in for the production negatives no curator has flagged for the case, and
`declined` negatives for the share of them that a curator considered and declined, weighted by that share among
comprehensively reviewed docs. Declined docs are the hardest negatives and are enriched in the eval set, so pooling them
unweighted would overstate production FPR, while leaving them out understates it. They also add resolution in the tail,
where few reviewed negatives score above precision-target thresholds.

Per-fold thresholds are aggregated in logit space because BERT softmax scores saturate near 0/1, making probability-
space statistics meaningless there. The cross-fold agreement (std of logits) doubles as a trust diagnostic: the
recommended threshold for the final full-data model is sigmoid(mean + CONSERVATIVE_K * std) when folds agree, and the
most conservative (max) fold threshold with `stable: false` flagged when they don't.

Results are written as thresholds.json into each case's model version directory
(data/models/{model_version}/{case_id}/) — the same layout inference.get_threshold() loads from in production, and
where the final model's adapter will land once trained (train.py --train_final) and pulled from S3.

With --calibrate, the aggregation rule itself is evaluated instead (leave_one_fold_out): thresholds aggregated from
4 folds are applied to the 5th fold's scores, which came from a model the thresholds never saw — the same situation
as the final model in production. Results land in data/training/{model_version}/threshold_calibration.parquet for
notebooks/v4_thresholds.py, to calibrate CONSERVATIVE_K / MAX_STD_LOGIT.
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


def _frac_at_or_above(scores, thresholds):
    """Fraction of `scores` at or above each threshold (0 where there are no scores)"""
    if len(scores) == 0:
        return np.zeros(np.shape(thresholds))
    return 1 - np.searchsorted(np.sort(scores), thresholds, side="left") / len(scores)


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


def _metrics_at_threshold(
    threshold: float, fold_scores: list[dict], prevalence: float, declined_weight: float = 0.0
) -> dict:
    """Mean recall / corrected precision / raw (eval-set, all negatives) precision across folds at a fixed threshold.
    With declined_weight 0 the corrected precision's FPR comes from reviewed negatives alone."""
    recalls, corrected, raw = [], [], []
    for fold in fold_scores:
        pos, reviewed_neg, all_neg = fold["pos"], fold["reviewed_neg"], fold["all_neg"]
        if len(pos) == 0:
            continue
        r = (pos >= threshold).mean()
        recalls.append(r)
        if len(reviewed_neg) > 0:
            fpr = float(production_fpr(threshold, reviewed_neg, fold["declined_neg"], declined_weight))
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


def _aggregate_logits(logits: np.ndarray) -> tuple[float, float, float, bool]:
    """The production aggregation rule over per-fold threshold logits.
    :return: (threshold logit, mean logit, std logit, stable)"""
    mean_logit = float(np.mean(logits))
    std_logit = float(np.std(logits))
    stable = std_logit <= MAX_STD_LOGIT and len(logits) >= MIN_FOLDS_FOR_STABILITY
    if stable:
        return mean_logit + CONSERVATIVE_K * std_logit, mean_logit, std_logit, stable
    # Folds disagree too much to trust an average; fall back to the most conservative fold threshold
    return float(logits.max()), mean_logit, std_logit, stable


def _aggregate_operating_point(
    name: str,
    per_fold_thresholds: list[float | None],
    fold_scores: list[dict],
    prevalence: float,
    declined_weight: float,
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

    threshold_logit, mean_logit, std_logit, stable = _aggregate_logits(_logit(np.array(valid)))
    threshold = float(_sigmoid(threshold_logit))

    op.update({"threshold": threshold, "mean_logit": mean_logit, "std_logit": std_logit, "stable": stable})
    op.update(_metrics_at_threshold(threshold, fold_scores, prevalence, declined_weight))
    return op


def _fold_scores(doc_pred_df: pd.DataFrame, case_id: int) -> list[dict[str, Any]]:
    """Splits out-of-fold doc scores by fold into positives, reviewed- and declined-source negatives, and all
    negatives"""
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
                "declined_neg": negatives[negatives.source == "declined"].pred_score.values,
                "all_neg": negatives.pred_score.values,
            }
        )
    return fold_scores


def _per_fold_operating_points(
    fold_scores: list[dict], prevalence: float, declined_weight: float, case_id: int
) -> list[tuple[str, float | None, list[float | None]]]:
    """
    Each operating point's threshold picked independently on every fold's prevalence-corrected PR curve.
    :return: list of (operating point name, target precision or None for F-beta, per-fold thresholds aligned with
        `fold_scores`, None where the fold can't support one)
    """
    fold_curves: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
    for fold in fold_scores:
        if len(fold["pos"]) == 0 or len(fold["reviewed_neg"]) == 0:
            logger.warning(f"Case {case_id} fold {fold['fold']}: no positives or no reviewed negatives, skipping fold")
            fold_curves.append(None)
        else:
            fold_curves.append(
                corrected_pr_curve(fold["pos"], fold["reviewed_neg"], prevalence, fold["declined_neg"], declined_weight)
            )

    fbeta_folds = [_threshold_for_fbeta(*curve, FSCORE_BETA) if curve is not None else None for curve in fold_curves]
    points: list[tuple[str, float | None, list[float | None]]] = [("fbeta15", None, fbeta_folds)]
    for target in TARGET_PRECISIONS:
        per_fold = [_threshold_for_precision(*curve, target) if curve is not None else None for curve in fold_curves]
        points.append((f"precision{target * 100:.0f}", target, per_fold))
    return points


def compute_case_thresholds(doc_pred_df: pd.DataFrame, prevalence_entry: dict, case_id: int) -> dict:
    """
    :param doc_pred_df: the per-case doc dataset with out-of-fold `pred_score` for every fold (train.py output)
    :param prevalence_entry: this case's entry from compute_case_prevalence()
    :return: the thresholds.json artifact contents (see module docstring)
    """
    prevalence, declined_weight = prevalence_entry["prevalence"], prevalence_entry["declined_weight"]
    fold_scores = _fold_scores(doc_pred_df, case_id)
    operating_points = [
        _aggregate_operating_point(name, per_fold, fold_scores, prevalence, declined_weight, target_precision=target)
        for name, target, per_fold in _per_fold_operating_points(fold_scores, prevalence, declined_weight, case_id)
    ]

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
        "n_declined": prevalence_entry["n_declined"],
        "declined_weight": declined_weight,
        "stable": stable,
        "dataset_version": make_classification_datasets.LATEST_VERSION,
        "operating_points": operating_points,
    }


def mean_k_rule(k: float):
    """mean + k * std (ddof=0, as in thresholds.json) with no stability gate"""

    def rule(logits: np.ndarray) -> float | None:
        return float(np.mean(logits) + k * np.std(logits))

    return rule


def t_interval_rule(q: float):
    """
    One-sided upper prediction bound at coverage q: if fold threshold logits are normal and the final model's ideal
    threshold is one more draw from the same distribution, it lands at or below this bound with probability q. The
    t quantile and sqrt(1 + 1/n) widen the bound when there are few folds to estimate the mean and std from.
    """

    def rule(logits: np.ndarray) -> float | None:
        n = len(logits)
        if n < 2:
            return None
        return float(np.mean(logits) + stats.t.ppf(q, n - 1) * np.std(logits, ddof=1) * np.sqrt(1 + 1 / n))

    return rule


# Candidate aggregation rules for leave_one_fold_out, from per-fold threshold logits to a threshold logit (or None
# when the rule can't produce one). "production" is the rule thresholds.json currently uses.
CALIBRATION_RULES = {
    "production": lambda logits: _aggregate_logits(logits)[0],
    "max": lambda logits: float(logits.max()),
    **{f"mean_k{k:g}": mean_k_rule(k) for k in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]},
    **{f"t_q{q:g}": t_interval_rule(q) for q in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]},
}


def leave_one_fold_out(doc_pred_df: pd.DataFrame, prevalence_entry: dict, case_id: int, rules=None) -> pd.DataFrame:
    """
    Evaluates threshold aggregation rules the way production uses them: per-fold thresholds from all folds but one
    are aggregated, and the result is applied to the held-out fold's scores. Those scores come from a model the
    aggregated thresholds never saw, which is the situation the final full-data model is in.

    One row per (operating point, held-out fold, rule), with:
    - n_folds, mean_logit, std_logit: the per-fold thresholds the rule aggregated (std ddof=0, as in thresholds.json)
    - held_out_ideal_logit: the held-out fold's own threshold for the operating point (nan where unattainable), and
      covered = threshold_logit >= held_out_ideal_logit, a direct coverage test for t_interval_rule's q
    - recall / corrected_precision / raw_precision realized on the held-out fold, and precision_shortfall
      (target - corrected precision; positive means the target was missed). corrected_precision is nan when nothing
      scores above the threshold
    - n_pos / n_reviewed_neg / n_declined_neg and the counts above threshold, to judge how much each realized precision
      can be trusted
    """
    rules = CALIBRATION_RULES if rules is None else rules
    prevalence, declined_weight = prevalence_entry["prevalence"], prevalence_entry["declined_weight"]
    fold_scores = _fold_scores(doc_pred_df, case_id)
    rows = []
    for name, target, per_fold in _per_fold_operating_points(fold_scores, prevalence, declined_weight, case_id):
        for i, held_out in enumerate(fold_scores):
            if len(held_out["pos"]) == 0 or len(held_out["reviewed_neg"]) == 0:
                continue
            others = [t for j, t in enumerate(per_fold) if j != i and t is not None]
            if not others:
                continue
            logits = _logit(np.array(others))
            ideal = per_fold[i]
            ideal_logit = float(_logit(ideal)) if ideal is not None else np.nan
            base = {
                "case_id": int(case_id),
                "operating_point": name,
                "target_precision": target,
                "held_out_fold": held_out["fold"],
                "n_folds": len(others),
                "mean_logit": float(np.mean(logits)),
                "std_logit": float(np.std(logits)),
                "held_out_ideal_logit": ideal_logit,
                "n_pos": len(held_out["pos"]),
                "n_reviewed_neg": len(held_out["reviewed_neg"]),
                "n_declined_neg": len(held_out["declined_neg"]),
            }
            for rule_name, rule in rules.items():
                threshold_logit = rule(logits)
                if threshold_logit is None:
                    continue
                threshold = float(_sigmoid(threshold_logit))
                metrics = _metrics_at_threshold(threshold, [held_out], prevalence, declined_weight)
                precision = metrics["corrected_precision"]
                rows.append(
                    {
                        **base,
                        "rule": rule_name,
                        "threshold_logit": threshold_logit,
                        "threshold": threshold,
                        "covered": threshold_logit >= ideal_logit if ideal is not None else np.nan,
                        "recall": metrics["achieved_recall_mean"],
                        "corrected_precision": precision if precision is not None else np.nan,
                        "raw_precision": metrics["raw_precision"] if metrics["raw_precision"] is not None else np.nan,
                        "precision_shortfall": target - precision
                        if target is not None and precision is not None
                        else np.nan,
                        "n_pos_above": int((held_out["pos"] >= threshold).sum()),
                        "n_reviewed_neg_above": int((held_out["reviewed_neg"] >= threshold).sum()),
                        "n_declined_neg_above": int((held_out["declined_neg"] >= threshold).sum()),
                    }
                )
    return pd.DataFrame(rows)


def compute_case_prevalence(documents, points, comprehensive_threshold: int, en_only=True) -> dict[int, dict]:
    """
    Estimates per-case production prevalence: among comprehensively reviewed docs (same criteria as
    make_classification_datasets.make_doc_datasets), the fraction that have an approved point for the case. Used to
    convert eval-set precision into an estimate of precision against the production corpus of all documents.
    Computed on the fly from the cleaned DB dumps rather than serialized, so it can't go stale.
    Also estimates the declined stratum's share of production negatives (declined_weight for production_fpr): among
    comprehensively reviewed docs without an approved point for the case, the fraction with a declined one.
    Caveat: comprehensively reviewed docs skew toward popular services, so these are proxies, not corrected for that
    sampling bias.

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


def _load_cv_results(model_version: str) -> tuple[dict[int, pd.DataFrame], dict[int, dict]]:
    """:return: out-of-fold doc predictions by case (train.py CV output) and compute_case_prevalence() entries"""
    doc_pred_path = here / f"../../data/training/{model_version}/cv/doc_pred.pkl"
    logger.info(f"Loading doc predictions from {doc_pred_path}")
    with open(doc_pred_path, "rb") as f:
        doc_results: dict[int, pd.DataFrame] = pickle.load(f)

    logger.info("Computing case prevalence from the cleaned DB dumps")
    _, documents, points = make_classification_datasets.load_clean_dumps()
    prevalence = compute_case_prevalence(documents, points, make_classification_datasets.COMPREHENSIVE_THRESHOLD)
    return doc_results, prevalence


def calibration_path(model_version: str) -> Path:
    return here / f"../../data/training/{model_version}/threshold_calibration.parquet"


def calibrate(model_version: str) -> pd.DataFrame:
    """Runs leave_one_fold_out over every case of a CV run and writes the combined rows to calibration_path()"""
    doc_results, prevalence = _load_cv_results(model_version)
    frames = []
    for case_id, doc_pred_df in doc_results.items():
        if case_id not in prevalence:
            logger.warning(f"Case {case_id} has no prevalence estimate, skipping")
            continue
        case_df = leave_one_fold_out(doc_pred_df, prevalence[case_id], case_id)
        case_df["prevalence"] = prevalence[case_id]["prevalence"]
        case_df["declined_weight"] = prevalence[case_id]["declined_weight"]
        frames.append(case_df)
    calibration = pd.concat(frames, ignore_index=True)

    out_path = calibration_path(model_version)
    logger.info(f"Writing {len(calibration)} rows to {out_path}")
    calibration.to_parquet(out_path)
    return calibration


def run(model_version: str, upload: bool = False):
    doc_results, prevalence = _load_cv_results(model_version)

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
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Instead of writing thresholds.json, evaluate candidate aggregation rules by leave-one-fold-out and "
        "write data/training/{model_version}/threshold_calibration.parquet (see notebooks/v4_thresholds.py)",
    )
    args = parser.parse_args()
    if args.calibrate and args.upload:
        parser.error("--upload has nothing to upload with --calibrate")

    if args.calibrate:
        calibrate(args.model_version)
    else:
        run(args.model_version, args.upload)
