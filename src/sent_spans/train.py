import contextlib
import io
import logging
import os
import pickle
import random
import shutil
import socket
import threading
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import boto3
import botocore
import numpy as np
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

import wandb
from src import aws_auth, inference, make_classification_datasets, utils
from src.sent_spans import RESULTS_S3_BUCKET, TEST_CASE_IDS, train_push, trainer_callbacks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

CLASSIFICATION_VERSION = make_classification_datasets.LATEST_VERSION
# Fraction of docs held out of each training pool purely for early stopping / model selection
STOPPING_FRACTION = 0.1

# Predictions attached to the doc dataset and persisted in results. pred_label is deliberately excluded: it comes
# from an oracle threshold fit on the eval set itself (inference.attach_predictions), fine for relative model
# selection during training but misleading as a stored prediction.
DOC_PRED_COLS = ["pred_score", "pred_char_start", "pred_char_end", "pred_num_sents", "prefilter_rate"]


@dataclass
class TrainConfig:
    """Everything finetune()/finetune_final() need to know about how to train, so run-shape concerns
    (cases, folds, uploads) stay in argparse and don't thread through every function."""

    model_base: str
    batch_size: int
    learning_rate: float
    eval_steps: int
    early_stopping_patience: int
    num_train_epochs: int
    attempts_per_fold: int
    eval_training_set: bool = False
    device: str = "cuda"
    log_wandb: bool = True
    training_overrides: dict = field(default_factory=dict)


def case_models_dir(case_id) -> Path:
    # Scratch area for a case's training: HF Trainer checkpoints plus the tmp_* best-model dirs below.
    # Removed after a successful upload (see cleanup_case_models). Lives under data/training/ because it's
    # training ephemera — data/models/ holds only deployable artifacts that inference reads
    return here / f"../../data/training/scratch/{case_id}"


def tmp_attempt_best_loc(case_id) -> Path:
    # Best checkpoint within the current attempt, according to doc F1 on the stopping slice (DocEvalCallback)
    dir_path = case_models_dir(case_id) / "tmp_attempt_best"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def tmp_best_loc(case_id) -> Path:
    # Best model across attempts (the winner among tmp_attempt_best dirs); source for test-fold predictions
    # and S3 uploads
    return case_models_dir(case_id) / "tmp_best"


def cleanup_case_models(case_id):
    logger.info(f"Cleaning up {case_models_dir(case_id)}")
    shutil.rmtree(case_models_dir(case_id), ignore_errors=True)


def get_sent_boundaries(doc_datasets: dict[int, pd.DataFrame]) -> dict[int, list[int]]:
    """
    Each doc dataset has a column id_doc refering to the original id from the database dump `documents` table.
    The same doc can exist in multiple doc datasets (each is a single case), and in theory the texts should match,
    but we can confirm this here, and only extract sentences once per unique doc. The splitting itself (and its
    content-hashed cache) is shared with dataset creation via make_classification_datasets.get_sent_boundaries.
    :return: dict from doc id to list of char positions that start sentences
    """
    unique_texts = dict()
    for doc_dataset in doc_datasets.values():
        for id_doc, text in zip(doc_dataset.id_doc, doc_dataset.text):
            if id_doc in unique_texts:
                assert text == unique_texts[id_doc]
            else:
                unique_texts[id_doc] = text
    unique_docs = pd.DataFrame({"id_doc": list(unique_texts.keys()), "text": list(unique_texts.values())})
    return make_classification_datasets.get_sent_boundaries(unique_docs)


def _prep_inputs(sent_spans_df, tokenizer) -> tuple[pd.DataFrame, Dataset]:
    """Turns the sent span dataframe into int labels and a tokenized HF Dataset. Index alignment with
    `sent_spans_df` is preserved (row i of the Dataset is sent_spans_df row i)."""

    def tokenize(examples):
        # No padding here; batches are padded dynamically by DataCollatorWithPadding
        return tokenizer(examples["text"], truncation=True)

    # Keep only what training needs, so new dataset columns can't silently leak into the model inputs
    input_cases = sent_spans_df[["text", "fold"]].copy()
    input_cases["label"] = (sent_spans_df.label == "positive").astype(int)

    dataset = Dataset.from_pandas(input_cases)
    # cast: Dataset.map is stubbed to return DatasetDict, but for a Dataset input it returns a Dataset
    tokenized = cast(Dataset, dataset.map(tokenize, batched=True))
    return input_cases, tokenized


def _stopping_slice_doc_ids(sent_spans_df, doc_df, frac=STOPPING_FRACTION, seed=0) -> set:
    """
    Picks a small set of doc ids to hold out of a training pool, used only for early stopping / best-checkpoint /
    best-attempt selection (the training recipe needs eval data; it plays no role in thresholds or test-fold
    predictions). Grouped by service, like make_classification_datasets.assign_folds, so near-duplicate docs can't
    leak between the slice and the training data — and stratified the same way: services contributing any positive
    instance are sliced separately from negative-only services, so the slice gets a representative share of rare
    positives.
    """
    doc_service = dict(zip(sent_spans_df.document_id, sent_spans_df.service_id))
    doc_service.update(zip(doc_df.id_doc, doc_df.id_service))

    positive_doc_ids = set(sent_spans_df.loc[sent_spans_df.label == "positive", "document_id"]).union(
        doc_df.loc[doc_df.label == "positive", "id_doc"]
    )
    positive_services = sorted({doc_service[doc_id] for doc_id in positive_doc_ids})
    negative_only_services = sorted(set(doc_service.values()) - set(positive_services))

    rng = random.Random(seed)
    rng.shuffle(positive_services)
    rng.shuffle(negative_only_services)
    n_pos = max(1, round(len(positive_services) * frac))
    n_neg = max(1, round(len(negative_only_services) * frac))
    slice_services = set(positive_services[:n_pos]) | set(negative_only_services[:n_neg])
    return {doc_id for doc_id, service in doc_service.items() if service in slice_services}


def _load_best_model(model_base, adapter_dir: Path, device):
    base_model = AutoModelForSequenceClassification.from_pretrained(model_base, num_labels=2)
    # pyrefly: ignore[not-callable]  # merge_and_unload is delegated via PeftModel.__getattr__
    return PeftModel.from_pretrained(base_model, adapter_dir).merge_and_unload().to(device)


def _predict_logits(model, dataset, tokenizer, cfg, case_id) -> np.ndarray:
    """Batch predictions outside a training run; Trainer.predict handles batching/collation/device placement"""
    args = TrainingArguments(
        output_dir=case_models_dir(case_id).as_posix(), per_device_eval_batch_size=cfg.batch_size, report_to="none"
    )
    trainer = Trainer(model=model, args=args, data_collator=DataCollatorWithPadding(tokenizer))
    # cast: predictions is typed as ndarray | tuple, but sequence classification models return a single logit array
    return cast(np.ndarray, trainer.predict(dataset).predictions)


def _make_doc_eval_kwargs(case_id, tokenizer, doc_sent_boundaries, device):
    return dict(
        prefilter_kwargs=inference.load_prefilter_kwargs(case_id),
        # Prefilter masks depend only on the TF-IDF model and doc text, so share them across folds/attempts/evals
        prefilter_cache=dict(),
        tokenizer=tokenizer,
        sent_boundaries=doc_sent_boundaries,
        device=device,
    )


@dataclass
class TestFoldPreds:
    """A fold's held-out predictions from one attempt's checkpoint, plus the metrics derived from them."""

    sent_scores: np.ndarray  # positive-class probability, positionally aligned with the attempt's test_idxs
    doc_pred: pd.DataFrame  # test_doc_df with the DOC_PRED_COLS (and pred_label) attached
    metrics: dict


def _test_fold_metrics(doc_pred: pd.DataFrame, sent_scores, sent_labels, sent_sources) -> dict:
    """
    Metrics over a fold's held-out test data, named to mirror the stopping-slice metrics in
    trainer_callbacks.DocEvalCallback (stop/doc_f1 vs test/doc_f1, etc.) so the two can be read side by side.

    doc f1/prec/rec/accuracy inherit inference.attach_predictions' threshold, which is fit on the set being scored
    — fine for comparing folds and attempts, which is what these are for, but optimistic as absolute numbers. The
    stop/doc_* equivalents share that caveat. test/doc_roc_auc is threshold-free.
    """
    doc_labels = doc_pred.int_labels
    sent_preds = (sent_scores >= 0.5).astype(int)
    metrics = {
        "test/doc_f1": f1_score(doc_labels, doc_pred.pred_label, zero_division=0),
        "test/doc_prec": precision_score(doc_labels, doc_pred.pred_label, zero_division=0),
        "test/doc_rec": recall_score(doc_labels, doc_pred.pred_label, zero_division=0),
        "test/doc_accuracy": accuracy_score(doc_labels, doc_pred.pred_label),
        "test/doc_roc_auc": roc_auc_score(doc_labels, doc_pred.pred_score) if doc_labels.nunique() > 1 else np.nan,
        "test/sent_f1_pos": f1_score(sent_labels, sent_preds, zero_division=0),
        "test/sent_prec_pos": precision_score(sent_labels, sent_preds, zero_division=0),
        "test/sent_rec_pos": recall_score(sent_labels, sent_preds, zero_division=0),
        "test/sent_roc_auc": roc_auc_score(sent_labels, sent_scores) if len(np.unique(sent_labels)) > 1 else np.nan,
    }
    # Positive rate per source, as in DocEvalCallback, but over the whole fold instead of the ~10% stopping slice.
    # Rare sources (`declined` is single digits per fold) are reliably present here and frequently absent from the
    # stop/doc_pos_rate_* equivalents.
    for source, group in doc_pred.groupby("source"):
        metrics[f"test/doc_pos_rate_{source}"] = float(group.pred_label.mean())
    for source in np.unique(sent_sources):
        metrics[f"test/sent_pos_rate_{source}"] = float(sent_preds[sent_sources == source].mean())
    return metrics


def _predict_test_fold(
    sent_spans_df, tokenized, tokenizer, case_id, cfg: TrainConfig, doc_eval_kwargs, test_idxs, test_doc_df, adapter_dir
) -> TestFoldPreds:
    """
    Runs the model in `adapter_dir` over a fold's held-out test data. Called once per attempt, after that attempt
    has trained and its checkpoint is already chosen, so the test fold still plays no part in early stopping,
    checkpoint selection or attempt selection — it is measured here, never optimized against.
    """
    model = _load_best_model(cfg.model_base, adapter_dir, cfg.device)
    sent_scores = softmax(_predict_logits(model, tokenized.select(test_idxs), tokenizer, cfg, case_id), axis=1)[:, 1]

    logger.info(f"Predicting {len(test_doc_df)} held-out test docs")
    doc_pred = inference.attach_predictions(test_doc_df, **doc_eval_kwargs, model=model, batch_size=cfg.batch_size // 2)
    scored = doc_pred[doc_pred.pred_score.notna()]
    if len(scored) < len(doc_pred):
        logger.warning(f"{len(doc_pred) - len(scored)} test docs have no pred_score; excluded from test/ metrics")

    metrics = _test_fold_metrics(
        scored,
        sent_scores,
        np.asarray(tokenized.select(test_idxs)["label"]),
        sent_spans_df.source.to_numpy()[test_idxs],
    )
    for key, val in metrics.items():
        logger.info(f"\t{key}: {val:.4f}")
    return TestFoldPreds(sent_scores=sent_scores, doc_pred=doc_pred, metrics=metrics)


def finetune(sent_spans_df, doc_df, case_id, cfg: TrainConfig, doc_sent_boundaries, cv_folds=None):
    """
    Cross-validation training. For each fold, models are trained on the other folds minus a small stopping slice
    (used for early stopping and checkpoint/attempt selection). Each attempt then predicts the held-out test fold,
    and the attempt that won on the stopping slice supplies the returned predictions; the others' predictions are
    only summarized into wandb test/ metrics, so per-fold and per-attempt variance is visible. The test fold plays
    no part in any selection decision, so its predictions are honestly out-of-fold — threshold selection in
    thresholds.py depends on this.

    Returns a tuple:
    - a copy of `sent_spans_df`, with `pred` set to a softmax probability of positive classification
    - a copy of `doc_df` also with out-of-fold predictions of the document dataset
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_base)

    sent_spans_df = sent_spans_df.copy()
    doc_df = doc_df.copy()
    doc_df["int_labels"] = [0 if i == "negative" else 1 for i in doc_df.label]

    input_cases, tokenized = _prep_inputs(sent_spans_df, tokenizer)

    if cv_folds is None:
        cv_folds = input_cases.fold.nunique()
    # CV logic depends on a reset index
    assert input_cases.index.values.tolist() == list(range(len(input_cases)))

    doc_eval_kwargs = _make_doc_eval_kwargs(case_id, tokenizer, doc_sent_boundaries, cfg.device)

    sent_spans_df["pred"] = pd.Series(dtype=float)
    for col in DOC_PRED_COLS:
        doc_df[col] = pd.Series(dtype=float)

    for fold_i in range(cv_folds):
        pool_mask = input_cases.fold != fold_i
        doc_pool_mask = doc_df.fold != fold_i
        # Hold a slice of the training pool out for early stopping / model selection, seeded per fold so the
        # slices differ across folds
        stopping_doc_ids = _stopping_slice_doc_ids(sent_spans_df[pool_mask], doc_df[doc_pool_mask], seed=fold_i)
        in_stopping = sent_spans_df.document_id.isin(stopping_doc_ids)
        train_idxs = input_cases[pool_mask & ~in_stopping].index.values
        stop_idxs = input_cases[pool_mask & in_stopping].index.values
        stop_doc_df = doc_df[doc_pool_mask & doc_df.id_doc.isin(stopping_doc_ids)]
        test_idxs = input_cases[input_cases.fold == fold_i].index.values
        test_doc_df = doc_df[doc_df.fold == fold_i]

        _, _, test_preds = _run_attempts(
            sent_spans_df,
            tokenized,
            tokenizer,
            case_id,
            cfg,
            doc_eval_kwargs,
            train_idxs,
            stop_idxs,
            stop_doc_df,
            fold_label=f"fold{fold_i}",
            test_idxs=test_idxs,
            test_doc_df=test_doc_df,
        )

        # Out-of-fold predictions, from the attempt that won on the stopping slice. Every attempt already predicted
        # this fold for its test/ metrics, and the winner's checkpoint is exactly what gets promoted to
        # tmp_best_loc, so re-running the model here would only reproduce what _run_attempts already returned.
        assert test_preds is not None
        sent_spans_df.loc[test_idxs, "pred"] = test_preds.sent_scores
        fold_preds = test_preds.doc_pred.set_index("id_doc")
        fold_mask = doc_df.fold == fold_i
        doc_df.loc[fold_mask, DOC_PRED_COLS] = fold_preds.loc[doc_df.loc[fold_mask, "id_doc"], DOC_PRED_COLS].values

    return sent_spans_df, doc_df


def finetune_final(sent_spans_df, doc_df, case_id, cfg: TrainConfig, doc_sent_boundaries):
    """
    Trains the production model on (nearly) all data — everything except a small stratified stopping slice —
    using the decision thresholds derived beforehand from the CV fold models (see thresholds.py).
    The best model across attempts lands in tmp_best_loc(case_id), same as finetune().

    Returns the stopping slice of `doc_df` with predictions from the final model attached, so the final model's
    score distribution can be sanity-checked against the CV fold models' out-of-fold scores (pr_curves.ipynb).
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_base)

    sent_spans_df = sent_spans_df.copy()
    doc_df = doc_df.copy()
    doc_df["int_labels"] = [0 if i == "negative" else 1 for i in doc_df.label]

    input_cases, tokenized = _prep_inputs(sent_spans_df, tokenizer)
    assert input_cases.index.values.tolist() == list(range(len(input_cases)))

    stopping_doc_ids = _stopping_slice_doc_ids(sent_spans_df, doc_df)
    in_stopping = sent_spans_df.document_id.isin(stopping_doc_ids)
    train_idxs = input_cases[~in_stopping].index.values
    stop_idxs = input_cases[in_stopping].index.values
    stop_doc_df = doc_df[doc_df.id_doc.isin(stopping_doc_ids)]

    doc_eval_kwargs = _make_doc_eval_kwargs(case_id, tokenizer, doc_sent_boundaries, cfg.device)

    _, stop_doc_preds, _ = _run_attempts(
        sent_spans_df,
        tokenized,
        tokenizer,
        case_id,
        cfg,
        doc_eval_kwargs,
        train_idxs,
        stop_idxs,
        stop_doc_df,
        fold_label="final",
    )
    return stop_doc_preds.drop(columns=["pred_label"])


def _run_attempts(
    sent_spans_df,
    tokenized,
    tokenizer,
    case_id,
    cfg: TrainConfig,
    doc_eval_kwargs,
    train_idxs,
    stop_idxs,
    stop_doc_df,
    fold_label,
    test_idxs=None,
    test_doc_df=None,
) -> tuple[float, pd.DataFrame, TestFoldPreds | None]:
    """
    Trains cfg.attempts_per_fold models on `train_idxs`, using the stopping slice (`stop_idxs` sent spans,
    `stop_doc_df` docs) for early stopping and model selection. The best model across attempts, by doc F1 on the
    stopping slice, ends up in tmp_best_loc(case_id).

    In CV mode the caller also passes the fold's held-out test data (`test_idxs`, `test_doc_df`). Each attempt
    predicts it after training, purely so per-fold and per-attempt variance is visible in wandb under test/ —
    selection still runs off the stopping slice alone, which is what keeps those predictions honestly out-of-fold
    for thresholds.py.
    :return: the best stopping-slice doc F1, the corresponding stopping-slice doc predictions, and the winning
        attempt's test-fold predictions (None with --train_final, which has no test fold)
    """
    title = sent_spans_df.iloc[0].title
    best_metric = -1.0
    best_stop_doc_preds = None
    best_test_preds = None
    for attempt in range(cfg.attempts_per_fold):
        logger.info(f"Case id {case_id}: {title}")
        logger.info(f"{fold_label}, attempt {attempt + 1} / {cfg.attempts_per_fold}")
        logger.info(f"{len(train_idxs)} train instances, {len(stop_idxs)} stopping-slice instances")
        logger.info(f"{len(stop_doc_df)} stopping-slice doc instances")
        run = None
        if cfg.log_wandb:
            run = wandb.init(
                project="tosdr_cases",
                group=str(case_id),
                name=f"{case_id}-{fold_label}-a{attempt}",
                job_type="final" if fold_label == "final" else "cv",
                config=dict(
                    case_id=case_id,
                    case_title=title,
                    fold=fold_label,
                    attempt=attempt,
                    model_base=cfg.model_base,
                    dataset_version=CLASSIFICATION_VERSION,
                ),
            )
            wandb.define_metric("stop/doc_f1", summary="max")
            wandb.define_metric("eval/f1_pos", summary="max")
        with run if run is not None else contextlib.nullcontext():
            pos_f1, stop_doc_preds = _train_one_attempt(
                tokenized.select(train_idxs),
                tokenized.select(stop_idxs),
                stop_doc_df,
                case_id,
                cfg,
                tokenizer,
                doc_eval_kwargs,
                stop_sources=sent_spans_df.source.to_numpy()[stop_idxs],
                seed=42 + attempt,
            )
            attempt_test_preds = None
            if test_idxs is not None:
                attempt_test_preds = _predict_test_fold(
                    sent_spans_df,
                    tokenized,
                    tokenizer,
                    case_id,
                    cfg,
                    doc_eval_kwargs,
                    test_idxs,
                    test_doc_df,
                    tmp_attempt_best_loc(case_id),
                )
            if run is not None:
                # The metric attempt selection actually uses (doc F1 of the attempt's best checkpoint)
                run.summary["attempt_stop_doc_f1"] = pos_f1
                if attempt_test_preds is not None:
                    # Summary rather than log: one value per run, so wandb can table/scatter them across the
                    # fold and attempt config fields to show variance
                    run.summary.update(attempt_test_preds.metrics)
        if pos_f1 > best_metric:
            best_metric = pos_f1
            best_stop_doc_preds = stop_doc_preds
            best_test_preds = attempt_test_preds
            # Promote this attempt's best checkpoint to the case-level best. Full replace, never a merge, so a
            # leftover file from an older run (possibly in an older serialization format) can't survive here.
            shutil.rmtree(tmp_best_loc(case_id), ignore_errors=True)
            shutil.copytree(tmp_attempt_best_loc(case_id), tmp_best_loc(case_id))

    # Always set on the first attempt since pos_f1 >= 0 > best_metric's initial -1.0
    assert best_stop_doc_preds is not None
    return best_metric, best_stop_doc_preds, best_test_preds


def _train_one_attempt(
    train_dataset, stop_dataset, stop_doc_df, case_id, cfg: TrainConfig, tokenizer, doc_eval_kwargs, stop_sources, seed
) -> tuple[float, pd.DataFrame]:
    """
    A single training run. Early stopping uses sent-span F1 on the stopping slice; the best checkpoint by doc F1
    on the stopping slice (tracked by DocEvalCallback) is saved to tmp_attempt_best_loc(case_id).
    :return: that checkpoint's stopping-slice doc F1 and doc predictions
    """
    # Seed everything (including the head/LoRA init below) per attempt, so attempts are controlled random
    # restarts rather than depending on incidental global RNG state
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_base, num_labels=2)
    lora_args = dict(r=16, lora_alpha=16, lora_dropout=0.1, bias="all")
    logger.info(f"Training Lora with args {lora_args}")
    # pyrefly: ignore[bad-argument-type]  # bias literal is widened to str through **lora_args unpacking
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, **lora_args)
    model = get_peft_model(model, peft_config)

    training_kwargs = dict(
        output_dir=case_models_dir(case_id).as_posix(),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.eval_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_train_epochs,
        save_total_limit=1,
        learning_rate=cfg.learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="f1_pos",
        seed=seed,
        bf16=cfg.device == "cuda",  # bert-base trains fine in bf16, roughly doubling throughput on modern GPUs
    )
    # Explicit "none" matters: TrainingArguments' default report_to is all installed integrations, which would
    # attach WandbCallback (and call wandb.init) even when log_wandb is off
    training_kwargs["report_to"] = "wandb" if cfg.log_wandb else "none"
    training_kwargs.update(cfg.training_overrides)
    training_args = TrainingArguments(**training_kwargs)

    stop_sources = np.asarray(stop_sources)
    stop_labels = np.asarray(stop_dataset["label"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metrics = {
            "f1_pos": f1_score(labels, predictions, zero_division=0),
            "prec_pos": precision_score(labels, predictions, zero_division=0),
            "rec_pos": recall_score(labels, predictions, zero_division=0),
            "accuracy": accuracy_score(labels, predictions),
            # The logit difference is monotone in the softmax positive probability; logits[:, 1] alone is not
            "roc_auc": roc_auc_score(labels, logits[:, 1] - logits[:, 0]) if len(np.unique(labels)) > 1 else np.nan,
        }
        # Positive prediction rate per instance source (approved/declined/surrounding/topical/...), to audit
        # negative sources for mislabeled true negatives. This closure is also invoked on the training subset by
        # EvalTrainingSetCallback (same length as the stopping slice by construction), where stop_sources wouldn't
        # align — the label check skips that case.
        if np.array_equal(labels, stop_labels):
            for source in np.unique(stop_sources):
                metrics[f"pos_rate_{source}"] = float(predictions[stop_sources == source].mean())
        return metrics

    # Start from a clean slate so a checkpoint from a previous attempt or run can never leak into this one
    shutil.rmtree(tmp_attempt_best_loc(case_id), ignore_errors=True)
    doc_eval_callback = trainer_callbacks.DocEvalCallback(
        stop_doc_df,
        case_id,
        **doc_eval_kwargs,
        # Doc eval is throttled by only running every `eval_every`-th eval, and the prefilter masks are cached,
        # so we can afford the full stopping slice
        eval_every=3,
        save_to_dir=tmp_attempt_best_loc,
        log_wandb=cfg.log_wandb,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=stop_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience), doc_eval_callback],
    )
    if cfg.eval_training_set:
        training_eval_dataset = train_dataset.shuffle(seed=0).select(range(len(stop_dataset)))
        trainer.add_callback(trainer_callbacks.EvalTrainingSetCallback(training_eval_dataset, trainer))
    trainer.train()

    assert doc_eval_callback.best_doc_preds is not None, (
        "DocEvalCallback never evaluated; training may be too short for the configured eval_steps"
    )
    return doc_eval_callback.best_f1, doc_eval_callback.best_doc_preds


def results_dir(results_key, final=False) -> Path:
    """Local prediction cache dir for a training run. `results_key` is the model version, or a timestamp for
    throwaway experiments. Lives under data/training/ — data/results/ holds production inference outputs
    (apply_docbot.py). CV and final live in separate subdirs because both write doc_pred.pkl and train_serial
    uses its presence to skip completed cases — sharing a dir would make a final run mistake CV results for
    its own."""
    return here / f"../../data/training/{results_key}" / ("final" if final else "cv")


def results_by_case(results_key) -> pd.DataFrame:
    """
    For later analysis; not used during training
    :return: case DF with new float columns: sent_span_f1, doc_f1
    """
    sent_span_output_path = results_dir(results_key) / "sent_span_pred.pkl"
    doc_output_path = results_dir(results_key) / "doc_pred.pkl"
    sent_span_results = pickle.load(open(sent_span_output_path, "rb"))
    doc_results = pickle.load(open(doc_output_path, "rb"))
    dump_version = make_classification_datasets.DB_DUMP_VERSION
    cases = pickle.load(open(here / f"../../data/db_dumps/{dump_version}/cases_clean.pkl", "rb"))
    for case_id in sent_span_results:
        # The sent span eval logic sets a "pred" score float. We'll find the optimal threshold that maximizes f1
        eval_set = sent_span_results[case_id][~sent_span_results[case_id].pred.isna()]
        int_labels = [0 if i == "negative" else 1 for i in eval_set.label]
        cases.at[case_id, "sent_span_f1"] = utils.optimal_threshold(eval_set.pred, int_labels)[3]

    for case_id in doc_results:
        eval_set = doc_results[case_id][~doc_results[case_id].pred_score.isna()]
        int_labels = [0 if i == "negative" else 1 for i in eval_set.label]
        cases.at[case_id, "doc_f1"] = utils.optimal_threshold(eval_set.pred_score, int_labels)[3]

    return cases


def _adapter_files(case_id) -> list[Path]:
    """The files needed to use a trained model in production (LoRA adapter; the base model isn't uploaded).
    Built from directory contents rather than hardcoded names because the weights filename depends on the
    serialization format (adapter_model.safetensors on current peft, adapter_model.bin historically)."""
    model_dir = tmp_best_loc(case_id)
    weights = list(model_dir.glob("adapter_model.*"))
    if len(weights) != 1:
        raise RuntimeError(f"Expected exactly one adapter weights file in {model_dir}, found {weights}")
    return [model_dir / "adapter_config.json", weights[0]]


def _upload_file(s3_client, filepath: Path, key, metadata):
    content_type = "application/json" if filepath.suffix == ".json" else "application/octet-stream"
    logger.info(f"Uploading to s3://{RESULTS_S3_BUCKET}/{key}")
    s3_client.upload_file(
        filepath.as_posix(), RESULTS_S3_BUCKET, key, ExtraArgs={"Metadata": metadata, "ContentType": content_type}
    )


def _upload_df(s3_client, df: pd.DataFrame, key, metadata):
    # Parquet rather than pickle so results survive pandas version bumps and are readable outside python
    buffer = io.BytesIO()
    df.to_parquet(buffer)
    buffer.seek(0)
    logger.info(f"Uploading to s3://{RESULTS_S3_BUCKET}/{key}")
    s3_client.upload_fileobj(
        buffer, RESULTS_S3_BUCKET, key, ExtraArgs={"Metadata": metadata, "ContentType": "application/octet-stream"}
    )


def upload_results(model_version, case_id, sent_pred_df, doc_pred_df):
    """
    Uploads a case's CV artifacts (out-of-fold predictions plus the last fold's adapter) under
    {model_version}/cv/, so threshold selection can be revisited later without retraining. The cv/ subdir keeps
    fold adapters away from {model_version}/{case_id}/ — the production path apply_docbot.py pulls from — so a
    partially completed final run can never serve a CV fold model.
    """
    s3_client = boto3.client("s3", region_name=train_push.AWS_REGION)
    metadata = {"host": socket.gethostname()}
    prefix = f"{model_version}/cv/{case_id}"
    _upload_df(s3_client, sent_pred_df, f"{prefix}/{case_id}_sents.parquet", metadata)
    _upload_df(s3_client, doc_pred_df, f"{prefix}/{case_id}_docs.parquet", metadata)
    for filepath in _adapter_files(case_id):
        _upload_file(s3_client, filepath, f"{prefix}/{filepath.name}", metadata)


def upload_final_results(model_version, case_id, final_eval_doc_df):
    """
    Uploads everything needed to use the final full-data model in production: the LoRA adapter, the thresholds
    menu (written into the model version dir by thresholds.py after a CV run), and the stopping-slice doc
    predictions for calibration sanity checks in pr_curves.ipynb. These land at {model_version}/{case_id}/,
    the layout production inference (apply_docbot.py) reads.
    """
    check_final_trainable(case_id, model_version)
    thresholds_path = inference.model_dir(case_id, model_version) / "thresholds.json"

    s3_client = boto3.client("s3", region_name=train_push.AWS_REGION)
    metadata = {"host": socket.gethostname()}
    prefix = f"{model_version}/{case_id}"
    _upload_df(s3_client, final_eval_doc_df, f"{prefix}/{case_id}_final_eval_docs.parquet", metadata)
    for filepath in _adapter_files(case_id) + [thresholds_path]:
        _upload_file(s3_client, filepath, f"{prefix}/{filepath.name}", metadata)


def check_final_trainable(case_id, model_version):
    """Fail fast — before hours of training — if threshold selection hasn't been run for this case"""
    thresholds_path = inference.model_dir(case_id, model_version) / "thresholds.json"
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"{thresholds_path} not found. Run thresholds.py with --model_version {model_version} on the CV "
            "results before training/uploading a final model"
        )


def already_uploaded(model_version, case_id, final=False) -> bool:
    s3_client = boto3.client("s3", region_name=train_push.AWS_REGION)
    if final:
        s3_obj_name = f"{model_version}/{case_id}/{case_id}_final_eval_docs.parquet"
    else:
        s3_obj_name = f"{model_version}/cv/{case_id}/{case_id}_sents.parquet"
    try:
        s3_client.head_object(Bucket=RESULTS_S3_BUCKET, Key=s3_obj_name)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise e


@contextlib.contextmanager
def _visibility_heartbeat(sqs_client, receipt_handle, interval_s=240, visibility_s=900):
    """Extends the SQS message's visibility timeout every `interval_s` while the body runs, so a message whose
    case is mid-training (hours) isn't redelivered to another worker when the queue's own timeout lapses."""
    stop = threading.Event()

    def beat():
        while True:
            try:
                sqs_client.change_message_visibility(
                    QueueUrl=train_push.parallel_queue_name(),
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=visibility_s,
                )
            except Exception as e:
                logger.warning(f"Visibility heartbeat failed (continuing): {e}")
            if stop.wait(interval_s):
                return

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


def train_serial(args, cfg: TrainConfig, dataset_dict, doc_dataset_dict, doc_sent_boundaries):
    """
    Method 1 of running: single process
    Looping through all cases. May take days to train. Results are saved locally as pickled dictionaries from
    case_id to dataframes that include predictions using the best model (as measured by F1 on the doc dataset).
    Results are written/cached after each case model, so if this crashes you can restart with the same
    --model_version and already-trained cases will be skipped. Rerunning a version after a code or dataset change
    hits the same cache — delete data/training/{model_version}/ first to retrain from scratch.
    """
    results_key = args.model_version or ("test" if args.debug else str(int(time.time())))
    out_dir = results_dir(results_key, final=args.train_final)
    logger.info(f"Caching results in {out_dir}")
    if not args.model_version:
        logger.info(f"(pass --model_version {results_key} to resume this run after a crash)")
    sent_span_output_path = out_dir / "sent_span_pred.pkl"
    doc_output_path = out_dir / "doc_pred.pkl"

    # Cache case trainings, in case of crash
    try:
        sent_pred_local = pickle.load(open(sent_span_output_path, "rb"))
    except FileNotFoundError:
        sent_pred_local = dict()
    try:
        docs_pred_local = pickle.load(open(doc_output_path, "rb"))
    except FileNotFoundError:
        docs_pred_local = dict()

    # For reproducibility, a seed was set above for a consistent visitation order
    case_ids = list(dataset_dict.keys())
    random.shuffle(case_ids)
    for case_id in case_ids:
        # Final mode produces no sent span predictions, so completion is tracked via the doc results
        if case_id in (docs_pred_local if args.train_final else sent_pred_local):
            logger.info(f"Skipping {case_id}: already in local results")
            continue
        if args.upload and already_uploaded(args.model_version, case_id, final=args.train_final):
            logger.info(f"Skipping {case_id}: results already in S3")
            continue
        if args.train_final and args.upload:
            check_final_trainable(case_id, args.model_version)

        doc_output_path.parent.mkdir(exist_ok=True, parents=True)
        if args.train_final:
            final_eval_doc_df = finetune_final(
                dataset_dict[case_id], doc_dataset_dict[case_id], case_id, cfg, doc_sent_boundaries
            )
            docs_pred_local[case_id] = final_eval_doc_df.copy()
            pickle.dump(docs_pred_local, open(doc_output_path, "wb"))

            if args.upload:
                upload_final_results(args.model_version, case_id, final_eval_doc_df)
                cleanup_case_models(case_id)
        else:
            sent_pred_df, doc_pred_df = finetune(
                dataset_dict[case_id],
                doc_dataset_dict[case_id],
                case_id,
                cfg,
                doc_sent_boundaries,
                cv_folds=args.cv_folds,
            )
            sent_pred_local[case_id] = sent_pred_df.copy()
            docs_pred_local[case_id] = doc_pred_df.copy()

            pickle.dump(sent_pred_local, open(sent_span_output_path, "wb"))
            pickle.dump(docs_pred_local, open(doc_output_path, "wb"))

            if args.upload:
                upload_results(args.model_version, case_id, sent_pred_df, doc_pred_df)
                cleanup_case_models(case_id)


def train_parallel(args, cfg: TrainConfig, dataset_dict, doc_dataset_dict, doc_sent_boundaries):
    """
    Method 2 of running: distributed training with --parallel
    Instead of looping through case IDs, we'll dequeue them from SQS. This assumes train_push.py was run first
    with the same --model_version.
    """
    sqs_client = boto3.client("sqs", region_name=train_push.AWS_REGION)

    receives = 0
    max_poll_time_s = 300
    while True:
        logger.info(f"Polling {train_push.parallel_queue_name()} for {max_poll_time_s}s")
        start_time = time.time()
        found = False
        while time.time() - start_time < max_poll_time_s:
            response = sqs_client.receive_message(
                QueueUrl=train_push.parallel_queue_name(), WaitTimeSeconds=20, AttributeNames=["MessageGroupId"]
            )
            if "Messages" in response:
                found = True
                break
        if not found:
            if receives == 0:
                logger.error("No messages found upon startup")
            else:
                logger.info("No messages found, exiting")
            break
        else:
            receives += 1
            message = response["Messages"][0]
            logger.info(f"Received case ID {message['Body']}")
            receipt_handle = message["ReceiptHandle"]

            try:
                case_id = int(message["Body"])
                # Sanity check the model_version used in train_push.py matches
                found_key = message["Attributes"]["MessageGroupId"].split("/")[0]
                if found_key != args.model_version:
                    raise Exception(f"Different model_version found {found_key} != {args.model_version}")

                if args.upload and already_uploaded(args.model_version, case_id, final=args.train_final):
                    logger.info(f"Skipping case {case_id}: results already in S3")
                    sqs_client.delete_message(QueueUrl=train_push.parallel_queue_name(), ReceiptHandle=receipt_handle)
                    continue
                if args.train_final and args.upload:
                    check_final_trainable(case_id, args.model_version)

                # Trainings run far longer than any reasonable queue visibility timeout, so keep the message
                # invisible to other workers while we work on it
                with _visibility_heartbeat(sqs_client, receipt_handle):
                    if args.train_final:
                        final_eval_doc_df = finetune_final(
                            dataset_dict[case_id], doc_dataset_dict[case_id], case_id, cfg, doc_sent_boundaries
                        )
                        if args.upload:
                            upload_final_results(args.model_version, case_id, final_eval_doc_df)
                            cleanup_case_models(case_id)
                    else:
                        sent_pred_df, doc_pred_df = finetune(
                            dataset_dict[case_id],
                            doc_dataset_dict[case_id],
                            case_id,
                            cfg,
                            doc_sent_boundaries,
                            cv_folds=args.cv_folds,
                        )
                        if args.upload:
                            upload_results(args.model_version, case_id, sent_pred_df, doc_pred_df)
                            cleanup_case_models(case_id)

                # Delete received message from queue
                sqs_client.delete_message(QueueUrl=train_push.parallel_queue_name(), ReceiptHandle=receipt_handle)
            except Exception as e:
                # Put the message back on the queue (or let it go to the DLQ) immediately, instead of waiting for
                # the message's default visibility timeout
                logger.info("Problem encountered; putting message back on queue")
                sqs_client.change_message_visibility(
                    QueueUrl=train_push.parallel_queue_name(), ReceiptHandle=receipt_handle, VisibilityTimeout=0
                )
                raise e


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Fine-tune BERT (LoRA) on the sentence span dataset created in make_classification_datasets.py. "
        "Also uses the document dataset during evaluation."
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all ~150 case models, instead of just enough to optimize. If --parallel is "
        "set this is ignored, and training assignments come from SQS",
    )
    parser.add_argument("--eval_training", action="store_true", help="Evaluate training set alongside the test set")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use multiple hosts or GPUs/processes to train cases. Note you first have to run train_push.py with "
        "the same --model_version, which will push case IDs to SQS. Requires --model_version",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload results to S3 under the model version: CV runs put LORA models and test-fold predictions in "
        "{model_version}/cv/, --train_final puts production artifacts in {model_version}/. Requires --model_version",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        choices=["bert-base-uncased", "roberta-base", "nlpaueb/legal-bert-base-uncased"],
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--train_final",
        action="store_true",
        help="Train production models on all data except a small stopping slice, instead of running CV. "
        "Requires thresholds.json to already exist per case (run thresholds.py on a prior CV run first)",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        required=False,
        help="Production model version being prepared (e.g. v4). Names the local results cache "
        "(data/training/{model_version}/) and the S3 prefix for uploads; final uploads read each case's "
        "thresholds.json from data/models/{model_version}/{case_id}/. Omit for throwaway experiments, which "
        "cache under a timestamp instead. Required with --upload or --parallel",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Default is ideal for 24gb GPU memory")
    parser.add_argument("--learning_rate", type=float, default=5e-04)
    parser.add_argument("--attempts_per_fold", type=int, default=3)
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=None,
        help="Number of CV folds to train/evaluate (ignored with --train_final). Defaults to all pre-assigned folds, "
        "which is what threshold selection in thresholds.py needs",
    )
    parser.add_argument("--eval_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=28)
    parser.set_defaults(debug=False, all=False, eval_training=False)
    args = parser.parse_args()

    if (args.upload or args.parallel) and not args.model_version:
        parser.error("--model_version is required with --upload or --parallel")

    # Cloud hosts authenticate as the bootstrap IAM user (env-var keys) and assume tosdr-trainer for the actual
    # SQS/S3 work. Only needed when we touch AWS. Set AWS_SKIP_ASSUME_ROLE=1 to use ambient creds instead (e.g.
    # running locally from a root profile that already has access).
    if (args.upload or args.parallel) and not os.environ.get("AWS_SKIP_ASSUME_ROLE"):
        # Role ARN comes from AWS_ACCOUNT (or a full AWS_TRAINER_ROLE_ARN override); the host itself only needs the
        # tosdr-trainer-bootstrap user's keys in the environment.
        trainer_role_arn = aws_auth.role_arn("tosdr-trainer", "AWS_TRAINER_ROLE_ARN")
        aws_auth.install_assumed_role_session(trainer_role_arn, train_push.AWS_REGION, session_prefix="train")

    dataset_dict = make_classification_datasets.load_sent_span()

    # For evaluation purposes, run sentence span predictor on entire documents
    doc_dataset_dict = make_classification_datasets.load_docs()
    doc_sent_boundaries = get_sent_boundaries(doc_dataset_dict)

    if args.debug:
        cases_to_train = {232, 216}
    elif args.all:
        cases_to_train = dataset_dict.keys()
    else:
        # Enough to reason about hyperparameters, not enough to take forever
        cases_to_train = set(TEST_CASE_IDS)

    for key in list(dataset_dict.keys()):
        if key in cases_to_train:
            if args.debug:
                dataset_dict[key] = dataset_dict[key].sample(200, random_state=0).reset_index(drop=True)
                doc_dataset_dict[key] = doc_dataset_dict[key].groupby(["fold", "label"]).head(10).reset_index(drop=True)
        else:
            del dataset_dict[key]
            del doc_dataset_dict[key]

    random.seed(0)
    np.random.seed(0)
    trainer_callbacks.logger = logger

    log_wandb = True
    device = "cuda"
    if args.debug:
        log_wandb = False
        device = "mps"
        args.batch_size = 4
        args.attempts_per_fold = 2
        args.eval_steps = 10
        args.learning_rate = 1e-04
        args.num_train_epochs = 1
        args.early_stopping_patience = 2
        if args.parallel:
            logger.info("Testing SQS workflow")
            train_push.push(list(dataset_dict.keys()), args.model_version)

    cfg = TrainConfig(
        model_base=args.model_base,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
        early_stopping_patience=args.early_stopping_patience,
        num_train_epochs=args.num_train_epochs,
        attempts_per_fold=args.attempts_per_fold,
        eval_training_set=args.eval_training,
        device=device,
        log_wandb=log_wandb,
    )

    if args.parallel:
        train_parallel(args, cfg, dataset_dict, doc_dataset_dict, doc_sent_boundaries)
    else:
        train_serial(args, cfg, dataset_dict, doc_dataset_dict, doc_sent_boundaries)
