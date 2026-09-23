import logging
from typing import Callable

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import TrainerCallback

import wandb
from src import inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Our training loop in train.py runs evaluation of the sentence span dataset, but we'd also like to run against
the doc dataset, hence the callback.
There's also one to eval against the training set.
"""


class DocEvalCallback(TrainerCallback):
    def __init__(
        self,
        doc_df: pd.DataFrame,
        case_id,
        sent_boundaries,
        tokenizer,
        prefilter_kwargs,
        save_to_dir: Callable,
        prefilter_cache=None,
        eval_every=3,
        log_wandb=True,
        device="cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.doc_df = doc_df
        self.prefilter_kwargs = prefilter_kwargs
        self.prefilter_cache = prefilter_cache
        self.tokenizer = tokenizer
        self.case_id = case_id
        self.sent_boundaries = sent_boundaries
        self.eval_every = eval_every
        self.eval_count = 0
        self.save_to_dir = save_to_dir
        self.log_wandb = log_wandb
        self.device = device
        self.best_f1 = -1.0
        # Predictions from the best checkpoint so far; train.py uses (best_f1, best_doc_preds) to pick the
        # best attempt without re-running doc inference
        self.best_doc_preds: pd.DataFrame | None = None

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if self.eval_count % self.eval_every != 0:
            self.eval_count += 1
            return
        self.eval_count += 1
        logger.info("======= Evaluating doc dataset =======")
        doc_df = self.doc_df.copy()
        # For some reason this takes more memory than during training, so cut batch size in half
        doc_df = inference.attach_predictions(
            doc_df,
            self.prefilter_kwargs,
            self.tokenizer,
            self.sent_boundaries,
            model,
            batch_size=args.per_device_eval_batch_size // 2,
            device=self.device,
            prefilter_cache=self.prefilter_cache,
        )

        labels = doc_df.int_labels
        # stop/ marks the split these come from (the stopping slice), so they read alongside train.py's test/doc_*
        # metrics on the held-out fold. f1/prec/rec/accuracy use inference.attach_predictions' threshold, fit on
        # this slice itself — the same oracle convention as test/doc_*, so the two are directly comparable.
        metrics = {
            "stop/doc_f1": f1_score(labels, doc_df.pred_label, zero_division=0),
            "stop/doc_prec": precision_score(labels, doc_df.pred_label, zero_division=0),
            "stop/doc_rec": recall_score(labels, doc_df.pred_label, zero_division=0),
            "stop/doc_accuracy": accuracy_score(labels, doc_df.pred_label),
            "stop/doc_roc_auc": roc_auc_score(labels, doc_df.pred_score) if labels.nunique() > 1 else float("nan"),
        }
        # Positive prediction rate per instance source (approved/declined/reviewed), to audit negative
        # sources for mislabeled true negatives. Sources absent from this slice simply don't appear; the
        # test/doc_pos_rate_* equivalents cover the whole fold and are populated far more reliably.
        for source, group in doc_df.groupby("source"):
            metrics[f"stop/doc_pos_rate_{source}"] = float(group.pred_label.mean())
        for key, val in metrics.items():
            logger.info(f"\t{key}: {val:.4f}")
        if self.log_wandb:
            wandb.log(metrics)

        # Normally the Trainer class keeps track of the best model using its standard evaluation loop on the test
        # set. In our case we want to use the doc dataset to select the best model, so we'll keep track here.
        f1 = float(metrics["stop/doc_f1"])
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_doc_preds = doc_df
            save_dir = self.save_to_dir(self.case_id)
            logger.info(f"****** Found a new top f1 score of {f1:.4f}, saving model to {save_dir}")
            model.save_pretrained(save_dir)
            # Should probably save trainer state here too


class EvalTrainingSetCallback(TrainerCallback):
    def __init__(self, test_dataset: Dataset, trainer, **kwargs):
        super().__init__(**kwargs)
        self.test_dataset = test_dataset
        self.trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        logger.info("======= Evaluating training subset =======")
        output = self.trainer.predict(self.test_dataset, metric_key_prefix="train")
        self.trainer.log(output.metrics)
