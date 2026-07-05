import logging
from typing import Callable

import evaluate
import pandas as pd
from datasets import Dataset
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
        save_to_dir: Callable,
        eval_every=3,
        log_wandb=True,
        device="cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.doc_df = doc_df
        self.prefilter_kwargs = inference.load_prefilter_kwargs(case_id)
        self.tokenizer = tokenizer
        self.case_id = case_id
        self.sent_boundaries = sent_boundaries
        self.eval_every = eval_every
        self.eval_count = 0
        self.save_to_dir = save_to_dir
        self.log_wandb = log_wandb
        self.device = device
        self.best_f1 = -1.0

        self.acc_metric = evaluate.load("accuracy")
        self.auc_metric = evaluate.load("roc_auc")
        self.prec_metric = evaluate.load("precision")
        self.rec_metric = evaluate.load("recall")
        self.f1_metric = evaluate.load("f1")

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
        )

        def _compute(metric, key, **compute_kwargs):
            result = metric.compute(**compute_kwargs)
            assert result is not None, f"{key} metric returned no result"
            return result[key]

        labels = {"predictions": doc_df.pred_label, "references": doc_df.int_labels}
        metrics = {
            "doc/f1": _compute(self.f1_metric, "f1", **labels),
            "doc/prec": _compute(self.prec_metric, "precision", **labels),
            "doc/rec": _compute(self.rec_metric, "recall", **labels),
            "doc/accuracy": _compute(self.acc_metric, "accuracy", **labels),
            "doc/roc_auc": _compute(
                self.auc_metric, "roc_auc", prediction_scores=doc_df.pred_score, references=doc_df.int_labels
            ),
        }
        for key, val in metrics.items():
            logger.info(f"\t{key}: {val:.4f}")
        if self.log_wandb:
            wandb.log(metrics)

        # Normally the Trainer class keeps track of the best model using its standard evaluation loop on the test
        # set. In our case we want to use the doc dataset to select the best model, so we'll keep track here.
        f1 = metrics["doc/f1"]
        if f1 > self.best_f1:
            self.best_f1 = f1
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
