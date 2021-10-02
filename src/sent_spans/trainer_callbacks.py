import logging
from typing import Callable

import pandas as pd
import wandb
from datasets import load_metric, Dataset
from transformers import TrainerCallback

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
            self, doc_df: pd.DataFrame, case_id, sent_boundaries, tokenizer, save_to_dir: Callable,
            eval_every=3, log_wandb=True, device='cuda', **kwargs
    ):
        super().__init__(**kwargs)
        self.doc_df = doc_df
        self.tokenizer = tokenizer
        self.case_id = case_id
        self.sent_boundaries = sent_boundaries
        self.eval_every = eval_every
        self.eval_count = 0
        self.save_to_dir = save_to_dir
        self.log_wandb = log_wandb
        self.device = device
        self.best_f1 = -1.0

        self.acc_metric = load_metric('accuracy')
        self.auc_metric = load_metric('roc_auc')
        self.prec_metric = load_metric('precision')
        self.rec_metric = load_metric('recall')
        self.f1_metric = load_metric('f1')

    def on_evaluate(self, args, state, control, model, **kwargs):
        if self.eval_count % self.eval_every != 0:
            self.eval_count += 1
            return
        self.eval_count += 1
        logger.info(f"======= Evaluating doc dataset =======")
        doc_df = self.doc_df.copy()
        # For some reason this takes more memory than during training, so cut batch size in half
        doc_df = inference.attach_predictions(
            doc_df, self.tokenizer, self.sent_boundaries, model,
            batch_size=args.per_device_eval_batch_size // 2, device=self.device
        )

        metrics = {
            'doc/f1': self.f1_metric.compute(
                predictions=doc_df.pred_label, references=doc_df.int_labels)['f1'],
            'doc/prec': self.prec_metric.compute(
                predictions=doc_df.pred_label, references=doc_df.int_labels)['precision'],
            'doc/rec': self.rec_metric.compute(
                predictions=doc_df.pred_label, references=doc_df.int_labels)['recall'],
            'doc/accuracy': self.acc_metric.compute(
                predictions=doc_df.pred_label, references=doc_df.int_labels)['accuracy'],
            'doc/roc_auc': self.auc_metric.compute(
                prediction_scores=doc_df.pred_score, references=doc_df.int_labels)['roc_auc'],
        }
        for key, val in metrics.items():
            logger.info(f"\t{key}: {val:.4f}")
        if self.log_wandb:
            wandb.log(metrics)

        # Normally the Trainer class keeps track of the best model using its standard evaluation loop on the test
        # set. In our case we want to use the doc dataset to select the best model, so we'll keep track here.
        f1 = metrics['doc/f1']
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
        logger.info(f"======= Evaluating training subset =======")
        output = self.trainer.predict(self.test_dataset, metric_key_prefix='train')
        self.trainer.log(output.metrics)
