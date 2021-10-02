import contextlib
import random
from argparse import ArgumentParser
import io
import logging
from pathlib import Path
import pickle
import shutil
import socket
import time

import boto3
import botocore
from datasets import load_metric, Dataset
import numpy as np
import pandas as pd
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from sklearn.metrics import f1_score
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    EarlyStoppingCallback
import wandb

from src import make_classification_datasets, inference, utils
from src.sent_spans import train_push, trainer_callbacks, TEST_CASE_IDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

DB_DUMP_VERSION = '211222'
CLASSIFICATION_VERSION = make_classification_datasets.LATEST_VERSION
RESULTS_S3_BUCKET = 'tosdr-training'

def best_model_loc(case_id):
    # Stores the best model of the latest fold of a run, according to F1 on the doc dataset
    # Will be overwritten by future trainings and folds, but good for saving "the best" model to be used in production
    dir_path = here / f'../../data/models/best/{case_id}'
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def best_model_loc_attempt(case_id):
    # Stores the best model during an individual run of a single attempt of a fold, according to F1 on the doc dataset
    # Will be overwritten by future trainings, folds, and attempt within the same fold
    dir_path = here / f'../../data/models/{case_id}/best'
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def get_sent_boundaries(doc_datasets: dict[int, pd.DataFrame]) -> dict[int, list[int]]:
    """
    Sentence splitting strategy was worked out in sent_splitting_benchmarks.py
    Each doc dataset has a column id_doc refering to the original id from the database dump `documents` table.
    The same doc can exist in multiple doc datasets (each is a single case), and in theory the texts should match,
    but we can confirm this here, and only extract sentences once per unique doc.
    :return: dict from doc id to list of char positions that start sentences
    """
    # Since this can take several minutes, cache
    cache_loc = here / f'../data/doc_classification_{make_classification_datasets.LATEST_VERSION}_sents.pkl'
    try:
        return pickle.load(open(cache_loc, mode='rb'))
    except FileNotFoundError:
        logger.info(f"Cached sentence boundaries not found")
        unique_texts = dict()
        for doc_dataset in doc_datasets.values():
            for id_doc, text in zip(doc_dataset.id_doc, doc_dataset.text):
                if id_doc in unique_texts:
                    assert text == unique_texts[id_doc]
                else:
                    unique_texts[id_doc] = text

        nlp = spacy.load('en_core_web_md', disable=['attribute_ruler', 'lemmatizer', 'ner'])
        sent_boundaries = dict()
        ids, texts = zip(*unique_texts.items())
        for doc_id, spacy_doc in tqdm(zip(ids, nlp.pipe(texts, n_process=1, batch_size=10)),
                                     total=len(unique_texts),
                                     desc='Splitting sentences'):
            # Filter sentences down to those with actual content
            sents = list(filter(lambda sent: sent.text != '' and not sent.text.isspace(), spacy_doc.sents))
            sent_boundaries[doc_id] = list(sorted(map(lambda s: s.start_char, sents)))
        pickle.dump(sent_boundaries, open(cache_loc, 'wb'))
        return sent_boundaries

def finetune(
        sent_spans_df, doc_df, case_id, model_base, lora, doc_sent_boundaries, train_kwargs,
        cv_folds=None, training_overrides=None, attempts_per_fold=1, device='cuda', log_wandb=True
):
    """
    Returns a tuple:
    - a copy of `sent_spans_df`, with `pred` set to a softmax probability of positive classification
    - a copy of `doc_df` also with predictions of the document dataset
    """
    if training_overrides is None:
        training_overrides = dict()

    tokenizer = AutoTokenizer.from_pretrained(model_base)
    def tokenize(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    sent_spans_df = sent_spans_df.copy()
    doc_df = doc_df.copy()
    doc_df['int_labels'] = [0 if i == 'negative' else 1 for i in doc_df.label]

    input_cases = sent_spans_df.copy()
    # To make logs quieter
    input_cases = input_cases.drop(
        ['service_id', 'quoteStart', 'char_end', 'status', 'point_id', 'quoteEnd', 'source', 'sent_idx_start',
         'sent_idx_end', 'quoteText', 'case_id', 'char_start', 'document_id', 'num_sents', 'lang', 'title'], axis=1)
    input_cases['label'] = (input_cases.label == 'positive').astype(int)

    dataset = Dataset.from_pandas(input_cases)
    tokenized = dataset.map(tokenize, batched=True)

    if cv_folds is None:
        cv_folds = input_cases.fold.nunique()
    # CV logic depends on a reset index
    assert input_cases.index.values.tolist() == list(range(len(input_cases)))

    doc_eval_kwargs = dict(tokenizer=tokenizer, sent_boundaries=doc_sent_boundaries, log_wandb=log_wandb, device=device)

    sent_spans_df['pred'] = pd.Series(dtype=float)
    for fold_i in range(cv_folds):
        train_idxs = input_cases[input_cases.fold != fold_i].index.values
        test_idxs = input_cases[input_cases.fold == fold_i].index.values
        test_doc_df = doc_df[doc_df.fold == fold_i]

        best_metric = -1.0
        best_pred_docs = None
        best_pred = None
        for attempt in range(attempts_per_fold):
            logger.info(f"Case id {case_id}: {sent_spans_df.iloc[0].title}")
            logger.info(f"Fold {fold_i + 1} / {cv_folds}, attempt {attempt + 1} / {attempts_per_fold}")
            logger.info(f"{len(train_idxs)} train instances, {len(test_idxs)} test instances")
            logger.info(f"{len(test_doc_df)} test doc instances")
            if log_wandb:
                run = wandb.init(project=f'tosdr_cases', group=str(case_id),
                                 config=dict(fold=fold_i, dataset_version=CLASSIFICATION_VERSION))
                wandb.define_metric('doc/f1', summary='max')
                wandb.define_metric('eval/f1_pos', summary='max')
            else:
                run = contextlib.nullcontext()
            with run:
                predictions, predictions_docs, pos_f1 = finetune_fold(
                    tokenized.select(train_idxs), tokenized.select(test_idxs), test_doc_df, case_id, model_base, lora,
                    training_overrides, doc_eval_kwargs, **train_kwargs
                )
            if pos_f1 > best_metric:
                best_metric = pos_f1
                best_pred_docs = predictions_docs
                best_pred = predictions
                # Copy the new best model for this fold to a more persistent location (note this will overwrite folds,
                # but it's fine for training a model for production based on one fold)
                shutil.copytree(best_model_loc_attempt(case_id), best_model_loc(case_id), dirs_exist_ok=True)

        # Turn the sentence span dataset's prediction activations into probabilities, save in `pred` column
        for i, n in enumerate(best_pred):
            softmax_probs = np.exp(n) / sum(np.exp(n))
            sent_spans_df.iloc[test_idxs[i], sent_spans_df.columns.get_loc('pred')] = softmax_probs[1]

        doc_df = doc_df.merge(best_pred_docs, 'left')

    return sent_spans_df, doc_df


def finetune_fold(
        train_dataset, test_dataset, test_doc_df, case_id, model_base, lora, training_overrides, doc_eval_kwargs,
        batch_size, learning_rate, eval_steps, early_stopping_patience, num_train_epochs, log_wandb,
        eval_training_set, device
):
    model = AutoModelForSequenceClassification.from_pretrained(model_base, num_labels=2)
    if lora:
        lora_args = dict(r=16, lora_alpha=16, lora_dropout=0.1, bias="all")
        logger.info(f"Training Lora with args {lora_args}")
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, **lora_args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    training_kwargs = dict(
        output_dir=(here / f'../../data/models/{case_id}'),
        evaluation_strategy='steps',
        save_strategy='steps',
        eval_steps=eval_steps,
        save_steps=eval_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        use_mps_device=device == 'mps',
        run_name=(model_base + '_lora' if lora else model_base),
        save_total_limit=1,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model='f1_pos',
    )
    if log_wandb:
        training_kwargs['report_to'] = 'wandb'
    training_kwargs.update(training_overrides)
    training_args = TrainingArguments(**training_kwargs)

    acc_metric = load_metric('accuracy')
    auc_metric = load_metric('roc_auc')
    prec_metric = load_metric('precision')
    rec_metric = load_metric('recall')
    f1_metric = load_metric('f1')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'f1_pos': f1_metric.compute(predictions=predictions, references=labels)['f1'],
            'prec_pos': prec_metric.compute(predictions=predictions, references=labels)['precision'],
            'rec_pos': rec_metric.compute(predictions=predictions, references=labels)['recall'],
            'accuracy': acc_metric.compute(predictions=predictions, references=labels)['accuracy'],
            'roc_auc': auc_metric.compute(prediction_scores=logits[:, 1], references=labels)['roc_auc'],
        }
    # Doc evaluation takes a long time (it applies the model to every single sentence) so during the training loop
    # cap the number of docs
    callback_doc_df = test_doc_df.groupby('label')\
        .apply(lambda x: x.sample(50) if x.shape[0]>=50 else x)\
        .sample(frac=1)     # shuffles

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            trainer_callbacks.DocEvalCallback(
                callback_doc_df, case_id, **doc_eval_kwargs, eval_every=3, save_to_dir=best_model_loc_attempt
            )
        ],
    )
    if eval_training_set:
        training_eval_dataset = train_dataset.shuffle(seed=0).select(range(len(test_dataset)))
        trainer.add_callback(
            trainer_callbacks.EvalTrainingSetCallback(training_eval_dataset, trainer)
        )
    trainer.train()

    best_dir = best_model_loc_attempt(case_id)
    logger.info(f"****** Loading final best model from {best_dir}")

    if lora:
        best_model = PeftModel.from_pretrained(
            AutoModelForSequenceClassification.from_pretrained(model_base, num_labels=2), best_dir
        ).merge_and_unload().to(device)
    else:
        best_model = trainer.model.from_pretrained(best_dir).to(device)

    # Doc dataset. F1 can later be used to pick the best one from multiple attempts
    test_doc_df = inference.attach_predictions(
        test_doc_df, **doc_eval_kwargs, model=best_model, batch_size=batch_size // 2
    )
    f1 = f1_metric.compute(predictions=test_doc_df.pred_label, references=test_doc_df.int_labels)['f1']

    # Sent span dataset. Might as well use the trainer.predict() API
    trainer.model = best_model
    pred_output = trainer.predict(test_dataset)

    return pred_output.predictions, test_doc_df, f1


def results_by_case(results_key) -> pd.DataFrame:
    """
    For later analysis; not used during training
    :return: case DF with new float columns: sent_span_f1, doc_f1
    """
    sent_span_output_path = here / f'../../data/results/{results_key}/sent_span_pred.pkl'
    doc_output_path = here / f'../../data/results/{results_key}/doc_pred.pkl'
    sent_span_results = pickle.load(open(sent_span_output_path, 'rb'))
    doc_results = pickle.load(open(doc_output_path, 'rb'))
    cases = pickle.load(open(here / f'../../data/cases_{DB_DUMP_VERSION}_clean.pkl', 'rb'))
    for case_id in sent_span_results:
        # The sent span eval logic at the end sets a "pred" score float. We'll find the optimal threshold that maximizes f1
        eval_set = sent_span_results[case_id][~sent_span_results[case_id].pred.isna()]
        int_labels = [0 if i == 'negative' else 1 for i in eval_set.label]
        cases.at[case_id, 'sent_span_f1'] = utils.optimal_threshold(eval_set.pred, int_labels)[3]

    for case_id in doc_results:
        # For the doc dataset eval, instead of a score it sets the optimal 0/1 label (this could be changed for consistency)
        eval_set = doc_results[case_id][~doc_results[case_id].pred_label.isna()]
        labels = [0 if i == 'negative' else 1 for i in eval_set.label]
        cases.at[case_id, 'doc_f1'] = f1_score(labels, eval_set.pred_label)

    return cases

def upload_results(upload_key, case_id, sent_pred_df, doc_pred_df):
    s3_client = boto3.client('s3', region_name=train_push.AWS_REGION)
    metadata = {'host': socket.gethostname()}

    # Dataframes with predictions
    for df, key in [
        (sent_pred_df, f'{upload_key}/{case_id}/{case_id}_sents.pkl'),
        (doc_pred_df, f'{upload_key}/{case_id}/{case_id}_docs.pkl'),
    ]:
        buffer = io.BytesIO()
        pickle.dump(df, buffer)
        buffer.seek(0)
        logger.info(f"Uploading to s3://{RESULTS_S3_BUCKET}/{key}")
        s3_client.upload_fileobj(
            buffer, RESULTS_S3_BUCKET, key,
            ExtraArgs={'Metadata': metadata, 'ContentType': 'application/x-binary'}
        )

    # Files necessary to use models in production (LoRA specific; skipping the base model)
    for filepath, key, content_type in [
        (best_model_loc(case_id) / 'adapter_model.bin', f'{upload_key}/{case_id}/adapter_model.bin',
         'application/x-binary'),
        (best_model_loc(case_id) / 'adapter_config.json', f'{upload_key}/{case_id}/adapter_config.json',
         'application/json'),
    ]:
        logger.info(f"Uploading to s3://{RESULTS_S3_BUCKET}/{key}")
        s3_client.upload_file(
            filepath.as_posix(), RESULTS_S3_BUCKET, key,
            ExtraArgs={'Metadata': metadata, 'ContentType': content_type}
        )

def check_uploadable(upload_key, case_id):
    s3_client = boto3.client('s3', region_name=train_push.AWS_REGION)
    s3_sents_obj_name = f"{upload_key}/{case_id}/{case_id}_sents.pkl"
    try:
        response = s3_client.head_object(Bucket=RESULTS_S3_BUCKET, Key=s3_sents_obj_name)
        raise Exception(f"S3 result object already exists: {s3_sents_obj_name}")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            pass
        else:
            raise e

def train_serial(args, device, train_fold_kwargs, dataset_dict, doc_dataset_dict, doc_sent_boundaries, log_wandb):
    """
    Method 2 of running: single process
    Looping through all cases. May take days to train. Results are saved locally as pickled dictionaries from
    case_id to dataframes that include predictions using the best model (as measured by F1 on the doc dataset).
    Results are written/cached after each case model, so if this crashes you can just restart.
    """
    results_key = 'test' if args.debug else str(int(time.time()))
    sent_span_output_path = here / f'../../data/results/{results_key}/sent_span_pred.pkl'
    doc_output_path = here / f'../../data/results/{results_key}/doc_pred.pkl'

    # Cache case trainings, in case of crash
    try:
        sent_pred_local = pickle.load(open(sent_span_output_path, 'rb'))
    except FileNotFoundError:
        sent_pred_local = dict()
    try:
        docs_pred_local = pickle.load(open(doc_output_path, 'rb'))
    except FileNotFoundError:
        docs_pred_local = dict()

    # For reproducability, a seed was set above for a consistent visitation order
    case_ids = list(dataset_dict.keys())
    random.shuffle(case_ids)
    for case_id in case_ids:
        if case_id in sent_pred_local:
            logger.info(f"Skipping {case_id}")
            continue

        # Sanity check results don't exist in S3
        if args.upload_key:
            check_uploadable(args.upload_key, case_id)

        sent_pred_df, doc_pred_df = finetune(
            dataset_dict[case_id], doc_dataset_dict[case_id], case_id, args.model_base, args.lora,
            doc_sent_boundaries, train_fold_kwargs, cv_folds=args.cv_folds,
            attempts_per_fold=args.attempts_per_fold, log_wandb=log_wandb, device=device
        )
        sent_pred_local[case_id] = sent_pred_df.copy()
        docs_pred_local[case_id] = doc_pred_df.copy()

        sent_span_output_path.parent.mkdir(exist_ok=True, parents=True)
        pickle.dump(sent_pred_local, open(sent_span_output_path, 'wb'))
        pickle.dump(docs_pred_local, open(doc_output_path, 'wb'))

        # Upload results to S3
        if args.upload_key:
            upload_results(args.upload_key, case_id, sent_pred_df, doc_pred_df)

def train_parallel(args, device, train_fold_kwargs, dataset_dict, doc_dataset_dict, doc_sent_boundaries, log_wandb):
    """
    Method 1 of running: distributed training by setting --parallel_key
    Instead of looping through case IDs, we'll dequeue them from SQS. This assumes train_push.py was run first.
    """
    sqs_client = boto3.client('sqs', region_name=train_push.AWS_REGION)

    receives = 0
    max_poll_time_s = 300
    while True:
        logger.info(f"Polling {train_push.parallel_queue_name()} for {max_poll_time_s}s")
        start_time = time.time()
        found = False
        while time.time() - start_time < max_poll_time_s:
            response = sqs_client.receive_message(
                QueueUrl=train_push.parallel_queue_name(), WaitTimeSeconds=20, AttributeNames=['MessageGroupId']
            )
            if 'Messages' in response:
                found = True
                break
        if not found:
            if receives == 0:
                logger.error(f"No messages found upon startup")
            else:
                logger.info(f"No messages found, exiting")
            break
        else:
            receives += 1
            message = response['Messages'][0]
            logger.info(f"Received case ID {message['Body']}")
            receipt_handle = message['ReceiptHandle']

            try:
                case_id = int(message['Body'])
                # Sanity check the parallel_key used in train_push.py matches
                found_key = message['Attributes']['MessageGroupId'].split('/')[0]
                if found_key != args.parallel_key:
                    raise Exception(f"Different parallel_key found {found_key} != {args.parallel_key}")

                # Sanity check results don't exist in S3
                if args.upload_key:
                    check_uploadable(args.upload_key, case_id)

                sent_pred_df, doc_pred_df = finetune(
                    dataset_dict[case_id], doc_dataset_dict[case_id], case_id, args.model_base, args.lora,
                    doc_sent_boundaries, train_fold_kwargs, cv_folds=args.cv_folds,
                    attempts_per_fold=args.attempts_per_fold, log_wandb=log_wandb, device=device
                )

                # Upload results to S3
                if args.upload_key:
                    upload_results(args.upload_key, case_id, sent_pred_df, doc_pred_df)

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


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Fine-tune BERT on the sentence span dataset created in make_classification_datasets.py. "
                    "Also uses the document dataset during evaluation."
    )
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--all", action='store_true',
                        help="Train all ~150 case models, instead of just enough to optimize. If --parallel_key is "
                             "set this is ignored, and training assignments come from SQS")
    parser.add_argument("--eval_training", action='store_true', help="Evaluate training set alongside the test set")
    parser.add_argument("--parallel_key", type=str, required=False,
                        help="Use multiple hosts or GPUs/processes to train cases. Note you first "
                             "have to use the same key while running train_push.py, which will push case IDs to SQS")
    parser.add_argument("--upload_key", type=str, required=False,
                        help="If provided, LORA models and test set predictions will be uploaded to S3")
    parser.add_argument("--model_base", type=str,
                        choices=['bert-base-uncased', 'roberta-base', 'nlpaueb/legal-bert-base-uncased'],
                        default='bert-base-uncased')
    parser.add_argument("--lora", action='store_true', help="Instead of fine-tuning the full model, train a lora")
    parser.add_argument("--no_lora", action='store_false', help="Fine-tune the whole model")
    parser.add_argument("--batch_size", type=int, default=32, help="Default is ideal for 24gb GPU memory")
    parser.add_argument("--learning_rate", type=float, default=5e-04)
    parser.add_argument("--attempts_per_fold", type=int, default=3)
    parser.add_argument("--cv_folds", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=24)
    parser.set_defaults(debug=False, all=False, eval_training=False, lora=True)
    args = parser.parse_args()

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
                dataset_dict[key] = dataset_dict[key].sample(200).reset_index(drop=True)
                doc_dataset_dict[key] = doc_dataset_dict[key].groupby(['fold', 'label']).head(10).reset_index(drop=True)
        else:
            del dataset_dict[key]
            del doc_dataset_dict[key]

    random.seed(0)
    np.random.seed(0)
    trainer_callbacks.logger = logger

    log_wandb = True
    device = 'cuda'
    if args.debug:
        log_wandb = False
        device = 'mps'
        args.batch_size = 4
        args.attempts_per_fold = 2
        args.eval_steps = 10
        args.learning_rate = 1e-04
        args.num_train_epochs = 1
        args.early_stopping_patience = 2
        if args.parallel_key:
            logger.info("Testing SQS workflow")
            train_push.push(dataset_dict.keys(), 'debug')

    train_fold_kwargs = dict(
        batch_size=args.batch_size, learning_rate=args.learning_rate, eval_steps=args.eval_steps, log_wandb=log_wandb,
        early_stopping_patience=args.early_stopping_patience, num_train_epochs=args.num_train_epochs,
        eval_training_set=args.eval_training, device=device
    )

    if args.parallel_key is None:
        # Method 1 of running: for loop through cases
        train_serial(args, device, train_fold_kwargs, dataset_dict, doc_dataset_dict, doc_sent_boundaries, log_wandb)
    else:
        # Method 2 of running: distributed training via SQS by setting --parallel_key
        train_parallel(args, device, train_fold_kwargs, dataset_dict, doc_dataset_dict, doc_sent_boundaries, log_wandb)
