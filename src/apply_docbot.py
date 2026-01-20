from argparse import ArgumentParser
from collections import defaultdict
from contextlib import contextmanager
from io import BytesIO
import logging
import operator
import os
from pathlib import Path
import pickle
import re
import tempfile
import time

import boto3
import pandas as pd
from peft import PeftModel
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import inference, phoenix, utils, apply_local
from src.inference import MODEL_VERSION

"""
Script run weekly within ToS;DR to apply our case models to any recently crawled privacy/terms documents.
Uses Phoenix API enpoints (phoenix.py) to retrieve docs and points, and to POST new points.
"""

here = Path(__file__).parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# logging.getLogger('src').setLevel(logging.DEBUG)

MODEL_S3_BUCKET = 'tosdr-training'
AWS_REGION = 'us-east-1'

DOCBOT_USER_ID = 21032


LOG_DIR = here / '../logs'
def get_results_dir(timestamp):
    return here / f'../data/results/{timestamp}'


class DocStore:
    def __init__(self, local_data: bool, phoenix_client):
        logger.info("Loading spacy model")
        # Minimal pipeline for sentence splitting
        self.spacy_model = spacy.load('en_core_web_md', disable=['attribute_ruler', 'lemmatizer', 'ner'])

        self.local_data = local_data
        self.phoenix_client = phoenix_client
        self.local_docs = pd.read_pickle(here / f'../data/documents_{apply_local.LOCAL_DUMP_VERSION}.pkl')

        # Since we plan to use one machine we can just cache docs here. It we scale up to a big cluster we can have them
        # share a cache.
        # Update: since fetching/prepping docs is the slowest step, and we're seeing intermittent crashes from API DNS
        # failures, I'm adding a cache on disk so this is preserved across restarts.
        self.cache_loc = here / '../data/docbot_doc_cache.pkl'
        try:
            self.docs = pickle.load(open(self.cache_loc, 'rb'))
        except FileNotFoundError:
            self.docs = dict()

    def __getitem__(self, doc_id):
        if doc_id not in self.docs:
            self.docs[doc_id] = self.fetch_doc(doc_id)
            pickle.dump(self.docs, open(self.cache_loc, 'wb'))
        return self.docs[doc_id]

    def fetch_doc(self, doc_id):
        if self.local_data:
            doc = self.local_docs.loc[doc_id].to_dict()
        else:
            doc = self.phoenix_client.get_doc(int(doc_id))

        # Precompute language, HTML-free content, and sentence boundaries
        if doc is not None and doc['text'] is not None:
            content = utils.preprocess_doc_text(doc['text'])
            doc['content'] = content
            doc['lang'] = inference.detect_lang(content)

            if doc['lang'] == 'en':
                # Spacy has a default limit of 1mil characters. This can be increased, but probably safest to have a ceiling
                spacy_doc = self.spacy_model(content[:1000000])
                sents = list(filter(lambda sent: sent.text != '' and not sent.text.isspace(), spacy_doc.sents))
                sent_boundaries = list(sorted(map(lambda s: s.start_char, sents)))
                doc['sent_boundaries'] = sent_boundaries

        return doc


@contextmanager
def s3_fileobj(bucket, key, s3_client):
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    yield BytesIO(obj['Body'].read())

def get_aws_creds() -> dict[str, str]:
    client = boto3.client('sts')
    response = client.assume_role(
        RoleArn=os.environ['AWS_ROLE'],
        RoleSessionName=f'docbot-infer-{int(time.time())}',
        DurationSeconds=43200   # 12 hour max
    )
    return {
        'aws_access_key_id': response['Credentials']['AccessKeyId'],
        'aws_secret_access_key': response['Credentials']['SecretAccessKey'],
        'aws_session_token': response['Credentials']['SessionToken']
    }

def load_peft_model(case_id, base_model, local_models):
    if local_models:
        peft_model_loc = apply_local.peft_path(case_id)
        logger.info(f"Initializing PEFT adapter from {peft_model_loc}")
    else:
        # Each case could take a long time, so get new IAM Role credentials to be safe (12 hour limit)
        s3_client = boto3.client('s3', **get_aws_creds(), region_name=AWS_REGION)

        tmpdir = tempfile.TemporaryDirectory()
        logger.info(f"Pulling PEFT adapter from: s3://{MODEL_S3_BUCKET}/{MODEL_VERSION}/{case_id}/adapter_model.bin")
        for filename in ['adapter_model.bin', 'adapter_config.json']:
            with s3_fileobj(MODEL_S3_BUCKET, f'{MODEL_VERSION}/{case_id}/{filename}', s3_client) as f:
                with open(f'{tmpdir.name}/{filename}', 'wb') as to_file:
                    to_file.write(f.read())
        peft_model_loc = tmpdir.name

    model = PeftModel.from_pretrained(base_model, peft_model_loc)
    # https://github.com/huggingface/peft/issues/217#issuecomment-1506224612
    return model.merge_and_unload()

def run_case(
        case_id: int, local_models: bool, local_data: bool, dont_post: bool,
        doc_list: list[tuple[int, str]], doc_store, phoenix_client, threshold, batch_size, device
):
    if local_data:
        local_points = pd.read_pickle(here / f'../data/points_{apply_local.LOCAL_DUMP_VERSION}.pkl')
        points = local_points[local_points.case_id == case_id].copy()
    else:
        points_list = phoenix_client.get_points_for_case(case_id)
        # Turn list of dicts into a dataframe, to match the access pattern when local_data is True
        points = pd.DataFrame(points_list)
    logger.info(f"Found {len(points)} points")
    # Points document_id is a float, we'll want to turn it into a int for comparison (ideally this should be done earlier)
    points['document_id'] = points.document_id.apply(lambda doc_id: None if pd.isna(doc_id) else int(doc_id))

    # Model loading
    prefilter_kwargs = inference.load_prefilter_kwargs(case_id)
    base_model = AutoModelForSequenceClassification.from_pretrained(inference.BASE_MODEL_NAME)
    model = load_peft_model(case_id, base_model, local_models)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(inference.BASE_MODEL_NAME)

    # In theory this should let us test how big we can make our batches, though I've noticed it can still run OOM
    inference.test_gpu_memory(batch_size, model, device, tokenizer.model_max_length)

    if local_data:
        visited_docs = set()
    else:
        visited_docs = phoenix_client.get_docbot_records(case_id, MODEL_VERSION)
        logger.info(f"{len(visited_docs)} records found for case {case_id} version {MODEL_VERSION}")

    result_counts = defaultdict(int)
    result_scores = dict()
    prefilter_rates = []
    for doc_id, text_version in doc_list:
        try:
            if (doc_id, text_version) in visited_docs:
                logger.debug(f"Skipping {doc_id, text_version}")
                result_counts['skip_visited'] += 1
                continue

            doc = doc_store[doc_id]
            if doc is None:
                # Edge case that should only come up when the original GET all doc IDs call returns some ID that can no
                # longer be retreived from phoenix. Don't mark it as visited.
                result_counts['skip_notfound'] += 1
                continue

            if 'text' not in doc or doc['text'] is None or doc['content'].strip() == '':
                result_counts['skip_empty'] += 1
                # Mark as visited
                if not dont_post:
                    phoenix_client.add_docbot_record(case_id, doc_id, text_version, MODEL_VERSION)
                continue

            if doc['lang'] != 'en':
                result_counts['skip_non_en'] += 1
                if not dont_post:
                    # If the doc is not english, mark visited so we don't re-consider it next time
                    phoenix_client.add_docbot_record(case_id, doc_id, text_version, MODEL_VERSION)
                continue


            # If doc has points of any status except for declined or draft (approved, pending, disputed, changes-requested)
            # we don't need to apply the model
            doc_points = points[points.document_id == doc_id]
            skipable_statuses = set(doc_points.status.values) - {'declined', 'draft'}
            if len(skipable_statuses) > 0:
                logger.debug(f"Skipping due to existing {skipable_statuses} points")
                # Don't mark visited because point statuses can change.
                # We could probably mark visited if there are approved points, but it's not a big deal to keep re-checking
                result_counts['skip_points'] += 1
                continue

            # Mark declined points as offlimits, so those character spans don't get suggested again
            #TODO if the document changes, the declined point spans will be incorrect. Really, we should add
            # text_version to the Point schema and only invalidate if they are current.
            declined_points = doc_points[doc_points.status == 'declined']
            off_limits = [
                (point.quote_start, point.quote_end) for i, point in declined_points.iterrows()
                if not pd.isna(point.quote_start) and not pd.isna(point.quote_end)
            ]

            score, best_start, best_end, _, filter_rate = inference.apply_sent_span_model(
                doc['content'], doc['sent_boundaries'],
                prefilter_kwargs,
                tokenizer, model, batch_size, device,
                off_limits
            )
            prefilter_rates.append(filter_rate)

            # TODO we shouldn’t suggest case 216 if it conflicts with case 220; maybe other high-level rules
            # or should we handle this downstream, either in curator guide (/suggestion in UI) or in grading?
            # If we want to handle it here, things might get tricky since we're decoupling reaching this point
            # vs POSTing results (we could always resolve these overlaps later before posting)

            success = score >= threshold
            if success:
                logger.info(f"✅ Doc {doc_id} scored {score:.3f} with span:\n{doc['content'][best_start:best_end]}")
            result_counts['ran_found' if success else 'ran_notfound'] += 1
            result_scores[doc_id] = score

            if not dont_post:
                # Submit a quote using the original text (with HTML) as opposed to doc['content']
                if success:
                    analysis = f'Created by Docbot version {MODEL_VERSION}'
                    phoenix_client.add_point(
                        case_id, DOCBOT_USER_ID, doc_id, doc['service_id'], doc['url'], analysis,
                        doc['text'][best_start:best_end], best_start, best_end, MODEL_VERSION, score
                    )
                phoenix_client.add_docbot_record(
                    case_id, doc_id, text_version, MODEL_VERSION, best_start, best_end, score
                )
        except Exception as e:
            raise RuntimeError(f"Issue processing doc {doc_id} text_version {text_version} case {case_id}") from e

    return result_counts, result_scores, sum(prefilter_rates) / len(prefilter_rates) if len(prefilter_rates) > 0 else 0.

def get_all_docs(local_data, phoenix_client) -> list[tuple[int, str]]:
    if local_data:
        ids = pd.read_pickle(here / f'../data/documents_{apply_local.LOCAL_DUMP_VERSION}.pkl').id.values
        # There was no text_version in the older documents schema, so just pretend they're all '0'
        return [(i, '0') for i in ids]
    else:
        return phoenix_client.get_docs()

def list_case_models_s3(s3_client) -> list[int]:
    case_ids = set()
    list_result = s3_client.list_objects(Bucket=MODEL_S3_BUCKET, Prefix=f'{MODEL_VERSION}/')
    for object_key in map(operator.itemgetter('Key'), list_result['Contents']):
        match = re.match(f'^{MODEL_VERSION}/(\d+)/adapter_model.bin', object_key)
        if match is not None:
            case_ids.add(int(match.groups()[0]))
    return list(sorted(case_ids))

def save_results(case_result_scores, case_result_counts, results_dir, s3_client, timestamp):
    for obj, name in [(case_result_scores, 'result_scores'), (case_result_counts, 'result_counts')]:
        # Save locally
        pickle.dump(obj, open(results_dir / f'{name}.pkl', 'wb'))

        # Save to s3, optionally
        if s3_client is not None:
            buffer = BytesIO()
            pickle.dump(obj, buffer)
            buffer.seek(0)
            s3_key = f'infer/{timestamp}/{name}.pkl'
            logger.info(f"Uploading to s3://{MODEL_S3_BUCKET}/{s3_key}")
            s3_client.upload_fileobj(
                buffer, MODEL_S3_BUCKET, s3_key,
                ExtraArgs={'ContentType': 'application/x-binary'}
            )

def run_all_cases(limit: int, local_models: bool, local_data: bool, dont_post: bool, batch_size: int, device: str):
    start_s = time.time()
    LOG_DIR.mkdir(exist_ok=True)
    logger.addHandler(logging.FileHandler(LOG_DIR / f'{int(start_s)}.log'))
    logger.info(f"Starting at {int(start_s)}")

    phoenix_client = None if (local_data and dont_post) else phoenix.Client()

    if not dont_post or not local_models:
        s3_client = boto3.client('s3', **get_aws_creds(), region_name=AWS_REGION)

    # Load a list of case IDs
    if local_models:
        case_id_strs = filter(lambda f: f.isdigit(), os.listdir(apply_local.LOCAL_PEFT_PATH))
        case_ids = list(sorted(list(map(int, case_id_strs))))
    else:
        case_ids = list_case_models_s3(s3_client)

    logger.info(f"Found {len(case_ids)} case IDs: {case_ids}")

    doc_list = get_all_docs(local_data, phoenix_client)
    logger.info(f"Found {len(doc_list)} docs")
    if limit:
        case_ids = [134]
        # case_ids = case_ids[:25]
        doc_list = doc_list[:limit]
        logger.info(f"Limiting to {len(doc_list)} docs, cases {case_ids}")

    doc_store = DocStore(local_data, phoenix_client)

    total_result_counts = defaultdict(int)
    # Maps from case ID to status to count
    case_result_counts: dict[int, dict[str, int]] = dict()
    # Maps from case ID to doc ID to score
    case_result_scores: dict[int, dict[int, float]] = dict()
    timestamp_key = int(start_s)
    results_dir = get_results_dir(timestamp_key)
    results_dir.mkdir(exist_ok=True, parents=True)

    for case_idx, case_id in enumerate(case_ids):
        logger.info(f"====================== Case {case_id} ({case_idx}/{len(case_ids)}) ========================")
        case_start_s = time.time()

        result_counts, result_scores, mean_prefilter_rate = run_case(
            case_id, local_models, local_data, dont_post,
            doc_list, doc_store, phoenix_client, inference.THRESHOLDS[case_id], batch_size, device
        )
        for result, count in result_counts.items():
            total_result_counts[result] += count
        case_result_counts[case_id] = result_counts
        case_result_scores[case_id] = result_scores

        logger.info(f"Case {case_id} took {time.time() - case_start_s:.2f} seconds to process")
        logger.info(f"Mean filter rate for TF-IDF prefilter: {100 * mean_prefilter_rate:.2f}%")
        logger.info(f"Results:\n\t{dict(result_counts)}")
        if len(result_scores) > 0:
            logger.info(f"Prediction scores:\n{pd.Series(list(result_scores.values())).describe()}")

        # Serialize latest results in case of crash
        # Get new S3 credentials in case the 12 hour limit ran out
        if not dont_post or not local_models:
            s3_client = boto3.client('s3', **get_aws_creds(), region_name=AWS_REGION)
        save_results(case_result_scores, case_result_counts, results_dir, None if dont_post else s3_client, timestamp_key)

    end_s = time.time()
    logger.info(f"Ending at {int(end_s)} for a duration of {end_s - start_s:.3f} seconds")
    logger.info(f"Results:\n\t{dict(total_result_counts)}")
    save_results(case_result_scores, case_result_counts, results_dir, None if dont_post else s3_client, timestamp_key)

def resolve_device(args):
    if torch.cuda.is_available():
        device = 'cuda'
    elif args.cuda_only:
        raise RuntimeError("Cuda not available")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info(f"Using device {device}")
    return device

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--limit', type=int, required=False,
        help='Run one case model on some docs, for testing'
    )
    parser.add_argument(
        '--local_models', action='store_true',
        help='Load HF models from disk instead of S3'
    )
    parser.add_argument(
        '--local_data', action='store_true',
        help='Load docs, points, and docbot history from disk instead of calling Phoenix APIs'
    )
    parser.add_argument(
        '--dont_post', action='store_true',
        help='Instead of POSTing new docbot history and new points to Phoenix, just print'
    )
    parser.add_argument(
        '--cuda_only', action='store_true',
        help='Only run if cuda hardware is available. Will still run on CUDA if not set, this just enforces it.'
    )
    parser.add_argument('--batch_size', type=int, default=16)
    parser.set_defaults(local_models=False, local_data=False, dont_post=False, cuda_only=False)
    args = parser.parse_args()

    device = resolve_device(args)
    run_all_cases(args.limit, args.local_models, args.local_data, args.dont_post, args.batch_size, device)
