from argparse import ArgumentParser
import logging
import operator
from pathlib import Path
import pickle
import re

import boto3
import tqdm

from src.sent_spans import train_push, train


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

def list_case_models_s3(s3_client, upload_key) -> list[int]:
    case_ids = set()
    list_result = s3_client.list_objects(Bucket=train.RESULTS_S3_BUCKET, Prefix=f'{upload_key}/')
    for object_key in map(operator.itemgetter('Key'), list_result['Contents']):
        match = re.match(f'^{upload_key}/(\d+)/adapter_model.bin', object_key)
        if match is not None:
            case_ids.add(int(match.groups()[0]))
    return list(sorted(case_ids))


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Pull results from S3 after running train.py with a --upload_key"
    )
    parser.add_argument("--upload_key", type=str, required=True)
    args = parser.parse_args()

    output_dir = here / f'../../data/results/{args.upload_key}'
    output_dir.mkdir(parents=True, exist_ok=False)

    s3_client = boto3.client('s3', region_name=train_push.AWS_REGION)
    case_ids = list_case_models_s3(s3_client, args.upload_key)

    sent_res = dict()
    doc_res = dict()
    for case_id in tqdm.tqdm(case_ids):
        res = s3_client.get_object(Bucket=train.RESULTS_S3_BUCKET, Key=f'{args.upload_key}/{case_id}/{case_id}_sents.pkl')
        sent_res[case_id] = pickle.loads(res['Body'].read())
        res = s3_client.get_object(Bucket=train.RESULTS_S3_BUCKET, Key=f'{args.upload_key}/{case_id}/{case_id}_docs.pkl')
        doc_res[case_id] = pickle.loads(res['Body'].read())

    logger.info(f"Found {len(sent_res)} case results: {list(sorted(list(sent_res.keys())))}")

    sent_out = output_dir / 'sent_span_pred.pkl'
    logger.info(f"Writing to {sent_out}")
    pickle.dump(sent_res, open(sent_out, 'wb'))
    doc_out = output_dir / 'doc_pred.pkl'
    logger.info(f"Writing to {doc_out}")
    pickle.dump(doc_res, open(doc_out, 'wb'))
