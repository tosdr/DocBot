import io
import logging
import pickle
import re
from argparse import ArgumentParser

import boto3
import pandas as pd
import tqdm

from src.sent_spans import RESULTS_S3_BUCKET, train, train_push

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_case_results_s3(s3_client, model_version) -> list[int]:
    """Case IDs with doc predictions under {model_version}/ in S3"""
    case_ids = set()
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=RESULTS_S3_BUCKET, Prefix=f"{model_version}/"):
        for obj in page.get("Contents", []):
            match = re.match(rf"^{model_version}/(\d+)/\1_docs\.parquet$", obj["Key"])
            if match is not None:
                case_ids.add(int(match.groups()[0]))
    return sorted(case_ids)


def _download_parquet(s3_client, key) -> pd.DataFrame:
    res = s3_client.get_object(Bucket=RESULTS_S3_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(res["Body"].read()))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Pull training results from S3 (uploaded by train.py --upload) and rebuild the local prediction "
        "caches in data/training/{model_version}/, e.g. to recompute thresholds or audit labels without re-training"
    )
    parser.add_argument("--model_version", type=str, required=True)
    args = parser.parse_args()

    output_dir = train.results_dir(args.model_version)
    sent_out = output_dir / "sent_span_pred.pkl"
    doc_out = output_dir / "doc_pred.pkl"
    for path in [sent_out, doc_out]:
        if path.exists():
            raise FileExistsError(f"{path} already exists; move it aside to pull fresh results")
    output_dir.mkdir(parents=True, exist_ok=True)

    s3_client = boto3.client("s3", region_name=train_push.AWS_REGION)
    case_ids = list_case_results_s3(s3_client, args.model_version)

    sent_res = dict()
    doc_res = dict()
    for case_id in tqdm.tqdm(case_ids):
        prefix = f"{args.model_version}/{case_id}"
        sent_res[case_id] = _download_parquet(s3_client, f"{prefix}/{case_id}_sents.parquet")
        doc_res[case_id] = _download_parquet(s3_client, f"{prefix}/{case_id}_docs.parquet")

    logger.info(f"Found {len(sent_res)} case results: {sorted(sent_res.keys())}")

    logger.info(f"Writing to {sent_out}")
    pickle.dump(sent_res, open(sent_out, "wb"))
    logger.info(f"Writing to {doc_out}")
    pickle.dump(doc_res, open(doc_out, "wb"))
