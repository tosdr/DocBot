from argparse import ArgumentParser
import logging
import os
import random
import time

import boto3

from src import make_classification_datasets
from src.sent_spans import TEST_CASE_IDS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AWS_REGION = 'us-east-1'

def parallel_queue_name():
    if 'AWS_ACCOUNT' not in os.environ:
        raise RuntimeError(f"Set AWS_ACCOUNT env variable to use parallel training")
    return f'https://sqs.us-east-1.amazonaws.com/{os.environ["AWS_ACCOUNT"]}/tosdr-training.fifo'

def push(case_ids: list, parallel_key: str, empty_assert_timeout=30):
    sqs_client = boto3.client('sqs', region_name=AWS_REGION)

    # Assert nothing is in the queue. We have to do this for a few minutes because messages can be delayed
    #   https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/confirm-queue-is-empty.html
    logger.info(f"Confirming queue {parallel_queue_name()} is empty for {empty_assert_timeout} seconds")
    poll_start = time.time()
    while time.time() - poll_start < empty_assert_timeout:
        attrs = ['ApproximateNumberOfMessagesDelayed',
                 'ApproximateNumberOfMessagesNotVisible',
                 'ApproximateNumberOfMessages']
        response = sqs_client.get_queue_attributes(QueueUrl=parallel_queue_name(), AttributeNames=attrs)
        for attr in attrs:
            if int(response['Attributes'][attr]) > 0:
                raise Exception(f"Queue {parallel_queue_name()} not empty")
        time.sleep(10)

    for case_id in case_ids:
        logger.info(f"Pushing case {case_id} to queue {parallel_queue_name()}")
        # In order for multiple cases to be processed in parallel, we have to give them a different MessageGroupID
        # since this is a FIFO queue
        sqs_client.send_message(
            QueueUrl=parallel_queue_name(), MessageBody=str(case_id), MessageGroupId=f'{parallel_key}/{case_id}'
        )


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Add case IDs to an SQS queue to be trained on"
    )
    parser.add_argument(
        "--all", action='store_true',
        help="Train all ~150 case models, instead of a smaller group for testing"
    )
    parser.add_argument("--parallel_key", type=str, required=True)
    args = parser.parse_args()

    if args.all:
        dataset_dict = make_classification_datasets.load_sent_span()
        case_ids = list(dataset_dict.keys())
    else:
        case_ids = TEST_CASE_IDS

    # Consistent order for reproducability
    random.seed(0)
    random.shuffle(case_ids)

    push(case_ids, args.parallel_key)
