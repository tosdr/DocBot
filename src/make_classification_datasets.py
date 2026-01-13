import bisect
import itertools
import logging
from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

from src import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

MIN_APPROVED = 40     # Focus on cases with enough approved points to fine-tunen and evaluate
NUM_FOLDS = 5         # Pre-determined train-test splits for downstream ML

DB_DUMP_VERSION = '211222'
LATEST_VERSION = '211222_v2_corrected_5-28-23'

# Outputs are saved here
SENT_SPAN_LOC = here / f'../data/sent_span_classification_{LATEST_VERSION}.pkl'
DOC_LOC = here / f'../data/doc_classification_{LATEST_VERSION}.pkl'

"""
This module takes database dumps (cleaned up by explore.ipynb) and forms two binary text classification datasets
for each case: 1) sentence spans that look similar to points, 2) documents
We create both at the same time to ensure that we use the same pre-split folds for cross validation. This allows us 
to compare full document classification approaches (e.g. sequence tagging) with those that work on sentence spans.

For sentence span datasets, instances are similar to points, but with [sent_idx_start, sent_idx_end, sent_text, label,
source, fold] fields, and with boundaries stretched to start/end on sentence boundaries (to match the input format 
during inference).
    `fold` is an int from 0-4
    `label` is `positive` or `negative`
    `source` is one of [
        'approved', annotated by a human and approved by curator
        'declined', annotated by a human and declined by curator
        'surrounding', sentences on either side of approved points, which in theory should not contain evidence
        'topical', approved points of other cases in the same topic area
        'doc_random', random sentence segments from the docs of approved points
        'reviewed_random', random sentence segments from comprehensively reviewed docs
    ]
Negative instances were designed to allow models to learn the exact patterns showing evidence for cases, without 
just learning what topical words are.

Instances of the document datasets look very similar to the `documents` table exported from the DB, but with ['fold', 
'text', 'label', 'source']
`source` is one of [
        'approved', annotated by a human and approved by curator
        'declined', annotated by a human and declined by curator
        'reviewed', comprehensively reviewed docs that don't have an approved point for the case
    ] 
"""


# Utility functions, to load after this script has been run
def load_sent_span() -> dict[int, pd.DataFrame]:
    return pickle.load(open(SENT_SPAN_LOC, 'rb'))

def load_docs() -> dict[int, pd.DataFrame]:
    return pickle.load(open(DOC_LOC, 'rb'))


def _get_sent_boundaries(documents) -> dict[int, list[int]]:
    """
    Sentence splitting strategy was worked out in sent_splitting_benchmarks.py
    :return: dict from doc id to list of char positions that start sentences
    """
    # Since this can take ~10 minutes, cache
    cache_loc = here / f'../data/documents_{DB_DUMP_VERSION}_sents.pkl'
    try:
        return pickle.load(open(cache_loc, mode='rb'))
    except FileNotFoundError:
        logger.info(f"Cached sentence boundaries not found")
        nlp = spacy.load('en_core_web_md', disable=['attribute_ruler', 'lemmatizer', 'ner'])
        sent_boundaries = dict()
        for doc_id, doc in tqdm(zip(documents.id_doc, nlp.pipe(documents.text, n_process=4, batch_size=10)),
                                total=len(documents),
                                desc='Splitting sentences'):
            # Filter sentences down to those with actual content
            sents = list(filter(lambda sent: sent.text != '' and not sent.text.isspace(), doc.sents))
            sent_boundaries[doc_id] = list(sorted(map(lambda s: s.start_char, sents)))

        pickle.dump(sent_boundaries, open(cache_loc, 'wb'))
        return sent_boundaries


def _stretch_points(points: pd.DataFrame, sent_boundaries):
    """
    Expands the boundaries of points to start/end at sentence boundaries.

    :return: a copy of `points` with sent_idx_start, sent_idx_end, and num_sents set.
    Indices refer to sent_boundaries. sent_idx_end is exclusive right bound (to match python indexing), e.g.
    sent_idx_start and sent_idx_end of 3 & 5 means the instance spans two sentences
    """
    points = points.copy()
    for doc_id in sent_boundaries:
        for point_i, point in points[points.document_id == doc_id].iterrows():
            sent_idx_start = bisect.bisect_right(sent_boundaries[doc_id], point.quoteStart) - 1
            sent_idx_end = bisect.bisect_left(sent_boundaries[doc_id], point.quoteEnd)

            points.at[point_i, 'sent_idx_start'] = sent_idx_start
            points.at[point_i, 'sent_idx_end'] = sent_idx_end
            points.at[point_i, 'num_sents'] = sent_idx_end - sent_idx_start
    points.sent_idx_start = points.sent_idx_start.astype('Int64')
    points.sent_idx_end = points.sent_idx_end.astype('Int64')
    points.num_sents = points.num_sents.astype('Int64')
    return points

def _get_sample_window_size_fn(points):
    """
    Some negative instances will be randomly sampled from elsewhere in the doc or other random docs. These could be
    sentence windows of any length, but we'd like the distribution of windows lengths to be
    about the same between positive and negative instances so the model doesn't learn it as a feature.
    This distribution can vary by case. So create numpy arrays representing a multimodal probability distribution
    for different window lengths.
    :return: function that takes a case ID and return a realistic num sentences
    """
    case_size_proportions = dict()
    for case_id in points.case_id.unique():
        props = points[points.case_id == case_id].num_sents.value_counts(normalize=True)
        prop_array = np.array([0.0] * max(props.index))
        prop_array[[i - 1 for i in props.index]] = props
        case_size_proportions[case_id] = prop_array

    def sample_window_size(case_id):
        return np.random.choice(range(1, len(case_size_proportions[case_id]) + 1), p=case_size_proportions[case_id])

    return sample_window_size

def _make_surrounding(approved_df, sample_window_size_fn, sent_boundaries, target_case):
    surrounding = []
    for _, point in approved_df[approved_df.case_id == target_case.id].iterrows():
        if point.sent_idx_start >= 1:
            before_point = point.copy()
            before_point.sent_idx_end = point.sent_idx_start
            # Back the start up a realistic number of sentences
            before_point.sent_idx_start = max(point.sent_idx_start - sample_window_size_fn(target_case.id), 0)
            surrounding.append(before_point)
        if point.sent_idx_end < len(sent_boundaries[point.document_id]):
            after_point = point.copy()
            after_point.sent_idx_start = point.sent_idx_end
            # Advance the end a realistic number of sentences
            after_point.sent_idx_end = min(point.sent_idx_end + sample_window_size_fn(target_case.id),
                                           len(sent_boundaries[point.document_id]))
            surrounding.append(after_point)
    surrounding_df = pd.DataFrame(surrounding)
    surrounding_df = surrounding_df.assign(label='negative', source='surrounding', point_id=np.NaN,
                                           status=np.NaN, quoteText=np.NaN, quoteStart=np.NaN, quoteEnd=np.NaN)
    surrounding_df = surrounding_df.drop('id', axis=1)
    return surrounding_df

def _make_random_reviewed(approved, documents, instance_df, points, sample_window_size_fn, sent_boundaries, target_case):
    # There is an is_comprehensively_reviewed feature, but also make sure at least 10 points were submitted for
    # the doc. Avoid any docs that have a point for the case in question.
    # Of those elligible, choose 3*len(approved) random doc IDs, then for each of those choose a sent start/end
    elligible_docs = documents[(documents.is_comprehensively_reviewed) &
                               (documents.num_points >= 10) &
                               (~documents.id_doc.isin(points[points.case_id == target_case.id].document_id))]
    # Use one of the other instances as a template
    instance_template = instance_df.iloc[0]

    reviewed_random = []
    desired_num = 3 * len(approved)
    for doc_id in np.random.choice(elligible_docs.index, size=desired_num, replace=True):
        instance_num_sents = min(sample_window_size_fn(target_case.id), len(sent_boundaries[doc_id]))
        instance = instance_template.copy()
        random_sent_start = len(sent_boundaries[doc_id]) - instance_num_sents
        instance.sent_idx_start = np.random.randint(random_sent_start) if random_sent_start > 0 else 0
        instance.sent_idx_end = instance.sent_idx_start + instance_num_sents
        instance.document_id = doc_id
        instance.service_id = documents.loc[doc_id].service_id
        instance.lang = documents.loc[doc_id].lang
        instance.point_id = np.NaN
        reviewed_random.append(instance)
    reviewed_random_df = pd.DataFrame(reviewed_random)
    reviewed_random_df = reviewed_random_df.assign(label='negative', source='reviewed_random', status=np.NaN,
                                                   quoteText=np.NaN, quoteStart=np.NaN, quoteEnd=np.NaN)
    return reviewed_random_df


def _make_random_from_approved(approved, instance_df, sample_window_size_fn, sent_boundaries, target_case):
    doc_random = []
    for doc_id in approved.document_id.unique():
        doc_instances = instance_df[instance_df.document_id == doc_id]
        existing_ranges = doc_instances[['sent_idx_start', 'sent_idx_end']].values
        # Higher is more diverse, but worse class imbalance
        for i in range(5):
            # We want to avoid sentence spans already represented by previous instances. It's totally possible to
            # write logic to do this perfectly, but the chances of collisions are small (around (5 / n_sentences))
            # so just try it randomly up to 5 times and then move on
            instance_num_sents = sample_window_size_fn(target_case.id)
            disallowed_sent_idx = set(itertools.chain.from_iterable([range(start - instance_num_sents, end)
                                                                     for start, end in existing_ranges]))
            for sent_idx_start in np.random.randint(len(sent_boundaries[doc_id]), size=5):
                if sent_idx_start not in disallowed_sent_idx:
                    # Use one of the other instances as a template
                    instance = doc_instances.iloc[0].copy()
                    instance.sent_idx_start = int(sent_idx_start)
                    instance.sent_idx_end = int(sent_idx_start) + instance_num_sents
                    doc_random.append(instance)
                    break
    doc_random_df = pd.DataFrame(doc_random)
    doc_random_df = doc_random_df.assign(
        label='negative', source='doc_random', point_id=np.NaN, status=np.NaN, quoteText=np.NaN,
        quoteStart=np.NaN, quoteEnd=np.NaN
    )
    return doc_random_df


def _make_topical(approved, cases, points, target_case):
    topical = points[(points.case_id != target_case.id) &
                     (points.case_id.isin(cases[cases.topic_id == target_case.topic_id].id)) &
                     (points.status == 'approved')]
    # Remove those that overlap with approved points of the target case
    to_remove = []
    for i, point in topical.iterrows():
        for i2, approved_point in approved[approved.document_id == point.document_id].iterrows():
            # If overlapping sentence indices
            if approved_point.sent_idx_start <= (point.sent_idx_end - 1) and\
                    point.sent_idx_start <= (approved_point.sent_idx_end - 1):
                to_remove.append(i)
    topical = topical.drop(to_remove)

    # to limit class imbalance
    desired_num = 3 * len(approved)
    if len(topical) > desired_num:
        topical = topical.sample(desired_num, random_state=0)
    topical = topical.assign(label='negative', source='topical', point_id=topical.id,
                             status=np.NaN, quoteText=np.NaN, quoteStart=np.NaN, quoteEnd=np.NaN)
    topical = topical.drop('id', axis=1)
    return topical


def make_sent_span_datasets(cases, documents, points, topics, services, en_only=True) -> dict[int, pd.DataFrame]:
    """
    :return: dict from case id to DF of instances.
        An instance is similar to point, but with [sent_idx_start, sent_idx_end, sent_text, label, source] fields
            `label` is `positive` or `negative`
            `source` is one of ['approved', 'declined', 'surrounding', 'topical', 'random']
    """
    if en_only:
        documents = documents[documents.lang == 'en']
        points = points[points.lang == 'en']
    point_counts = points[points.status == 'approved'].case_id.value_counts()
    cases.at[point_counts.index, 'num_approved'] = point_counts

    # Join service info onto documents (we'll use is_comprehensively_reviewed), attach num points
    documents = pd.merge(documents, services, left_on='service_id', right_index=True, suffixes=['_doc', '_service'])
    points.document_id = points.document_id.astype(np.int64)
    point_counts = points.document_id.value_counts()
    documents.at[point_counts.index, 'num_points'] = point_counts

    # Clean up html, which is necessary for good sentence splitting. This should be done for inference as well.
    documents['text'] = documents.text.apply(utils.preprocess_doc_text)
    # Pre-compute sentence-splitting info (no need for docs without Points)
    # Maps doc id to list of char positions of sentence starts
    sent_boundaries: dict[int, list[int]] = _get_sent_boundaries(documents[documents.index.isin(points.document_id)])

    # Points with some quoteStart and quoteEnd will act as a starting place for instance scheme.
    # We want to stretch the boundaries to start/end on sentences (because we will perform inference on sentence boundaries)
    points = _stretch_points(points, sent_boundaries)

    sample_window_size_fn = _get_sample_window_size_fn(points)
    case_datasets = dict()
    for case_i, target_case in cases[cases.num_approved >= MIN_APPROVED].iterrows():
        approved = points[(points.case_id == target_case.id) & (points.status == 'approved')]
        approved = approved.assign(label='positive', source='approved')
        # Use approved points as a starting point for the dataset
        instance_df = approved.copy()

        # Add declined points
        #TODO don't add declined pionts if there was also an approved one in the same doc; it may have been declined for
        # being a duplicate
        declined = points[(points.case_id == target_case.id) & (points.status == 'declined')]
        declined = declined.assign(label='negative', source='declined')
        instance_df = pd.concat([instance_df, declined])
        # rename `id` to `point_id` to make it clear it's not a dataset instance id
        instance_df['point_id'] = instance_df.id
        instance_df = instance_df.drop('id', axis=1)

        # Add segments just before and after positive examples. The thinking here is it will sharpen the discriminative
        # power of models, leading them to learn actual evidence for the Case rather than learning the topic.
        # There is a chance for true negatives here, so we should manually look over
        surrounding_df = _make_surrounding(approved, sample_window_size_fn, sent_boundaries, target_case)
        instance_df = pd.concat([instance_df, surrounding_df])

        # Add random approved points of the same topic but different case
        topical = _make_topical(approved, cases, points, target_case)
        instance_df = pd.concat([instance_df, topical])

        # Add random segments from the rest of docs with approved points
        doc_random_df = _make_random_from_approved(
            approved, instance_df, sample_window_size_fn, sent_boundaries, target_case
        )
        instance_df = pd.concat([instance_df, doc_random_df])

        # Add random segments of comprehensively reviewed docs.
        reviewed_random_df = _make_random_reviewed(
            approved, documents, instance_df, points, sample_window_size_fn, sent_boundaries, target_case
        )
        instance_df = pd.concat([instance_df, reviewed_random_df])

        # The logic above was not diligent about setting num_sents so reset it
        instance_df['num_sents'] = instance_df.sent_idx_end - instance_df.sent_idx_start

        # Extract the string content for each instance based on sent positions
        instance_df['char_start'] = instance_df.apply(
            lambda i: int(sent_boundaries[i.document_id][i.sent_idx_start]), axis=1)

        def _char_end(i):
            # If an instance ends at the very end of the doc, the "end sentence" position doesn't exist
            # but we can just set end char to something past the end (we're just using this to slice a str)
            sent_pos = sent_boundaries[i.document_id]
            if i.sent_idx_end < len(sent_pos):
                return int(sent_pos[i.sent_idx_end])
            else:
                return len(documents.loc[i.document_id].text)
        instance_df['char_end'] = instance_df.apply(_char_end, axis=1)
        instance_df['text'] = instance_df.apply(
            lambda i: documents.loc[i.document_id].text[i.char_start:i.char_end], axis=1)

        case_datasets[target_case.id] = instance_df.drop(
            ['analysis', 'created_at', 'updated_at', 'service_needs_rating_update', 'user_id', 'point_change'], axis=1)\
            .reset_index(drop=True)

    return case_datasets


def make_doc_datasets(cases, documents, points, services, en_only=True) -> dict[int, pd.DataFrame]:
    """
    :return: dict from case id to DF of instances.
        An instance is a document, but with [label, source] fields
            `label` is `positive` or `negative`
            `source` is one of ['approved', 'declined', 'random']
    """
    if en_only:
        documents = documents[documents.lang == 'en']
        points = points[points.lang == 'en']
    # point_counts = points[points.status == 'approved'].case_id.value_counts()
    # cases.at[point_counts.index, 'num_approved'] = point_counts

    # Join service info onto documents (we'll use is_comprehensively_reviewed), attach num points
    documents = pd.merge(documents, services, left_on='service_id', right_index=True, suffixes=['_doc', '_service'])
    points['document_id'] = points.document_id.astype(np.int64)
    point_counts = points.document_id.value_counts()
    documents.at[point_counts.index, 'num_points'] = point_counts

    # Filter out docs without points
    documents = documents[documents.index.isin(points.document_id)]
    # Clean up html, which is necessary for good sentence splitting. This should be done for inference as well.
    documents['text'] = documents.text.apply(utils.preprocess_doc_text)
    comprehensively_reviewed_docs = documents[(documents.is_comprehensively_reviewed) & (documents.num_points >= 10)]

    case_datasets = dict()
    for case_i, target_case in cases[cases.num_approved >= MIN_APPROVED].iterrows():
        approved = points[(points.case_id == target_case.id) & (points.status == 'approved')]
        approved_docs = documents.loc[approved.document_id.unique()]
        approved_docs = approved_docs.assign(label='positive', source='approved')
        # Use approved docs as a starting point for the dataset
        instance_df = approved_docs.copy()

        # Add docs with declined points (and no approved points)
        declined = points[(points.case_id == target_case.id) & (points.status == 'declined')]
        declined_docs = documents.loc[list(set(declined.document_id.unique()) - set(approved_docs.id_doc))]
        declined_docs = declined_docs.assign(label='negative', source='declined')
        instance_df = pd.concat([instance_df, declined_docs])

        # Add random comprehensively reviewed docs. There is an is_comprehensively_reviewed feature, but
        # also make sure at least 10 points were submitted for the doc. Avoid any docs that have a point for
        # the case in question.
        reviewed_docs = comprehensively_reviewed_docs[
            ~comprehensively_reviewed_docs.id_doc.isin(points[points.case_id == target_case.id].document_id)
        ]
        reviewed_docs = reviewed_docs.assign(label='negative', source='reviewed')
        instance_df = pd.concat([instance_df, reviewed_docs])

        case_datasets[target_case.id] = instance_df[['id_doc', 'text', 'id_service', 'label', 'source']]\
            .reset_index(drop=True)

    return case_datasets


def assign_folds(sent_span_datasets, doc_datasets):
    # Make folds assigned by document id
    # Ideally they'd be stratefied by sent span source, but I don't think there's a way to accomplish both
    # Since we are training models per case, we can do the split separately per case
    def _gen_fold_slice(n, fold_i):
        return slice(round(fold_i * (n / NUM_FOLDS)), round((fold_i+1) * (n / NUM_FOLDS)))

    for case_id in set(sent_span_datasets.keys()).union(set(doc_datasets.keys())):
        doc_ids = set()
        if case_id in sent_span_datasets:
            doc_ids = doc_ids.union(set(sent_span_datasets[case_id].document_id))
        if case_id in doc_datasets:
            doc_ids = doc_ids.union(set(doc_datasets[case_id].id_doc))
        doc_ids = list(doc_ids)
        random.seed(0)
        random.shuffle(doc_ids)
        sent_df = sent_span_datasets[case_id]
        doc_df = doc_datasets[case_id]
        for fold_i, fold_doc_ids in enumerate([doc_ids[_gen_fold_slice(len(doc_ids), i)] for i in range(NUM_FOLDS)]):
            sent_df.loc[sent_df[sent_df.document_id.isin(set(fold_doc_ids))].index, 'fold'] = fold_i
            doc_df.loc[doc_df[doc_df.id_doc.isin(set(fold_doc_ids))].index, 'fold'] = fold_i
        sent_df.fold = sent_df.fold.astype(int)
        doc_df.fold = doc_df.fold.astype(int)
        sent_span_datasets[case_id] = sent_df
        doc_datasets[case_id] = doc_df

    return sent_span_datasets, doc_datasets

def run():
    cases = pickle.load(open(here / f'../data/cases_{DB_DUMP_VERSION}_clean.pkl', 'rb'))
    documents = pickle.load(open(here / f'../data/documents_{DB_DUMP_VERSION}_clean.pkl', 'rb'))
    points = pickle.load(open(here / f'../data/points_{DB_DUMP_VERSION}_clean.pkl', 'rb'))
    topics = pickle.load(open(here / f'../data/topics_{DB_DUMP_VERSION}_clean.pkl', 'rb'))
    services = pickle.load(open(here / f'../data/services_{DB_DUMP_VERSION}_clean.pkl', 'rb'))

    sent_span_datasets = make_sent_span_datasets(cases, documents, points, topics, services)
    doc_datasets = make_doc_datasets(cases, documents, points, services)
    sent_span_datasets, doc_datasets = assign_folds(sent_span_datasets, doc_datasets)

    logger.info(f"Saving sentence span classification datasets to {SENT_SPAN_LOC}")
    pickle.dump(sent_span_datasets, open(SENT_SPAN_LOC, 'wb'))
    logger.info(f"Saving document classification datasets to {DOC_LOC}")
    pickle.dump(doc_datasets, open(DOC_LOC, 'wb'))

    np.random.seed(0)
    case_id = np.random.choice(list(sent_span_datasets.keys()))
    logger.info(f"Example structure for sent spans, case id {case_id}:")
    sent_span_datasets[case_id].info()
    logger.info(f"Negative case breakdown for sent spans, case {case_id}")
    logger.info(sent_span_datasets[case_id].source.value_counts())
    logger.info(f"Example structure for documents, case id {case_id}:")
    doc_datasets[case_id].info()

    for case_id in sent_span_datasets:
        logger.info(f"{'=' * 30} Case {case_id} {'=' * 30}")
        logger.info("Sentence spans:")
        logger.info(f"\t{sent_span_datasets[case_id].source.value_counts()}")
        logger.info(f"Documents:")
        logger.info(f"\t{doc_datasets[case_id].source.value_counts()}")
        logger.info(f"Sent span folds: {sent_span_datasets[case_id].fold.value_counts()}")
        logger.info(f"Doc folds: {doc_datasets[case_id].fold.value_counts()}")


if __name__ == '__main__':
    run()
