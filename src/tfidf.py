import copy
import logging
from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
import spacy
from textacy.representations.vectorizers import Vectorizer
from textacy import extract
from tqdm import tqdm

from src import utils, inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

DB_DUMP_VERSION = '211222'
# We'll only train models for Cases that have deep learning models trained
CASE_IDS = list(sorted(inference.THRESHOLDS.keys()))
RANDOM_STATE = 0
random.seed(RANDOM_STATE)

# When updating, be sure to modify .dockerignore so the models are copied into the inference Docker image
MODEL_VERSION = 'tfidf_v1'

"""
This script trains high recall TF-IDF classification models that can be used as prefilters, to avoid unnecessary 
inference with our deep learning models for completely irrelevant sentences. For most cases it reduces inference 
by 90-95% without dropping any positive examples in the test set.

Models are written to data/models/{MODEL_VERSION} and later added to the inference Docker image
"""

def model_output_dir(case_id) -> Path:
    return here / f'../data/models/{MODEL_VERSION}/{case_id}'

def _find_pareto_frontier(precision, recall, thresholds, critical_recalls):
    """
    Find Pareto-optimal points on precision-recall curve. We also ensure we capture key recall levels.
    Returns indices of Pareto-optimal points
    """
    if len(recall) == 0:
        return np.array([])

    # Sort by recall (ascending) then precision (descending)
    sorted_idx = np.lexsort((-precision, recall))
    precision = precision[sorted_idx]
    recall = recall[sorted_idx]
    thresholds = thresholds[sorted_idx]

    # Find Pareto frontier: keep point if it has higher precision than all previous points
    # (since we're moving in increasing recall direction)
    pareto_idx = []
    max_precision_seen = -1

    for i in range(len(precision)):
        if precision[i] > max_precision_seen:
            pareto_idx.append(i)
            max_precision_seen = precision[i]

    pareto_idx = np.array(pareto_idx)

    # Add critical recall thresholds if not already present
    for target_recall in critical_recalls:
        if recall.min() <= target_recall <= recall.max():
            # Find closest point to target recall
            closest_idx = np.argmin(np.abs(recall - target_recall))
            if closest_idx not in pareto_idx:
                pareto_idx = np.append(pareto_idx, closest_idx)

    pareto_idx = np.unique(pareto_idx)
    pareto_idx = np.sort(pareto_idx)

    return sorted_idx[pareto_idx]


def make_vectorizer() -> Vectorizer:
    return Vectorizer(tf_type='sqrt', idf_type='smooth', norm='l1', min_df=4, max_df=0.8)


def summarize_metrics(metrics: dict, cv_folds: int, critical_recalls: list[float]) -> dict:
    """
    Transforms raw threshold_metrics from CV into a summary dict.

    Args:
        metrics: Dict with 'avg_precisions' and 'threshold_metrics' keys
        cv_folds: Number of CV folds (used for validation)
        critical_recalls: List of recall levels to extract (e.g., [1.0, 0.98, 0.95])

    Returns:
        Dict with:
            'avg_precisions': [0.54, 0.48, ...]  # One per fold
            'recall100_filterrates': [0.85, 0.87, ...]  # One per fold
            'recall100_thresholds': [0.45, 0.43, ...]   # One per fold
            'recall98_filterrates': [...]
            'recall98_thresholds': [...]
            ...
    """
    # Step 1: Organize threshold_metrics by fold
    all_recall_filterrates = [[] for _ in range(cv_folds)]
    for threshold_option in metrics['threshold_metrics']:
        fold_idx = threshold_option['fold']
        all_recall_filterrates[fold_idx].append((
            threshold_option['recall'],
            threshold_option['filter_rate'],
            threshold_option['threshold']
        ))

    # Step 2: Sort each fold's options by recall (ascending)
    all_recall_filterrates = [
        list(sorted(options, key=lambda m: m[0]))
        for options in all_recall_filterrates
    ]

    # Step 3: For each critical recall, find best point per fold
    critical_folds = {recall: [] for recall in critical_recalls}
    for recall in critical_recalls:
        for fold_options in all_recall_filterrates:
            # Find first option that meets or exceeds target recall
            for option in fold_options:
                if option[0] >= recall:
                    critical_folds[recall].append((option[1], option[2]))
                    break

    # Step 4: Build summary dict
    summary = {'avg_precisions': metrics['avg_precisions']}
    for recall in critical_recalls:
        recall_str = f'recall{recall * 100:.0f}'
        summary[f'{recall_str}_filterrates'] = [pair[0] for pair in critical_folds[recall]]
        summary[f'{recall_str}_thresholds'] = [pair[1] for pair in critical_folds[recall]]

    return summary


def summarize_all_metrics(cv_results: list[dict], critical_recalls: list[float], cv_folds: int) -> pd.DataFrame:
    """
    Converts CV results list into a DataFrame for easy model comparison.

    Args:
        cv_results: List of dicts, each with model_class, model_kwargs, rebalance_to, and metrics
        critical_recalls: List of recall levels (e.g., [1.0, 0.98, 0.95])
        cv_folds: Number of CV folds

    Returns:
        DataFrame with columns:
            - model_class_name: str (e.g., 'LogisticRegression')
            - model_class: class object (for easy instantiation)
            - model_kwargs: dict (original kwargs)
            - rebalance_to: int or None
            - avg_precision: float (mean across folds)
            - recall100_filterrate: float (mean across folds)
            - recall100_threshold_cv: float (coefficient of variation)
            - recall98_filterrate, recall98_threshold_cv, etc.
    """
    res_df = []
    for idx, model in enumerate(cv_results):
        # Step 1: Extract model configuration (keep as objects/dicts, don't expand)
        model_dict = {
            'cv_result_idx': idx,  # Track index for easy lookup
            'model_class_name': model['model_class'].__name__,
            'model_class': model['model_class'],
            'model_kwargs': model['model_kwargs'],
            'rebalance_to': model['rebalance_to'],
        }

        # Step 2: Summarize metrics for this model
        metrics_summary = summarize_metrics(model['metrics'], cv_folds, critical_recalls)

        # Step 3: Aggregate across folds
        model_dict['avg_precision'] = np.mean(metrics_summary['avg_precisions'])

        for recall in critical_recalls:
            recall_str = f"recall{recall * 100:.0f}"
            # Mean filter rate across folds
            model_dict[f'{recall_str}_filterrate'] = np.mean(metrics_summary[f'{recall_str}_filterrates'])

            thresholds = metrics_summary[f'{recall_str}_thresholds']
            model_dict[f'{recall_str}_thresholds'] = thresholds

            # Coefficient of variation for thresholds (std / mean)
            model_dict[f'{recall_str}_threshold_cv'] = np.std(thresholds) / np.mean(thresholds)

        res_df.append(model_dict)

    return pd.DataFrame(res_df)


def gen_setups():
    # TODO also loop through the dataset labels (one with other topical positives, one without)
    for model_class, model_kwargs in [
        # saga solver is good for sparse data
        (LogisticRegression, dict(penalty='l2', max_iter=5000, solver='saga')),
        (LogisticRegression, dict(penalty='elasticnet', l1_ratio=0.5, max_iter=5000, solver='saga')),
        (SGDClassifier, dict(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=5000)),
        # There was no clear winner for optimal regularization params, just try a bunch
        (RandomForestClassifier, dict(
            n_estimators=100, max_depth=3, min_samples_split=8, min_samples_leaf=3, max_features='sqrt'
        )),
        (RandomForestClassifier, dict(
            n_estimators=100, max_depth=5, min_samples_split=8, min_samples_leaf=3, max_features='sqrt'
        )),
        (RandomForestClassifier, dict(
            n_estimators=100, max_depth=10, min_samples_split=8, min_samples_leaf=3, max_features='sqrt'
        )),
        (RandomForestClassifier, dict(
            n_estimators=100, max_depth=5, min_samples_split=4, min_samples_leaf=3, max_features='sqrt'
        )),
        (RandomForestClassifier, dict(
            n_estimators=100, max_depth=5, min_samples_split=12, min_samples_leaf=3, max_features='sqrt'
        )),
        # For some reason ComplementNB does very poorly. I don't think I have negative values so that's not it
        # (ComplementNB, dict(alpha=1.0, norm=False)),
        # (ComplementNB, dict(alpha=0.1, norm=False)),
        # TODO try GradientBoostingClassifier if it's not overkill
    ]:
        for rebalance_to in [20, None]:
            setup = {
                'model_class': model_class,
                'model_kwargs': model_kwargs,
                'rebalance_to': rebalance_to,
            }
            if model_class.__name__ == 'ComplementNB':
                yield setup
            else:
                setup['random_state'] = RANDOM_STATE
                if model_class.__name__ == 'RandomForestClassifier':
                    weight_options = ['balanced_subsample']
                else:
                    weight_options = [{0: 2.0, 1: 1.0}, {0: 1.0, 1: 1.0}, {0: 1.0, 1: 3.0}, {0: 1.0, 1: 5.0}]
                for class_weight in weight_options:
                    new_setup = copy.deepcopy(setup)
                    new_setup['model_kwargs']['class_weight'] = class_weight
                    yield new_setup


def train_cv(
        ngrams: list[list[str]], sentence_labels: list[int], model_outdir: Path, critical_recalls, cv_folds=5, min_recall=0.8,
) -> list[dict]:
    """
    Returns a list of results describing different TF-IDF models/setups that we could use for a single case filter.
    It also serializes the list as we iterate, in case of crash

    A single result is a dictionary, with setup parameters used and a `metrics` dict, format:
    metrics = {
        'avg_precisions': [.54, .483, ...],
        'threshold_metrics': [
            {
                # Threshold option 1
                'fold': 1,
                'threshold': threshold,
                'precision': precisions[idx],
                'recall': recalls[idx],
                # Key metric: what % of data can we filter while maintaining high recall
                'filter_rate': tn / len(y_val)
            }, {
                # Threshold option 2
                'fold': 1,
                'threshold': threshold,
                'precision': precisions[idx],
                'recall': recalls[idx],
                # Key metric: what % of data can we filter while maintaining high recall
                'filter_rate': tn / len(y_val)
            },
            ...
        ]
    }
    """
    results_filepath = model_outdir / 'cv_results.pkl'

    vectorizer = make_vectorizer()
    logger.info("Vectorizing")
    sentence_vectors = vectorizer.fit_transform(ngrams)
    logger.info(f"Vectorized shape {sentence_vectors.shape}")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    results = []
    setups = list(gen_setups())
    for setup_dict in tqdm(setups, desc="Trying setups"):
        metrics = {
            'avg_precisions': [],
            'threshold_metrics': []
        }

        logger.info(f"Training {setup_dict['model_class'].__name__} {setup_dict['model_kwargs']}")
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(sentence_vectors, sentence_labels)):
            X_train, X_val = sentence_vectors[train_idx], sentence_vectors[val_idx]
            y_train, y_val = sentence_labels[train_idx], sentence_labels[val_idx]

            if setup_dict['rebalance_to'] is not None:
                pos_idx = np.where(y_train == 1)[0]
                neg_idx = np.where(y_train == 0)[0]

                # Keep all positives, sample negatives at desired ratio
                # For 1:150 imbalance, might want 1:10 or 1:20 for training
                target_ratio = min(setup_dict['rebalance_to'], len(neg_idx) // len(pos_idx))
                neg_sample_size = int(len(pos_idx) * target_ratio)

                neg_sample_idx = np.random.choice(neg_idx, size=neg_sample_size, replace=False)
                balanced_idx = np.concatenate([pos_idx, neg_sample_idx])
                np.random.shuffle(balanced_idx)

                X_train = X_train[balanced_idx]
                y_train = y_train[balanced_idx]
                logger.debug(
                    f"Rebalanced training: {len(pos_idx)} pos, {len(neg_sample_idx)} neg (1:{target_ratio} ratio)"
                )

            model = setup_dict['model_class'](**setup_dict['model_kwargs'])
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]

            # Get the full PR curve so we can consider all thresholds
            precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
            #  precision_recall_curve returns n+1 precision/recall values but n thresholds. The last p/r is threshold=0
            #  (all positive), add it to thresholds for completeness
            thresholds = np.append(thresholds, 0)

            # Filter by minimum recall
            valid_idx = recalls >= min_recall
            precisions = precisions[valid_idx]
            recalls = recalls[valid_idx]
            thresholds = thresholds[valid_idx]
            pareto_idx = _find_pareto_frontier(precisions, recalls, thresholds, critical_recalls)

            # Evaluate at different thresholds
            for idx in pareto_idx:
                threshold = thresholds[idx]
                y_pred = (y_proba >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

                threshold_metrics = {
                    'threshold': threshold,
                    'fold': fold_idx,
                    'precision': precisions[idx],
                    'recall': recalls[idx],
                    # Key metric: what % of data can we filter while maintaining high recall
                    'filter_rate': tn / len(y_val)
                }

                metrics['threshold_metrics'].append(threshold_metrics)

            # For an overall score of each fold model we can use average precision, which apparently is a better fit than
            # ROCAUC for very imbalanced datasets like mine.
            # We'll also use the filter rate achieved at different recalls (to be computed after-the-fact), and the std
            # dev of CV fold thresholds
            metrics['avg_precisions'].append(average_precision_score(y_val, y_proba))

        results.append(copy.copy(setup_dict) | {'metrics': metrics})
        pickle.dump(results, open(results_filepath.as_posix(), 'wb'))

    pickle.dump(results, open(results_filepath.as_posix(), 'wb'))
    return results


def train_full(
    ngrams: list[list[str]], sentence_labels: list[int], cv_results: list[dict], model_outdir: Path, critical_recalls: list[float], cv_folds: int
):
    """
    Selects the best model from CV results and trains it on the full dataset.

    Args:
        ngrams: ngrams to use for training
        sentence_labels: int labels for each sentence
        cv_results: List of CV result dicts from train_cv()
        model_outdir: Directory to save final model and vectorizer
        critical_recalls: List of recall levels used in CV
        cv_folds: Number of CV folds used
    """
    # Summarize CV results into DataFrame
    res_df = summarize_all_metrics(cv_results, critical_recalls, cv_folds)

    # Calculate score for models factoring filter rate and threshold stability, select best model
    res_df['score'] = (res_df['recall100_filterrate'] + (1 - res_df['recall100_threshold_cv']))
    res_df = res_df.sort_values('score', ascending=False)
    best_model = res_df.iloc[0]

    # Log top 3 for reference
    logger.info("Top 3 models by score:")
    display_cols = [
        'model_class_name', 'rebalance_to', 'score', 'recall100_filterrate', 'recall100_threshold_cv', 'avg_precision'
    ]
    for i, (idx, row) in enumerate(res_df.head(3).round(4).iterrows()):
        logger.info(f"    {i+1}. {row[display_cols].to_dict()}")

    if res_df.iloc[0].recall100_threshold_cv >= .2:
        raise ValueError("Model does not have stable enough CV fold thresholds to trust")
    logger.info("Selecting best model and training on full dataset")

    # Extract model configuration directly from DataFrame
    model_class = best_model['model_class']
    model_kwargs = best_model['model_kwargs']
    rebalance_to = best_model['rebalance_to']

    # Create and fit vectorizer
    vectorizer = make_vectorizer()
    sentence_vectors = vectorizer.fit_transform(ngrams)
    logger.info(f"Vectorized shape: {sentence_vectors.shape}")

    # Apply rebalancing if needed (same logic as train_cv)
    X_train = sentence_vectors
    y_train = sentence_labels

    if rebalance_to is not None and pd.notna(rebalance_to):
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]

        target_ratio = min(int(rebalance_to), len(neg_idx) // len(pos_idx))
        neg_sample_size = int(len(pos_idx) * target_ratio)

        neg_sample_idx = np.random.choice(neg_idx, size=neg_sample_size, replace=False)
        balanced_idx = np.concatenate([pos_idx, neg_sample_idx])
        np.random.shuffle(balanced_idx)

        X_train = X_train[balanced_idx]
        y_train = y_train[balanced_idx]
        logger.info(f"Rebalanced: {len(pos_idx)} pos, {len(neg_sample_idx)} neg (1:{target_ratio} ratio)")

    logger.info(f"Training final {model_class.__name__} on full dataset...")
    final_model = model_class(**model_kwargs)
    final_model.fit(X_train, y_train)

    # Calculate recommended threshold for 100% recall, being conservative with mean - 0.2 * std
    threshold_mean = np.mean(best_model['recall100_thresholds'])
    threshold_std = np.std(best_model['recall100_thresholds'])
    recommended_threshold = threshold_mean - 0.2 * threshold_std
    logger.info(f"Mean threshold: {threshold_mean:.4f}, recommended threshold: {recommended_threshold:.4f}")

    # STEP 10: Save artifacts
    logger.info(f"Saving model artifacts to {model_outdir}...")

    pickle.dump(vectorizer, open((model_outdir / 'vectorizer.pkl').as_posix(), 'wb'))
    pickle.dump(final_model, open((model_outdir / 'model.pkl').as_posix(), 'wb'))
    pickle.dump(res_df, open((model_outdir / 'cv_summary.pkl').as_posix(), 'wb'))

    final_metrics = {
        'threshold': recommended_threshold,
        'recall100_filterrate': best_model['recall100_filterrate'],
        'recall100_threshold_cv': best_model['recall100_threshold_cv'],
        'score': best_model['score'],
        'avg_precision': best_model['avg_precision'],
    }
    pickle.dump(final_metrics, open((model_outdir / 'final_metrics.pkl').as_posix(), 'wb'))


def make_case_dataset(case_id, positives, negatives):
    #TODO try including topical points in the positives, or at least exclude from the negatives?

    positives = positives[case_id].copy()
    positives['label'] = 1

    to_drop = []
    for i, sent in negatives.iterrows():
        if 'offlimits_cases' in sent and case_id in sent['offlimits_cases']:
            to_drop.append(i)
    logger.info(f"Dropping {len(to_drop)} negative sentences due to points")
    negatives = negatives.drop(to_drop, axis=0)
    negatives['label'] = 0

    dataset = pd.concat([positives, negatives])
    logger.info(f"Case {case_id}: {len(positives)} positives, {len(negatives)} negatives")

    return dataset.reset_index(drop=True)


def prep_datasets(documents, points, services, num_negative_docs=600, sents_per_doc=18) \
        -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """
    :return: make dataframes that can later be refined into datasets of sentences for any Case:
        1) dict from case_id to df of positive instances
        2) a huge dataframe of sentences that can be used as negative instances, with a `offlimits_cases` field
            so I can filter out sentences when they were positive points
    """
    documents = documents[documents.lang == 'en']
    points = points[points.lang == 'en'].copy()

    # Join service info onto documents (we'll use is_comprehensively_reviewed), attach num points
    documents = pd.merge(documents, services, left_on='service_id', right_index=True, suffixes=['_doc', '_service'])
    points['document_id'] = points.document_id.astype(np.int64)

    # Filter out docs without enough points
    point_counts = points.document_id.value_counts()
    documents.at[point_counts.index, 'num_points'] = point_counts
    # documents = documents[documents.index.isin(points.document_id)]
    documents = documents[(documents.is_comprehensively_reviewed) & (documents.num_points >= 8)]

    # Clean up html, which is necessary for good sentence splitting. This should be done for inference as well.
    documents['text'] = documents.text.apply(utils.preprocess_doc_text)

    logger.info("Loading spacy model")
    spacy_model = spacy.load('en_core_web_md', disable=['attribute_ruler', 'lemmatizer', 'ner'])

    approved_points = points[points.status == 'approved']
    positives = dict()
    for case_id in CASE_IDS:
        approved = approved_points[approved_points.case_id == case_id].copy()
        approved['text'] = approved['quoteText']
        approved['point_id'] = approved['id']
        positives[case_id] = approved[['point_id', 'case_id', 'quoteStart', 'quoteEnd', 'document_id', 'text']]

    negatives = []
    for i, (doc_id, doc) in tqdm(
            enumerate(documents.sample(num_negative_docs, random_state=RANDOM_STATE).iterrows()),
            total=num_negative_docs,
    ):
        spacy_doc = spacy_model(doc.text)
        doc_sents = list(spacy_doc.sents)

        # Exclude any existing points in case they are positives
        #TODO if slow, add index document_id
        doc_points = points[(points.document_id == doc_id)]
        # Start/end positions for points, grouped by case for fast lookup
        case_offlimits: dict[int, list[tuple[int, int]]] = dict()
        for case_id in doc_points.case_id.unique():
            case_points = doc_points[doc_points.case_id == case_id]
            case_offlimits[case_id] = [(int(s), int(e)) for s, e in zip(case_points.quoteStart, case_points.quoteEnd)]

        random.shuffle(doc_sents)
        for sent in doc_sents[:sents_per_doc]:
            if sent.text != '' and not sent.text.isspace():
                offlimits_cases = []
                for case_id, boundaries in case_offlimits.items():
                    for start, end in boundaries:
                        if start < sent.end_char and end > sent.start_char:
                            offlimits_cases.append(case_id)
                negatives.append({
                    'document_id': doc_id,
                    'quoteStart': sent.start_char,
                    'quoteEnd': sent.end_char,
                    'text': sent.text,
                    'offlimits_cases': offlimits_cases
                })

    pos_lens = [len(pos) for pos in positives.values()]
    logger.info(f"Positive lengths: {pd.Series(pos_lens).describe()}")

    return positives, pd.DataFrame(negatives)

def extract_ngrams(case_dataset):
    # Shuffle
    case_dataset = case_dataset.sample(n=len(case_dataset), random_state=RANDOM_STATE)

    # Process text into ngrams
    logger.info("Loading spacy model and extracting ngrams")
    # Disable all components except tokenizer for faster ngram extraction
    nlp = spacy.load('en_core_web_md', disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
    ngrams = [
        [span.text.lower() for span in extract.ngrams(nlp(text), n=[1, 2])]
        for text in tqdm(case_dataset['text'], total=len(case_dataset))
    ]

    sentence_labels = case_dataset.label.values
    s = '\n'.join(map(str, ngrams[:5]))
    logger.info(f"ngrams preview: {s}")
    s = '\n'.join(map(str, [ngrams[i] for i in np.where(sentence_labels)[0][:5]]))
    logger.info(f"ngrams preview (pos only): {s}")

    return ngrams, sentence_labels

def run():
    documents = pickle.load(open(here / f'../data/documents_{DB_DUMP_VERSION}_clean.pkl', 'rb'))
    points = pickle.load(open(here / f'../data/points_{DB_DUMP_VERSION}_clean.pkl', 'rb'))
    services = pickle.load(open(here / f'../data/services_{DB_DUMP_VERSION}_clean.pkl', 'rb'))

    # Create cache and results directories
    cache_dir = Path(here / '../data/tfidf')
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Try to load cached prep_datasets result
    prep_cache = cache_dir / 'prep_datasets.pkl'
    try:
        positives, negatives = pickle.load(open(prep_cache, 'rb'))
        logger.info("Loaded prep_datasets from cache")
    except FileNotFoundError:
        logger.info("Running prep_datasets (this may take a while)")
        positives, negatives = prep_datasets(documents, points, services)
        pickle.dump((positives, negatives), open(prep_cache, 'wb'))

    critical_recalls = [1.0, .98, .95]

    # Loop over all cases
    for case_id in CASE_IDS:
        logger.info(f"Processing case {case_id}")

        # Create dataset for this case
        case_dataset: pd.DataFrame = make_case_dataset(case_id, positives, negatives)
        ngrams, sentence_labels = extract_ngrams(case_dataset)

        # Train and save results
        model_outdir = model_output_dir(case_id)
        model_outdir.mkdir(parents=True, exist_ok=True)
        cv_folds = 5
        cv_results = train_cv(ngrams, sentence_labels, model_outdir, critical_recalls, cv_folds)
        logger.info(f"Completed CV for case {case_id}")

        # Train final model on full dataset
        train_full(ngrams, sentence_labels, cv_results, model_outdir, critical_recalls, cv_folds)
        logger.info(f"Completed full training for case {case_id}")


if __name__ == '__main__':
    run()

