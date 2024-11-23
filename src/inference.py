import logging
from pathlib import Path

import numpy as np
import torch.nn.functional as f
from tqdm import tqdm

from src import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

MAX_EXPANSION_SENTENCES = 5

def apply_sent_span_model(
        text, sent_boundaries: list[int], tokenizer, hf_model, batch_size, device, off_limits=None
) -> tuple[float, int, int, int]:
    """
    Takes a model that was trained for text classification on sentence spans (usually 1 or 2, but potentially 5+)
    and applies it to full documents.
    First applies the model to every sentence, finding the maximum score. Then tries expanding the bounaries on
    either side, looking for an even higher score.
    Ideally we'll end up with the minimal representative sentence span that contains evidence for the case, and we don't
    know in advance how long that will be.

    Char spans can be marked as off limits with list[tuple(int, int)] off_limits. This is so that in production we can
    avoid re-suggesting a point that was declined by a curator.
    """
    if off_limits is None:
        off_limits = []
    sent_boundaries = sent_boundaries + [None]
    sent_texts = [text[sent_boundaries[i]: sent_boundaries[i+1]] for i in range(len(sent_boundaries) - 1)]
    def _is_overlapping(start, end):
        return any([((off_end is None or off_end > start) and (end is None or off_start < end))
                    for off_start, off_end in off_limits])

    sent_off_limits = [_is_overlapping(sent_boundaries[i], sent_boundaries[i+1])
                       for i in range(len(sent_boundaries) - 1)]

    def _batch(iterable, n=batch_size):
        l = len(iterable)
        for idx in range(0, l, n):
            yield iterable[idx: min(idx + n, l)]

    softmax_probs = []
    for batch in _batch(sent_texts):
        tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        tokens.to(device)
        preds = hf_model(**tokens.data)
        softmax_probs += f.softmax(preds.logits, dim=1)[:,1].tolist()
    softmax_probs = np.array(softmax_probs)
    softmax_probs[sent_off_limits] = -np.inf
    best_score = softmax_probs.max()
    best_sent_idx = softmax_probs.argmax()

    # We'll now try extending the best scoring span up to 5 sentences to the left and right, looking for a higher score.

    prior_sents_considered = min(MAX_EXPANSION_SENTENCES, best_sent_idx)
    num_prior = 0

    # There's probably a way to combine tokens from above if this is slow
    texts_with_priors = []
    for i in range(prior_sents_considered):
        if sent_off_limits[best_sent_idx - (i + 1)]:
            break
        texts_with_priors.append(
            text[sent_boundaries[best_sent_idx - (i + 1)]:sent_boundaries[best_sent_idx + 1]]
        )
    if len(texts_with_priors) > 0:
        softmax_probs = []
        for batch in _batch(texts_with_priors):
            tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            tokens.to(device)
            preds = hf_model(**tokens.data)
            softmax_probs += f.softmax(preds.logits, dim=1)[:,1].tolist()
        softmax_probs = np.array(softmax_probs)
        best_score_with_priors = softmax_probs.max()
        if best_score_with_priors > best_score:
            best_score = best_score_with_priors
            num_prior = softmax_probs.argmax() + 1

    # Example: sent_boundaries are [0, 5, 10, 15, 20, None] and best_sent_idx is 2 (chars 10-15). There should only
    # be two more sentences to potentially add at the end, 15-20 and 20-end. So we can't go beyond len(sent_boundaries) - (best + 2)
    after_sents_considered = min(MAX_EXPANSION_SENTENCES, len(sent_boundaries) - (best_sent_idx + 2))
    num_after = 0
    texts_with_afters = []
    for i in range(after_sents_considered):
        if sent_off_limits[best_sent_idx + i + 1]:
            break
        texts_with_afters.append(
            text[sent_boundaries[best_sent_idx - num_prior]:sent_boundaries[best_sent_idx + i + 2]]
        )

    if len(texts_with_afters) > 0:
        softmax_probs = []
        for batch in _batch(texts_with_afters):
            tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            tokens.to(device)
            preds = hf_model(**tokens.data)
            softmax_probs += f.softmax(preds.logits, dim=1)[:,1].tolist()

        softmax_probs = np.array(softmax_probs)
        best_score_with_afters = softmax_probs.max()
        if best_score_with_afters > best_score:
            best_score = best_score_with_afters
            num_after = softmax_probs.argmax() + 1

    # Return winning score, char span, and how many sentences it spans
    return best_score,\
        sent_boundaries[best_sent_idx - num_prior],\
        sent_boundaries[best_sent_idx + num_after + 1],\
        1 + num_prior + num_after


def attach_predictions(doc_df, tokenizer, sent_boundaries, model, batch_size=4, device='cpu'):
    """
    Supports evaluation during train.py
    Instead of selecting the class with a higher logit, this gathers post-softmax probabilities and
    uses whatever threshold optimizes F1 of the dataset
    """
    doc_df = doc_df.copy()
    for idx, instance in tqdm(doc_df.iterrows(), total=len(doc_df), desc=f'Applying model to docs on {device}'):
        score, start, end, num_sents = apply_sent_span_model(
            instance.text, sent_boundaries[instance.id_doc], tokenizer, model,
            batch_size=batch_size, device=device
        )
        doc_df.at[idx, 'pred_score'] = score
        doc_df.at[idx, 'pred_char_start'] = start
        doc_df.at[idx, 'pred_char_end'] = end
        doc_df.at[idx, 'pred_num_sents'] = num_sents

    doc_df['int_labels'] = [0 if i == 'negative' else 1 for i in doc_df.label]
    threshold = utils.optimal_threshold(doc_df.pred_score, doc_df.int_labels)[0]
    doc_df['pred_label'] = [1 if bool_pred else 0 for bool_pred in doc_df.pred_score >= threshold]
    return doc_df
