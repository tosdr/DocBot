import logging
from pathlib import Path
import pickle

import langdetect
from langdetect import DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import numpy as np
import spacy
from textacy import extract
import torch
import torch.nn.functional as f
from tqdm import tqdm

from src import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

DetectorFactory.seed = 0
MAX_EXPANSION_SENTENCES = 5

# Used for huggingface's from_pretrained()
BASE_MODEL_NAME = 'bert-base-uncased'

# This has to match the upload key specified when training. It will also be used by apply_docbot.py to record whether
# docbot has already ran docs, so if we train a new model and want to rerun, this should be updated.
MODEL_VERSION = 'v3'

# Case-specific positive prediction thresholds. These come from pr_curves.ipynb, and optimize fscore with beta=1.5
THRESHOLDS = {
    117: 0.9313029646873474,
    118: 0.9996505975723267,
    121: 0.9194583892822266,
    122: 0.9512377381324768,
    124: 0.999946117401123,
    126: 0.8929231762886047,
    127: 0.31597447395324707,
    128: 0.8838386535644531,
    129: 0.9749407768249512,
    130: 0.9065191149711609,
    134: 0.7489349842071533,
    138: 0.8780727386474609,
    139: 0.9998829364776611,
    140: 0.9612037539482117,
    143: 0.9117705225944519,
    145: 0.9865456819534302,
    146: 0.06113763153553009,
    147: 0.9860677123069763,
    148: 0.9632765650749207,
    149: 0.9952959418296814,
    150: 0.9619890451431274,
    151: 0.9394944906234741,
    152: 0.5705567002296448,
    162: 0.9346338510513306,
    163: 0.31521105766296387,
    164: 0.404929518699646,
    166: 0.4554845690727234,
    170: 0.690903902053833,
    175: 0.9982500672340393,
    177: 0.979971706867218,
    178: 0.939058244228363,
    182: 0.9997735619544983,
    183: 0.9350226521492004,
    187: 0.7722886800765991,
    188: 0.2906496822834015,
    190: 0.4945354163646698,
    193: 0.9803260564804077,
    195: 0.9073813557624817,
    196: 0.9988189339637756,
    197: 0.9934123158454895,
    199: 0.5093461871147156,
    201: 0.21304339170455933,
    202: 0.2907284200191498,
    203: 0.9348453879356384,
    205: 0.5213550925254822,
    207: 0.6749431490898132,
    208: 0.9928780198097229,
    210: 0.9505872130393982,
    211: 0.9587513208389282,
    214: 0.9999645948410034,
    215: 0.9992538094520569,
    216: 0.43725141882896423,
    217: 0.9967785477638245,
    218: 0.970154881477356,
    219: 0.18621566891670227,
    220: 0.9325112104415894,
    223: 0.9068081378936768,
    226: 0.7020467519760132,
    227: 0.7985442280769348,
    228: 0.05485598370432854,
    229: 0.9920095205307007,
    230: 0.9913994073867798,
    231: 0.999704897403717,
    232: 0.9997180104255676,
    233: 0.5125555992126465,
    239: 0.8622395396232605,
    241: 0.534091591835022,
    242: 0.955049455165863,
    243: 0.9986709356307983,
    278: 0.9924724102020264,
    279: 0.9935654997825623,
    280: 0.34931480884552,
    281: 0.867756187915802,
    283: 0.9267996549606323,
    284: 0.9948242902755737,
    285: 0.9964261651039124,
    286: 0.5064982175827026,
    287: 0.2425031214952469,
    288: 0.34972575306892395,
    289: 0.7902135848999023,
    290: 0.5488492250442505,
    291: 0.9992871880531311,
    292: 0.9986388087272644,
    293: 0.7992552518844604,
    294: 0.9807372093200684,
    295: 0.24001362919807434,
    297: 0.9939224123954773,
    298: 0.9999438524246216,
    299: 0.9987969398498535,
    300: 0.6722489595413208,
    303: 0.8706561326980591,
    306: 0.8188225626945496,
    307: 0.9365635514259338,
    310: 0.9998196959495544,
    311: 0.6947453022003174,
    313: 0.9986419081687927,
    314: 0.995353102684021,
    315: 0.5673277378082275,
    323: 0.29618099331855774,
    325: 0.9868621826171875,
    326: 0.7446432113647461,
    329: 0.7189320921897888,
    331: 0.7955061197280884,
    333: 0.9513622522354126,
    336: 0.6547670960426331,
    339: 0.3487551510334015,
    373: 0.8815233707427979,
    374: 0.9988458156585693,
    375: 0.9324948191642761,
    376: 0.45920634269714355,
    377: 0.9996823072433472,
    382: 0.6362333297729492,
    383: 0.9639233946800232,
    384: 0.48562130331993103,
    387: 0.9989765882492065,
    399: 0.7867100834846497,
    400: 0.8498532176017761,
    402: 0.3530770242214203,
    403: 0.9664086699485779,
    481: 0.8873093724250793,
    482: 0.996204674243927,
    484: 0.9723506569862366,
    486: 0.9942421317100525
}

def detect_lang(text: str):
    try:
        return langdetect.detect(text)
    except LangDetectException as e:
        logger.error(f"Problem detecting lang: {e}")
        return None

def test_gpu_memory(batch_size, model, device, model_max_length):
    # Ensure we have enough GPU memory for the largest possible batch size
    tokens = torch.zeros(size=(batch_size, model_max_length), dtype=torch.int64, device=device)
    model(tokens)
    return

def load_prefilter_kwargs(case_id):
    from tfidf import model_output_dir
    prefilter_dir = model_output_dir(case_id)
    return {
        'vectorizer': pickle.load(open(prefilter_dir / 'vectorizer.pkl', 'rb')),
        'model': pickle.load(open(prefilter_dir / 'model.pkl', 'rb')),
        'threshold': pickle.load(open(prefilter_dir / 'final_metrics.pkl', 'rb'))['threshold'],
        # Disable all components except tokenizer for faster ngram extraction
        'spacy_model': spacy.load(
            'en_core_web_md', disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
        )
    }

def apply_prefilter(
        text: str, sent_boundaries: list[int], vectorizer, model, threshold, spacy_model
) -> tuple[np.ndarray, float]:
    """
    Applies a high-recall TF-IDF classification model as a prefilter. Must let at least one sentence through the filter,
    because the caller's job is to find the most likely sentence.
    :return: a boolean array (True = sentence is filtered/offlimits) indexed by sentence, and a float filter rate
        so we can track how much inference we're saving.
    """
    num_sentences = len(sent_boundaries) - 1
    terms: list[list[str]] = []
    for sent_idx in range(num_sentences):
        sent_text = text[sent_boundaries[sent_idx]:sent_boundaries[sent_idx + 1]]
        terms.append([span.text.lower() for span in extract.ngrams(spacy_model(sent_text), n=[1, 2])])

    # Sparse matrix of term counts
    sentence_vectors = vectorizer.transform(terms)
    pred_proba = model.predict_proba(sentence_vectors)[:, 1]
    max_prob_idx = pred_proba.argmax()

    # Boolean array: True means sentence is filtered out
    prefilter_mask = pred_proba < threshold
    # Always let through the highest-probability sentence
    prefilter_mask[max_prob_idx] = False

    filter_rate = prefilter_mask.sum() / num_sentences
    return prefilter_mask, filter_rate

def apply_sent_span_model(
        text: str, sent_boundaries: list[int],
        prefilter_kwargs,
        tokenizer, hf_model, batch_size, device,
        off_limits: list[tuple[int, int]]=None
) -> None | tuple[float, int, int, int, float]:
    """
    Takes a model that was trained for text classification on sentence spans (usually 1 or 2, but potentially 5+)
    and applies it to full documents to find the highest scoring span.
    First applies a two-phase model to every sentence, finding the maximum score: phase 1 is a fast TF-IDF model that
    narrows it down to relevant sentences, phase 2 is a fine-tuned BERT classifier.

    Then tries expanding the boundaries on either side, looking for an even higher score.
    Ideally we'll end up with the minimal representative sentence span that contains evidence for the case, and we don't
    know in advance how long that will be.

    Char spans can be marked as off limits with list[tuple(int, int)] off_limits. This is so that in production we can
    avoid re-suggesting a point that was declined by a curator.

    Returns the winning score, the winning char span, how many sentences it spans, and the prefilter filter rate.
    In rare cases returns None (only if the entire doc was marked as off_limits)
    """
    if off_limits is None:
        off_limits = []
    sent_boundaries = sent_boundaries + [None]
    num_sentences = len(sent_boundaries) - 1

    # Most of the sentences we can quickly disqualify by testing for key terms, via a TF-IDF classifier prefilter
    #TODO precompute / use doc cache for prefilter tokenization so they are shared between cases?
    prefilter_mask, filter_rate = apply_prefilter(text, sent_boundaries, **prefilter_kwargs)
    logger.debug(f"TF-IDF prefilter applied with filter rate {filter_rate:.2f}")

    # Convert user-provided off_limits spans to a sentence mask
    user_mask = np.zeros(num_sentences, dtype=bool)
    for off_start, off_end in off_limits:
        for i in range(num_sentences):
            sent_start = sent_boundaries[i]
            sent_end = sent_boundaries[i + 1]
            # Check overlap: sentence overlaps with off_limits span if they intersect
            if (sent_end is None or off_start < sent_end) and (off_end is None or sent_start < off_end):
                user_mask[i] = True

    # Combine prefilter mask with user-provided off_limits
    sent_off_limits = prefilter_mask | user_mask

    def _batch(iterable, n=batch_size):
        l = len(iterable)
        for idx in range(0, l, n):
            yield iterable[idx: min(idx + n, l)]

    valid_sent_indices = np.where(~np.array(sent_off_limits))[0]
    # Edge case: if the entire document is off limits, there's nothing to return
    if len(valid_sent_indices) == 0:
        return None

    sent_texts = [text[sent_boundaries[i]: sent_boundaries[i + 1]] for i in valid_sent_indices]
    softmax_probs = []
    for batch in _batch(sent_texts):
        tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        tokens.to(device)
        preds = hf_model(**tokens)
        softmax_probs += f.softmax(preds.logits, dim=1)[:,1].tolist()
    softmax_probs = np.array(softmax_probs)
    best_score = softmax_probs.max()
    best_sent_idx = valid_sent_indices[softmax_probs.argmax()]

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
            preds = hf_model(**tokens)
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
            preds = hf_model(**tokens)
            softmax_probs += f.softmax(preds.logits, dim=1)[:,1].tolist()

        softmax_probs = np.array(softmax_probs)
        best_score_with_afters = softmax_probs.max()
        if best_score_with_afters > best_score:
            best_score = best_score_with_afters
            num_after = softmax_probs.argmax() + 1

    # Return winning score, char span, how many sentences it spans, and the prefilter filter rate
    return best_score,\
        sent_boundaries[best_sent_idx - num_prior],\
        sent_boundaries[best_sent_idx + num_after + 1],\
        1 + num_prior + num_after,\
        filter_rate


def attach_predictions(doc_df, tokenizer, sent_boundaries, model, batch_size=4, device='cpu'):
    """
    Supports evaluation during train.py
    Instead of selecting the class with a higher logit, this gathers post-softmax probabilities and
    uses whatever threshold optimizes F1 of the dataset
    """
    doc_df = doc_df.copy()
    for idx, instance in tqdm(doc_df.iterrows(), total=len(doc_df), desc=f'Applying model to docs on {device}'):
        ret = apply_sent_span_model(
            instance.text, sent_boundaries[instance.id_doc], tokenizer, model,
            batch_size=batch_size, device=device
        )
        if ret is not None:
            score, start, end, num_sents, filter_rate = ret
            doc_df.at[idx, 'pred_score'] = score
            doc_df.at[idx, 'pred_char_start'] = start
            doc_df.at[idx, 'pred_char_end'] = end
            doc_df.at[idx, 'pred_num_sents'] = num_sents
            doc_df.at[idx, 'prefilter_rate'] = filter_rate

    doc_df['int_labels'] = [0 if i == 'negative' else 1 for i in doc_df.label]
    threshold = utils.optimal_threshold(doc_df.pred_score, doc_df.int_labels)[0]
    doc_df['pred_label'] = [1 if bool_pred else 0 for bool_pred in doc_df.pred_score >= threshold]
    return doc_df

