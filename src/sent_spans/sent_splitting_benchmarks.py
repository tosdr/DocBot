import bisect
from pathlib import Path
import pickle
import time

import pandas as pd
import spacy

from src import utils

here = Path(__file__).parent

VERSION = '181021'
USE_PARSER_SENTS = True
USE_MD = True

"""
Benchmark tests for spacy's sentence splitting. There are two methods to try: a simple rule-based
one ('senter' pipeline component) and a smarter one built into the parser pipeline component.
Spacy also has two pretrained models that should work well, sm and md.
Results: we're going with the parser-based medium model. It does still tend to not add enough sentence boundaries.
It espeically struggles when a period is not followed up by whitespace, like "asdf.</p>"

For 200 docs, single process runtime on a laptop was
senter sm: 21.132 seconds
senter md: 41.096 seconds
parser sm: 88.955 seconds
parser md: 112.923 seconds
"""

if __name__ == '__main__':
    cases = pickle.load(open(here / f'../data/cases_{VERSION}_clean.pkl', 'rb'))
    documents = pickle.load(open(here / f'../data/documents_{VERSION}_clean.pkl', 'rb'))
    points = pickle.load(open(here / f'../data/points_{VERSION}_clean.pkl', 'rb'))

    documents = documents[documents.lang == 'en']
    points = points[points.lang == 'en']

    # Join Points with Documents, so we can check the quote contexts
    points = pd.merge(points, documents, how='left', left_on='document_id', right_index=True, suffixes=['_point', '_doc'])
    # Also join to attach Case info
    points = pd.merge(points, cases, left_on='case_id', right_index=True, suffixes=['_point', '_case'])

    points['quote_len'] = points.quoteEnd - points.quoteStart
    approved_points = points[points.status == 'approved']

    # There's no need to analyze docs without Points
    docs_with_points = documents[documents.index.isin(points.document_id)].head(200)

    spacy_model = 'en_core_web_md' if USE_MD else 'en_core_web_sm'
    if USE_PARSER_SENTS:
        nlp = spacy.load(spacy_model, disable=['attribute_ruler', 'lemmatizer', 'ner'])
    else:
        nlp = spacy.load(spacy_model, disable=['attribute_ruler', 'lemmatizer', 'ner', 'tok2vec', 'parser'])
        nlp.enable_pipe("senter")

    """
    Since we have not removed html, all sentence splitters attempted have failed to split on ".</p>" which leads to 
    lots of extraneous content being added to points. Using spacy's is_sent_start attribute won't work because the 
    tokenizer doesn't keep the tag in one token, splitting 'foo.</p>' into ['foo.</p', '>']
    """
    docs_with_points['text_preprocessed'] = docs_with_points.text.apply(utils.preprocess_doc_text)

    start = time.time()
    for doc_id, doc in zip(docs_with_points.id, nlp.pipe(docs_with_points.text_preprocessed, n_process=4, batch_size=10)):
        # Get sentences with actual content
        sents = list(filter(lambda sent: sent.text != '' and not sent.text.isspace(), doc.sents))
        sent_starts = list(sorted(map(lambda s: s.start_char, sents)))
        for point_id, point in points[points.document_id == doc_id].iterrows():
            num_sents = (1 + (bisect.bisect_right(sent_starts, point.quoteEnd) - bisect.bisect_right(sent_starts, point.quoteStart)))
            points.at[point_id, 'num_sents'] = num_sents

    end = time.time()
    print(f"Time elapsed: {end - start:.3f}")

    print(points.num_sents.describe())

    for text in docs_with_points.text_preprocessed.sample(20):
        print('=' * 60)
        print('🀫'.join(map(str, nlp(text).sents))[250:1250])
