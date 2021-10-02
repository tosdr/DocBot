
TAGS_SENT_ENDS = ['</p>', '</li>']
TAGS_IGNORE = ['<p>', '<li>', '<b>', '</b>', '<i>', '</i>', '<u>', '</u>', '<ul>', '</ul>', '<strong>', '</strong>']
def preprocess_doc_text(text):
    # The primary motivation here is to fix sentence splitting on </p>, but we might as well clean up other tags too.
    # It's important to keep the string lengths the same, so char_start and char_end annotations don't change.
    # So instead of using a third-party HTML removal lib, we'll just string replace the most common tags
    for tag in TAGS_SENT_ENDS:
        text = text.replace(tag, '.' + (' ' * (len(tag) - 1)))
    for tag in TAGS_IGNORE:
        text = text.replace(tag, ' ' * len(tag))
    return text

def optimal_threshold(preds, int_labels) -> tuple[float, float, float, float]:
    """
    Given predicted scores and ground truth labels, finds the threshold to maximize F1 and returns thresh/prec/rec/f1
    """
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(int_labels, preds)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], precision[best_idx], recall[best_idx], f1_scores[best_idx]
