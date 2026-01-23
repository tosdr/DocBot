import logging
import os
from pathlib import Path

from peft import PeftModel
import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import inference, utils
from src.inference import MODEL_VERSION


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

LOCAL_PEFT_PATH = here / f'../data/models/{MODEL_VERSION}/'
LOCAL_DUMP_VERSION = '211222'


def list_case_models() -> set[int]:
    case_ids = set()
    for dirname in os.listdir(LOCAL_PEFT_PATH):
        dirpath = Path(LOCAL_PEFT_PATH / dirname)
        if dirpath.is_dir() and (dirpath / 'adapter_config.json').exists():
            case_ids.add(int(dirname))

    # Make sure these have thresholds
    missing_thresholds = case_ids - set(inference.THRESHOLDS.keys())
    if len(missing_thresholds) > 0:
        raise RuntimeError(f"Classification threshold not found for {missing_thresholds}")

    return case_ids

def peft_path(case_id):
    return (LOCAL_PEFT_PATH / str(case_id)).as_posix()

def run_ad_hoc(text_file_paths: list, device='mps', batch_size=16):
    """
    Run case models on a group of docs. Finds whatever PEFT adapters exist locally in LOCAL_PEFT_PATH.
    :param text_file_paths: list of string filenames or Path objects to text files to be analyzed
    :param device: cpu / mps / cuda
    :param batch_size:
    :return:
    """
    # case_ids = list(sorted(list_case_models()))
    case_ids = [201]

    logger.info(f"Loading spacy model")
    spacy_model = spacy.load('en_core_web_md', disable=['attribute_ruler', 'lemmatizer', 'ner'])

    texts: list[str] = []
    sent_boundaries: list[list[int]] = []
    filenames: list[str] = []
    for path in text_file_paths:
        text = open(path, 'r').read()

        # Preprocess by getting rid of html. Remove blank docs
        content = utils.preprocess_doc_text(text)
        if content.strip() == '' or inference.detect_lang(content) != 'en':
            continue
        texts.append(content)
        filenames.append(Path(path).name)

        spacy_doc = spacy_model(content)
        sents = list(filter(lambda sent: sent.text != '' and not sent.text.isspace(), spacy_doc.sents))
        sent_boundaries.append(list(sorted(map(lambda s: s.start_char, sents))))

    logger.info(f"Running {len(text_file_paths)} docs on {len(case_ids)} case IDs: {case_ids}")
    for case_idx, case_id in enumerate(case_ids):
        logger.info(f"====================== Case {case_id} ({case_idx + 1}/{len(case_ids)}) ========================")

        base_model = AutoModelForSequenceClassification.from_pretrained(inference.BASE_MODEL_NAME)
        # https://github.com/huggingface/peft/issues/217#issuecomment-1506224612
        model = PeftModel.from_pretrained(base_model, peft_path(case_id)).merge_and_unload()
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(inference.BASE_MODEL_NAME)
        prefilter_kwargs = inference.load_prefilter_kwargs(case_id)

        for filename, boundaries, text in zip(filenames, sent_boundaries, texts):
            ret = inference.apply_sent_span_model(
                text, boundaries, prefilter_kwargs, tokenizer, model, batch_size, device, off_limits=None
            )
            if ret is not None:
                score, best_start, best_end, _, __ = ret
                evidence_str = text[best_start:best_end]

                if score >= inference.THRESHOLDS[case_id]:
                    logger.info(f"{filename} scored {score:.3f} from {best_start}-{best_end} with:\n{evidence_str}")

if __name__ == '__main__':
    run_ad_hoc([
        'doc.txt',
    ])
