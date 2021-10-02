import backoff
import json
import logging
import os

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Client for Phoenix APIs necessary to run our models in production 
"""


def _backoff_hdlr(details):
    logger.info("Backing off {wait:0.1f} seconds after {tries} tries calling phoenix with args {args} and kwargs "
                "{kwargs}".format(**details))

def _assert_int(n):
    if not isinstance(n, int):
        raise ValueError(f"{n} is not an int?")
    return n

class Client:
    """
    Assumes ENV variables have been set with the necessary appwrite URLs
    """
    def __init__(self):
        endpoints = dict()
        for endpoint_key in ['get_doc', 'get_points', 'post_point', 'get_docbotrecords', 'post_docbotrecord']:
            env_name = f'PHOENIX_{endpoint_key.upper()}'
            try:
                endpoints[endpoint_key] = os.environ[env_name]
            except KeyError:
                raise EnvironmentError(f"Environment variable {env_name} not set to {endpoint_key} URL")
        try:
            self.api_key = os.environ['PHOENIX_API_KEY']
        except KeyError:
            raise EnvironmentError(f"Environment variable PHOENIX_API_KEY not set")

        self.endpoints = endpoints

    @backoff.on_exception(backoff.expo, RuntimeError, max_time=600, on_backoff=_backoff_hdlr)
    def _post(
            self, endpoint: str, desc: str, allow_404: bool=False, params: dict=None, payload: dict=None,
    ):
        if payload is None:
            payload = dict()
        if params is None:
            params = dict()
        res = requests.post(
            f'http://{endpoint}',
            params=params,
            json=payload,
            headers={'X-Appwrite-Key': self.api_key}
        )
        if res.status_code < 400 or (allow_404 and res.status_code == 404):
            return res
        else:
            err_str = f"Phoenix client error: unable to {desc}:\n{res.status_code}"
            if res.text is not None and res.text.strip() != '':
                err_str += f":\n{res.text}"
            raise RuntimeError(err_str)

    def get_docs(self) -> list[tuple[int, str]]:
        res = self._post(self.endpoints['get_doc'], "get all Doc IDs")
        return json.loads(res.text)['parameters']['documents']

    def get_doc(self, doc_id) -> dict:
        res = self._post(
            self.endpoints['get_doc'], 'GET doc by ID', allow_404=True, params={'id': _assert_int(doc_id)}
        )
        if res.status_code == 200:
            return json.loads(res.text)['parameters']
        elif res.status_code == 404:
            logger.warning(f"Doc {doc_id} not found in Phoenix")
            return None
        else:
            raise RuntimeError(f"{res.status_code} from Phoenix for doc {doc_id} GET?")

    def get_points_for_case(self, case_id) -> list[dict]:
        points = []
        current_page = 1
        max_pages = 1
        while current_page <= max_pages:
            # https://github.com/tosdr/API/blob/docbot/models/src/models/Points.ts#L47
            res = self._post(
                self.endpoints['get_points'],
                'GET Points for case',
                # Technically 404 isn't an abnormal state if this was run against a fresh staging env,
                # but it's unexpected and safer to throw an error
                allow_404=False,
                params={'case_id': _assert_int(case_id), 'page': current_page}
            )

            res_parameters = json.loads(res.text)['parameters']
            points += res_parameters['points']
            max_pages = res_parameters['_page']['end']
            current_page += 1
        return points

    def add_point(
            self, case_id, user_id, doc_id, service_id, source, analysis, quote_text, quote_start, quote_end,
            docbot_version, ml_score
    ):
        data_dict = {
            'case_id': case_id,
            'user_id': user_id,
            'document_id': doc_id,
            'service_id': service_id,
            'source': source,
            'analysis': analysis,
            'quote_text': quote_text,
            'quote_start': quote_start,
            'quote_end': quote_end,
            'docbot_version': docbot_version,
            'ml_score': ml_score
        }
        res = self._post(self.endpoints['post_point'], desc='POST new Point', payload=data_dict)
        if res.status_code != 201:
            raise RuntimeError("Not a 201 when POSTing point?")

    def get_docbot_records(self, case_id, docbot_version) -> set[tuple]:
        res = self._post(
            self.endpoints['get_docbotrecords'],
            'GET DocbotRecords',
            allow_404=True,
            params={'case_id': _assert_int(case_id), 'docbot_version': docbot_version}
        )
        if res.status_code == 200:
            return set(map(tuple, json.loads(res.text)['parameters']['documents']))
        elif res.status_code == 404:
            return set()
        else:
            raise RuntimeError(f"Unexpected response code {res.status_code} from GET DocbotRecords?")

    def add_docbot_record(
            self, case_id, doc_id, text_version, docbot_version,
            char_start=None, char_end=None, ml_score=None
    ):
        res = self._post(
            self.endpoints['post_docbotrecord'],
            desc="POST DocbotRecord",
            params={
                'case_id': case_id,
                'document_id': doc_id,
                'text_version': text_version,
                'docbot_version': docbot_version,
                'char_start': char_start,
                'char_end': char_end,
                'ml_score': ml_score
            }
        )
        if res.status_code != 201:
            raise RuntimeError("Not a 201 when POSTing DocbotRecord?")
