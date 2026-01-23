import backoff
import json
import logging
import os
import time

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Client for Phoenix APIs necessary to run our models in production 
"""

def _backoff_handler(details):
    msg = f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries calling {details['args'][1]}"
    if 'params' in details['kwargs']:
        msg += f" with kwargs {details['kwargs']['params']}"
    logger.debug(msg)

def _assert_int(n):
    if not isinstance(n, int):
        raise ValueError(f"{n} is not an int?")
    return n


class Client:
    def __init__(self):
        try:
            # e.g. https://api.staging.tosdr.org
            self.endpoint = os.environ['PHOENIX_API_URL']
            self.api_key = os.environ['PHOENIX_API_KEY']
        except KeyError:
            raise EnvironmentError(f"Environment variables PHOENIX_API_URL and PHOENIX_API_KEY not set")

    def _request(self, request_kwargs: dict):
        # Timeout for requests - handles SSL hangs that can occur during do_handshake()
        # See: https://bugs.python.org/issue34438
        # (connect_timeout, read_timeout) in seconds
        request_kwargs.setdefault('timeout', (30, 120))
        return requests.request(**request_kwargs)

    @backoff.on_exception(backoff.expo, RuntimeError, max_time=300, on_backoff=_backoff_handler, logger=None)
    def _call(
            self, path: str, method: str, is_private: bool, desc: str, allow_404: bool=False, params: dict=None,
            payload: dict=None
    ):
        if payload is None:
            payload = dict()
            headers = dict()
        else:
            headers = {'Content-Type': 'application/json'}
        if params is None:
            params = dict()
        if is_private:
            headers['apikey'] = self.api_key

        res = self._request(
            dict(method=method, url=f'{self.endpoint}{path}', params=params, json=payload, headers=headers)
        )
        if res.status_code < 400 or (allow_404 and res.status_code == 404):
            return res
        else:
            err_str = f"Phoenix client error: unable to {desc}:\t{res.status_code}\n\t{self.endpoint}{path}"
            if res.text is not None and res.text.strip() != '':
                err_str += f"\n\t{res.text}"
            raise RuntimeError(err_str)

    def get_docs(self) -> list[tuple[int, str]]:
        res = self._call('/document/v2', 'get', False, "get all Doc IDs")
        return json.loads(res.text)['documents']

    def get_doc(self, doc_id) -> dict:
        res = self._call(
            '/document/v2', 'get', False, 'GET doc by ID', allow_404=True,
            params={'id': _assert_int(doc_id)}
        )
        if res.status_code == 200:
            return json.loads(res.text)
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
            res = self._call(
                '/point/v1',
                'get',
                False,
                'GET Points for case',
                # Technically 404 isn't an abnormal state if this was run against a fresh staging env,
                # but it's unexpected and safer to throw an error
                allow_404=False,
                params={'case_id': _assert_int(case_id), 'page': current_page}
            )

            res_parameters = json.loads(res.text)
            points += res_parameters['points']
            max_pages = res_parameters['page']['end']
            current_page += 1
            if current_page < max_pages:
                time.sleep(0.5)
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
        res = self._call('/point/v1', 'post', True, 'POST new Point', payload=data_dict)
        if res.status_code != 201:
            raise RuntimeError(f"Not a 201 when POSTing point, got {res.status_code}")

    def get_docbot_records(self, case_id, docbot_version) -> set[tuple]:
        res = self._call(
            '/docbotrecord/v1',
            'get',
            True,
            'GET DocbotRecords',
            allow_404=True,
            params={'case_id': _assert_int(case_id), 'docbot_version': docbot_version}
        )
        if res.status_code == 404 or len(json.loads(res.text)) == 0:
            return set()
        elif res.status_code == 200:
            return set(map(tuple, json.loads(res.text)['documents']))
        else:
            raise RuntimeError(f"Unexpected response code {res.status_code} from GET DocbotRecords?")


    def add_docbot_records(self, records: list[dict]):
        if not records:
            return

        res = self._call(
            '/docbotrecord/v1',
            'post',
            True,
            desc="POST DocbotRecords",
            payload={'records': records}
        )
        if res.status_code != 201:
            raise RuntimeError(f"Not a 201 when bulk POSTing DocbotRecords, returned {res.status_code}")
