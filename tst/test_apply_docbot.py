"""
Integration tests for src/apply_docbot.py::run_all_cases()

These tests cover the main document inference scenarios without making real API calls or loading ML models.

Fixed parameters:
- local_models=True: Load models from disk, not S3
- local_data=False: Use Phoenix client (stubbed) for docs/points
- dont_post=True: Don't POST results to Phoenix
"""

import pytest
from unittest.mock import Mock, patch


# Test case ID used across tests
TEST_CASE_ID = 134


@pytest.fixture
def temp_dirs(tmp_path):
    """Set up temporary LOG_DIR and results_dir"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return log_dir, results_dir


def make_doc(doc_id, lang='en', text='Sample document text for testing.', service_id=1, url='http://example.com'):
    """Helper to create a document dict with required fields"""
    return {
        'id': doc_id,
        'text': text,
        'content': text,  # In real code this is preprocessed, but for tests we use same text
        'lang': lang,
        'sent_boundaries': [0, len(text)],
        'service_id': service_id,
        'url': url,
    }


def make_empty_points_list():
    """
    Helper to create an empty points list that will result in a DataFrame with required columns.
    The real Phoenix API returns at least one point or the columns must be present.
    """
    # Return a list with a dummy point that will never match any doc_id
    # This ensures DataFrame has the right columns
    return [{
        'document_id': -999,
        'status': 'declined',
        'quote_start': 0,
        'quote_end': 0,
        'case_id': TEST_CASE_ID,
    }]


class TestRunCase:
    """Test the run_case function directly for more precise control"""

    def test_ran_found_creates_result(self, temp_dirs):
        """
        Scenario: Document passes inference with score above threshold
        - Mock apply_sent_span_model to return (0.95, 10, 50, 2, 0.5) (high score)
        - Verify result_counts['ran_found'] == 1
        - Verify the document ID appears in result_scores with correct score
        """
        from src import apply_docbot

        doc_id = 123
        doc_list = [(doc_id, '1')]
        doc = make_doc(doc_id)
        threshold = 0.5

        # High score - should be found
        inference_return = (0.95, 10, 50, 2, 0.5)

        with patch('src.apply_docbot.inference.apply_sent_span_model', return_value=inference_return), \
             patch('src.apply_docbot.inference.load_prefilter_kwargs', return_value={}), \
             patch('src.apply_docbot.inference.test_gpu_memory'), \
             patch('src.apply_docbot.load_peft_model') as mock_peft, \
             patch('src.apply_docbot.AutoModelForSequenceClassification.from_pretrained') as mock_auto, \
             patch('src.apply_docbot.AutoTokenizer.from_pretrained'):

            # Set up model mock
            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_peft.return_value = mock_model
            mock_auto.return_value = mock_model

            # Mock DocStore
            mock_doc_store = Mock()
            mock_doc_store.__getitem__ = Mock(return_value=doc)

            # Mock Phoenix client
            mock_client = Mock()
            mock_client.get_points_for_case.return_value = make_empty_points_list()
            mock_client.get_docbot_records.return_value = set()

            result_counts, result_scores, mean_prefilter_rate = apply_docbot.run_case(
                case_id=TEST_CASE_ID,
                local_models=True,
                local_data=False,
                dont_post=True,
                doc_list=doc_list,
                doc_store=mock_doc_store,
                phoenix_client=mock_client,
                threshold=threshold,
                batch_size=1,
                device='cpu'
            )

        assert result_counts['ran_found'] == 1
        assert result_counts.get('ran_notfound', 0) == 0
        assert doc_id in result_scores
        assert result_scores[doc_id] == 0.95

    def test_ran_notfound_below_threshold(self, temp_dirs):
        """
        Scenario: Document runs inference but score is below threshold
        - Mock apply_sent_span_model to return (0.3, 10, 50, 2, 0.5) (low score)
        - Verify result_counts['ran_notfound'] == 1
        - Verify document appears in result_scores but wasn't marked as found
        """
        from src import apply_docbot

        doc_id = 456
        doc_list = [(doc_id, '1')]
        doc = make_doc(doc_id)
        threshold = 0.5

        # Low score - should not be found
        inference_return = (0.3, 10, 50, 2, 0.5)

        with patch('src.apply_docbot.inference.apply_sent_span_model', return_value=inference_return), \
             patch('src.apply_docbot.inference.load_prefilter_kwargs', return_value={}), \
             patch('src.apply_docbot.inference.test_gpu_memory'), \
             patch('src.apply_docbot.load_peft_model') as mock_peft, \
             patch('src.apply_docbot.AutoModelForSequenceClassification.from_pretrained') as mock_auto, \
             patch('src.apply_docbot.AutoTokenizer.from_pretrained'):

            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_peft.return_value = mock_model
            mock_auto.return_value = mock_model

            mock_doc_store = Mock()
            mock_doc_store.__getitem__ = Mock(return_value=doc)

            mock_client = Mock()
            mock_client.get_points_for_case.return_value = make_empty_points_list()
            mock_client.get_docbot_records.return_value = set()

            result_counts, result_scores, _ = apply_docbot.run_case(
                case_id=TEST_CASE_ID,
                local_models=True,
                local_data=False,
                dont_post=True,
                doc_list=doc_list,
                doc_store=mock_doc_store,
                phoenix_client=mock_client,
                threshold=threshold,
                batch_size=1,
                device='cpu'
            )

        assert result_counts.get('ran_found', 0) == 0
        assert result_counts['ran_notfound'] == 1
        assert doc_id in result_scores
        assert result_scores[doc_id] == 0.3

    def test_skip_non_english_doc(self, temp_dirs):
        """
        Scenario: Document is skipped because it's not in English
        - Mock DocStore to return a document with lang='fr'
        - Verify result_counts['skip_non_en'] == 1
        - Verify apply_sent_span_model was NOT called
        """
        from src import apply_docbot

        doc_id = 789
        doc_list = [(doc_id, '1')]
        doc = make_doc(doc_id, lang='fr')  # French document
        threshold = 0.5

        with patch('src.apply_docbot.inference.apply_sent_span_model') as mock_inference, \
             patch('src.apply_docbot.inference.load_prefilter_kwargs', return_value={}), \
             patch('src.apply_docbot.inference.test_gpu_memory'), \
             patch('src.apply_docbot.load_peft_model') as mock_peft, \
             patch('src.apply_docbot.AutoModelForSequenceClassification.from_pretrained') as mock_auto, \
             patch('src.apply_docbot.AutoTokenizer.from_pretrained'):

            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_peft.return_value = mock_model
            mock_auto.return_value = mock_model

            mock_doc_store = Mock()
            mock_doc_store.__getitem__ = Mock(return_value=doc)

            mock_client = Mock()
            mock_client.get_points_for_case.return_value = make_empty_points_list()
            mock_client.get_docbot_records.return_value = set()

            result_counts, result_scores, _ = apply_docbot.run_case(
                case_id=TEST_CASE_ID,
                local_models=True,
                local_data=False,
                dont_post=True,
                doc_list=doc_list,
                doc_store=mock_doc_store,
                phoenix_client=mock_client,
                threshold=threshold,
                batch_size=1,
                device='cpu'
            )

        assert result_counts['skip_non_en'] == 1
        assert result_counts.get('ran_found', 0) == 0
        assert result_counts.get('ran_notfound', 0) == 0
        # Verify inference was not called
        mock_inference.assert_not_called()

    def test_skip_doc_with_existing_points(self, temp_dirs):
        """
        Scenario: Document skipped because it already has approved/pending points
        - Set up mock points data with status='approved' for the document
        - Verify result_counts['skip_points'] == 1
        - Verify apply_sent_span_model was NOT called
        """
        from src import apply_docbot

        doc_id = 321
        doc_list = [(doc_id, '1')]
        doc = make_doc(doc_id)
        threshold = 0.5

        # Points with approved status for this doc
        points_data = [
            {
                'document_id': int(doc_id),
                'status': 'approved',
                'quote_start': 0,
                'quote_end': 10,
                'case_id': TEST_CASE_ID,
            }
        ]

        with patch('src.apply_docbot.inference.apply_sent_span_model') as mock_inference, \
             patch('src.apply_docbot.inference.load_prefilter_kwargs', return_value={}), \
             patch('src.apply_docbot.inference.test_gpu_memory'), \
             patch('src.apply_docbot.load_peft_model') as mock_peft, \
             patch('src.apply_docbot.AutoModelForSequenceClassification.from_pretrained') as mock_auto, \
             patch('src.apply_docbot.AutoTokenizer.from_pretrained'):

            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_peft.return_value = mock_model
            mock_auto.return_value = mock_model

            mock_doc_store = Mock()
            mock_doc_store.__getitem__ = Mock(return_value=doc)

            mock_client = Mock()
            mock_client.get_points_for_case.return_value = points_data
            mock_client.get_docbot_records.return_value = set()

            result_counts, result_scores, _ = apply_docbot.run_case(
                case_id=TEST_CASE_ID,
                local_models=True,
                local_data=False,
                dont_post=True,
                doc_list=doc_list,
                doc_store=mock_doc_store,
                phoenix_client=mock_client,
                threshold=threshold,
                batch_size=1,
                device='cpu'
            )

        assert result_counts['skip_points'] == 1
        assert result_counts.get('ran_found', 0) == 0
        assert result_counts.get('ran_notfound', 0) == 0
        # Verify inference was not called
        mock_inference.assert_not_called()

    def test_multiple_docs_mixed_outcomes(self, temp_dirs):
        """
        Scenario: Process multiple documents with different outcomes in one run
        - Doc 1: English, runs inference, passes threshold -> ran_found
        - Doc 2: English, runs inference, below threshold -> ran_notfound
        - Doc 3: Non-English -> skip_non_en
        - Verify all counts are correct
        """
        from src import apply_docbot

        doc1_id = '100'
        doc2_id = '200'
        doc3_id = '300'
        doc_list = [(doc1_id, '1'), (doc2_id, '1'), (doc3_id, '1')]

        doc1 = make_doc(doc1_id, lang='en')
        doc2 = make_doc(doc2_id, lang='en')
        doc3 = make_doc(doc3_id, lang='fr')  # French

        docs = {doc1_id: doc1, doc2_id: doc2, doc3_id: doc3}
        threshold = 0.5

        # Track which doc inference is called with, return different scores
        inference_calls = []

        def inference_side_effect(text, sent_boundaries, prefilter_kwargs, tokenizer, model, batch_size, device, off_limits):
            # Determine which doc based on the call order
            call_num = len(inference_calls)
            inference_calls.append(call_num)

            if call_num == 0:
                # Doc 1: high score
                return (0.95, 10, 50, 2, 0.5)
            else:
                # Doc 2: low score
                return (0.3, 10, 50, 2, 0.5)

        with patch('src.apply_docbot.inference.apply_sent_span_model', side_effect=inference_side_effect) as mock_inference, \
             patch('src.apply_docbot.inference.load_prefilter_kwargs', return_value={}), \
             patch('src.apply_docbot.inference.test_gpu_memory'), \
             patch('src.apply_docbot.load_peft_model') as mock_peft, \
             patch('src.apply_docbot.AutoModelForSequenceClassification.from_pretrained') as mock_auto, \
             patch('src.apply_docbot.AutoTokenizer.from_pretrained'):

            mock_model = Mock()
            mock_model.to = Mock(return_value=mock_model)
            mock_peft.return_value = mock_model
            mock_auto.return_value = mock_model

            mock_doc_store = Mock()
            mock_doc_store.__getitem__ = Mock(side_effect=lambda doc_id: docs.get(doc_id))

            mock_client = Mock()
            mock_client.get_points_for_case.return_value = make_empty_points_list()
            mock_client.get_docbot_records.return_value = set()

            result_counts, result_scores, _ = apply_docbot.run_case(
                case_id=TEST_CASE_ID,
                local_models=True,
                local_data=False,
                dont_post=True,
                doc_list=doc_list,
                doc_store=mock_doc_store,
                phoenix_client=mock_client,
                threshold=threshold,
                batch_size=1,
                device='cpu'
            )

        # Verify counts
        assert result_counts['ran_found'] == 1
        assert result_counts['ran_notfound'] == 1
        assert result_counts['skip_non_en'] == 1

        # Verify scores
        assert doc1_id in result_scores
        assert result_scores[doc1_id] == 0.95
        assert doc2_id in result_scores
        assert result_scores[doc2_id] == 0.3
        # Doc 3 should not be in scores (skipped)
        assert doc3_id not in result_scores

        # Inference should have been called only twice (for doc1 and doc2)
        assert mock_inference.call_count == 2
