from src import phoenix

DOC_ID = 1378
SERVICE_ID = 502
CASE_ID = 235
DOCBOT_USER_ID = 21032

def test_get_docs():
    client = phoenix.Client()
    docs = client.get_docs()
    assert len(docs) > 0

def test_get_doc():
    client = phoenix.Client()
    doc = client.get_doc(DOC_ID)
    assert doc['id'] == DOC_ID
    assert len(doc['text']) > 200

def test_get_points_for_case():
    client = phoenix.Client()
    points = client.get_points_for_case(CASE_ID)
    assert len(points) > 100
    print(points[100])

def test_add_point():
    client = phoenix.Client()
    client.add_point(
        CASE_ID, DOCBOT_USER_ID, DOC_ID, SERVICE_ID, 'test_source', 'Docbot test', 'This policy applies', 4, 23, '1', .1
    )

def test_get_records():
    client = phoenix.Client()
    record_set = client.get_docbot_records(CASE_ID, '1')

def test_add_record():
    client = phoenix.Client()
    client.add_docbot_record(CASE_ID, DOC_ID, '1', '1', 4, 23, 1.)
