from embkit.lib.planning import Document, multi_hop_retrieve, planned_search


def test_planned_search_filters_by_location():
    docs = [
        Document(id="1", text="George Washington, born in Virginia", score=0.8),
        Document(id="2", text="Theodore Roosevelt, born in New York", score=0.9),
        Document(id="3", text="Barack Obama, born in Hawaii", score=0.7),
    ]

    def search_fn(query: str, k: int = 10):
        return docs

    filtered = planned_search("presidents born in New York", search_fn)
    assert any("New York" in doc.text for doc in filtered)
    assert all("Virginia" not in doc.text for doc in filtered)


def test_multi_hop_retrieve_runs_second_query():
    question = "In what year was the president of France (who was born in 1977) elected?"
    macron_doc = Document(id="m1", text="Emmanuel Macron, president of France, born in 1977.", score=0.9)
    election_doc = Document(id="m2", text="Emmanuel Macron was elected in 2017.", score=0.95)

    def search_fn(query: str, k: int = 10):
        if "born in 1977" in query:
            return [macron_doc]
        if "elected" in query:
            return [election_doc]
        return []

    results = multi_hop_retrieve(question, search_fn, k=5)
    assert any("2017" in doc.text for doc in results)
