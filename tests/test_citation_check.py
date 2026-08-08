import pytest
from src.generation.citation_check import (
    extract_cited_claims,
    verify_citations,
    filter_hallucinated_citations,
    _verify_one,
)


class MockDoc:
    def __init__(self, page_content: str, source: str, page: int | str):
        self.page_content = page_content
        self.metadata = {"source": source, "page": page}


class MockLLMClient:
    def __init__(self, responses: dict[str, str] | None = None, default_response: str = "yes"):
        self.responses = responses or {}
        self.default_response = default_response
        self.calls = []

    def complete(self, messages, **kwargs):
        content = messages[0]["content"] if messages else ""
        self.calls.append(content)
        for key, resp in self.responses.items():
            if key.lower() in content.lower():
                return resp
        return self.default_response


class FailingLLMClient:
    def complete(self, messages, **kwargs):
        raise RuntimeError("API Rate Limit exceeded")


def test_extract_cited_claims_standard_format():
    answer = "Aryan studied Computer Science [Aryan Shah Resume.pdf - Page 1]."
    claims = extract_cited_claims(answer)
    assert len(claims) == 1
    assert claims[0]["source"] == "Aryan Shah Resume.pdf"
    assert claims[0]["page"] == 1
    assert claims[0]["raw_citation"] == "[Aryan Shah Resume.pdf - Page 1]"
    assert "Aryan studied Computer Science" in claims[0]["text"]


def test_extract_cited_claims_dash_variations():
    # En-dash, em-dash, colon, comma
    answer = (
        "Project A was successful [doc1.pdf – Page 2]. "
        "Project B had high ROI [doc2.pdf — Page 3]. "
        "Project C was finished early [doc3.pdf: Page 4]."
    )
    claims = extract_cited_claims(answer)
    assert len(claims) == 3
    assert claims[0]["source"] == "doc1.pdf" and claims[0]["page"] == 2
    assert claims[1]["source"] == "doc2.pdf" and claims[1]["page"] == 3
    assert claims[2]["source"] == "doc3.pdf" and claims[2]["page"] == 4


def test_extract_cited_claims_multiple_in_one_sentence():
    answer = "Python was used for backend [backend.pdf - Page 1] and React for frontend [frontend.pdf - Page 2]."
    claims = extract_cited_claims(answer)
    assert len(claims) == 2
    assert claims[0]["source"] == "backend.pdf" and claims[0]["page"] == 1
    assert claims[1]["source"] == "frontend.pdf" and claims[1]["page"] == 2


def test_extract_cited_claims_markdown_bullets():
    answer = """Here is the breakdown:
- Fast response times achieved [metrics.pdf - Page 1]
- 99.9% uptime guaranteed [sla.pdf - Page 5]
- Simple documentation [readme.txt]
"""
    claims = extract_cited_claims(answer)
    assert len(claims) == 3
    assert claims[0]["source"] == "metrics.pdf" and claims[0]["page"] == 1
    assert claims[1]["source"] == "sla.pdf" and claims[1]["page"] == 5
    assert claims[2]["source"] == "readme.txt" and claims[2]["page"] == 1


def test_extract_cited_claims_ignores_unverified_tags():
    answer = "Some unverified fact [unverified]. Valid fact [valid.pdf - Page 2]."
    claims = extract_cited_claims(answer)
    assert len(claims) == 1
    assert claims[0]["source"] == "valid.pdf"


def test_verify_citations_successful():
    client = MockLLMClient(default_response="yes")
    claims = [
        {"text": "Python engineer", "source": "resume.pdf", "page": 1, "raw_citation": "[resume.pdf - Page 1]"},
        {"text": "Built RAG systems", "source": "portfolio.pdf", "page": 2, "raw_citation": "[portfolio.pdf - Page 2]"},
    ]
    docs = [
        MockDoc("Aryan is a Python engineer with 5 years experience.", "resume.pdf", 1),
        MockDoc("Aryan built several RAG systems with LangChain and Groq.", "portfolio.pdf", 2),
    ]

    results = verify_citations(client, claims, docs)
    assert isinstance(results, list)
    assert len(results) == 2
    assert isinstance(results[0], dict)
    assert results[0]["verified"] is True
    assert results[1]["verified"] is True


def test_verify_citations_rejected_by_judge():
    client = MockLLMClient(default_response="no")
    claims = [
        {"text": "Expert in Rust", "source": "resume.pdf", "page": 1, "raw_citation": "[resume.pdf - Page 1]"}
    ]
    docs = [MockDoc("Aryan is a Python engineer.", "resume.pdf", 1)]

    results = verify_citations(client, claims, docs)
    assert len(results) == 1
    assert results[0]["verified"] is False


def test_verify_citations_missing_doc_or_page():
    client = MockLLMClient(default_response="yes")
    claims = [
        {"text": "Fact from page 3", "source": "resume.pdf", "page": 3, "raw_citation": "[resume.pdf - Page 3]"},
        {"text": "Fact from missing file", "source": "other.pdf", "page": 1, "raw_citation": "[other.pdf - Page 1]"},
    ]
    docs = [MockDoc("Page 1 content", "resume.pdf", 1)]

    results = verify_citations(client, claims, docs)
    assert len(results) == 2
    assert results[0]["verified"] is False
    assert results[1]["verified"] is False
    assert len(client.calls) == 0  # Should not even call LLM if doc is missing


def test_verify_citations_handles_case_and_path_differences():
    client = MockLLMClient(default_response="yes")
    claims = [
        {"text": "Fact", "source": "Resume.pdf", "page": "1", "raw_citation": "[Resume.pdf - Page 1]"}
    ]
    docs = [MockDoc("Relevant content", "/tmp/uploads/resume.pdf", 1)]

    results = verify_citations(client, claims, docs)
    assert len(results) == 1
    assert results[0]["verified"] is True


def test_verify_citations_handles_llm_exception_gracefully():
    client = FailingLLMClient()
    claims = [
        {"text": "Fact", "source": "resume.pdf", "page": 1, "raw_citation": "[resume.pdf - Page 1]"}
    ]
    docs = [MockDoc("Content", "resume.pdf", 1)]

    # Should not crash with exception, but mark verified as False
    results = verify_citations(client, claims, docs)
    assert len(results) == 1
    assert results[0]["verified"] is False


def test_filter_hallucinated_citations():
    answer = (
        "Fact 1 [verified.pdf - Page 1]. "
        "Fact 2 [hallucinated.pdf – Page 2]. "
        "Fact 3 [third.pdf — Page 3]."
    )
    verified_claims = [
        {"source": "verified.pdf", "page": 1, "raw_citation": "[verified.pdf - Page 1]", "verified": True},
        {"source": "hallucinated.pdf", "page": 2, "raw_citation": "[hallucinated.pdf – Page 2]", "verified": False},
        {"source": "third.pdf", "page": 3, "raw_citation": "[third.pdf — Page 3]", "verified": False},
    ]

    filtered = filter_hallucinated_citations(answer, verified_claims)
    assert "[verified.pdf - Page 1]" in filtered
    assert "[hallucinated.pdf – Page 2]" not in filtered
    assert "[third.pdf — Page 3]" not in filtered
    assert filtered.count("[unverified]") == 2
