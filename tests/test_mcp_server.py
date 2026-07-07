import pytest
from mcp_server import server


def test_successful_call_returns_response(monkeypatch):
    class FakeAgent:
        def invoke(self, input_data):
            return {"answer": "fake answer", "context": "fake context"}

    monkeypatch.setattr(server, "get_agent", lambda: FakeAgent())

    result = server.invoke_with_retry("What is GDPR?")

    assert result["answer"] == "fake answer"

def test_empty_question_raises_error():
    with pytest.raises(ValueError):
        server.invoke_with_retry("")

def test_retries_then_succeeds(monkeypatch):
    class FakeAgent:
        call_count = 0

        def invoke(self, input_data):
            FakeAgent.call_count += 1
            if FakeAgent.call_count == 1:
                raise Exception("simulated transient failure")
            return {"answer": "success on retry", "context": "..."}

    monkeypatch.setattr(server, "get_agent", lambda: FakeAgent())
    monkeypatch.setattr(server.time, "sleep", lambda x: None)  # skip real sleep delays

    result = server.invoke_with_retry("What is GDPR?")
    assert result["answer"] == "success on retry"
