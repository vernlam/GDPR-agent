from Ingestion.utils.translation_utils import chunk_text
import pytest


def test_short_text_stays_in_one_chunk():
    result = chunk_text("Hello world.", max_chars=3000)

    assert len(result) == 1

def test_empty_string_returns_empty_list():
    result = chunk_text("", max_chars=3000)

    assert result == []

def test_long_text_splits_into_multiple_chunks():

    text = "First sentence here. Second sentence here."

    result = chunk_text(text, max_chars = 30)

    assert len(result) >= 2

def test_no_content_is_lost_during_chunking():

    text = "Apple. Banana. Cherry. Date. Elderberry."

    result = chunk_text(text,max_chars = 30)

    for word in ["Apple", "Banana", "Cherry", "Date", "Elderberry"]:
        assert word in " ".join(result)

@pytest.mark.parametrize("text", [
    "Single sentence.",
    "Two sentences here. And another one.",
    "Question? Yes! Period.",
])

def test_result_is_always_a_list(text):
    result = chunk_text(text)
    
    assert isinstance(result,list)
