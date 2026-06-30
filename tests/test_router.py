from unittest.mock import patch, MagicMock
from gdpr_agent.router import route_query
import json
import pytest

def test_invalid_json():

    fake_response = MagicMock()
    fake_response.choices[0].message.content = 'not valid json'

    with patch("gdpr_agent.router.config") as mock_config:
        mock_config.openai_client.chat.completions.create.return_value = fake_response
        with pytest.raises(json.JSONDecodeError):
            route_query("Some Question")

def test_failed_openai_call():

    with patch("gdpr_agent.router.config") as mock_config:
        mock_config.openai_client.chat.completions.create.side_effect = Exception("API Failed")
        with pytest.raises(Exception):
            route_query("Some Question")
