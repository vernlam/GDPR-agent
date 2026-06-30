"""
In order to run the integration test, update your databricks config file with the below:

[DEFAULT]
host=https://[YOUR_WORKSPACE_HERE].cloud.databricks.com/
auth_type=databricks-cli
token = YOUR_TOKEN_HERE
"""

from databricks.vector_search.client import VectorSearchClient

def test_databricks_client_connects():
    client = VectorSearchClient(disable_notice=True)
    assert client is not None