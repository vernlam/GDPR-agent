from unittest.mock import MagicMock
from Ingestion.utils.spark_helpers import validate_columns
from Ingestion.utils.spark_helpers import table_exists
from Ingestion.utils.spark_helpers import list_to_dataframe
import pytest

def test_empty_dataframe_raises_error():
    fake_df = MagicMock()
    fake_df.columns = []
    required_columns = ["name","email"]

    with pytest.raises(ValueError):
        validate_columns(fake_df,required_columns)

def test_all_columns_present():
    fake_df = MagicMock()
    fake_df.columns = ["name", "age"]

    result = validate_columns(fake_df, ["name", "age"])

    assert result is True


def test_missing_column_raises_error():
    fake_df = MagicMock()
    fake_df.columns = ["name"]
    required_columns = ["name","email"]

    with pytest.raises(ValueError):
        validate_columns(fake_df,required_columns)

def test_required_columns_empty():
    fake_df = MagicMock()
    fake_df.columns = ["name"]
    required_columns = []

    result = validate_columns(fake_df,required_columns)

    assert result is True


def test_table_exists_returns_true():
    fake_spark = MagicMock()
    # No setup needed — mock methods succeed silently by default

    result = table_exists("catalog.schema.my_table", fake_spark)

    assert result is True

def test_table_does_not_exist_returns_false():
    fake_spark = MagicMock()
    fake_spark.table.side_effect = Exception("Table not found")
    result = table_exists("catalog.schema.my_table", fake_spark)
    assert result is False


def test_list_to_dataframe_empty_records():
    fake_spark = MagicMock()
    with pytest.raises(ValueError):
        list_to_dataframe([],fake_spark)

def test_list_to_dataframe_valid_records():
    fake_df = MagicMock()
    fake_spark = MagicMock()
    fake_spark.createDataFrame.return_value = fake_df
    result = list_to_dataframe([{"name":"Alice"}],fake_spark)
    assert result is fake_df