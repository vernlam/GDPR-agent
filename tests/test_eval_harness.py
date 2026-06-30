from eval_harness.utils import load_test_dataset
import pytest

def test_load_test_dataset():
    fake_path = 'fake_path/fake_path/test'

    with pytest.raises(FileNotFoundError):
        load_test_dataset(fake_path)
