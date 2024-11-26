import pytest
import os
import tempfile
from typing import Generator, Dict, Any
from src.model_tuner.pickleObjects import dumpObjects, loadObjects

@pytest.fixture
def sample_object() -> Dict[str, Any]:
    """Fixture for providing a sample object for testing"""
    return {'key1': 'value1', 'key2': 'value2', 'key3': [1, 2, 3]}

@pytest.fixture
def temp_filename() -> Generator[str, None, None]:
    """Fixture for creating and cleaning up a temporary file for testing"""
    fd, path = tempfile.mkstemp()
    os.close(fd)  # Close to make it available elsewhere
    yield path
    os.remove(path)

def test_pickle_dump_and_load(sample_object: Dict[str, Any], temp_filename: str) -> None:
    """Test dumping and loading using pickle"""
    dumpObjects(sample_object, temp_filename, use_pickle=True)
    loaded_object = loadObjects(temp_filename, use_pickle=True)
    assert loaded_object == sample_object

def test_joblib_dump_and_load(sample_object: Dict[str, Any], temp_filename: str) -> None:
    """Test dumping and loading using joblib"""
    dumpObjects(sample_object, temp_filename, use_pickle=False)
    loaded_object = loadObjects(temp_filename, use_pickle=False)
    assert loaded_object == sample_object