import pytest
from pathlib import Path
import sys

# extend path for imports
filepath = Path(__file__).parents[1]
sys.path.append(str(filepath))
import example2transformers as e2t  # noqa

TEST_URL = "iulusoy/test-transformers-pipeline"
# TEST_URL = "iulusoy/test-images"


# fixture to instantiate the class
@pytest.fixture(scope="module")
def da():
    return e2t.DataAnalysis(TEST_URL)


# write tests for the DataAnalysis class
def test_data_analysis_instantiation(da):
    # test that the class is instantiated correctly
    assert isinstance(da, e2t.DataAnalysis)
    da.get_dataset()
    assert hasattr(da, "dataset")


# test that the dataset is loaded correctly
def test_data_analysis_print_info(da):
    da.get_dataset()
    info, features = da.print_info()
    features = features.to_dict()
    assert info is not None
    assert info.dataset_name == "test-transformers-pipeline"
    assert features["sentence"]["dtype"] == "string"


# test that the pipeline is set correctly
# test that data is processed through the pipeline (fragments)
# test that data is processed through the pipeline (all data)
def test_data_analysis_set_pipeline(da):
    da.get_dataset()
    da.set_pipeline("sentiment-analysis")
    assert hasattr(da, "pipe")
    sentiment = da.analyze_fragment(1, "sentence")
    assert sentiment[0]["label"] == "POSITIVE"
    assert sentiment[0]["score"] == pytest.approx(0.997, 0.001)
    sentiment_all = da.analyze_all("sentence")
    assert sentiment_all[2]["label"] == "POSITIVE"
    assert sentiment_all[2]["score"] == pytest.approx(0.999, 0.001)
