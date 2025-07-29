import pytest
import zipfile
import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.preprocessing import OneHotEncoder

from model.lightgbm_202504 import file_handler as fh
from model.lightgbm_202504.train import (
    disparate_impact, fairness_metrics
)
from model.lightgbm_202504.preprocessing_tools import (
    get_region, impute_missing_values, assign_default_values
)

def test_extract_quarter():
    assert fh.extract_quarter("abc_Q3.txt") == "3"
    assert fh.extract_quarter("nothing.txt") is None

def test_unzip_and_load(tmp_path):
    zip_path = tmp_path / "sample.zip"
    log_dir = tmp_path / "logdir"
    file1 = log_dir / "Q1_data.txt"
    file1.parent.mkdir(parents=True, exist_ok=True)
    file1.write_text("12345|100000|3.5")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(file1, arcname=file1.name)
    unzip_dir = fh.unzip_file(str(zip_path))
    files = fh.find_all_text_files(unzip_dir)
    df = fh.load_with_quarter(files, ["loan_sequence_num", "loan_amount", "interest_rate"], unzip_dir)
    assert isinstance(df, dd.DataFrame)

def test_assign_default_values_structure():
    defaults = assign_default_values()
    assert "ELTV" in defaults and isinstance(defaults, dict)

def test_impute_missing_values_basic():
    df = pd.DataFrame({"ELTV": [70.0, 80.0, 999.0]})
    result = impute_missing_values(df.copy(), "ELTV", 999.0)
    assert 999.0 not in result.values

def test_get_region_mapping():
    assert get_region("CA") == "West"
    assert get_region("ZZ") == "Other"

@pytest.fixture
def dummy_encoder():
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    fit_data = pd.DataFrame({"property_valuation_method": ["1", "2"], "channel": ["1", "2"], "region": ["West", "South"]})
    ohe.fit(fit_data.astype(str))
    return ohe

@pytest.fixture
def dummy_scaling():
    return {"cur_actual_UPB": {"min": 100000, "max": 200000}}

def test_disparate_impact_valid_output():
    y_pred = np.array([1, 0, 1, 0])
    sensitive = np.array([1, 0, 1, 0])
    assert isinstance(disparate_impact(y_pred, sensitive), float)

def test_fairness_metrics_keys():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    attr = np.array([[1], [0], [1], [0]])
    results = fairness_metrics(y_true, y_pred, attr)
    for key in ["demographic_parity_difference", "equalized_odds_difference", "disparate_impact"]:
        assert key in results

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50),
        'feature3': np.random.randint(0, 2, 50)
    })
    y = np.random.randint(0, 2, 50)
    attr = np.random.randint(0, 2, (50, 1))
    return X, y, attr