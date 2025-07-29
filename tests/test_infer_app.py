import zipfile, tempfile as tmp, pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from blueprints.infer_app import background_infer_logic
import pandas as pd
import dask.dataframe as dd
from model.lightgbm_202504.file_handler import unzip_file, find_all_text_files, load_with_quarter, merge_data


def create_test_zip():
    with tmp.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
        with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
            zipf.writestr("data_2020Q1.txt", "loan_id,field\n1,foo")
            zipf.writestr("data_time_2020Q1.txt", "loan_id,month,field\n1,1,bar")
        return open(temp_zip.name, 'rb')

def test_unzip_file():
    with tmp.NamedTemporaryFile(delete=False, suffix='.zip') as zf:
        with zipfile.ZipFile(zf.name, 'w') as zipf:
            zipf.writestr("test.txt", "content")
        output = unzip_file(zf.name)
        assert Path(output).exists()
        assert (Path(output) / "test.txt").exists()

def test_find_text_files():
    with tmp.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "orig.txt").write_text("test")
        Path(tmpdir, "perf.txt").write_text("test")
        files = find_all_text_files(tmpdir)
        assert len(files) == 2

@patch('os.path.exists', return_value=True)
def test_load_with_quarter_real_dask(mock_exists, tmp_path):
    # Create fake text files
    data1 = tmp_path / "data_2020Q1.txt"
    data2 = tmp_path / "data_time_2020Q1.txt"
    data1.write_text("1|2")
    data2.write_text("3|4")

    files = [data1.name, data2.name]
    col_names = ["col1", "col2"]
    result = load_with_quarter(files, col_names, str(tmp_path))

    df = result.compute()
    assert "quarter" in df.columns

def test_merge_data():
    df1_pd = pd.DataFrame({"loan_sequence_num": [1, 2], "quarter": [1, 1]})
    df2_pd = pd.DataFrame({"loan_sequence_num": [1, 2], "quarter": [1, 1]})
    df1 = dd.from_pandas(df1_pd, npartitions=1)
    df2 = dd.from_pandas(df2_pd, npartitions=1)

    merged = merge_data(df1, df2)

    assert hasattr(merged, 'compute')  # ensure it's still a Dask object
    result = merged.compute()

    assert isinstance(result, pd.DataFrame)
    assert "quarter" in result.columns
    assert len(result) == 2

@patch("model.lightgbm_202504.infer.find_all_text_files", return_value=[])
@patch("model.lightgbm_202504.infer.unzip_file", return_value="mock_path")
def test_background_infer_logic_missing_files(mock_unzip, mock_find):
    with pytest.raises(RuntimeError):
        background_infer_logic("dummy.zip", 5)

@patch("model.lightgbm_202504.infer.unzip_file", side_effect=Exception("Unzip failed"))
def test_background_infer_logic_unzip_fail(mock_unzip):
    with pytest.raises(RuntimeError):
        background_infer_logic("broken.zip", 5)

@patch("joblib.load", side_effect=FileNotFoundError("Model not found"))
@patch("model.lightgbm_202504.infer.load_scaling", return_value="scale")
@patch("model.lightgbm_202504.infer.load_encoders", return_value="enc")
@patch("model.lightgbm_202504.infer.preprocess", return_value=pd.DataFrame(columns=["loan_sequence_num", "quarter"]))
@patch("model.lightgbm_202504.infer.merge_data", return_value=MagicMock(compute=lambda: pd.DataFrame()))
@patch("model.lightgbm_202504.infer.load_with_quarter")
@patch("model.lightgbm_202504.infer.find_all_text_files", return_value=["monthly.txt", "origination.txt"])
@patch("model.lightgbm_202504.infer.unzip_file", return_value="mock")
def test_background_infer_logic_model_load_fail(*_):
    with pytest.raises(RuntimeError):
        background_infer_logic("file.zip", 1)
