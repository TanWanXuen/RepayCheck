# === Configure module import paths for testing or modular structure === #
import sys
from pathlib import Path  

# Resolve the current file location
current_file = Path(__file__).resolve()
current_dir, target_root = current_file.parent, current_file.parents[1]

# Ensure the project root is in sys.path for module imports
if str(target_root) not in sys.path:
    sys.path.append(str(target_root))

# Optionally remove current directory from sys.path to avoid conflicts
try:
    sys.path.remove(str(current_dir))
except ValueError:
    pass  

# Setup additional import paths 
from utils.import_path_setup import setup_paths
setup_paths("app.py")
# ===================================================================== #

import os
import zipfile
import dask.dataframe as dd
import re

from constant import DTYPE_OVERRIDES

QUARTER_COL = "quarter"

# To unzip the ZIP file
def unzip_file(zip_path, extract_to=None):
    if extract_to is None:
        extract_to = zip_path.replace(".zip", "")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            os.makedirs(extract_to, exist_ok=True)
            zip_ref.extractall(extract_to)
        return extract_to

    except FileNotFoundError:
        raise RuntimeError("ZIP file not found.")
    except PermissionError as e:
        raise RuntimeError(f"Permission error: {e}")
    except zipfile.BadZipFile:
        raise RuntimeError("Invalid ZIP file.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while unzipping: {e}")
    
# To find all .txt files recursively
def find_all_text_files(base_dir, valid_exts=".txt"):
    if base_dir is None:
        raise ValueError("No file found in the directory.")
    all_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                full_path = os.path.join(root, f)
                all_files.append(os.path.relpath(full_path, base_dir))
    return all_files

# To extract quarter 
def extract_quarter(filename):
    match = re.search(r'q([1-4])', filename.lower())
    return match.group(1) if match else None

# To load and assign quarter
def load_with_quarter(files, col_names, base_dir):
    dfs = []
    for file in files:
        quarter = extract_quarter(file)
        if not quarter:
            raise ValueError(f"Filename '{file}' must contain 'Q1', 'Q2', 'Q3', or 'Q4'.")
        
        file_path = os.path.join(base_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist!")

        df = dd.read_csv(
            file_path,
            delimiter="|",
            names=col_names,
            skipinitialspace=True,
            assume_missing=True,
            dtype=DTYPE_OVERRIDES
        ).assign(quarter=quarter)
        #df[QUARTER_COL] = quarter
        dfs.append(df)
    return dd.concat(dfs, axis=0)


# Merge datasets
def merge_data(df_mp, df_ori):
    return dd.merge(df_mp, df_ori, on=["loan_sequence_num","quarter"], how="left")
