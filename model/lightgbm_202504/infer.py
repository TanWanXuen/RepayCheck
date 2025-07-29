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

from model_interface import ModelInterface, DatasetStage
from preprocessing_tools import get_region, impute_missing_values
import os 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle
import lightgbm as lgb
from dataclasses import dataclass
import joblib
from constant import RESOURCE_APP_METADATA_DIR, FEATURE_LIST1

# To load encoder metadata
def load_encoders():
    lgbencoder: OneHotEncoder
    with open(os.path.join(RESOURCE_APP_METADATA_DIR, "one_hot_encoder.pkl"), "rb") as f:
        lgbencoder = pickle.load(f)
    return lgbencoder

# To load scaling metadata
def load_scaling():
    scaling: DataScalar
    with open(os.path.join(RESOURCE_APP_METADATA_DIR, "scaling_metadata.pkl"),"rb") as f:
        scaling = pickle.load(f)
    return scaling

@dataclass
class DataScalar:
    min: float
    max: float 

class DatasetBundle:
    def __init__(
        self,
        origination: pd.DataFrame,
        monthly_performance: pd.DataFrame,
    ):
        self.origination = origination
        self.monthly_performance = monthly_performance

# To preprocess the data
def preprocess(df_merged: pd.DataFrame, columns: list[str], encoder: OneHotEncoder, scaling: dict[str, DataScalar], other_cols: list[str]|None=None, df_target: pd.DataFrame|None=None, process_from: DatasetStage=DatasetStage.RAW_TEXT) -> pd.DataFrame:
    df_merged: pd.DataFrame
    if process_from is DatasetStage.RAW_TEXT:
        assert all(col in df_merged.columns for col in columns), "some required columns are not found in the input dataframe"
        df_merged_processed = df_merged.drop_duplicates(subset=other_cols)
        
        if df_target is not None: 
            df_target = df_target.drop_duplicates()
            
        df_is_empty = df_merged_processed.shape[0] == 0
        if df_is_empty:
            return pd.DataFrame(columns=df_merged_processed.columns)

        # Feature engineering
        df_merged_processed['region'] = df_merged_processed['property_state'].apply(get_region)
        df_merged_processed = df_merged_processed.drop(columns='property_state')

        # Handling missing values
        df_merged_processed = df_merged_processed[df_merged_processed['credit_score']!=9999.0]
        df_merged_processed = df_merged_processed[df_merged_processed['ori_DTI']!=999.0]
        df_merged_processed = df_merged_processed[df_merged_processed['property_valuation_method']!=9]
        df_merged_processed = df_merged_processed[df_merged_processed['channel']!=9]
        df_merged_processed['ELTV'] = impute_missing_values(df_merged_processed, 'ELTV', 999.0)

        # Handling outliers
        df_merged_processed = df_merged_processed[(df_merged_processed['ori_DTI'] >= 0) & (df_merged_processed['ori_DTI'] <= 65)]
        df_merged_processed = df_merged_processed[(df_merged_processed['credit_score'] >= 300) & (df_merged_processed['credit_score'] <= 850)]
        df_merged_processed = df_merged_processed[(df_merged_processed['MI(%)'] >= 0) & (df_merged_processed['MI(%)'] <= 55)]
        df_merged_processed = df_merged_processed[(df_merged_processed['ELTV'] >= 1) & (df_merged_processed['ELTV'] <= 998)]

    #if df_target is not None:
    #    df_merged_processed = df_merged_processed.loc[lambda x: x["loan_sequence_num"].isin(df_target['loan_sequence_num'])]
    
    # One Hot encoding
    df_merged_processed['property_valuation_method']= df_merged_processed['property_valuation_method'].astype('object')
    transformed_dfs = []

    df_merged_processed = df_merged_processed.astype({col: 'object' for col in df_merged_processed.select_dtypes(include=['object', 'string']).columns})
    categorical_columns = df_merged_processed.select_dtypes(include=['object', 'string']).columns.tolist()
    df_merged_processed[categorical_columns] = df_merged_processed[categorical_columns].astype(object)

    if other_cols is not None:    
        df_merged_processed['loan_sequence_num'] = df_merged_processed['loan_sequence_num'].astype(str)
        df_target['loan_sequence_num'] = df_target['loan_sequence_num'].astype(str)
    
    CAT_COLUMNS = [
        col for col in df_merged_processed.select_dtypes(include=['object']).columns
        if col != 'loan_sequence_num'
    ]
    print(f"Categorical columns: {CAT_COLUMNS}")
    print(df_merged_processed.info())

    # Transform all categorical columns at once
    transformed = encoder.transform(df_merged_processed[CAT_COLUMNS])
    feature_names = encoder.get_feature_names_out(CAT_COLUMNS)

    # Create DataFrame with transformed data
    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=df_merged_processed.index
    )
    transformed_dfs.append(transformed_df)

    # Combine all transformed columns with the rest of the data
    df_encoded = pd.concat(transformed_dfs, axis=1)
    df_merged_processed = pd.concat([df_merged_processed.drop(columns=CAT_COLUMNS), df_encoded], axis=1)

    # Sacling
    columns_to_normalize = ['cur_actual_UPB', 'cur_int_rate', 'cur_deferred_UPB', 'ELTV', 'credit_score', 'MI(%)', 'num_of_units', 'ori_DTI','num_borrowers', 'loan_age']
    for column in columns_to_normalize:
        min = scaling[column]['min']
        max = scaling[column]['max']
        df_merged_processed[column] = (df_merged_processed[column] - min) / (max - min)

    if df_target is not None:
        df_merged_processed = df_merged_processed.merge(df_target, how="left", on="loan_sequence_num")

    df_merged_processed = df_merged_processed.reset_index(drop=True)
    return df_merged_processed

# Model interface for LightGBM
class LightGBMModel(ModelInterface):
    def __init__(self):
        self._ckpt_dir = os.path.join(os.path.dirname(__file__), "./resources")

        self._model = lgb.LGBMClassifier()
        self._model = joblib.load(os.path.join(self._ckpt_dir, "model_best.pkl"))
        
        with open(os.path.join(self._ckpt_dir, "one_hot_encoder.pkl"), "rb") as f:
            self._lgbencoder: OneHotEncoder = pickle.load(f)

        with open(os.path.join(self._ckpt_dir, "scaling_metadata.pkl"), "rb") as n:
            self._normalization: dict[str, DataScalar] = pickle.load(n)
        
        self._input_features = FEATURE_LIST1
        
        super().__init__()

    @property
    def identifier_raw(self):
        return "loan_sequence_number"

    @property
    def input_features(self):
        return self._input_features

    @property
    def model(self):
        return self._model

    @property
    def categorical_encoder(self):
        return self._encoder

    def inference_func(self, datasets: DatasetBundle, process_from: DatasetStage):
        df = self.preprocess_data(datasets, process_from)
        probs = self._model.predict_proba(df[self._input_features])[:, 1]
        return pd.DataFrame({
            "loan_sequence_number": df["loan_sequence_number"],
            "target": probs
        })
