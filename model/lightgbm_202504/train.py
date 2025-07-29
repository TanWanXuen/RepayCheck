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
import pandas as pd
import numpy as np

from datetime import datetime
import json
import socket

from imblearn.metrics import geometric_mean_score
import lightgbm as lgb
import pickle
import joblib

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

from preprocessing_tools import get_region, impute_missing_values
from constant import RESOURCE_APP_METADATA_DIR, RESOURCE_APP_RETRAIN_DIR, THRESHOLDS

"""
1. Read dataset from the upload directory
2. Merge monthly performance and origination datasets
3. Obtain desired dataset
4. Feature Selection
5. Get target variable
6. Check and remove duplicates
7. Handle missing values
8. Handle outliers
9. One-hot encoding
10. Standardisation
11. Data splitting
12. Model training
13. Model evaluation 
14. Save the model and metadata if and only if the the latest model meet the requirements. 
"""

# To get target variables
def get_target(col: str):
    if col == '0' or col == '1':
        return True
    else:
        return False

# To preprocess raw data 
def preprocess_data(df: pd.DataFrame) ->  tuple[pd.DataFrame, OneHotEncoder, list]:
    df["cur_LDS"] = df["cur_LDS"].astype(str).str.replace(" ", "", regex=True)
    df['target'] = df['cur_LDS'].apply(get_target)

    df = df.drop_duplicates()

    df['region'] = df['property_state'].apply(get_region)
    df = df.drop(columns=["cur_LDS", "property_state"])

    df = df[df['credit_score']!=9999.0]
    df = df[df['ori_DTI']!=999.0]
    df = df[df['property_valuation_method']!=9]
    df = df[df['channel']!=9]
    df['ELTV'] = impute_missing_values(df,'ELTV', 999.0)

    df = df[(df['ori_DTI'] >= 0) & (df['ori_DTI'] <= 65)]
    df = df[(df['credit_score'] >= 300) & (df['credit_score'] <= 850)]
    df = df[(df['MI(%)'] >= 0) & (df['MI(%)'] <= 55)]
    df = df[(df['ELTV'] >= 1) & (df['ELTV'] <= 998)]

    df['property_valuation_method']= df['property_valuation_method'].astype('object')
    CAT_COLUMNS = df.select_dtypes(include=['object', 'string']).columns.tolist()
    df[CAT_COLUMNS] = df[CAT_COLUMNS].astype(object)

    encoder = OneHotEncoder(drop='first', sparse_output=False)  
    encoded_array = encoder.fit_transform(df[CAT_COLUMNS])

    feature_names = encoder.get_feature_names_out(CAT_COLUMNS)
    df_encoded = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)

    df = df.drop(columns=CAT_COLUMNS)
    df = pd.concat([df, df_encoded], axis=1)
    
    scaling_metadata = {}
    cols_to_save = ['cur_actual_UPB', 'cur_int_rate', 'cur_deferred_UPB', 'ELTV', 'credit_score', 'MI(%)', 'num_of_units', 'ori_DTI','num_borrowers', 'loan_age']

    for column in cols_to_save:
        scaling_metadata[column]={
            'min': df[column].min(),
            'max': df[column].max()
        }

    scaler = MinMaxScaler()
    df[['cur_actual_UPB', 'cur_int_rate', 'cur_deferred_UPB', 'ELTV','credit_score', 'MI(%)', 'num_of_units', 'ori_DTI','num_borrowers', 'loan_age']]=scaler.fit_transform(df[['cur_actual_UPB', 'cur_int_rate', 'cur_deferred_UPB', 'ELTV', 'credit_score', 'MI(%)', 'num_of_units', 'ori_DTI','num_borrowers', 'loan_age']])

    return df, encoder, scaling_metadata

# Hyperparameter tuning
def objective_lgb(trial, X_resampled, y_resampled, X_val, y_val, X_test, y_test):
    params = {
        "objective": "binary",
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "rf", "dart"]),
        "metric": "auc",
        "verbose": -1,
        "random_seed": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),  # Ensuring valid bagging frequency
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),  # Ensuring valid fraction
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_resampled, y_resampled, 
        eval_set=[(X_val, y_val)], 
        eval_metric="auc"
    )

    y_pred_proba = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_proba)

    return accuracy

# To obtain the best predictor
def get_best_predictor(model, X_val_sample, y_val_sample, y_pred_sample):
    # Get all predictors
    predictors = list(model.predictors_)
    
    # Get accuracy scores for each predictor
    scores = []
    for predictor in predictors:
        y_pred_sample = predictor.predict(X_val_sample)
        score = accuracy_score(y_val_sample, y_pred_sample)
        scores.append(score)
    
    # Find index of best predictor
    best_idx = np.argmax(scores)
    return predictors[best_idx]

# To get model result
def result(model, X_sample, X_sample_val, X_sample_test, y_val_sample, y_test_sample, y_resampled_sample):
    assert len(X_sample_val) == len(y_val_sample), f"X_val/y_val length mismatch: {len(X_sample_val)} vs {len(y_val_sample)}"
    assert len(X_sample_test) == len(y_test_sample), f"X_test/y_test length mismatch: {len(X_sample_test)} vs {len(y_test_sample)}"
    train_pred = model.predict(X_sample)
    val_pred = model.predict(X_sample_val)
    y_pred = model.predict(X_sample_test)

    # Confusion Matrix
    conf_matrix_val = confusion_matrix(y_val_sample, val_pred)
    conf_matrix_test = confusion_matrix(y_test_sample, y_pred)
    conf_matrix_train = confusion_matrix(y_resampled_sample, train_pred)
    TN_train, FP_train= conf_matrix_train[0][0], conf_matrix_train[0][1]
    TN_val, FP_val= conf_matrix_val[0][0], conf_matrix_val[0][1]
    TN_test, FP_test= conf_matrix_test[0][0], conf_matrix_test[0][1]

    final_model = get_best_predictor(model, X_sample_val, y_val_sample, y_pred) 

    # Predictions
    train_pred_proba = final_model.predict_proba(X_sample)[:, 1]
    val_pred_proba = final_model.predict_proba(X_sample_val)[:, 1]
    test_pred_proba = final_model.predict_proba(X_sample_test)[:, 1]

    metrics = {
        "accuracy": {
            "train": accuracy_score(y_resampled_sample, train_pred),
            "val": accuracy_score(y_val_sample, val_pred),
            "test": accuracy_score(y_test_sample, y_pred),
        },
        "precision": {
            "train": precision_score(y_resampled_sample, train_pred),
            "val": precision_score(y_val_sample, val_pred),
            "test": precision_score(y_test_sample, y_pred),
        },
        "recall": {
            "train": recall_score(y_resampled_sample, train_pred),
            "val": recall_score(y_val_sample, val_pred),
            "test": recall_score(y_test_sample, y_pred),
        },
        "f1_score": {
            "train": f1_score(y_resampled_sample, train_pred),
            "val": f1_score(y_val_sample, val_pred),
            "test": f1_score(y_test_sample, y_pred),
        },
        "auc": {
            "train": roc_auc_score(y_resampled_sample, train_pred_proba),
            "val": roc_auc_score(y_val_sample, val_pred_proba),
            "test": roc_auc_score(y_test_sample, test_pred_proba),
        },
        "g_mean": {
            "train": geometric_mean_score(y_resampled_sample, train_pred),
            "val": geometric_mean_score(y_val_sample, val_pred),
            "test": geometric_mean_score(y_test_sample, y_pred),
        },
        "specificity": {
            "train": TN_train / (TN_train + FP_train),
            "val": TN_val / (TN_val + FP_val),
            "test": TN_test / (TN_test + FP_test),
        },
        "confusion_matrix": {
            "train": confusion_matrix(y_resampled_sample, train_pred).tolist(),
            "val": confusion_matrix(y_val_sample, val_pred).tolist(),
            "test": confusion_matrix(y_test_sample, y_pred).tolist(),
        },
        "classification_report": {
            "train": classification_report(y_resampled_sample, train_pred, output_dict=True),
            "val": classification_report(y_val_sample, val_pred, output_dict=True),
            "test": classification_report(y_test_sample, y_pred, output_dict=True),
        }
    }

    return metrics

# To compute Disparate Impact 
def disparate_impact(y_pred, sensitive_feature):

    sensitive_feature = np.array(sensitive_feature).flatten()
    
    privileged_group = sensitive_feature == 1  
    unprivileged_group = sensitive_feature == 0  

    rate_privileged = np.mean(y_pred[privileged_group])
    rate_unprivileged = np.mean(y_pred[unprivileged_group])

    return rate_unprivileged / rate_privileged if rate_privileged > 0 else np.nan

# To calculate fairness metrics
def fairness_metrics(y_true, y_pred, attr):
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=attr)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=attr)
    
    disparate_impact_scores = []
    protected_attr_array = np.array(attr)
    num_attributes = protected_attr_array.shape[1]
    for i in range(num_attributes):
        di = disparate_impact(y_pred, protected_attr_array[:, i])
        disparate_impact_scores.append(di)
    overall_disparate_impact = np.nanmean(disparate_impact_scores)

    return {
        "demographic_parity_difference": dpd,
        "equalized_odds_difference": eod,
        "disparate_impact": overall_disparate_impact
    }


'''
Only models that achieve at least 80% accuracy and maintain less than 20% for each fairness metric 
are eligible to be saved and marked as passed.
'''
# To evaluate all metrics and save them into a JSON file
def evaluate_and_save(model, model_name, X_sample, X_sample_val, X_sample_test, 
                      y_resampled, y_val, y_test, protected_attributes_test, protected_attributes_val,
                      encoder, scaling_metadata, threshold_acc=THRESHOLDS["acc"], threshold_auc=THRESHOLDS["auc"], threshold_f1=THRESHOLDS["f1"], threshold_fairness=THRESHOLDS['fairness']):
    performance = result(model, X_sample, X_sample_val, X_sample_test, y_val, y_test, y_resampled)
    fairness_test = fairness_metrics(y_test, model.predict(X_sample_test), protected_attributes_test)
    fairness_val = fairness_metrics(y_val, model.predict(X_sample_val), protected_attributes_val)
    
    val_acc = performance["accuracy"]["val"]
    val_auc = performance["auc"]["val"]
    val_f1 = performance["f1_score"]["val"]

    test_acc = performance["accuracy"]["test"]
    test_auc = performance["auc"]["test"]
    test_f1 = performance["f1_score"]["test"]

    val_metrics = (val_acc >= threshold_acc) and (val_auc >= threshold_auc) and (val_f1 >= threshold_f1)
    test_metrics = (test_acc >= threshold_acc) and (test_auc >= threshold_auc) and (test_f1 >= threshold_f1)

    passed_metrics = val_metrics and test_metrics

    fair_val_dp = abs(fairness_val.get("demographic_parity_difference", 0)) <= threshold_fairness
    fair_val_eod = abs(fairness_val.get("equalized_odds_difference", 0)) <= threshold_fairness
    
    fair_test_dp = abs(fairness_test.get("demographic_parity_difference", 0)) <= threshold_fairness
    fair_test_eod = abs(fairness_test.get("equalized_odds_difference", 0)) <= threshold_fairness

    passed_fairness = fair_val_dp and fair_val_eod and fair_test_dp and fair_test_eod
    passed = passed_metrics and passed_fairness

    if passed:
        save_encoder(encoder)
        save_scaler(scaling_metadata)
        save_model(model)

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename_str = datetime.now().strftime("%Y%m%d")
    metadata = {
        "model_name": model_name,
        "host": socket.gethostname(),
        "evaluation_timestamp": date_str,
        "train_samples": len(y_resampled),
        "validation_samples": len(y_val),
        "test_samples": len(y_test),
        "validation_auc": round(val_auc, 4),
        "validation_f1": round(val_f1, 4),
        "status": "PASS" if passed else "FAIL",
        "should_update_model": passed
    }

    full_results = {
        "metadata": metadata,
        "performance_metrics": performance,
        "fairness_metrics_test": fairness_test,
        "fairness_metrics_val": fairness_val
    }
    
    full_results = convert_np_types(full_results)

    output_filename = f"evaluation-{filename_str}.json"
    with open(os.path.join(RESOURCE_APP_RETRAIN_DIR, output_filename), "w") as f:
        json.dump(full_results, f, indent=4)
    
    print(f"Saved evaluation results to {output_filename}")
    return output_filename, passed

# To convert the data types 
def convert_np_types(obj):
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

# To save the encoder
def save_encoder(encoder):
    with open(os.path.join(RESOURCE_APP_METADATA_DIR,'one_hot_encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)

# To save the normalisation metadata
def save_scaler(scaling_metadata):
    with open(os.path.join(RESOURCE_APP_METADATA_DIR,'scaling_metadata.pkl'), 'wb') as f:
        pickle.dump(scaling_metadata, f)

# To save the model
def save_model(model):
    joblib.dump(model, os.path.join(RESOURCE_APP_METADATA_DIR, "model_best.pkl"))
