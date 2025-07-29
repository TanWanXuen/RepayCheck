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

from flask import Blueprint, request, render_template, redirect, url_for, flash
import os
import pandas as pd
from datetime import datetime

import optuna
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
import threading 
import shutil

from sklearn.model_selection import train_test_split
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

from utils.log import configure_logger
from model.lightgbm_202504.file_handler import unzip_file, find_all_text_files, load_with_quarter, merge_data
from model.lightgbm_202504.constant import (
    RETRAIN_UPLOAD_DIR, RESOURCE_APP_METADATA_DIR, MP_COL, ORI_COL,
    MONTHLY_PERF_KEYWORD, ORIGINATION_KEYWORD, FEATURE_LIST2, TARGET, PROTECTED_ATTRIBUTES, ORIGINATION_KEYWORD, MONTHLY_PERF_KEYWORD, RANDOM_STATE
)
from model.lightgbm_202504.train import preprocess_data, objective_lgb, evaluate_and_save
from database.db_task import finalize_retrain_and_model_version, add_retrain_and_model_version_to_db
from flask_login import login_required, current_user
from database.db_base import db_queue
from database.db_handler import  ModelStatus, RetrainStatusEnum

retrain_bp = Blueprint('retrain', __name__)
logger = configure_logger("retrain_app")

# To display retrain page
@retrain_bp.route('/retrain')
@login_required
def retrain():
    logger.info("Rendering retrain.html page.")
    return render_template('retrain.html')

# To upload ZIP file for retraining
@retrain_bp.route("/retrain/result", methods=["POST"])
@login_required
def retrain_data():
    try:
        if not os.path.exists(RETRAIN_UPLOAD_DIR):
            os.makedirs(RETRAIN_UPLOAD_DIR, exist_ok=True)
            logger.debug(f"Created upload directory at: {RETRAIN_UPLOAD_DIR}")

        file = request.files["file"]
        if not file.filename.endswith(".zip"):
            logger.warning("Uploaded file is not a zip file.")
            raise ValueError("Uploaded file must be a ZIP file.")
        
        if not os.access(RETRAIN_UPLOAD_DIR, os.W_OK):
            logger.error(f"No write permission for directory: {RETRAIN_UPLOAD_DIR}")
            raise PermissionError(f"Cannot write to {RETRAIN_UPLOAD_DIR}")

        filename = datetime.now().strftime("%Y%m%d") + "_" + file.filename
        file_path = os.path.join(RETRAIN_UPLOAD_DIR, filename)
        file.save(file_path)
        logger.info(f"File uploaded and saved at: {file_path}")

        admin_id = current_user.admin_id

        # Run background logic in thread
        thread = threading.Thread(target=background_retrain_logic, args=(file_path, admin_id), daemon=True)
        thread.start()
        logger.info("Started background retraining thread.")

        message = "Your dataset has been successfully uploaded. The training process may take some time. The results will be available on the Download page." 
        flash(message)
        return render_template("retrain.html", retrain_message=message)

    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        flash(f"Inference failed: {e}")
        return redirect(url_for('error'))

# To conduct retrain logic in the background
def background_retrain_logic(file_path, admin_id):
    try:
        logger.info("Starting background retraining logic.")
        unzip_lock = threading.Lock()

        with unzip_lock:
            extracted_path = unzip_file(file_path) 
            logger.debug("Zip file extracted successfully.")

        os.remove(file_path)
        logger.debug("Zip file removed after extraction.")

        started_at = datetime.now()
        all_files = find_all_text_files(extracted_path)

        mp_files = [f for f in all_files if MONTHLY_PERF_KEYWORD in f.lower()]
        orig_files = [
            f for f in all_files 
            if ORIGINATION_KEYWORD in f.lower() and f not in mp_files
        ]

        if not orig_files or not mp_files:
            logger.error("Missing required origination or monthly files.")
            raise ValueError("Must include at least one 'origination' and one 'monthly' file.")

        df_ori = load_with_quarter(orig_files, ORI_COL, extracted_path)
        df_mp = load_with_quarter(mp_files, MP_COL, extracted_path)

        ddf_merged = merge_data(df_mp, df_ori)

        ddf_merged = ddf_merged[FEATURE_LIST2]

        df = ddf_merged.compute()
        logger.debug(f"Final merged dataframe shape: {df.shape}")

        df, encoders,scaling = preprocess_data(df)
        logger.debug("Preprocessing completed.")

        df_train: pd.DataFrame
        df_test: pd.DataFrame

        # 60% Train, 40% Test+Validation
        df_train, df_temp = train_test_split(df, test_size=0.40, random_state=RANDOM_STATE, shuffle=True) #, stratify=df["target"]

        # Split df_temp into 20% Test, 20% Validation
        df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=RANDOM_STATE, shuffle=True) #, stratify=df_temp["target"]

        X_train = df_train.drop(TARGET, axis=1)
        y_train = df_train[TARGET]

        X_test = df_test.drop(TARGET, axis=1)
        y_test = df_test[TARGET]

        X_val = df_val.drop(TARGET, axis=1)
        y_val = df_val[TARGET]
        
        logger.debug(f"Training samples: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        
        random = RandomUnderSampler(sampling_strategy="auto", random_state=RANDOM_STATE)
        X_resampled, y_resampled = random.fit_resample(X_train, y_train)
        logger.info("Performed random undersampling.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"LightGBM_{timestamp}"
        
        db_path = os.path.join(RESOURCE_APP_METADATA_DIR, "retrain.sqlite3")
        storage_url = f"sqlite:///{db_path.replace(os.sep, '/')}"
        study_lgb = optuna.create_study(storage=storage_url, study_name=study_name, direction="maximize")
        study_lgb.optimize(lambda trial: objective_lgb(trial, X_resampled, y_resampled, X_val, y_val, X_test, y_test), n_trials=20, show_progress_bar=True)

        best_params_lgb = study_lgb.best_trial.params

        logger.info(f"Best parameters found: {best_params_lgb}")
        logger.info(f"Best accuracy score: {study_lgb.best_value}")
        
        protected_attr = X_resampled[PROTECTED_ATTRIBUTES]
        protected_attr_val = X_val[PROTECTED_ATTRIBUTES]
        protected_attr_test = X_test[PROTECTED_ATTRIBUTES]
       
        db_queue.put({
            "func": add_retrain_and_model_version_to_db,
            "args": (),
            "kwargs": {
                "admin_id": admin_id,
                "version": study_name,
                "started_at": started_at,
            }
        })

        lgb_model_best = lgb.LGBMClassifier(**best_params_lgb)
        lgb_model_best.fit(X_resampled, y_resampled)
        fair_lgb = ExponentiatedGradient(
            lgb_model_best,
            constraints=EqualizedOdds()
        )

        fair_lgb.fit(X_resampled, y_resampled, sensitive_features=protected_attr)
        logger.info("Fairness-constrained training completed.")

        output_filename, status = evaluate_and_save(fair_lgb,"LightGBM", X_resampled, X_val, X_test, y_resampled, y_val, y_test, protected_attr_test, protected_attr_val, encoders, scaling)
        finished_at = datetime.now()
       
        if status is None:
            retrain_status = RetrainStatusEnum.failed

        if status == True:
            model_status = ModelStatus.passed
            retrain_status = RetrainStatusEnum.completed
        else:
            model_status =  ModelStatus.failed
            retrain_status = RetrainStatusEnum.completed
        
        db_queue.put({
            "func": finalize_retrain_and_model_version,
            "args": (),
            "kwargs": {
                "admin_id": admin_id,
                "version": study_name,
                "model_status": model_status,
                "retrain_status": retrain_status,
                "output_filename": output_filename,
                "finished_at": finished_at
            }
        })
        shutil.rmtree(extracted_path)
        logger.info("Evaluation and model saving completed.")
    except Exception as e:
        logger.error(f"Background retrain error: {e}")
