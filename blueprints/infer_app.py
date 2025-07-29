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
import threading 
import joblib
import shutil

from model.lightgbm_202504.infer import load_encoders, load_scaling, preprocess
from model.lightgbm_202504.file_handler import unzip_file, find_all_text_files, load_with_quarter, merge_data
from model.lightgbm_202504.constant import (
    PREDICTION_UPLOAD_DIR, RESOURCE_APP_METADATA_DIR, RESOURCE_APP_PREDICTION_DIR, MP_COL, ORI_COL,
    MONTHLY_PERF_KEYWORD, ORIGINATION_KEYWORD, FEATURE_LIST1, ORIGINATION_KEYWORD, MONTHLY_PERF_KEYWORD
)
from utils.log import configure_logger
from database.db_task import add_prediction_to_db
from flask_login import current_user, login_required
from database.db_base import db_queue

prediction_admin_bp = Blueprint('prediction', __name__)
logger = configure_logger("infer_app")

# To display prediction page
@prediction_admin_bp.route('/prediction')
@login_required
def prediction():
    logger.info("Accessed /prediction page.")
    return render_template('prediction.html')

# To upload ZIP file for prediction
@prediction_admin_bp.route("/prediction/result", methods=["POST"])
@login_required
def predict_data():
    logger.info("Prediction upload request received.")
    try:
        if not os.path.exists(PREDICTION_UPLOAD_DIR):
            os.makedirs(PREDICTION_UPLOAD_DIR, exist_ok=True)
            logger.debug(f"Created upload directory: {PREDICTION_UPLOAD_DIR}")

        file = request.files["file"]
        if not file.filename.endswith(".zip"):
            logger.warning("Invalid file type uploaded.")
            raise ValueError("Uploaded file must be a ZIP file.")
        
        if not os.access(PREDICTION_UPLOAD_DIR, os.W_OK):
            raise PermissionError(f"Cannot write to {PREDICTION_UPLOAD_DIR}")

        filename = datetime.now().strftime("%Y%m%d") + "_" + file.filename
        file_path = os.path.join(PREDICTION_UPLOAD_DIR, filename)
        file.save(file_path)
        logger.info(f"Uploaded file saved to {file_path}")
        
        admin_id = current_user.admin_id
        
        # Run background logic in thread
        thread = threading.Thread(target=background_infer_logic, args=(file_path, admin_id,), daemon=True)
        thread.start()
        logger.info("Background inference thread started.")
        
        
        message = "Your dataset has been successfully uploaded. The prediction may take some time. The results will be available on the Download page." 
        flash(message)
        
        return render_template("prediction.html", prediction_message=message)

    except Exception as e:
        logger.error(f"Inference upload failed: {e}", exc_info=True)
        flash(f"Inference failed: {e}")
        return redirect(url_for('error'))

# To conduct inference logic in the background
def background_infer_logic(file_path, admin_id):
    try:
        unzip_lock = threading.Lock()

        with unzip_lock:
            unzip_file(file_path)
            logger.debug(f"Unzipped file at {file_path}")   

        extracted_path = unzip_file(file_path)
        os.remove(file_path)
        logger.debug(f"Removed uploaded zip file: {file_path}")

        # Get all files
        all_files = find_all_text_files(extracted_path)
        logger.debug(f"Total files found after extraction: {len(all_files)}")

        # Separate by type
        mp_files = [f for f in all_files if MONTHLY_PERF_KEYWORD in f.lower()]
        orig_files = [
            f for f in all_files 
            if ORIGINATION_KEYWORD in f.lower() and f not in mp_files
        ]

        if not orig_files or not mp_files:
            logger.error("Missing required origination or monthly performance files.")
            raise ValueError("Must include at least one 'origination' and one 'monthly' file.")

        # Load and tag by quarter
        df_ori = load_with_quarter(orig_files, ORI_COL, extracted_path)
        df_mp = load_with_quarter(mp_files, MP_COL, extracted_path)
        logger.debug("Loaded data.")

        # Final merged DataFrame
        ddf_merged = merge_data(df_mp, df_ori)
        ddf_merged = ddf_merged[FEATURE_LIST1]
        logger.debug("Data merged and filtered for feature list.")
        
        df_init = ddf_merged[['loan_sequence_num','quarter']].compute()
        df=ddf_merged.compute()

        # One hot encoding
        encoder = load_encoders()
        logger.debug("Loaded one-hot encoders.")

        # Scaling
        scaling = load_scaling()
        logger.debug("Loaded feature scalers.")

        # Data preprocessing
        df_test = preprocess(df, FEATURE_LIST1, encoder, scaling, ['loan_sequence_num','quarter'], df_init)
        logger.debug("Data preprocessing completed.")

        X_test = df_test.drop(columns=['loan_sequence_num','quarter'])
        lgb_model = joblib.load(os.path.join(RESOURCE_APP_METADATA_DIR,"model_best.pkl"))
        logger.debug("Model loaded for prediction.")
        
        target_test = lgb_model.predict(X_test)

        df_test_result = pd.DataFrame({
            "loan_sequence_number": df_test["loan_sequence_num"],
            "target": target_test
        })
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_result_{timestamp}.csv"
       
        df_test_result.to_csv(os.path.join(RESOURCE_APP_PREDICTION_DIR,filename), index=False)
        db_queue.put({
            "func": add_prediction_to_db,
            "args": (),
            "kwargs": {
                "admin_id": admin_id,
                "file_name": filename
            }
        })
        shutil.rmtree(extracted_path)
        logger.info(f"Prediction result saved to {filename}")
        return admin_id, filename
    except Exception as e:
        logger.error("Background inference failed.", exc_info=True)
        raise RuntimeError("Inference failed due to invalid dataset structure.") from e