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
from flask import Blueprint, render_template, send_from_directory
from flask_login import login_required
from utils.log import configure_logger
from database.db_task import add_download_to_db 
from database.db_handler import FileType
import model.lightgbm_202504.constant as constant

download_bp = Blueprint('download', __name__)
logger = configure_logger("download_app")

# To display download page
@download_bp.route('/downloads')
@login_required
def downloads():
    logger.info("Accessing /downloads route.")
    try:
        excel_files = []
        json_files = []

        if os.path.exists(constant.RESOURCE_APP_PREDICTION_DIR):
            excel_files = [f for f in os.listdir(constant.RESOURCE_APP_PREDICTION_DIR) if f.endswith('.csv')]
            logger.debug(f"Found {len(excel_files)} CSV files in {constant.RESOURCE_APP_PREDICTION_DIR}.")
        else:
            logger.warning(f"Prediction directory does not exist: {constant.RESOURCE_APP_PREDICTION_DIR}")

        if os.path.exists(constant.RESOURCE_APP_RETRAIN_DIR):
            json_files = [f for f in os.listdir(constant.RESOURCE_APP_RETRAIN_DIR) if f.endswith('.json')]
            logger.debug(f"Found {len(json_files)} JSON files in {constant.RESOURCE_APP_RETRAIN_DIR}.")
        else:
            logger.warning(f"Retrain directory does not exist: {constant.RESOURCE_APP_RETRAIN_DIR}")

        logger.info("Rendering download.html with available files.")
        return render_template('download.html', excel_files=excel_files, json_files=json_files)

    except Exception as e:
        logger.error(f"Error loading downloads: {e}", exc_info=True)
        return "Error loading downloads"

# To fetch downloadable prediction CSV file
@download_bp.route('/download/excel/<filename>')
@login_required
def download_excel(filename):
    logger.info(f"User requested download of Excel file: {filename}")
    try:
        add_download_to_db(filename, FileType.PREDICTION)
    except Exception as e:
        logger.error(f"Failed to log CSV download: {e}", exc_info=True)
    return send_from_directory(constant.RESOURCE_APP_PREDICTION_DIR, filename, as_attachment=True)

# To fetch downloadable retrain JSON file
@download_bp.route('/download/json/<filename>')
@login_required
def download_json(filename):
    logger.info(f"User requested download of JSON file: {filename}")
    try:
        add_download_to_db(filename, FileType.RETRAIN)
    except Exception as e:
        logger.error(f"Failed to log JSON download: {e}", exc_info=True)
    return send_from_directory(constant.RESOURCE_APP_RETRAIN_DIR, filename, as_attachment=True)