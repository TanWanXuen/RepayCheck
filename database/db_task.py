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

'''
This module handles data insertion into the database from
various pages of the application, including those related 
to uploads, predictions, retraining, and user interactions (contact us).
'''

from database.db_base import db
from database.db_handler import ContactUs, RetrainTracking, RetrainStatusEnum, Prediction, Downloads, Retrain, ModelVersion, ModelStatus
#UserInfer,
from datetime import datetime
from flask_login import current_user
from utils.log import configure_logger
logger = configure_logger("db_task")

def add_download_to_db(filename, file_type):
    logger.info(f"Logging download: {filename} as type {file_type}")
    new_download = Downloads(
        admin_id=current_user.admin_id,
        file_type=file_type,
        file_name=filename,
        downloaded_at=datetime.now()
    )
    db.session.add(new_download)
    db.session.commit()
    logger.info("Download record committed to DB")

def add_prediction_to_db(admin_id, file_name):
    logger.info(f"Logging prediction: {file_name} by admin_id={admin_id}")
    new_prediction = Prediction(
        admin_id=admin_id,
        file_name=file_name,
        predicted_at=datetime.now()
    )
    db.session.add(new_prediction)
    db.session.commit()
    logger.info("Prediction record committed to DB")

def add_retrain_and_model_version_to_db(admin_id, version, started_at):
    logger.info(f"Creating new ModelVersion and Retrain entry for version: {version}")
    
    # Insert model version with minimal info
    model_version = ModelVersion(
        version=version,
        created_at=datetime.now(),
        status=ModelStatus.in_progress  
    )
    db.session.add(model_version)
    db.session.flush() 

    # Insert retrain record with partial info
    retrain_record = Retrain(
        admin_id=admin_id,
        model_version_id=model_version.m_id,
        status=RetrainStatusEnum.in_progress,
        started_at=started_at
    )
    db.session.add(retrain_record)
    db.session.flush()

    tracking = RetrainTracking(
        version=version,
        model_version_id=model_version.m_id,
        retrain_id=retrain_record.r_id
    )
    db.session.add(tracking)
    db.session.commit()

    logger.info(f"ModelVersion ID: {model_version.m_id}, Retrain ID: {retrain_record.r_id} stored with tracking.")


def finalize_retrain_and_model_version(admin_id, 
    version,
    model_status,
    retrain_status,
    output_filename,
    finished_at
):
    logger.info(f"Finalizing retrain results for version: {version}")
    tracking = db.session.query(RetrainTracking).filter_by(version=version).first()

    if not tracking:
        logger.error(f"No tracking info found for version {version}")
        return

    model_version = db.session.get(ModelVersion, tracking.model_version_id)
    retrain_record = db.session.get(Retrain, tracking.retrain_id)

    if model_version:
        model_version.status = model_status
        logger.info(f"Updated ModelVersion {model_version.m_id} with status {model_status}")

    if retrain_record:
        retrain_record.admin_id = admin_id
        retrain_record.status = retrain_status
        retrain_record.file_name = output_filename
        retrain_record.finished_at = finished_at
        logger.info(f"Updated Retrain {retrain_record.r_id} with status {retrain_status} and file {output_filename}")

    db.session.commit()
    logger.info("Finalized retrain and model version updates.")


#def add_user_infer_to_db(**kwargs):
 #   kwargs["inferred_at"] = datetime.now()
  #  logger.info(f"Adding UserInfer with data: {kwargs}")
  #  new_infer = UserInfer(**kwargs)
   # db.session.add(new_infer)
   # db.session.commit()
   # logger.info("User inference committed to DB")

def add_contact_us_to_db(name, email, enquiry_type, message):
    try:
        new_entry = ContactUs(
            name=name,
            email=email,
            enquiry_type=enquiry_type,
            message=message
        )
        db.session.add(new_entry)
        db.session.commit()
        return True, "Thank you for your response. We will get back to you as soon as possible."
    except Exception as e:
        db.session.rollback()
        return False, f"Database error: {e}"
