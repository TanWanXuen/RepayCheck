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

from database.db_base import db
from sqlalchemy import Enum as SQLAlchemyEnum
from datetime import datetime
import enum

class FileType(enum.Enum):
    PREDICTION = "prediction"
    RETRAIN = "retrain"

class RetrainStatusEnum(enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"

class ModelStatus(enum.Enum):
    passed = "passed"
    failed = "failed"
    in_progress = "in_progress"

class Admin(db.Model):
    __tablename__ = "admin"
    admin_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    predictions = db.relationship("Prediction", back_populates="admin")
    retrains = db.relationship("Retrain", back_populates="admin")
    downloads = db.relationship("Downloads", back_populates="admin")

class ModelVersion(db.Model):
    __tablename__ = "model_version"
    m_id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(50), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    status = db.Column(SQLAlchemyEnum(ModelStatus), nullable=True)

    retrains = db.relationship("Retrain", back_populates="models")
    retrain_trackings = db.relationship("RetrainTracking", back_populates="models")

class Prediction(db.Model):
    __tablename__ = "prediction"
    p_id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey("admin.admin_id"), nullable=False)
    file_name = db.Column(db.String(50), nullable=False)
    predicted_at = db.Column(db.DateTime, nullable=False, default=datetime.now)

    admin = db.relationship("Admin", back_populates="predictions")

#class UserInfer(db.Model):
 #   __tablename__ = "user_infer"
  #  ui_id = db.Column(db.Integer, primary_key=True)
  #  inferred_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
  #  monthly_income = db.Column(db.Integer, nullable=False)
   # loan_term_years = db.Column(db.Integer, nullable=False)
   # cur_actual_UPB = db.Column(db.Integer, nullable=False)
   # cur_int_rate = db.Column(db.Float, nullable=False)
  #  mi_percentage = db.Column(db.Float, nullable=False)
  #  num_borrowers = db.Column(db.Integer, nullable=False)
  #  loan_start_date = db.Column(db.Date, nullable=False)
  #  property_state = db.Column(db.String(5), nullable=False)
  #  credit_score = db.Column(db.Integer, nullable=False)
  #  loan_purpose = db.Column(db.String(5), nullable=False)
  #  channel = db.Column(db.String(5), nullable=False)
   # first_time_homebuyer = db.Column(db.String(5), nullable=False)
   # property_type = db.Column(db.String(5), nullable=False)
  #  occupancy_status = db.Column(db.String(5), nullable=False)
   # num_of_units = db.Column(db.Integer, nullable=False)
   # target = db.Column(db.Integer, nullable=False)

class Downloads(db.Model):
    __tablename__ = "downloads"
    d_id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey("admin.admin_id"), nullable=False)
    file_type = db.Column(SQLAlchemyEnum(FileType), nullable=False)
    file_name = db.Column(db.String(50), nullable=False)
    downloaded_at = db.Column(db.DateTime, nullable=False, default=datetime.now)

    admin = db.relationship("Admin", back_populates="downloads")

class Retrain(db.Model):
    __tablename__ = "retrain"
    r_id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey("admin.admin_id"), nullable=False)
    model_version_id = db.Column(db.Integer, db.ForeignKey("model_version.m_id"), nullable=False)
    file_name = db.Column(db.String(255), nullable=True) 
    status = db.Column(SQLAlchemyEnum(RetrainStatusEnum), nullable=False, default=RetrainStatusEnum.pending)
    started_at = db.Column(db.DateTime, nullable=False)
    finished_at = db.Column(db.DateTime, nullable=True)

    admin = db.relationship("Admin", back_populates="retrains")
    models = db.relationship("ModelVersion", back_populates="retrains")
    retrain_trackings = db.relationship("RetrainTracking", back_populates="retrains")

class RetrainTracking(db.Model):
    __tablename__ = "retrain_tracking"
    t_id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(255), unique=True, nullable=False)
    model_version_id = db.Column(db.Integer, db.ForeignKey("model_version.m_id"), nullable=False)
    retrain_id = db.Column(db.Integer, db.ForeignKey("retrain.r_id"), nullable=False)

    models = db.relationship("ModelVersion", back_populates="retrain_trackings")
    retrains = db.relationship("Retrain", back_populates="retrain_trackings")

class ContactUs(db.Model):
    __tablename__ = "contact_us"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    enquiry_type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)