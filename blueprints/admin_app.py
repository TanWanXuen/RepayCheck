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

from flask import Blueprint, render_template
from flask_login import login_required
from database.db_handler import Admin, Prediction, Retrain, Downloads, ModelVersion  
#, UserInfer
from flask import request, redirect, url_for
from werkzeug.security import generate_password_hash,check_password_hash
from database.db_task import db
from utils.log import configure_logger
from urllib.parse import urlencode


admin_bp = Blueprint('admin_debug', __name__, url_prefix='/admin')
logger = configure_logger("admin_app")

# Admin debug screen
@admin_bp.route('/debug')
@login_required
def debug_dashboard():
    try:
        logger.debug("Fetching admin, prediction, retrain, download, and user_infer records from database.")
        admins = db.session.query(Admin).all()
        predictions = db.session.query(Prediction).all()
        retrains = db.session.query(Retrain).all()
        downloads = db.session.query(Downloads).all()
        #user_infer = db.session.query(UserInfer).all()
        model_version = db.session.query(ModelVersion).all()
        logger.info("Successfully fetched all records for debug dashboard.")
    except Exception as e:
        logger.error(f"Error while fetching data for debug dashboard: {e}", exc_info=True)
        return render_template("debug.html", error_message="Error while fetching data for debug dashboard"), 200

    return render_template("debug.html", admins=admins, predictions=predictions, retrains=retrains,
                           downloads=downloads, model_version=model_version)
# user_infer=user_infer, 

# To add new admin
@admin_bp.route('/add_admin', methods=['POST'])
@login_required
def add_admin():
    message = ""
    try:
        username = request.form['username']
        password = request.form['password']
        logger.debug(f"Received new admin data: username={username}")

        # Check if username already exists
        if Admin.query.filter_by(username=username).first():
            message = f"Admin username '{username}' already exists."
            logger.warning(message)
        else:
            hashed_password = generate_password_hash(password)
            new_admin = Admin(username=username, password_hash=hashed_password)
            db.session.add(new_admin)
            db.session.commit()
            message = f"New admin '{username}' added successfully."
            logger.info(message)

    except Exception as e:
        db.session.rollback()
        message = "Failed to add new admin due to an unexpected error."
        logger.error(f"{message}: {e}", exc_info=True)

    finally:
        db.session.close()
        logger.debug("Database session closed after admin creation.")

    return redirect(url_for('admin_debug.debug_dashboard') + "?" + urlencode({"message": message}))

# To delete admin
@admin_bp.route('/delete_admin', methods=['POST'])
@login_required
def delete_admin():
    message = ""
    try:
        username = request.form['username']
        admin = Admin.query.filter_by(username=username).first()

        if admin:
            logger.info(f"Deleting admin: {admin.username}")
            db.session.delete(admin)
            db.session.commit()
            logger.info(f"Admin '{admin.username}' successfully deleted.")
            message = f"Admin '{username}' successfully deleted."
        else:
            logger.warning(f"No admin found with username: {username}")
            message = f"No admin found with username '{username}'."

    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to delete admin '{username}': {e}", exc_info=True)
        message = f"Error deleting admin '{username}'."

    finally:
        db.session.close()
        logger.debug("Database session closed after admin deletion.")

    # Redirect with query message
    return redirect(url_for('admin_debug.debug_dashboard') + "?" + urlencode({"message": message}))

# To modify admin
@admin_bp.route('/modify_admin', methods=['POST'])
@login_required
def modify_admin():
    message = ""
    try:
        username = request.form['username']
        old_password = request.form['old-password']
        new_password = request.form['new-password']

        admin = Admin.query.filter_by(username=username).first()

        if admin:
            if check_password_hash(admin.password_hash, old_password):
                if old_password == new_password:
                    message = "New password cannot be the same as the old password."
                    logger.warning(f"{message} for admin '{username}'.")
                else:
                    admin.password_hash = generate_password_hash(new_password)
                    db.session.commit()
                    message = "Password updated successfully."
                    logger.info(f"{message} for admin '{username}'.")
            else:
                message = "Incorrect old password."
                logger.warning(f"{message} for admin '{username}'.")
        else:
            message = f"No admin found with username: {username}"
            logger.warning(message)

    except Exception as e:
        db.session.rollback()
        message = "An unexpected error occurred."
        logger.error(f"Failed to modify admin '{username}': {e}", exc_info=True)
    finally:
        db.session.close()
        logger.debug("Database session closed after admin modification.")

    return redirect(url_for('admin_debug.debug_dashboard') + "?" + urlencode({"message": message}))
