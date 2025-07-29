# === package imports boilerplate === #
import sys
from pathlib import Path # if you haven't already done so
current_file = Path(__file__).resolve()
current_dir, target_root = current_file.parent, current_file.parents[1]

# append grandparent directory if not yet added into sys.path
if str(target_root) not in sys.path:
    sys.path.append(str(target_root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(current_dir))
except ValueError: # Already removed
    pass

from utils.import_path_setup import setup_paths
setup_paths("app.py") 
# =============================== #

from database.db_task import db
from database.db_handler import Admin 

# To obtain the current user
def get_admin_user(username):
    try:
        admin = db.session.query(Admin).filter_by(username=username).first()
        return admin
    except Exception as e:
        print(f"Error fetching admin user: {e}")  # Replace with proper logging if needed
        return None
    finally:
        db.session.remove()

