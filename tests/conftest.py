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

import pytest
from app import create_app
from database.db_base import db as _db
from database.db_handler import Admin
from werkzeug.security import generate_password_hash
from model.lightgbm_202504 import constant
import io
import zipfile


# Setup application fixture
@pytest.fixture(scope="session")
def app():
    app = create_app(testing=True)
    app.config.update({
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
        'LOGIN_DISABLED': False,
    })
    with app.app_context():
        _db.create_all()
        yield app
        # _db.drop_all()  # Optional cleanup

# Fixture to return a client
@pytest.fixture()
def client(app):
    return app.test_client()

# Fixture to return a CLI runner
@pytest.fixture()
def runner(app):
    return app.test_cli_runner()

# Clear Admin table before each test
@pytest.fixture(autouse=True)
def clear_admins_before_test(app):
    with app.app_context():
        _db.session.query(Admin).delete()
        _db.session.commit()

# Creates a dummy admin
@pytest.fixture()
def logged_in_client(app):
    # create admin in DB
    with app.app_context():
        admin = Admin(username="adminuser", password_hash=generate_password_hash("adminpass"))
        _db.session.add(admin)
        _db.session.commit()

    client = app.test_client()
    
    # actual login
    response = client.post('/admin/login', data={
        'username': 'adminuser',
        'password': 'adminpass'
    }, follow_redirects=True)

    assert b"Menu" in response.data or b"Logout" in response.data  # optional debug
    return client

# Patch the test resource directories
@pytest.fixture(autouse=True)
def patch_resource_dirs(monkeypatch):
    monkeypatch.setattr(constant, "RESOURCE_TEST_PREDICTION_DIR", "resource_test/prediction")
    monkeypatch.setattr(constant, "RESOURCE_TEST_RETRAIN_DIR", "resource_test/retrain")

@pytest.fixture
def sample_zip_bytes():
    """Return sample valid zip bytes containing origination and monthly performance files."""
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zipf:
        zipf.writestr("data_2020Q1.txt", "some,data,here\n1,2,3")
        zipf.writestr("data_time_2020Q1.txt", "another,line\n4,5,6")
    memory_file.seek(0)
    return memory_file.read()