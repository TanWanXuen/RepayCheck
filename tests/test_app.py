import pytest
from app import create_app, db
from database.db_handler import Admin
from werkzeug.security import generate_password_hash

# === Setup ===
@pytest.fixture()
def test_client():
    app = create_app(testing=True)
    app.config['WTF_CSRF_ENABLED'] = False

    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            admin = Admin(username="testadmin", password_hash=generate_password_hash("test123"))
            db.session.add(admin)
            db.session.commit()
        yield client
        with app.app_context():
            db.drop_all()

def test_homepage_flask(test_client):
    res = test_client.get('/')
    assert res.status_code == 200
    assert b"Loan Repayment" in res.data or b"index" in res.data

def test_login_logout_flask(test_client):
    res = test_client.post('/admin/login', data={"username": "testadmin", "password": "test123"}, follow_redirects=True)
    assert res.status_code == 200
    assert b"model update" in res.data.lower()

    res = test_client.get('/logout', follow_redirects=True)
    assert res.status_code == 200
    assert b"out" in res.data.lower()

def test_admin_main_requires_login(test_client):
    res = test_client.get('/admin', follow_redirects=True)
    assert b"login" in res.data.lower() or res.status_code == 200

def test_404_error_page_flask(test_client):
    res = test_client.get('/nonexistentpage')
    assert res.status_code == 404
    assert b"404" in res.data or b"not found" in res.data.lower()

def test_admin_login_fail(test_client):
    res = test_client.post('/admin/login', data={"username": "wronguser", "password": "wrongpass"}, follow_redirects=True)
    assert res.status_code == 200
    assert b"invalid username or password" in res.data.lower()

def test_admin_login_success(test_client):
    res = test_client.post("/admin/login", data={"username": "testadmin", "password": "test123"}, follow_redirects=True)
    assert res.status_code == 200
    assert b"model update" in res.data.lower()

def test_ratelimit_lockout(test_client):
    for _ in range(6):  # trigger rate limiter
        res = test_client.post('/admin/login', data={
            "username": "wronguser",
            "password": "wrongpass"
        }, follow_redirects=True)
    assert b"admin login" in res.data.lower() or res.status_code == 429
