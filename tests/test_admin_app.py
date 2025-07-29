from database.db_handler import Admin
from database.db_base import db
from werkzeug.security import generate_password_hash

def create_admin(db, username, password):
    hashed = generate_password_hash(password)
    admin = Admin(username=username, password_hash=hashed)
    db.session.add(admin)
    db.session.commit()
    return admin  

def login_as(client, admin):
    with client.session_transaction() as session:
        session['_user_id'] = str(admin.admin_id)

def test_admin_debug(logged_in_client):
    response = logged_in_client.get('/admin/debug')
    assert response.status_code == 200

def test_login_failure(client):
    response = client.post("/admin/login", data={
        "username": "adminuser",
        "password": "wrongpass"
    }, follow_redirects=True)
    assert b"Login" in response.data

def test_delete_admin_success(app, logged_in_client):
    with app.app_context():
        admin = create_admin(db, 'todelete', 'pass123')
        login_as(logged_in_client, admin)

    response = logged_in_client.post('/admin/delete_admin', data={
        'username': 'todelete'
    }, follow_redirects=True)

    assert response.status_code == 200
    with app.app_context():
        deleted = Admin.query.filter_by(username='todelete').first()
        assert deleted is None
