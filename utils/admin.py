from werkzeug.security import check_password_hash
from flask_login import UserMixin

# Admin interface
class AdminUser(UserMixin):
    def __init__(self, admin_id, username, password_hash):
        self.id = str(admin_id)  
        self.admin_id = admin_id
        self.username = username
        self.password_hash = password_hash
    
    def get_id(self):
        return str(self.admin_id)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)
    