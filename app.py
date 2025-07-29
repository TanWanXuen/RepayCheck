from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import CSRFProtect
from apscheduler.schedulers.background import BackgroundScheduler
from threading import Thread
from werkzeug.security import check_password_hash, generate_password_hash
from sqlalchemy.exc import OperationalError
from pathlib import Path
import os, secrets
from dotenv import load_dotenv
import getpass
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_session import Session
import redis

# Blueprints
from blueprints.user_app import inference_bp
from blueprints.retrain_app import retrain_bp
from blueprints.infer_app import prediction_admin_bp
from blueprints.download_app import download_bp
from blueprints.admin_app import admin_bp

# Local modules
from utils.log import rotate_all_loggers, configure_logger, schedule_log_cleanup
from utils.admin import AdminUser
from database.db_base import init_db, db, db_queue
from database.db_handler import Admin
from model.lightgbm_202504.constant import (
    RESOURCE_APP_PREDICTION_DIR, RESOURCE_APP_RETRAIN_DIR,
    RETRAIN_UPLOAD_DIR, PREDICTION_UPLOAD_DIR
)

env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

LOG_DIR = "/log"

log_dirs = [
    (RETRAIN_UPLOAD_DIR, 180),
    (PREDICTION_UPLOAD_DIR, 30),
    (RESOURCE_APP_RETRAIN_DIR, 30),
    (RESOURCE_APP_PREDICTION_DIR, 30),
    (LOG_DIR, 7)
]

def create_app(testing=False):
    app = Flask(__name__)
    init_db(app, testing=testing)
    
    #session["SESSION_SECRET_TOKEN"] = secrets.token_hex(16)

    # Configure app
    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY"),
        WTF_CSRF_SECRET_KEY=os.getenv("WTF_CSRF_SECRET_KEY"),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SECURE=not app.debug,
        SESSION_COOKIE_SAMESITE='Strict',
        PERMANENT_SESSION_LIFETIME=1800,
        SESSION_PERMANENT=False,
        TESTING=testing,
        WTF_CSRF_ENABLED=not testing, 
        SESSION_TYPE='redis',
        SESSION_REDIS=redis.from_url("redis://redis:6379"),
        SESSION_USE_SIGNER=True,
        SESSION_KEY_PREFIX='flask_sess:'                    
    )

    # Logging
    logger = configure_logger("main_app")
    app.logger = logger

    # Extensions
    CSRFProtect(app)
    Session(app)
    login_manager = LoginManager(app)
    login_manager.login_view = "admin_login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "warning"

    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )

    # Login user loader
    @login_manager.user_loader
    def load_user(user_id):
        admin = db.session.query(Admin).filter_by(admin_id=user_id).first()
        if admin:
            return AdminUser(admin.admin_id, admin.username, admin.password_hash)
        return None

    # Register blueprints
    app.register_blueprint(inference_bp)
    app.register_blueprint(retrain_bp)
    app.register_blueprint(prediction_admin_bp)
    app.register_blueprint(download_bp)
    app.register_blueprint(admin_bp)

    # Routes
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/admin')
    def admin():
        if current_user.is_authenticated:
            logger.info(f"Admin '{current_user.get_id()}' accessed admin main page.")
            return redirect(url_for('retrain.retrain'))
        return render_template('admin.html')

    @app.route('/admin/login', methods=['GET', 'POST'])
    @limiter.limit("5 per minute", deduct_when=lambda response: not current_user.is_authenticated)
    def admin_login():
        if current_user.is_authenticated:
            return redirect(url_for('retrain.retrain'))

        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            if not username or not password:
                flash("Both username and password are required.", "danger")
                return render_template('admin.html')

            admin = get_admin_user_by_username(username)
            if admin and check_password_hash(admin.password_hash, password):
                user = AdminUser(admin.admin_id, admin.username, admin.password_hash)
                login_user(user)
                session["SESSION_SECRET_TOKEN"] = secrets.token_hex(16)
                session["fresh"] = True
                flash("Logged in successfully!", "success")
                return redirect(request.args.get('next') or url_for('retrain.retrain'))
            else:
                return render_template('admin.html', error_message="Invalid username or password.")

        return render_template('admin.html')

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        session.clear()
        flash("You have been logged out successfully.", "info")
        return redirect(url_for('index'))

    @app.before_request
    def enforce_session_freshness():
        if current_user.is_authenticated:
            if not session.get("SESSION_SECRET_TOKEN"):
                logout_user()
                session.clear()
                flash("Session expired. Please log in again.", "warning")
                logger.info(f"Session expired for user: {current_user.get_id()}")
                return redirect(url_for("admin"))

    # Error handlers
    @app.errorhandler(401)
    def unauthorized(e):
        flash("Please log in to access this page.", "warning")
        return redirect(url_for('admin', next=request.path))

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(429)
    def ratelimit_handler(e):
        flash("Too many login attempts. Please try again later.", "danger")
        return redirect(url_for('admin'))

    @app.route('/error')
    def error():
        return render_template('error.html', message=request.args.get('message', "An error occurred."))

    @app.route('/user_error')
    def user_error():
        return render_template('user_error.html', message=request.args.get('message', "An error occurred."))

    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    
    # Only the first Gunicorn worker should start these
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or os.environ.get("GUNICORN_WORKER_ID", "0") == "0":
        scheduler = BackgroundScheduler()
        schedule_log_cleanup(scheduler, log_dirs)
        scheduler.start()
        app.logger.info("Background log cleanup job scheduled.")
        Thread(target=db_writer, args=(app,), daemon=True).start()

    initialize_database(app)
    return app

def initialize_database(app, retries=10):
    from sqlalchemy import text
    import time

    with app.app_context(): 
        for attempt in range(retries):
            try:
                with db.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                print("Database connection successful.")
                break
            except OperationalError as e:
                print(f"DB connection failed (attempt {attempt+1}/{retries}): {e}")
                time.sleep(3)
        else:
            app.logger.critical("Could not connect to database after retries.")
            raise SystemExit("Database initialization failed.")

        db.create_all()
        app.logger.info("All tables created or already exist.")

def get_admin_user_by_username(username):
    return db.session.query(Admin).filter_by(username=username).first()

def db_writer(app):
    with app.app_context():
        while True:
            task = db_queue.get()
            if task is None:
                break
            try:
                task['func'](*task['args'], **task['kwargs'])
            except Exception as e:
                app.logger.error(f"DB task failed: {e}", exc_info=True)
            finally:
                db_queue.task_done()
app = create_app()
# To create admin 
# Can be used in the terminal when newly setting up the db 
def create_admin():
    username = input("Enter admin username: ")
    password = getpass.getpass("Enter password: ")
    hashed = generate_password_hash(password)

    new_admin = Admin(username=username, password_hash=hashed)
    with app.app_context():
        db.session.add(new_admin)
        db.session.commit()
        print("Admin created successfully.")

# if __name__ == "__main__":
#     rotate_all_loggers()
#     app = create_app()
    
    # Run app with waitress
    # from waitress import serve
    # serve(app,
    #       host='0.0.0.0',
    #       port=int(os.getenv("PORT", 5000)),
    #       threads=int(os.getenv("WAITRESS_THREADS", 4)),
    #       _quiet=True
    # )