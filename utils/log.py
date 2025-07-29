import os
import logging
import shutil
from datetime import datetime, timedelta
import re
from pathlib import Path

LOG_FOLDER = "/log"
LOGGER_CACHE = {}  # Keep loggers to avoid reconfiguring inside same process


def get_log_file_path(process_name, current_date):
    date_folder = os.path.join(LOG_FOLDER, current_date)
    os.makedirs(date_folder, exist_ok=True)
    log_filename = f"{process_name}_{current_date}.log"
    return os.path.join(date_folder, log_filename)


def configure_logger(process_name):
    """
    Configure or reconfigure the logger to point to the current day's log file.
    Avoid clearing handlers unless rotating manually.
    """
    current_date = datetime.now().strftime('%Y%m%d')
    log_path = get_log_file_path(process_name, current_date)

    logger = logging.getLogger(process_name)

    if process_name in LOGGER_CACHE:
        return LOGGER_CACHE[process_name]

    logger.setLevel(logging.DEBUG)

    # Ensure no duplicate handlers
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - [%(filename)s] - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    LOGGER_CACHE[process_name] = logger
    return logger


def rotate_all_loggers():
    """
    Reconfigure all loggers to point to a new log file (for the new date).
    Called daily via APScheduler.
    """
    current_date = datetime.now().strftime('%Y%m%d')
    for name in [
        "admin_app",
        "db_task",
        "download_app",
        "file_removal_manager",
        "infer_app",
        "main_app",
        "retrain_app",
        "user_app"
    ]:
        log_path = get_log_file_path(name, current_date)
        logger = logging.getLogger(name)

        # Remove all handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Attach new file handler for current date
        handler = logging.FileHandler(log_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - [%(filename)s] - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.info(f"[AUTO-ROTATE] Logger '{name}' rotated for new day.")
        LOGGER_CACHE[name] = logger

# To clean the olf files after certain periods
def clean_old_files(relative_dir: str, num_days: int):
    logger = configure_logger("file_removal_manager")
    logger.info(f"Running cleanup for {relative_dir} (older than {num_days} days)")
    now = datetime.now()
    cutoff_date = now - timedelta(days=num_days)
    date_pattern = re.compile(r"(\d{8})")

    base_dir = Path(__file__).resolve().parent.parent
    target_dir = (base_dir / relative_dir).resolve()

    if not target_dir.exists():
        logger.warning(f"Directory does not exist: {target_dir}")
        return

    for entry in target_dir.iterdir():
        if entry.is_file():
            file_time = datetime.fromtimestamp(entry.stat().st_mtime)
            if file_time < cutoff_date:
                try:
                    entry.unlink()
                    logger.info(f"Deleted old file: {entry}")
                except Exception as e:
                    logger.error(f"Failed to delete file {entry}: {e}")
            continue

        if entry.is_dir():
            match = date_pattern.match(entry.name)
            if match:
                try:
                    folder_date = datetime.strptime(match.group(1), "%Y%m%d")
                    if folder_date < cutoff_date:
                        delete_folder(entry)
                except ValueError:
                    logger.warning(f"[SKIP] Invalid date in folder name: {entry}")
            else:
                logger.debug(f"[SKIP] Folder does not match date format: {entry}")

# To delete folder
def delete_folder(folder_path: Path):
    try:
        shutil.rmtree(folder_path, ignore_errors=True)
        print(f"Successfully deleted folder: {folder_path}")
    except Exception as e:
        print(f"Error while deleting folder {folder_path}: {e}")

# To schedule the reconfiguration of the log filesat 12:01am and cleaning of the old files at 12:05am
def schedule_log_cleanup(scheduler, jobs):
    for rel_path, days in jobs:
        scheduler.add_job(
            clean_old_files,
            'cron', hour=0, minute=5,
            args=[rel_path, days],
            id=f"cleanup_{rel_path}"
        )
        
    clean_old_files(rel_path, days)

    scheduler.add_job(
        rotate_all_loggers,
        'cron', hour=0, minute=1,
        id="rotate_logger"
    )
