from utils.log import configure_logger

def test_configure_logger_creates_log_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = configure_logger("test_logger")
    logger.info("test log message")

    log_dir = tmp_path / "log"
    log_files = list(log_dir.rglob("*.log"))
    assert len(log_files) == 1
    assert "test_logger" in log_files[0].name
    assert "test log message" in log_files[0].read_text()

def test_configure_logger_reuse_existing_logger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger1 = configure_logger("reuse_logger")
    logger2 = configure_logger("reuse_logger")
    logger2.info("reuse test")
    log_dir = tmp_path / "log"
    assert any(log_dir.rglob("*.log"))