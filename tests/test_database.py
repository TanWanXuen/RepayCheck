from datetime import datetime
from database.db_handler import (
    RetrainTracking, ModelVersion,
    Retrain, ModelStatus, RetrainStatusEnum, FileType
)
from database.db_task import (
    finalize_retrain_and_model_version,
    add_contact_us_to_db,
    add_retrain_and_model_version_to_db,
    add_prediction_to_db,
    add_download_to_db
)
from unittest.mock import patch, MagicMock

@patch("database.db_task.db")
@patch("database.db_task.current_user")
def test_add_download_to_db(mock_current_user, mock_db):
    mock_current_user.admin_id = 1
    mock_db.session = MagicMock()

    add_download_to_db("test.csv", FileType.PREDICTION)
    added = mock_db.session.add.call_args[0][0]
    assert added.file_name == "test.csv"
    assert added.file_type == FileType.PREDICTION
    assert mock_db.session.commit.called

@patch("database.db_task.db")
def test_add_prediction_to_db(mock_db):
    mock_db.session = MagicMock()
    add_prediction_to_db(admin_id=1, file_name="prediction.csv")
    added = mock_db.session.add.call_args[0][0]
    assert added.file_name == "prediction.csv"
    assert added.admin_id == 1

@patch("database.db_task.db")
def test_add_retrain_and_model_version_to_db(mock_db):
    mock_db.session = MagicMock()
    add_retrain_and_model_version_to_db(admin_id=1, version="v1.0", started_at=datetime.now())
    added_objects = [call[0][0] for call in mock_db.session.add.call_args_list]
    assert any(isinstance(x, ModelVersion) for x in added_objects)
    assert any(isinstance(x, Retrain) for x in added_objects)
    assert any(isinstance(x, RetrainTracking) for x in added_objects)

@patch("database.db_task.db")
def test_finalize_retrain_and_model_version(mock_db):
    mock_db.session = MagicMock()

    # Setup fake DB get results
    tracking = MagicMock(model_version_id=1, retrain_id=2)
    model = MagicMock()
    retrain = MagicMock()
    mock_db.session.query().filter_by().first.return_value = tracking
    mock_db.session.get.side_effect = lambda cls, id: { (ModelVersion, 1): model, (Retrain, 2): retrain }[(cls, id)]

    finalize_retrain_and_model_version(
        admin_id=2, version="v2.0",
        model_status=ModelStatus.passed,
        retrain_status=RetrainStatusEnum.completed,
        output_filename="retrain_model.pkl",
        finished_at=datetime.now()
    )

    assert model.status == ModelStatus.passed
    assert retrain.status == RetrainStatusEnum.completed
    assert retrain.file_name == "retrain_model.pkl"
    assert retrain.admin_id == 2

@patch("database.db_task.db")
def test_add_contact_us_to_db(mock_db):
    mock_db.session = MagicMock()
    success, message = add_contact_us_to_db(
        name="Test User",
        email="test@abc.com",
        enquiry_type="Feedback",
        message="Hello there"
    )
    assert success is True
    assert "Thank you" in message
