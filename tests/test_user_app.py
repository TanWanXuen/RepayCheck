from unittest.mock import patch

def test_checker_page(client):
    response = client.get("/checker", follow_redirects=True)
    assert response.status_code == 200
    assert b"checker" in response.data.lower()

@patch("blueprints.infer_app.load_scaling")
@patch("blueprints.infer_app.load_encoders")
@patch("blueprints.infer_app.joblib.load")
@patch("blueprints.infer_app.preprocess")
def test_checker_infer_from_form_success(mock_preprocess, mock_joblib, mock_enc, mock_scaling, client):
    mock_preprocess.return_value = [[0.1] * 10]
    mock_model = mock_joblib.return_value
    mock_model.predict.return_value = [1]

    data = {
        "monthly_income": "100000",
        "loan_age": "10",
        "cur_actual_UPB": "200000",
        "cur_int_rate": "4.5",
        "mi": "2.0",
        "num_borrowers": "1",
        "loan_start_date": "2020-01-01",
        "property_state": "CA",
        "credit_score": "720"
    }

    response = client.post("/checker/result", data=data, follow_redirects=True)
    assert response.status_code == 200
    assert b"eligible" in response.data

def test_credit_score_page(client):
    res = client.get("/credit_score")
    assert res.status_code == 200
    assert b"score" in res.data.lower()

@patch("blueprints.user_app.render_template")
def test_credit_score_form_valid(mock_render, client):
    mock_render.return_value = "ok"
    form_data = {
        "late_payments_last_2_years": "0",
        "ever_defaulted_or_collected": "N",
        "ever_declared_bankruptcy": "N",
        "estimated_total_debt": "5000",
        "estimated_total_credit_limit": "20000",
        "number_of_credit_accounts": "3",
        "oldest_credit_years": "2",
        "opened_new_accounts_past_year": "N",
        "recent_credit_inquiries": "1",
        "credit_types": ["mortgage", "credit_card"]
    }
    client.post("/checker/credit_score", data=form_data)
    mock_render.assert_called_once() 

def test_contact_us_page(client):
    res = client.get("/contact_us")
    assert res.status_code == 200
    assert b"contact" in res.data.lower()

@patch("blueprints.user_app.add_contact_us_to_db", return_value=(True, "Thanks for your message!"))
def test_contact_us_submit_success(mock_add, client):
    data = {
        "name": "Test User",
        "email": "test@example.com",
        "enquiry_type": "Support",
        "message": "Hello!"
    }
    res = client.post("/contact_us/support", data=data, follow_redirects=False)
    assert res.status_code == 302  # redirect
    assert "message=Thanks" in res.headers["Location"]

def test_faq_page(client):
    res = client.get("/faq")
    assert res.status_code == 200
    assert b"faq" in res.data.lower()