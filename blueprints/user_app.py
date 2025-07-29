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

from flask import Blueprint, request, render_template, redirect, url_for, flash
from pydantic import BaseModel, Field, ValidationError

from datetime import datetime, date
import pandas as pd
import joblib
import os
from urllib.parse import urlencode
#import traceback 

from model.lightgbm_202504.constant import RESOURCE_APP_METADATA_DIR, FEATURE_LIST3
from model.lightgbm_202504.preprocessing_tools import calculate_ori_dti, assign_default_values
from model.lightgbm_202504.infer import load_encoders, load_scaling, preprocess
from utils.log import configure_logger
from database.db_task import add_contact_us_to_db
#add_user_infer_to_db  

logger = configure_logger("user_app")
inference_bp = Blueprint('inference', __name__)

class LoanForm(BaseModel):
    monthly_income: int = Field(..., json_schema_extra={"example": 150000})
    loan_term_years: int = Field(..., alias="loan_age", ge=1, le=30, json_schema_extra={"example": 15})
    cur_actual_UPB: int = Field(..., json_schema_extra={"example": 150000})
    cur_int_rate: float = Field(..., json_schema_extra={"example": 3.5})
    mi_percentage: float = Field(..., alias="MI(%)", ge=0, le=55, json_schema_extra={"example": 3.5})
    num_borrowers: int
    loan_start_date: date
    property_state: str
    credit_score: int = Field(..., ge=300, le=850, json_schema_extra={"example": 650})

    class Config:
        populate_by_name = True

# To display user loan repayment eligibility checker page
@inference_bp.route('/checker')
def checker():
    score = request.args.get("score")
    return render_template('checker.html', prefill_score=score)

# To predict loan repayment eligibility
@inference_bp.route("/checker/result", methods=["POST"])
def infer_from_form():
    try:
        logger.info("Received form submission for loan prediction.")
        logger.debug(f"Raw form data: {request.form}")

        form_data = LoanForm(
            monthly_income=int(request.form['monthly_income']),
            loan_term_years=int(request.form['loan_age']),
            cur_actual_UPB=int(request.form['cur_actual_UPB']),
            cur_int_rate=float(request.form['cur_int_rate']),
            mi_percentage=float(request.form['mi']),
            num_borrowers=int(request.form['num_borrowers']),
            loan_start_date=datetime.strptime(request.form['loan_start_date'], "%Y-%m-%d").date(),
            property_state=request.form['property_state'],
            credit_score=int(request.form['credit_score']),
        )

        logger.debug(f"Parsed LoanForm data: {form_data}")

        # Optional advanced fields with default fallback
        loan_purpose = request.form.get("loan_purpose", "P")  # Default: Purchase
        channel = request.form.get("channel", "R")             # Default: Retail
        first_time_homebuyer = request.form.get("first_time_homebuyer", "Y")  # Default: No
        property_type = request.form.get("property_type", "SF")  # Default: Single-Family
        occupancy_status = request.form.get("occupancy_status", "P")
        num_of_units = int(request.form.get('num_of_units', 1))

        input_dict = form_data.dict(by_alias=True)
        full_data = {**assign_default_values(), 
            **input_dict,  
            "loan_purpose": loan_purpose,
            "channel": channel,
            "first_time_homebuyer": first_time_homebuyer,
            "property_type": property_type,
            "occupancy_status": occupancy_status,
            "num_of_units": num_of_units
        }

        full_data["ori_DTI"] = calculate_ori_dti(
            full_data["monthly_income"],
            full_data["cur_actual_UPB"],
            full_data["cur_int_rate"],
            full_data["loan_age"]
        )
        full_data["quarter"] = f"{(full_data['loan_start_date'].month - 1) // 3 + 1}"
        full_data["loan_start_date"] = full_data["loan_start_date"].strftime("%Y-%m-%d")
   
        logger.debug(f"Full input data with computed ori_DTI: {full_data}")

        df = pd.DataFrame([full_data])
        df = df[FEATURE_LIST3]
        logger.debug(f"Input DataFrame columns: {df.columns.tolist()}")

        encoder = load_encoders()
        scaling = load_scaling()
        logger.info("Encoders and scalers loaded.")
        
        sample = preprocess(df, FEATURE_LIST3, encoder, scaling, None, None)
        lgb_model = joblib.load(os.path.join(RESOURCE_APP_METADATA_DIR, "model_best.pkl"))
        prediction = lgb_model.predict(sample)[0]

        logger.info("Preprocessing completed.")
        logger.info("Loaded trained LightGBM model.")
        logger.info(f"Prediction result: {prediction}")

        message = "You are eligible to repay." if prediction == True else "You are not eligible to repay."
        
        # Store inference result in database
        #add_user_infer_to_db(
          #  monthly_income=form_data.monthly_income,
         #   loan_term_years=form_data.loan_term_years,
          #  cur_actual_UPB=form_data.cur_actual_UPB,
           # cur_int_rate=form_data.cur_int_rate,
          #  mi_percentage=form_data.mi_percentage,
          #  num_borrowers=form_data.num_borrowers,
          #  loan_start_date=form_data.loan_start_date,
          #  property_state=form_data.property_state,
          #  credit_score=form_data.credit_score,
          #  loan_purpose=loan_purpose,
          #  channel=channel,
          #  first_time_homebuyer=first_time_homebuyer,
         #   property_type=property_type,
         #   occupancy_status=occupancy_status,
          #  num_of_units=num_of_units,
          #  target=int(prediction)
        #)

        logger.info("User inference logged to database.")

        return render_template("checker.html", result_message=message)

    except ValidationError as ve:
        logger.warning(f"Validation error in form data: {ve}")
        #traceback.print_exc()
        flash(f"Validation error: {ve}")
        return redirect(url_for('user_error'))

    except Exception as e:
        logger.error(f"General error during inference: {e}")
        #traceback.print_exc()
        flash(f"Inference failed: {e}")
        return redirect(url_for('user_error'))

# To display credit score calculation page
@inference_bp.route("/credit_score")
def credit_score_checker():
    return render_template("credit_score.html")

# To calculate credit score 
@inference_bp.route("/checker/credit_score", methods=["POST"])
def calculate_credit_score():
    try:
        logger.info("Received form submission for credit score calculation.")
        filtered_form = {k: v for k, v in request.form.items() if k != "csrf_token"}
        logger.debug(f"Filtered form data (excluding CSRF token): {filtered_form}")

        # === Get form inputs ===
        late_payments = int(request.form.get("late_payments_last_2_years"))
        ever_defaulted = request.form.get("ever_defaulted_or_collected")
        ever_bankrupt = request.form.get("ever_declared_bankruptcy") 

        total_debt = float(request.form.get("estimated_total_debt"))
                           
        total_limit = float(request.form.get("estimated_total_credit_limit"))  
        num_credit_accounts = int(request.form.get("number_of_credit_accounts"))

        credit_history = int(request.form.get("oldest_credit_years"))
        opened_new = request.form.get("opened_new_accounts_past_year")
        inquiries = int(request.form.get("recent_credit_inquiries"))

        credit_type = request.form.getlist("credit_types")
        
        logger.debug("Parsed input values successfully.")

        # === Scoring Breakdown ===

        # Payment History (35%)
        payment_score = 35
        if late_payments == 1:
            payment_score -= 5
        elif late_payments == 2:
            payment_score -= 10
        elif late_payments >= 3:
            payment_score -= 15

        if ever_defaulted == "Y":
            payment_score -= 10
        if ever_bankrupt == "Y":
            payment_score -= 10
        payment_score = max(payment_score, 0)

        # Amounts Owed (30%)
        utilization = total_debt / total_limit
        owed_score = 30
        if utilization < 0.3:
            owed_score -= 0  # Ideal
        elif utilization < 0.5:
            owed_score -= 5
        else:
            owed_score -= 10
        if num_credit_accounts == 0:
            owed_score -= 5
        owed_score = max(owed_score, 0)

        # Length of Credit History (15%)
        history_score = map_credit_history_score(credit_history)
        
        # New Credit (10%)
        new_score = 10
        if opened_new == "Y":
            new_score -= 3
        if inquiries == 2:
            new_score -= 3
        elif inquiries >= 3:
            new_score -= 5
        new_score = max(new_score, 0)

        # Credit Mix (10%)
        mix_score = 10 if credit_type else 5  # 10% if user has any credit type

        # === Final Score Calculation ===
        raw_score = payment_score + owed_score + history_score + new_score + mix_score
        final_score = int(300 + (raw_score / 100) * 550)
        logger.info(f"Credit score calculated: {final_score}")
        logger.debug({
            "payment_score": payment_score,
            "owed_score": owed_score,
            "history_score": history_score,
            "new_score": new_score,
            "mix_score": mix_score,
            "raw_score": raw_score,
            "final_score": final_score
        })

        return render_template("credit_score.html", score=final_score)

    except Exception as e:
        logger.exception(f"Credit score calculation failed due to error: {e}")
        flash("Credit score calculation failed. Please check your inputs.")
        return redirect(url_for("inference.checker"))

def map_credit_history_score(years_code: int) -> int:
    if years_code == 0:
        return 2     # Less than 1 year
    elif years_code == 1:
        return 7     # 1–3 years
    elif years_code == 2:
        return 12    # 4–7 years
    elif years_code == 3:
        return 15    # 8+ years
    else:
        raise ValueError("Invalid value for oldest_credit_years.")

# To display contact us page
@inference_bp.route("/contact_us")
def contact_us():
    return render_template("contact_us.html")

# To display frequent ask question page
@inference_bp.route("/faq")
def faq():
    return render_template("faq.html")

# To collect user enquiries
@inference_bp.route("/contact_us/support", methods=["POST"])
def contact_us_submit():
    try:
        name = request.form.get("name")
        email = request.form.get("email")
        enquiry_type = request.form.get("enquiry_type")
        message = request.form.get("message")

        logger.debug(f"Received contact form: name={name}, email={email}, type={enquiry_type}")

        success, response_msg = add_contact_us_to_db(name, email, enquiry_type, message)

        if success:
            logger.info(f"Contact form submitted successfully by {email}: {response_msg}")
            message_param = response_msg
        else:
            logger.warning(f"Form submission failed for {email}: {response_msg}")
            message_param = "Submission failed: " + response_msg

        return redirect(url_for("inference.contact_us") + "?" + urlencode({"message": message_param}))

    except Exception as e:
        logger.error(f"Unexpected error during contact form submission: {e}", exc_info=True)
        return redirect(url_for("inference.contact_us") + "?" + urlencode({"message": f"Unexpected error: {e}"}))