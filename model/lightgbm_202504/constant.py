'''
This file contains shared constant values used across the system.
''' 

PREDICTION_UPLOAD_DIR = "uploads/prediction"
RESOURCE_APP_METADATA_DIR= "resource_app/metadata"
RESOURCE_APP_RETRAIN_DIR= "resource_app/retrain"
RESOURCE_APP_PREDICTION_DIR= "resource_app/prediction"
RESOURCE_TEST_METADATA_DIR= "resource_test/metadata"
RESOURCE_TEST_RETRAIN_DIR= "resource_test/retrain"
RESOURCE_TEST_PREDICTION_DIR= "resource_test/prediction"
RESOURCE_DIR='resources'
RETRAIN_UPLOAD_DIR = 'uploads/retrain'

DTYPE_OVERRIDES = {
    'del_due_to_disaster': 'object', 'modification_flag': 'object',
    'step_mod_flag': 'object', 'super_conforming_flag': 'object',
    'program_indicator': 'object', 'BASC': 'object', 'cur_LDS': 'object',
    'deferred_pay_plan': 'object'
}

# mp stands for monthly performance data
MP_COL = ["loan_sequence_num", "monthly_reporting_period", "cur_actual_UPB", "cur_LDS", "loan_age",
          "remaining_mths_to_legal_maturity", "defect_settlement_date", "modification_flag", "zero_bal_code",
          "zero_bal_effective_date", "cur_int_rate", "cur_deferred_UPB", "DDLPI", "MI_recover",
          "net_sales_proceeds", "non_MI_recover", "expenses", "legal_costs", "MPC", "taxes_and_insurance",
          "miscel_expenses", "actual_loss_calculation", "modification_cost", "step_mod_flag", "deferred_pay_plan",
          "ELTV", "zero_bal_removal_UPB", "del_accrued_int", "del_due_to_disaster", "BASC", "cur_mth_mod_cost",
          "int_bearing_UPB"]

# ori stands for origination data
ORI_COL = ["credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date", 
           "MSA", "MI(%)", "num_of_units", "occupancy_status", "ori_CLTV", "ori_DTI", "ori_UPB", 
           "ori_LTV", "ori_int_rate", "channel", "PPM_flag", "amorisation_type", "property_state",
           "property_type", "postal_code", "loan_sequence_num", "loan_purpose", "ori_loan_term", 
           "num_borrowers", "seller_name", "servicer_name", "super_conforming_flag", "pre-HARP_LSN",
           "program_indicator", "HARP_indicator", "property_valuation_method", "I/O_indicator", "MICI"]

FEATURE_LIST1=['loan_sequence_num','cur_actual_UPB', 'cur_int_rate', 'cur_deferred_UPB',
            'ELTV', 'credit_score', 'first_time_homebuyer', 'MI(%)', 'num_of_units',
            'occupancy_status', 'ori_DTI', 'channel', 'property_state', 'property_type',
            'loan_purpose', 'num_borrowers', 'property_valuation_method',
            'loan_age', 'quarter']  

FEATURE_LIST2=['cur_actual_UPB', 'cur_int_rate', 'cur_deferred_UPB',
       'ELTV', 'credit_score', 'first_time_homebuyer', 'MI(%)', 'num_of_units',
       'occupancy_status', 'ori_DTI', 'channel', 'property_state', 'property_type',
       'loan_purpose', 'num_borrowers', 'property_valuation_method',
       'loan_age', 'quarter', 'cur_LDS']

FEATURE_LIST3=['cur_actual_UPB', 'cur_int_rate', 'cur_deferred_UPB',
       'ELTV', 'credit_score', 'first_time_homebuyer', 'MI(%)', 'num_of_units',
       'occupancy_status', 'ori_DTI', 'channel', 'property_state', 'property_type',
       'loan_purpose', 'num_borrowers', 'property_valuation_method',
       'loan_age', 'quarter']

PROTECTED_ATTRIBUTES = ['region_Northeast', 'region_South', 'region_West','region_Other']

ORIGINATION_KEYWORD = "historical_data"
MONTHLY_PERF_KEYWORD = "historical_data_time"
TARGET = "target"
RANDOM_STATE = 42

THRESHOLDS = {
    "acc": 0.80,
    "auc": 0.85,
    "f1": 0.85,
    "fairness": 0.20
}

