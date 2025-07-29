def get_region(region: str)->str:
    group = {
        'West': ['WA','OR','CA','MT','ID','WY','NV','UT','CO','AZ','NM','HI','AK'],
        'Midwest': ['ND','SD','MN','NE','KS','MN','IA','MO','WI','IL','MI','IN','OH'],
        'South': ['TX','OK','AR','LA','MS','AL','TN','KY','GA','FL','SC','NC','VA','WV','DC','MD','DE'],
        'Northeast': ['PA','NY','NJ','RI','CT','MA','VT','NH','ME']
    }
    for category, keywords in group.items():
        if any(keyword.lower() in region.lower() for keyword in keywords):
            return category
    return 'Other' 

def impute_missing_values(df, column: str, num_to_replace: float):
    mean_value = df[df[column] != num_to_replace][column].mean().round(1)  
    df[column] = df[column].replace(num_to_replace, mean_value)  # Replace invalid values
    return df[column]


def calculate_monthly_payment(loan_amount, annual_interest_rate, years):
    r = annual_interest_rate / 100 / 12  # monthly interest rate
    n = years * 12  # total number of payments
    if r == 0:
        return loan_amount / n
    return loan_amount * (r * (1 + r)**n) / ((1 + r)**n - 1)

def calculate_ori_dti(monthly_income, loan_amount, annual_interest_rate, years):
    # Restriction 1: Monthly income must be positive
    if monthly_income <= 0:
        monthly_income = 1000  # fallback default

    # Restriction 2: Interest rate cap
    annual_interest_rate = max(min(annual_interest_rate, 10.0), 0.1)  # clamp to 0.1–10%

    # Restriction 3: Minimum loan term (increase term to lower monthly debt)
    years = max(years, 5)  # increase term if too short

    # Calculate initial monthly payment and ori_DTI
    monthly_debt = calculate_monthly_payment(loan_amount, annual_interest_rate, years)
    ori_dti = (monthly_debt / monthly_income) * 100

    # If DTI is too high, try to extend loan term up to a max of 30 years
    max_years = 30
    while ori_dti > 65 and years < max_years:
        years += 1
        monthly_debt = calculate_monthly_payment(loan_amount, annual_interest_rate, years)
        ori_dti = (monthly_debt / monthly_income) * 100

    # If still too high, scale down the loan amount
    if ori_dti > 65:
        max_monthly_debt = (65 / 100) * monthly_income
        loan_amount = max_monthly_debt * ((1 + (annual_interest_rate / 100 / 12)) ** (years * 12) - 1) / \
                      ((annual_interest_rate / 100 / 12) * ((1 + (annual_interest_rate / 100 / 12)) ** (years * 12)))
        monthly_debt = calculate_monthly_payment(loan_amount, annual_interest_rate, years)
        ori_dti = (monthly_debt / monthly_income) * 100

    return round(ori_dti, 5)

def assign_default_values():
    return{
        "ELTV": 71.67837, 
        "cur_deferred_UPB":0,
        "property_valuation_method": 2
    }
