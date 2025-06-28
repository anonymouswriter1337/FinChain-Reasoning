import random
from misc import companies, currencies

def basic_dividend_stable_policy():
    """
    1:Basic: Calculate the total dividend payable given the number of shares outstanding and the dividend per share.

    Generates a question and solution about calculating total dividend payment under a stable dividend policy.
    This function creates a problem scenario where a randomly selected company follows a stable dividend
    policy with a fixed dividend per share. It calculates the total dividend payment based on the number
    of shares outstanding.
    Returns:
        tuple: A pair of strings where:
            - First element is the question text describing the dividend calculation scenario
            - Second element is the detailed solution showing the step-by-step calculation
    """

    company = random.choice(companies)
    company_name = company[0]
    currency = random.choice(currencies)
    
    dividend_per_share = round(random.uniform(1, 10), 2)
    shares_outstanding = random.randint(500000, 5000000)
    
    question = f'''{company_name} follows a stable dividend policy and pays a fixed dividend of {currency}{dividend_per_share} per share annually. If {company_name} has {shares_outstanding:,} shares outstanding, how much total dividend will the company pay to its shareholders?'''
    
    total_dividend = round(dividend_per_share * shares_outstanding, 2)
    
    solution = f'''Step 1: Calculate total dividend using the formula:
    Total Dividend = Dividend per Share × Number of Shares Outstanding
                    = {currency}{dividend_per_share} × {shares_outstanding:,}
                    = {currency}{total_dividend:,.2f}'''
    
    return question, solution
    
def basic_residual_dividend_policy():
    """
    2:basic: Calculate the total dividend payable using the residual dividend policy given net income, and capital expenditure.

    Generates a question and solution about residual dividend policy calculation.
    This function creates a scenario where a company follows a residual dividend policy.
    It randomly selects company details and financial parameters to construct a problem
    about calculating dividend payments based on net income, capital expenditure, and
    target equity ratio.
    Returns:
        tuple: A pair containing:
            - question (str): A word problem describing the scenario and asking for dividend calculation
            - solution (str): Step-by-step solution showing equity funding and dividend calculations
    """

    company = random.choice(companies)
    company_name = company[0]
    currency = random.choice(currencies)
    
    equity_ratio = round(random.uniform(0.4, 0.6), 2)
    capex = random.randint(2000000, 6000000)
    
    equity_funding_needed = round(capex * equity_ratio, 2)
    
    while True:
        net_income = random.randint(5000000, 10000000)
        if net_income > equity_funding_needed:
            break
            
    available_for_dividend = round(net_income - equity_funding_needed, 2)
    
    question = f'''{company_name} follows a residual dividend policy. The company expects to generate {currency}{net_income} in net income this year and has capital expenditure (CapEx) plans of {currency}{capex}. The company wants to maintain a capital structure where {equity_ratio*100}% of its funding comes from equity and the remaining {(1-equity_ratio)*100}% comes from debt. How much total dividend will {company_name} pay to its shareholders this year?'''
    
    solution = f'''Step 1: Calculate required equity funding
    Equity Funding = CapEx × Equity Ratio
                    = {currency}{capex} × {equity_ratio}
                    = {currency}{equity_funding_needed}
    
    Step 2: Calculate residual dividend amount
    Dividend = Net Income - Required Equity Funding
                = {currency}{net_income} - {currency}{equity_funding_needed}
                = {currency}{available_for_dividend}'''
    
    return question, solution

def intermediate_dividend_constant_payout_ratio_policy():
    """
    3:Intermediate: Calculate the total dividend payable next year based on a stable policy and an expected net income.

    Generates a dividend policy problem involving constant payout ratio calculation.
    This function creates a scenario where a company maintains a constant dividend payout 
    ratio while dealing with changes in shares outstanding and projected net income. It 
    randomly selects company details and financial figures to create a realistic problem.
    Returns:
        tuple: A pair containing:
            - question (str): A word problem describing the dividend policy scenario
            - solution (str): A detailed step-by-step solution showing calculations for:
                1. Current total dividend payment
                2. Current dividend payout ratio
                3. Expected dividend payment for next year
    """

    company = random.choice(companies)
    company_name = company[0]
    currency = random.choice(currencies)
    
    dividend_per_share = round(random.uniform(2, 10), 2)
    current_shares = random.randint(500000, 2000000)
    share_increase = round(random.uniform(5, 20), 2)
    
    new_shares = round(current_shares * (1 + share_increase/100), 0)
    current_total_dividend = round(dividend_per_share * current_shares, 2)

    while True:
        net_income = random.randint(2000000, 50000000)
        if net_income > current_total_dividend:
            break
    
    # Add next year's net income with some random growth
    income_growth = round(random.uniform(0, 20),2)
    next_year_income = round(net_income * (1 + income_growth/100), 2)
    
    current_payout_ratio = current_total_dividend / net_income
    expected_dividend_next_year = current_payout_ratio * next_year_income

    current_payout_ratio = round(current_payout_ratio, 2)
    expected_dividend_next_year = round(expected_dividend_next_year, 2)
    
    question = f'''{company_name} follows a stable dividend policy and has been paying a dividend of {currency}{dividend_per_share} per share annually. The company currently has {current_shares} shares outstanding and expects a net income of {currency}{net_income} this year. For next year, the company projects a net income of {currency}{next_year_income}. {company_name} plans to issue {share_increase}% more shares next year, which will increase the total number of shares outstanding to {new_shares}. If {company_name} wants to maintain the same dividend payout ratio as this year, what will be the total dividend payment next year?'''
    
    solution = f'''Step 1: Calculate current total dividend payment:
    Current Total Dividend = Current Dividend per Share × Current Number of Shares
                            = {currency}{dividend_per_share} × {current_shares}
                            = {currency}{current_total_dividend}
    
    Step 2: Calculate current dividend payout ratio:
    Current Payout Ratio = Current Total Dividend / Net Income
                        = {currency}{current_total_dividend} / {currency}{net_income}
                        = {current_payout_ratio}
    
    Step 3: Calculate expected dividend payment for next year:
    Next Year's Total Dividend = Current Payout Ratio × Next Year's Net Income
                                = {current_payout_ratio} × {currency}{next_year_income}
                                = {currency}{expected_dividend_next_year}'''
    
    return question, solution

def intermediate_residual_dividend_policy_capex():
    """
    4:Intermediate: Calculate the total dividend payable using the residual dividend policy given net income, capital expenditure, and debt repayment amount.

    Generates a finance problem and solution related to residual dividend policy with capital expenditure.
    The function creates a scenario where a randomly selected company follows a residual dividend policy
    and needs to determine its dividend payment based on:
    - Expected net income
    - Capital expenditure requirements
    - Debt repayment obligations
    - Target capital structure (equity ratio)
    Returns:
        tuple: A pair containing:
            - question (str): A detailed problem statement with the company's financial data
            - solution (str): Step-by-step solution showing calculations for:
                1. Total funding needed
                2. Required equity funding
                3. Residual dividend amount
    """

    company = random.choice(companies)
    company_name = company[0]
    currency = random.choice(currencies)
    
    capex = random.randint(4000000, 8000000)
    debt_repayment = random.randint(1000000, 3000000)
    equity_ratio = round(random.uniform(0.4, 0.6), 2)
    
    total_funding_needed = capex + debt_repayment
    equity_funding_needed = round(total_funding_needed * equity_ratio, 2)

    while True:
        net_income = random.randint(10000000, 20000000)
        if net_income > equity_funding_needed:
            break

    available_for_dividend = round(net_income - equity_funding_needed, 2)
    
    question = f'''{company_name} follows a residual dividend policy. The company expects to generate {currency}{net_income} in net income this year. {company_name} has the following financial needs:
    • Capital expenditures (CapEx): {currency}{capex}
    • Debt repayment: {currency}{debt_repayment}
    • Target capital structure: {equity_ratio*100}% equity, {(1-equity_ratio)*100}% debt (i.e., the company wants to finance {equity_ratio*100}% of its capital requirements with equity and {(1-equity_ratio)*100}% with debt).

Given the company's capital expenditure and debt repayment plans, how much total dividend will {company_name} pay to its shareholders this year?'''
    
    solution = f'''Step 1: Calculate total funding needed
    Total Funding = CapEx + Debt Repayment
                    = {currency}{capex} + {currency}{debt_repayment}
                    = {currency}{total_funding_needed}

Step 2: Calculate required equity funding
    Equity Funding = Total Funding × Equity Ratio
                    = {currency}{total_funding_needed} × {equity_ratio}
                    = {currency}{equity_funding_needed}

Step 3: Calculate residual dividend amount
    Dividend = Net Income - Required Equity Funding
                = {currency}{net_income} - {currency}{equity_funding_needed}
                = {currency}{available_for_dividend}'''
    
    return question, solution

def advanced_hybrid_dividend_policy():
    """
    5:Advanced: Calculate total dividend payment under a hybrid dividend policy.

    This function generates a finance problem and its solution related to hybrid dividend policy,
    which combines both residual dividend policy and constant payout ratio policy. It randomly
    selects a company and currency, and generates realistic financial figures to create a
    scenario where a company needs to determine its dividend payment.
    The function considers:
    - Company's net income
    - Capital expenditure needs
    - Debt repayment requirements
    - Target capital structure (equity ratio)
    - Constant payout ratio
    Returns:
        tuple: A tuple containing two strings:
            - question (str): A detailed problem statement describing the company's financial
              situation and dividend policy.
            - solution (str): A step-by-step solution showing the calculation of the final
              dividend payment, considering both residual and constant payout approaches.
    """

    company = random.choice(companies)
    company_name = company[0]
    currency = random.choice(currencies)
    
    capex = random.randint(4000000, 8000000)
    debt_repayment = random.randint(1000000, 3000000)
    equity_ratio = round(random.uniform(0.4, 0.6), 2)
    payout_ratio = round(random.uniform(0.3, 0.5), 2)
    
    total_funding_needed = capex + debt_repayment
    equity_funding_needed = round(total_funding_needed * equity_ratio, 2)

    while True:
        net_income = random.randint(10000000, 20000000)
        if net_income > equity_funding_needed:
            break

    residual_dividend = round(net_income - equity_funding_needed, 2)
    constant_payout_dividend = round(net_income * payout_ratio, 2)
    final_dividend = min(residual_dividend, constant_payout_dividend)
    
    question = f'''{company_name} is evaluating its dividend policy for the current year. The company expects to generate {currency}{net_income} in net income and has the following financial needs:
    • Capital expenditures (CapEx): {currency}{capex}
    • Debt repayment: {currency}{debt_repayment}
    • Target capital structure: {equity_ratio*100}% equity, {(1-equity_ratio)*100}% debt

{company_name} follows a constant payout ratio policy, where the company pays {payout_ratio*100}% of its net income as dividends. However, if the residual dividend based on the capital expenditure and debt repayment needs is lower than the dividend implied by the constant payout ratio, {company_name} will pay the lower of the two amounts.

What will be the total dividend payment for {company_name} this year, considering both the constant payout ratio policy and the residual dividend policy?'''
    
    solution = f'''Step 1: Calculate total funding needed
    Total Funding = CapEx + Debt Repayment
                    = {currency}{capex} + {currency}{debt_repayment}
                    = {currency}{total_funding_needed}

Step 2: Calculate required equity funding
    Equity Funding = Total Funding × Equity Ratio
                    = {currency}{total_funding_needed} × {equity_ratio}
                    = {currency}{equity_funding_needed}

Step 3: Calculate residual dividend amount
    Residual Dividend = Net Income - Required Equity Funding
                    = {currency}{net_income} - {currency}{equity_funding_needed}
                    = {currency}{residual_dividend}

Step 4: Calculate constant payout ratio dividend
    Constant Payout Dividend = Net Income × Payout Ratio
                    = {currency}{net_income} × {payout_ratio}
                    = {currency}{constant_payout_dividend}

Step 5: Determine final dividend (lower of the two)
    Final Dividend = min(Residual Dividend, Constant Payout Dividend)
                    = min({currency}{residual_dividend}, {currency}{constant_payout_dividend})
                    = {currency}{final_dividend}'''
    
    return question, solution

def main():
    """
    Generate 10 instances of each template with different random seeds 
    and write the results to a JSON file.
    """
    import json
    # List of template functions
    templates = [
        basic_dividend_stable_policy,
        basic_residual_dividend_policy,
        intermediate_dividend_constant_payout_ratio_policy,
        intermediate_residual_dividend_policy_capex,
        advanced_hybrid_dividend_policy
    ]
    
    # List to store all generated problems
    all_problems = []
    
    # Generate 10 problems for each template
    for template_func in templates:
        id = template_func.__doc__.split(':')[0].strip()
        level = template_func.__doc__.split(':')[1].strip()
        
        for i in range(10):
            # Generate a unique seed for each problem
            seed = random.randint(1000000000, 4000000000)
            random.seed(seed)
            
            # Generate the problem and solution
            question, solution = template_func()
            
            # Create a JSON entry
            problem_entry = {
                "seed": seed,
                "id": id,
                "level": level,
                "question": question,
                "solution": solution
            }
            
            # Add to the list of problems
            all_problems.append(problem_entry)
            
            # Reset the random seed
            random.seed()
    
    random.shuffle(all_problems)
    # Write all problems to a .jsonl file
    output_file = "../../testset/corporate_finance/div_policies.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == '__main__':
    main()