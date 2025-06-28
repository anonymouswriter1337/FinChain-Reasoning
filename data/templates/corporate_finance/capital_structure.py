import random
from misc import companies, currencies

def basic_debt_to_equity_ratio():
    """
    1:Basic: Calculate the debt-to-equity ratio given the total debt and total equity for a company.

    Generate a question and solution for calculating a company's debt-to-equity ratio.
    This function randomly selects a company, currency and monetary values to create a
    financial analysis problem involving debt-to-equity ratio calculation.

    The debt-to-equity ratio is calculated as (Total Liabilities / Shareholders' Equity)

    Returns:
        tuple: A pair of strings containing:
            - question (str): A word problem describing a company's debt and equity values
            - solution (str): Step-by-step solution showing debt-to-equity ratio calculation
    """

    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]
    currency = random.choice(currencies)


    money_multiplier = "million" if random.choice([True, False]) else "billion"

    total_debt = random.randint(100, 100000)/100
    total_equity = random.randint(100, 100000)/100

    question = f'''{company_name}, a leading company in the {industry} sector, has recently released its financial statements. According to the balance sheet:

        - The company's total liabilities (debt) amount to {currency}{total_debt} {money_multiplier}.
        - The total shareholder equity is valued at {currency}{total_equity} {money_multiplier}.
        Based on this information, calculate the debt-to-equity ratio for {company_name}.'''

    debt_to_equity_ratio = total_debt / total_equity
    solution = f'''Step 1: Calculate the Debt-to-Equity Ratio using the formula:\nDebt-to-Equity Ratio = Total Liabilities / Shareholders' Equity\n                      = {total_debt} / {total_equity} = {debt_to_equity_ratio:.2f}'''

    return question, solution

def basic_debt_to_equity_ratio_different_multiplier():
    """
    2:Basic: Calculate debt-to-equity ratio given the total debt and total equity but in different units.

    Generate a question and solution for calculating debt-to-equity ratio with different unit multipliers.
    This function creates a financial problem involving debt-to-equity ratio calculation where
    the debt and equity values are given in different units (millions vs billions). It randomly 
    selects a company, currency, and generates appropriate financial values.
    Returns:
        tuple: A tuple containing two strings:
            - question (str): A word problem describing a company's financial situation with 
              debt and equity values in different units
            - solution (str): Detailed step-by-step solution showing:
                1. Unit conversion to make values comparable
                2. Final debt-to-equity ratio calculation
    """

    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]
    currency = random.choice(currencies)


    money_multipliers = ["milllion", "billion"]
    mm_1 = random.choice(money_multipliers)
    mm_2 = "billion" if mm_1 == "million" else "million"

    total_debt = random.randint(10, 1000000)/100
    total_equity = random.randint(10, 1000000)/100

    question = f'''{company_name}, a leading company in the {industry} sector, has recently released its financial statements. According to the balance sheet:

        - The company's total liabilities (debt) amount to {currency}{total_debt} {mm_1}.
        - The total shareholder equity is valued at {currency}{total_equity} {mm_2}.
        Based on this information, calculate the debt-to-equity ratio for {company_name}.'''

    debt_to_equity_ratio = (total_debt * (1000 if mm_1 == "billion" else 1)) / (total_equity *(1000 if mm_2 == "billion" else 1))
    
    solution_step_1 = f'''Step 1: Convert the total liabilities to the same unit as the total equity. Since the total liabilities are in {mm_1} {currency} and the total equity is in {mm_2} {currency}, we need to convert the {"total liabilities" if mm_1 == "billion" else "total debt"} to million {currency}:
        Total Debt = {currency}{total_debt} {mm_1} {f"= {currency}{total_debt * 1000} million" if mm_1 == "billion" else ""}
        Total Equity = {currency}{total_equity} {mm_2} {f"= {currency}{total_equity * 1000} million" if mm_2 == "billion" else ""}'''
        
    solution_step_2 = f'''Step 2: Calculate the Debt-to-Equity Ratio using the formula:\nDebt-to-Equity Ratio = Total Liabilities / Shareholders' Equity\n                      = {currency}{total_debt} {mm_1}/ {currency}{total_equity} {mm_2} = {debt_to_equity_ratio:.2f}'''

    solution = f"{solution_step_1}\n\n{solution_step_2}"

    return question, solution

def intermediate_debt_to_equity_ratio_equity_long_short_debt():
    """
    3:Intermediate: Calculate debt-to-equity ratio given long / short term debt, shareholder equity and treasury stock.

    Generates a financial problem and solution for calculating debt-to-equity ratio.
    The function creates a scenario for a randomly selected company with given:
    - Long-term debt
    - Short-term debt
    - Shareholder equity
    The function calculates the debt-to-equity ratio using the formula:
    Debt-to-Equity Ratio = Total Debt / Shareholder Equity
    where Total Debt = Long-term Debt + Short-term Debt
    Returns:
        tuple: A tuple containing two strings:
            - question (str): A problem statement with the company's financial data
            - solution (str): A detailed step-by-step solution showing the calculation
                             of the debt-to-equity ratio
    """

    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]
    currency = random.choice(currencies)


    money_multiplier = "million" if random.choice([True, False]) else "billion"

    long_debt = random.randint(10000, 10000000)/100
    short_debt = random.randint(100, 100000)/100
    shareholder_equity = random.randint(10000, 10000000)/100

    question = f'''{company_name}, a leading company in the {industry} sector, has recently released its financial statements. According to the balance sheet:
    
        - The company's long-term debt amounts to {currency}{long_debt} {money_multiplier}.
        - The company's short-term debt amounts to {currency}{short_debt} {money_multiplier}.
        - The total shareholder equity is valued at {currency}{shareholder_equity} {money_multiplier}.
        Based on this information, calculate the debt-to-equity ratio for {company_name}.'''

    total_debt = long_debt + short_debt

    debt_to_equity_ratio = total_debt / shareholder_equity

    solution = f'''Step 1: Calculate the Total Debt and Total Equity:
        Total Debt = Long-term Debt + Short-term Debt = {currency}{long_debt} {money_multiplier} + {currency}{short_debt} {money_multiplier} = {currency}{total_debt} {money_multiplier}
        Total Equity = Shareholder Equity = {currency}{shareholder_equity} {money_multiplier}
        Step 2: Calculate the Debt-to-Equity Ratio using the formula:
        Debt-to-Equity Ratio = Total Debt / Total Equity
        Debt-to-Equity Ratio = {currency}{total_debt} {money_multiplier} / {currency}{shareholder_equity} {money_multiplier} = {debt_to_equity_ratio:.2f}'''

    return question, solution

def intermediate_debt_to_equity_ratio_treasury_stock_equity_long_short_debt():
    """
    4:Intermediate: Calculate debt-to-equity ratio given long / short term debt, shareholder equity and treasury stock.

    Generates a financial problem and solution for calculating debt-to-equity ratio considering treasury stock.
    The function creates a scenario for a randomly selected company with given:
    - Long-term debt
    - Short-term debt
    - Shareholder equity
    - Treasury stock
    The function calculates the debt-to-equity ratio using the formula:
    Debt-to-Equity Ratio = Total Debt / (Shareholder Equity - Treasury Stock)
    where Total Debt = Long-term Debt + Short-term Debt
    Returns:
        tuple: A tuple containing two strings:
            - question (str): A problem statement with the company's financial data
            - solution (str): A detailed step-by-step solution showing the calculation
                             of the debt-to-equity ratio
    """

    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]
    currency = random.choice(currencies)


    money_multiplier = "million" if random.choice([True, False]) else "billion"

    long_debt = random.randint(10000, 10000000)/100
    short_debt = random.randint(100, 100000)/100
    shareholder_equity = random.randint(10000, 10000000)/100
    treasury_stock = random.randint(100, 100000)/100

    while treasury_stock > shareholder_equity:
        treasury_stock = random.randint(100, 100000)/100

    question = f'''{company_name}, a leading company in the {industry} sector, has recently released its financial statements. According to the balance sheet:
    
        - The company's long-term debt amounts to {currency}{long_debt} {money_multiplier}.
        - The company's short-term debt amounts to {currency}{short_debt} {money_multiplier}.
        - The total shareholder equity is valued at {currency}{shareholder_equity} {money_multiplier}.
        - The company has {currency}{treasury_stock} {money_multiplier} in treasury stock.
        Based on this information, calculate the debt-to-equity ratio for {company_name}.'''

    total_debt = long_debt + short_debt
    total_equity = shareholder_equity - treasury_stock

    debt_to_equity_ratio = total_debt / total_equity

    solution = f'''Step 1: Calculate the Total Debt and Total Equity:
        Total Debt = Long-term Debt + Short-term Debt = {currency}{long_debt} {money_multiplier} + {currency}{short_debt} {money_multiplier} = {currency}{total_debt} {money_multiplier}
        Total Equity = Shareholder Equity - Treasury Stock = {currency}{shareholder_equity} {money_multiplier} - {currency}{treasury_stock} {money_multiplier} = {currency}{total_equity} {money_multiplier}
        Step 2: Calculate the Debt-to-Equity Ratio using the formula:
        Debt-to-Equity Ratio = Total Debt / Total Equity
        Debt-to-Equity Ratio = {currency}{total_debt} {money_multiplier} / {currency}{total_equity} {money_multiplier} = {debt_to_equity_ratio:.2f}'''

    return question, solution

def advanced_debt_to_equity_ratio_target():
    """
    5:Advanced: Calculate the change in debt required to reach a target debt-to-equity ratio.

    Generate a question and solution about adjusting a company's debt-to-equity ratio to meet a target ratio.
    The function creates a scenario where a randomly selected company needs to either decrease debt
    or increase equity to achieve a desired debt-to-equity ratio. It provides step-by-step calculations
    for the required changes.
    Returns:
        tuple: A tuple containing two strings:
            - question (str): A problem statement with company details and financial information
            - solution (str): Detailed step-by-step solution showing calculations and required changes
    """


    
    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]
    currency = random.choice(currencies)

    money_multiplier = "million" if random.choice([True, False]) else "billion"

    total_debt = random.randint(100, 100000) / 100
    total_equity = random.randint(100, 100000) / 100
    target_ratio = random.uniform(0.5, 2.0)

    question = f'''{company_name}, a leading company in the {industry} sector, has recently released its financial statements. According to the balance sheet:

        - The company's total liabilities (debt) amount to {currency}{total_debt} {money_multiplier}.
        - The total shareholder equity is valued at {currency}{total_equity} {money_multiplier}.
        - The target debt-to-equity ratio for the company is {target_ratio:.2f}.
        Based on this information, determine whether the company needs to decrease its debt or increase its equity to reach the target debt-to-equity ratio. Calculate the required change.'''

    current_ratio = total_debt / total_equity
    if current_ratio > target_ratio:
        required_debt = target_ratio * total_equity
        change_needed = total_debt - required_debt
        solution = f'''Step 1: Calculate the current Debt-to-Equity Ratio:
        Current Debt-to-Equity Ratio = Total Liabilities / Shareholders' Equity = {currency}{total_debt} {money_multiplier} / {currency}{total_equity} {money_multiplier} = {current_ratio:.2f}

        Step 2: Since the current ratio ({current_ratio:.2f}) is higher than the target ratio ({target_ratio:.2f}), the company needs to decrease its debt.

        Step 3: Calculate the required debt to achieve the target ratio:
        Required Debt = Target Ratio * Total Equity = {target_ratio:.2f} * {currency}{total_equity} {money_multiplier} = {currency}{required_debt:.2f} {money_multiplier}

        Step 4: Calculate the change needed:
        Change Needed = Total Debt - Required Debt = {currency}{total_debt} {money_multiplier} - {currency}{required_debt:.2f} {money_multiplier} = {currency}{change_needed:.2f} {money_multiplier}

        Therefore, the company needs to decrease its debt by {currency}{change_needed:.2f} {money_multiplier} to reach the target debt-to-equity ratio of {target_ratio:.2f}.'''
    else:
        required_equity = total_debt / target_ratio
        change_needed = required_equity - total_equity
        solution = f'''Step 1: Calculate the current Debt-to-Equity Ratio:
        Current Debt-to-Equity Ratio = Total Liabilities / Shareholders' Equity = {currency}{total_debt} {money_multiplier} / {currency}{total_equity} {money_multiplier} = {current_ratio:.2f}

        Step 2: Since the current ratio ({current_ratio:.2f}) is lower than the target ratio ({target_ratio:.2f}), the company needs to increase its equity.

        Step 3: Calculate the required equity to achieve the target ratio:
        Required Equity = Total Debt / Target Ratio = {currency}{total_debt} {money_multiplier} / {target_ratio:.2f} = {currency}{required_equity:.2f} {money_multiplier}

        Step 4: Calculate the change needed:
        Change Needed = Required Equity - Total Equity = {currency}{required_equity:.2f} {money_multiplier} - {currency}{total_equity} {money_multiplier} = {currency}{change_needed:.2f} {money_multiplier}

        Therefore, the company needs to increase its equity by {currency}{change_needed:.2f} {money_multiplier} to reach the target debt-to-equity ratio of {target_ratio:.2f}.'''

    return question, solution

def main():
    """
    Generate 10 instances of each template with different random seeds 
    and write the results to a JSON file.
    """
    import json
    # List of template functions
    templates = [
        basic_debt_to_equity_ratio,
        basic_debt_to_equity_ratio_different_multiplier,
        intermediate_debt_to_equity_ratio_equity_long_short_debt,
        intermediate_debt_to_equity_ratio_treasury_stock_equity_long_short_debt,
        advanced_debt_to_equity_ratio_target
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
    output_file = "../../testset/corporate_finance/capital_structure.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == '__main__':
    main()