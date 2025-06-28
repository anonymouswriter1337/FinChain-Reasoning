import random
from misc import companies, currencies

def basic_eps():
    """
    1:Basic: Calculate earnings per share given the net income, and the number of shares.

    Generate a basic Earnings Per Share (EPS) calculation problem and its solution.
    The function creates a randomized EPS problem using:
    - A randomly selected currency and company (with its industry)
    - Random values for net income, and outstanding shares
    Returns:
        tuple: Contains two elements:
            - question (str): A formatted problem statement with company details and financial information
            - solution (str): A step-by-step solution showing the EPS calculation
    """

    currency = random.choice(currencies)
    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]

    net_income = random.randint(100000, 10000000)
    shares = random.randint(100000, 1000000)

    question = f'''{company_name}, operating in the {industry} industry, reported the following financial information:

        - Net Income: {currency}{net_income}
        - Common Shares Outstanding: {shares}

    Calculate the company's Earnings Per Share (EPS).'''

    eps = net_income / shares

    solution = f'''Step 1: Apply the EPS formula: Net Income / Number of Common Shares Outstanding
    EPS = {currency}{net_income} / {shares}
        = {currency}{eps} per share'''

    return question, solution

def basic_eps_preferred_dividends():
    """
    2:Basic: Calculate earnings per share given the net income, preferred dividends and the number of shares.

    Generate a basic Earnings Per Share (EPS) calculation problem and its solution.
    The function creates a randomized EPS problem using:
    - A randomly selected currency and company (with its industry)
    - Random values for net income, preferred dividends, and outstanding shares
    Returns:
        tuple: Contains two elements:
            - question (str): A formatted problem statement with company details and financial information
            - solution (str): A step-by-step solution showing the EPS calculation
    """

    currency = random.choice(currencies)
    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]

    net_income = random.randint(100000, 10000000)
    preferred_dividends = round(random.uniform(10000, 100000), 2)
    shares = random.randint(100000, 1000000)

    question = f'''{company_name}, operating in the {industry} industry, reported the following financial information:

        - Net Income: {currency}{net_income}
        - Preferred Dividends: {currency}{preferred_dividends}
        - Common Shares Outstanding: {shares}

    Calculate the company's Earnings Per Share (EPS).'''

    eps = (net_income - preferred_dividends) / shares

    solution = f'''Step 1: Apply the EPS formula: (Net Income - Preferred Dividends) / Number of Common Shares Outstanding
    EPS = ({currency}{net_income} - {currency}{preferred_dividends}) / {shares}
        = {currency}{eps} per share'''

    return question, solution


def intermediate_weighted_eps():
    """
    3:Intermediate: Calculate the weighted Earnings Per Share (EPS) for a randomly selected company.
    
    This function generates a financial scenario with random values for:
    - A company and its industry
    - Net income
    - Preferred dividends 
    - Initial number of shares outstanding
    - Number of shares repurchased during the year
    - Month when share repurchase occurred
    The function calculates the weighted average shares outstanding based on the timing
    of share repurchases and computes the EPS using:
    EPS = (Net Income - Preferred Dividends) / Weighted Average Shares
    Returns:
        tuple: A pair containing:
            - question (str): Problem statement with the financial scenario
            - solution (str): Detailed step-by-step solution showing the EPS calculation
    """

    currency = random.choice(currencies)
    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]

    net_income = random.randint(500000, 5000000)
    preferred_dividends = round(random.uniform(50000, 200000), 2)
    initial_shares = random.randint(200000, 1000000)
    repurchased_shares = random.randint(20000, 100000)
    repurchase_month = random.randint(1, 11)

    question = f'''{company_name}, operating in the {industry} industry, has the following financial details for the year:

        - Net Income: {currency}{net_income}
        - Preferred Dividends: {currency}{preferred_dividends}
        - Common Shares Outstanding at the Beginning of the Year: {initial_shares}
        - The Company Repurchased {repurchased_shares} Shares in the {repurchase_month}th month of the year

    Calculate the Earnings Per Share (EPS) for the year.'''

    weighted_shares = initial_shares * (repurchase_month/12) + (initial_shares - repurchased_shares) * ((12-repurchase_month)/12)
    eps = (net_income - preferred_dividends) / weighted_shares

    weighted_shares = round(weighted_shares, 2)
    eps = round(eps, 2)

    solution = f'''Step 1: Calculate the weighted average number of shares outstanding:
    - Shares for first {repurchase_month} months: {initial_shares}
    - Shares for remaining {12-repurchase_month} months: {initial_shares} - {repurchased_shares} = {initial_shares-repurchased_shares}
    Weighted Average Shares = {initial_shares} × ({repurchase_month}/12) + {initial_shares-repurchased_shares} × ({12-repurchase_month}/12) = {weighted_shares}

    Step 2: Apply the EPS formula: (Net Income - Preferred Dividends) / Weighted Average Shares
    EPS = ({currency}{net_income} - {currency}{preferred_dividends}) / {weighted_shares}
        = {currency}{eps} per share'''

    return question, solution

def intermediate_diluted_eps():
    """
    4:Intermediate: Calculate diluted Earnings Per Share (EPS) with convertible securities.
    
    This function generates a financial scenario with random values for:
    - Net income and preferred dividends
    - Common shares outstanding
    - Convertible bonds with tax-deductible interest
    - Convertible preferred shares
    Returns:
        tuple: Contains:
            - question (str): Problem with company financials and convertible securities
            - solution (str): Step-by-step solution showing diluted EPS calculation
    """
    
    currency = random.choice(currencies)
    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]

    net_income = random.randint(1000000, 8000000)
    shares = random.randint(200000, 800000)
    preferred_dividends = round(random.uniform(50000, 150000), 2)
    
    convertible_bonds = random.randint(5000, 15000)
    bond_interest_rate = round(random.uniform(5, 8), 2)
    conversion_ratio = random.randint(20, 40)
    tax_rate = 0.30
    
    convertible_preferred = random.randint(10000, 30000)
    preferred_dividend_rate = round(random.uniform(4, 7), 2)
    preferred_conversion_ratio = random.randint(2, 5)

    question = f'''{company_name}, operating in the {industry} industry, reports:

        - Net Income: {currency}{net_income}
        - Common Shares Outstanding: {shares}
        - Preferred Dividends: {currency}{preferred_dividends}
        
        Convertible Securities:
        - {convertible_bonds} bonds at {bond_interest_rate}% interest (tax rate: 30%)
            Each bond converts to {conversion_ratio} common shares
        - {convertible_preferred} preferred shares paying {preferred_dividend_rate}%
            Each preferred share converts to {preferred_conversion_ratio} common shares

    Calculate the Basic and Diluted EPS.'''

    bond_interest = convertible_bonds * (bond_interest_rate/100)
    tax_savings = bond_interest * tax_rate
    net_bond_effect = bond_interest - tax_savings
    
    preferred_adjustment = convertible_preferred * (preferred_dividend_rate/100)
    
    basic_eps = (net_income - preferred_dividends) / shares
    
    diluted_shares = shares + (convertible_bonds * conversion_ratio) + (convertible_preferred * preferred_conversion_ratio)
    diluted_eps = (net_income - preferred_dividends + net_bond_effect + preferred_adjustment) / diluted_shares
    
    basic_eps = round(basic_eps, 2)
    diluted_eps = round(diluted_eps, 2)

    solution = f'''Step 1: Calculate Basic EPS
    Basic EPS = (Net Income - Preferred Dividends) / Shares
                = ({currency}{net_income} - {currency}{preferred_dividends}) / {shares}
                = {currency}{basic_eps}

    Step 2: Adjust for convertible securities
    - Additional shares from bonds: {convertible_bonds} × {conversion_ratio} = {convertible_bonds * conversion_ratio}
    - Additional shares from preferred: {convertible_preferred} × {preferred_conversion_ratio} = {convertible_preferred * preferred_conversion_ratio}
    - Net bond interest after tax: {currency}{round(net_bond_effect, 2)}
    - Preferred dividends saved: {currency}{round(preferred_adjustment, 2)}

    Step 3: Calculate Diluted EPS
    Diluted EPS = {currency}{diluted_eps}'''

    return question, solution

def advanced_eps_stock_splits():
    """
    5:Advanced: Calculate EPS considering stock splits, reverse splits, and complex securities.
    
    Generates an EPS problem involving:
    - Stock splits and reverse splits during the year
    - Convertible securities with contingent conditions
    - Multiple classes of preferred stock
    Returns:
        tuple: Contains:
            - question (str): Complex EPS scenario with stock splits
            - solution (str): Detailed step-by-step solution
    """
    
    currency = random.choice(currencies)
    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]

    # Financial data
    net_income = random.randint(2000000, 10000000)
    initial_shares = random.randint(1000000, 3000000)
    
    # Stock split data
    split_ratio = random.choice([2, 3, 4])
    split_month = random.randint(3, 9)
    
    # Preferred stock data
    class_a_preferred = random.randint(20000, 50000)
    class_b_preferred = random.randint(10000, 30000)
    class_a_rate = round(random.uniform(5, 8), 2)
    class_b_rate = round(random.uniform(6, 9), 2)
    
    # Convertible bonds
    bonds = random.randint(2000, 5000)
    conversion_ratio = random.randint(20, 40)
    bond_rate = round(random.uniform(4, 7), 2)
    tax_rate = 0.30

    question = f'''{company_name}, operating in the {industry} industry, has the following financial scenario:

        Financial Information:
        - Net Income: {currency}{net_income}
        - Initial Common Shares: {initial_shares}
        - {split_ratio}-for-1 stock split in month {split_month}
        
        Preferred Stock:
        - Class A: {class_a_preferred} shares at {class_a_rate}% dividend rate
        - Class B: {class_b_preferred} shares at {class_b_rate}% dividend rate
        
        Convertible Securities:
        - {bonds} Convertible Bonds at {bond_rate}% interest
        - Each bond convertible to {conversion_ratio} common shares
        
        Calculate the Basic and Diluted EPS, adjusting for the stock split.'''

    # Calculations
    adjusted_shares = initial_shares * (split_month/12) + (initial_shares * split_ratio) * ((12-split_month)/12)
    preferred_dividends = (class_a_preferred * class_a_rate/100) + (class_b_preferred * class_b_rate/100)
    bond_interest_savings = bonds * (bond_rate/100) * (1 - tax_rate)
    adjusted_shares = round(adjusted_shares, 2)
    preferred_dividends = round(preferred_dividends, 2)
    bond_interest_savings = round(bond_interest_savings, 2)
    
    basic_eps = round((net_income - preferred_dividends) / adjusted_shares, 2)
    diluted_shares = adjusted_shares + (bonds * conversion_ratio)
    diluted_eps = round((net_income - preferred_dividends + bond_interest_savings) / diluted_shares, 2)
    diluted_shares = round(diluted_shares, 2)

    solution = f'''Step 1: Calculate split-adjusted weighted average shares
    - Pre-split ({split_month} months): {initial_shares} × ({split_month}/12)
    - Post-split ({12-split_month} months): {initial_shares * split_ratio} × ({12-split_month}/12)
    Adjusted Weighted Shares = {adjusted_shares}

    Step 2: Calculate preferred dividends
    - Class A: {currency}{round(class_a_preferred * class_a_rate/100, 2)}
    - Class B: {currency}{round(class_b_preferred * class_b_rate/100, 2)}
    Total Preferred Dividends = {currency}{preferred_dividends}

    Step 3: Calculate Basic EPS
    Basic EPS = ({currency}{net_income} - {currency}{preferred_dividends}) / {adjusted_shares}
                = {currency}{basic_eps}

    Step 4: Calculate Diluted EPS with convertible bonds
    Diluted EPS = {currency}{diluted_eps}'''

    return question, solution

def main():
    """
    Generate 10 instances of each template with different random seeds 
    and write the results to a JSON file.
    """
    import json
    # List of template functions
    templates = [
        basic_eps,
        basic_eps_preferred_dividends,
        intermediate_weighted_eps,
        intermediate_diluted_eps,
        advanced_eps_stock_splits
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
    output_file = "../../testset/corporate_finance/eps.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == '__main__':
    main()