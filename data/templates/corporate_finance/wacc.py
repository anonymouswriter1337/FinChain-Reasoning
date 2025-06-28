import random
from misc import companies, currencies

def basic_wacc():
    """
    1:Basic: Calculate the weighted average cost of capital.

    Generates a WACC (Weighted Average Cost of Capital) calculation problem with solution.
    This function creates a random WACC problem using a randomly selected company
    and currency. It generates random but realistic values for cost of equity,
    cost of debt, tax rate, and capital structure components.
    Returns
    -------
    tuple
        A tuple containing two strings:
        - question (str): A formatted problem statement with company details and financial data
        - solution (str): A detailed step-by-step solution showing WACC calculation
    """

    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]
    currency = random.choice(currencies)

    money_multiplier = "million" if random.choice([True, False]) else "billion"

    cost_of_equity = round(random.uniform(5, 15),2)
    cost_of_debt = round(random.uniform(3, 10), 2)
    tax_rate = round(random.uniform(20, 35), 2)
    total_equity = round(random.uniform(10, 100), 2)
    total_debt = round(random.uniform(10, 100), 2)

    question = f'''{company_name}, a leading company in the {industry} sector, has recently released its financial statements. According to the balance sheet:

        - Market value of equity (E) = {currency}{total_equity} {money_multiplier}
        - Market value of debt (D) = {currency}{total_debt} {money_multiplier}
        - Cost of equity (r_e) = {cost_of_equity}%
        - Cost of debt (r_d) = {cost_of_debt}%
        - Corporate tax rate (T) = {tax_rate}%

    Calculate the company’s Weighted Average Cost of Capital (WACC).'''

    total_capital = total_equity + total_debt
    equity_proportion = total_equity / total_capital
    debt_proportion = total_debt / total_capital
    wacc = (total_equity / total_capital) * cost_of_equity + ((total_debt / total_capital) * cost_of_debt * (1 - tax_rate / 100))

    total_capital = round(total_capital, 2)
    equity_proportion = round(equity_proportion, 2)
    debt_proportion = round(debt_proportion, 2)
    wacc = round(wacc, 2)

    solution = f'''Step 1: Calculate total capital using the formula V = E + D:
    Total Capital = Market Value of Equity (E) + Market Value of Debt (D) = {currency}{total_equity} {money_multiplier} + {currency}{total_debt} {money_multiplier} = {currency}{total_equity + total_debt} {money_multiplier}
    
    Step 2: Calculate the proportion of equity and debt in the company's capital structure:
    Proportion of Equity = E / V = {currency}{total_equity} {money_multiplier} / {currency}{total_capital} {money_multiplier} = {equity_proportion}
    Proportion of Debt = D / V = {currency}{total_debt} {money_multiplier} / {currency}{total_capital} {money_multiplier} = {debt_proportion}

    Step 3: Calculate the Weighted Average Cost of Capital (WACC) using the formula:
    WACC = (E / V) * r_e + (D / V) * r_d * (1 - T)
         = ({total_equity} / {total_capital}) * {cost_of_equity}% + (({total_debt} / {total_capital}) * {cost_of_debt}% * (1 - {tax_rate}% / 100))
         = {wacc}%'''

    return question, solution

def basic_capm_wacc():
    """
    2:Basic: Generates a WACC (Weighted Average Cost of Capital) calculation problem using CAPM (Capital Asset Pricing Model).
    The function creates a random scenario for a company with realistic financial values and calculates:
    - Cost of equity using CAPM formula
    - Capital structure weights
    - WACC incorporating tax effects
    Returns:
        tuple: A pair of strings containing:
            - question (str): Problem statement with company data and requirements
            - solution (str): Detailed step-by-step solution showing WACC calculation
    """
    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]
    currency = random.choice(currencies)

    # Simplified values with tighter ranges and rounded numbers
    market_value_equity = round(random.uniform(50, 100), 0) 
    market_value_debt = round(random.uniform(20, 50), 0)
    beta = round(random.uniform(0.9, 1.3), 1)
    risk_free_rate = round(random.uniform(2, 4), 0)
    market_return = round(random.uniform(8, 10), 0) 
    cost_of_debt = round(random.uniform(5, 8), 0)
    tax_rate = round(random.uniform(25, 30), 0)

    cost_of_equity = round(risk_free_rate + beta * (market_return - risk_free_rate), 1)
    total_capital = market_value_equity + market_value_debt

    question = f'''{company_name} has the following data:

    - Equity value = {currency}{market_value_equity} million
    - Debt value = {currency}{market_value_debt} million
    - Total capital = {currency}{total_capital} million
    - Beta = {beta}
    - Risk-free rate = {risk_free_rate}%
    - Market return = {market_return}%
    - Cost of debt = {cost_of_debt}%
    - Tax rate = {tax_rate}%

Calculate the WACC using CAPM for cost of equity.'''

    
    equity_weight = market_value_equity / total_capital
    debt_weight = market_value_debt / total_capital
    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate / 100))
    wacc = round(wacc, 1)

    solution = f'''1. Cost of equity using CAPM:
r_e = {risk_free_rate}% + {beta}({market_return}% - {risk_free_rate}%) = {cost_of_equity}%

2. Capital weights:
Equity weight = {market_value_equity}/{total_capital} = {round(equity_weight,2)}
Debt weight = {market_value_debt}/{total_capital} = {round(debt_weight,2)}

3. WACC = {round(equity_weight,2)} × {cost_of_equity}% + {round(debt_weight,2)} × {cost_of_debt}% × (1 - {tax_rate}%) = {wacc}%'''

    return question, solution

def intermediate_capm_wacc():
    """
    3:Intermediate: Calculate WACC using CAPM given all variables.

    Generates a WACC (Weighted Average Cost of Capital) problem using CAPM (Capital Asset Pricing Model).
    This function creates a randomized finance problem that:
    1. Selects a random company and currency
    2. Generates random but realistic financial metrics including:
        - Market values for equity and debt
        - Beta
        - Risk-free rate
        - Market return
        - Cost of debt
        - Tax rate
    3. Calculates the cost of equity using CAPM
    4. Determines the WACC
    Returns:
         tuple: A pair containing:
              - question (str): A formatted problem statement with the company's financial data
              - solution (str): A detailed step-by-step solution showing the WACC calculation
    """

    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]
    currency = random.choice(currencies)

    money_multiplier = "million" if random.choice([True, False]) else "billion"

    market_value_equity = round(random.uniform(50, 150), 2)
    market_value_debt = round(random.uniform(20, 80), 2)
    beta = round(random.uniform(0.8, 1.5), 2)
    risk_free_rate = round(random.uniform(2, 5), 2)
    market_return = round(random.uniform(6, 12), 2)
    cost_of_debt = round(random.uniform(5, 10), 2)
    tax_rate = round(random.uniform(20, 35), 2)

    cost_of_equity = round(risk_free_rate + beta * (market_return - risk_free_rate), 2)

    question = f'''{company_name}, a leading company in the {industry} sector, has the following financial data:

        - Market value of equity (E) = {currency}{market_value_equity} {money_multiplier}
        - Market value of debt (D) = {currency}{market_value_debt} {money_multiplier}
        - Beta (β) = {beta}
        - Risk-free rate (r_f) = {risk_free_rate}%
        - Market return (r_m) = {market_return}%
        - Cost of debt before tax (r_d) = {cost_of_debt}%
        - Corporate tax rate (T) = {tax_rate}%

    Using the Capital Asset Pricing Model (CAPM) to calculate the cost of equity (r_e), determine the Weighted Average Cost of Capital (WACC).'''

    total_capital = market_value_equity + market_value_debt
    equity_proportion = market_value_equity / total_capital
    debt_proportion = market_value_debt / total_capital
    
    # Calculate WACC with the original values (not rounded)
    wacc = (equity_proportion * cost_of_equity) + (debt_proportion * cost_of_debt * (1 - tax_rate / 100))
    
    # Round for display in the solution
    total_capital_rounded = round(total_capital, 2)
    equity_proportion_rounded = round(equity_proportion, 4)
    debt_proportion_rounded = round(debt_proportion, 4)
    wacc_rounded = round(wacc, 2)

    solution = f'''Step 1: Calculate the cost of equity (r_e) using the CAPM formula:
    r_e = r_f + β * (r_m - r_f)
        = {risk_free_rate}% + {beta} * ({market_return}% - {risk_free_rate}%)
        = {cost_of_equity}%

    Step 2: Calculate total capital using the formula V = E + D:
    Total Capital (V) = Market Value of Equity (E) + Market Value of Debt (D) = {currency}{market_value_equity} {money_multiplier} + {currency}{market_value_debt} {money_multiplier} = {currency}{total_capital_rounded} {money_multiplier}

    Step 3: Calculate the proportion of equity and debt in the company's capital structure:
    Proportion of Equity = E / V = {currency}{market_value_equity} {money_multiplier} / {currency}{total_capital_rounded} {money_multiplier} = {equity_proportion_rounded}
    Proportion of Debt = D / V = {currency}{market_value_debt} {money_multiplier} / {currency}{total_capital_rounded} {money_multiplier} = {debt_proportion_rounded}

    Step 4: Calculate the Weighted Average Cost of Capital (WACC) using the formula:
    WACC = (E / V) * r_e + (D / V) * r_d * (1 - T)
            = {equity_proportion_rounded} * {cost_of_equity}% + ({debt_proportion_rounded} * {cost_of_debt}% * (1 - {tax_rate}% / 100))
            = {wacc_rounded}%'''

    return question, solution

def intermediate_wacc():
    """
    4:Intermediate: Calculate WACC with multiple debt instruments and preferred stock.
    
    Generates a WACC problem involving common stock, preferred stock, and multiple types of debt.
    Returns:
        tuple: Contains:
            - question (str): Problem with multiple financing components
            - solution (str): Step-by-step WACC calculation
    """
    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]
    currency = random.choice(currencies)
    
    # Generate financial parameters
    common_equity = round(random.uniform(50, 150), 2)
    preferred_equity = round(random.uniform(10, 30), 2)
    lt_debt = round(random.uniform(20, 60), 2)
    st_debt = round(random.uniform(10, 30), 2)
    
    cost_of_equity = round(random.uniform(8, 16), 2)
    preferred_dividend_rate = round(random.uniform(5, 8), 2)
    lt_interest = round(random.uniform(6, 9), 2)
    st_interest = round(random.uniform(4, 7), 2)
    tax_rate = round(random.uniform(25, 35), 2)

    total_capital = common_equity + preferred_equity + lt_debt + st_debt
    
    question = f'''{company_name}, operating in the {industry} sector, has the following capital structure:

        - Common stock (E_c): {currency}{common_equity} million
        - Preferred stock (E_p): {currency}{preferred_equity} million
        - Long-term debt (D_l): {currency}{lt_debt} million at {lt_interest}%
        - Short-term debt (D_s): {currency}{st_debt} million at {st_interest}%
        
        Additional information:
        - Required return on common equity (r_e): {cost_of_equity}%
        - Preferred dividend rate (r_p): {preferred_dividend_rate}%
        - Corporate tax rate (T): {tax_rate}%
        
    Calculate the company's Weighted Average Cost of Capital (WACC).'''

    # Calculate weights
    w_common = common_equity / total_capital
    w_preferred = preferred_equity / total_capital
    w_lt_debt = lt_debt / total_capital
    w_st_debt = st_debt / total_capital
    
    # Calculate WACC
    wacc = (w_common * cost_of_equity +
            w_preferred * preferred_dividend_rate +
            w_lt_debt * lt_interest * (1 - tax_rate/100) +
            w_st_debt * st_interest * (1 - tax_rate/100))
    
    w_common = round(w_common, 2)
    w_preferred = round(w_preferred, 2)
    w_lt_debt = round(w_lt_debt, 2)
    w_st_debt = round(w_st_debt, 2)
    wacc = round(wacc, 2)
    
    solution = f'''Step 1: Calculate total capital (V)
    Total Capital = Common Equity + Preferred Equity + Long-term Debt + Short-term Debt
    V = E_c + E_p + D_l + D_s
    V = {common_equity} + {preferred_equity} + {lt_debt} + {st_debt} = {round(total_capital,2)} million

    Step 2: Calculate weights
    Weight of Common Equity (w_c) = E_c/V = {common_equity}/{round(total_capital,2)} = {w_common}
    Weight of Preferred Equity (w_p) = E_p/V = {preferred_equity}/{round(total_capital,2)} = {w_preferred}
    Weight of Long-term Debt (w_l) = D_l/V = {lt_debt}/{round(total_capital,2)} = {w_lt_debt}
    Weight of Short-term Debt (w_s) = D_s/V = {st_debt}/{round(total_capital,2)} = {w_st_debt}

    Step 3: Apply WACC formula
    WACC = (Weight of Common Equity × Cost of Common Equity) + 
           (Weight of Preferred Equity × Preferred Dividend Rate) +
           (Weight of Long-term Debt × Long-term Interest Rate × (1-Tax Rate)) +
           (Weight of Short-term Debt × Short-term Interest Rate × (1-Tax Rate))

    WACC = w_c × r_e + w_p × r_p + w_l × r_d_l × (1-T) + w_s × r_d_s × (1-T)
    where:
    - Common Equity Component = {w_common} × {cost_of_equity}% = {round(w_common * cost_of_equity, 2)}%
    - Preferred Equity Component = {w_preferred} × {preferred_dividend_rate}% = {round(w_preferred * preferred_dividend_rate, 2)}%
    - Long-term Debt Component = {w_lt_debt} × {lt_interest}% × (1 - {tax_rate}%) = {round(w_lt_debt * lt_interest * (1 - tax_rate/100), 2)}%
    - Short-term Debt Component = {w_st_debt} × {st_interest}% × (1 - {tax_rate}%) = {round(w_st_debt * st_interest * (1 - tax_rate/100), 2)}%

    Step 4: Sum all components
    WACC = {wacc}%'''

    return question, solution

def advanced_detailed_wacc():
    """
    5:Advanced: Calculate the Weighted Average Cost of Capital (WACC) by detailed calculation of intermediate steps.
    
    This function generates a comprehensive WACC calculation problem by randomly selecting:
    - A company and its industry
    - Currency
    - Common stock parameters (shares and price)
    - Preferred stock parameters (shares, price, and dividend)
    - Bond parameters (number, face value, price, coupon rate, YTM, and maturity)
    - Market parameters (beta, risk-free rate, market return, and tax rate)
    Returns:
        tuple: A pair containing:
            - question (str): Detailed problem statement with all given parameters
            - solution (str): Step-by-step solution showing calculations for:
                1. Market values of equity, preferred stock, and debt
                5. Final WACC calculation
    """
    company = random.choice(companies)
    company_name = company[0]
    industry = company[1]
    currency = random.choice(currencies)

    # Common stock parameters
    common_shares = round(random.uniform(1, 10), 2)
    common_price = round(random.uniform(20, 100), 2)
    
    # Preferred stock parameters
    preferred_shares = round(random.uniform(0.5, 2), 2)
    preferred_price = round(random.uniform(20, 50), 2)
    preferred_dividend = round(random.uniform(1, 4), 2)
    
    # Bond parameters
    num_bonds = random.randint(5000, 15000)
    face_value = 1000
    bond_price_percent = round(random.uniform(90, 98), 2)
    coupon_rate = round(random.uniform(4, 8), 2)
    ytm = round(random.uniform(5, 9), 2)
    maturity = random.randint(5, 15)
    
    # Market parameters
    beta = round(random.uniform(0.8, 1.8), 2)
    risk_free_rate = round(random.uniform(2, 5), 2)
    market_return = round(random.uniform(8, 12), 2)
    tax_rate = round(random.uniform(20, 35), 2)

    question = f'''{company_name}, a leading company in the {industry} sector, has the following financial information:

        Common stock:
        • {common_shares} million shares outstanding, trading at {currency}{common_price} per share
        
        Preferred stock:
        • {preferred_shares} million shares outstanding, trading at {currency}{preferred_price} per share
        • Dividend of {currency}{preferred_dividend} per share
        
        Bonds:
        • {num_bonds} bonds outstanding, each with face value of {currency}{face_value}
        • Trading at {bond_price_percent}% of face value
        • Coupon rate: {coupon_rate}% (paid annually)
        • Time to maturity: {maturity} years
        • Yield to Maturity (YTM): {ytm}%
        
        Other information:
        • Corporate tax rate (T): {tax_rate}%
        • Risk-free rate (r_f): {risk_free_rate}%
        • Market return (r_m): {market_return}%
        • Beta (β): {beta}

    Calculate:
    1. Market value of equity, preferred stock, and debt
    2. Cost of equity using CAPM
    3. Cost of preferred stock
    4. After-tax cost of debt
    5. Weighted Average Cost of Capital (WACC)'''

    # Calculations
    market_value_common = common_shares * common_price
    market_value_preferred = preferred_shares * preferred_price
    market_value_debt = (num_bonds * face_value * bond_price_percent / 100) / 1000000  # Convert to millions
    total_capital = market_value_common + market_value_preferred + market_value_debt

    cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
    cost_of_preferred = (preferred_dividend / preferred_price) * 100
    cost_of_debt = ytm * (1 - tax_rate/100)

    wacc = ((market_value_common/total_capital) * cost_of_equity + 
            (market_value_preferred/total_capital) * cost_of_preferred +
            (market_value_debt/total_capital) * cost_of_debt)
    
    market_value_common = round(market_value_common, 2)
    market_value_preferred = round(market_value_preferred, 2)
    market_value_debt = round(market_value_debt, 2)
    total_capital = round(total_capital, 2)

    cost_of_equity = round(cost_of_equity, 2)
    cost_of_preferred = round(cost_of_preferred, 2)
    cost_of_debt = round(cost_of_debt, 2)
    
    wacc = round(wacc, 2)

    solution = f'''Step 1: Calculate market values
    Common Equity (E) = {common_shares}M × {currency}{common_price} = {currency}{market_value_common}M
    Preferred Equity (P) = {preferred_shares}M × {currency}{preferred_price} = {currency}{market_value_preferred}M
    Debt (D) = ({num_bonds} × {currency}{face_value} × {bond_price_percent}%) / 1,000,000 = {currency}{market_value_debt}M
    Total Capital (V) = E + P + D = {currency}{total_capital}M

    Step 2: Calculate cost of equity using CAPM
    r_e = r_f + β(r_m - r_f)
    r_e = {risk_free_rate}% + {beta}({market_return}% - {risk_free_rate}%) = {cost_of_equity}%

    Step 3: Calculate cost of preferred stock
    r_p = (Dividend per share / Market price per share) × 100
    r_p = ({preferred_dividend} / {preferred_price}) × 100 = {cost_of_preferred}%

    Step 4: Calculate after-tax cost of debt
    r_d(1-T) = {ytm}% × (1 - {tax_rate/100}) = {cost_of_debt}%

    Step 5: Calculate WACC
    WACC = (E/V × r_e) + (P/V × r_p) + (D/V × r_d(1-T))
    WACC = ({market_value_common}/{total_capital} × {cost_of_equity}%) + 
            ({market_value_preferred}/{total_capital} × {cost_of_preferred}%) + 
            ({market_value_debt}/{total_capital} × {cost_of_debt}%)
    WACC = {wacc}%'''

    return question, solution

def main():
    """
    Generate 10 instances of each template with different random seeds 
    and write the results to a JSON file.
    """
    import json
    # List of template functions
    templates = [
        basic_wacc,
        basic_capm_wacc,
        intermediate_capm_wacc,
        intermediate_wacc,
        advanced_detailed_wacc
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
    output_file = "../../testset/corporate_finance/wacc.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == '__main__':
    main()