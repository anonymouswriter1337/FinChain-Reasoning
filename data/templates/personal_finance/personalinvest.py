import random

# Named entities for companies and industries
company_names = ["Tesla Inc.", "Apple Inc.", "Amazon.com", "SpaceX", "Google LLC"]
industry_names = ["automotive", "technology", "e-commerce", "aerospace", "internet services"]

# Template 1
def template_investment_growth_simple():
    """1:Basic: Generates a simple compound interest problem where an investor calculates 
    the future value of their investment based on an initial amount, annual growth rate, 
    and time period."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    initial_investment = round(random.randint(100000, 500000),2)  # Initial investment
    annual_growth_rate = round(random.uniform(5.0, 15.0), 2)  # Annual growth rate (%)
    years = random.randint(5, 15)  # Number of years

    # Step 1:
    future_value = round((initial_investment * (1 + annual_growth_rate / 100) ** years),2)
    
    # Question
    question = (
        f"{person_name} invests ${initial_investment} in a stock that is expected to grow by {annual_growth_rate:.2f}% annually, compounded annually. "
        f"How much will their investment be worth after {years} years?"
    )
    
    # Solution
    solution = (
        f"Step 1: Calculate the future value of the investment using the formula for compound interest:\n"
        f"  Future Value = Initial Investment × (1 + Annual Growth Rate) ^ Years\n"
        f"  Future Value = ${initial_investment} × (1 + {annual_growth_rate / 100:.4f})^{years}\n"
        f"  Future Value = ${future_value:.2f}\n\n"
        f"This formula assumes the growth rate is compounded annually."
    )
    
    return question, solution

# Template 2
def template_investment_portfolio_allocation():
    """2:Basic: Creates a portfolio allocation problem where an investor must 
    calculate how to distribute their total investment across different asset classes 
    (stocks, bonds, and real estate) based on specified percentages."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    age = random.randint(25, 55)
    total_investment = round(random.randint(1000000, 5000000),2)  # Total amount to invest
    stock_allocation = round(random.uniform(40.0, 60.0), 2)  # Percentage allocated to stocks
    bond_allocation = round(random.uniform(20.0, 40.0), 2)  # Percentage allocated to bonds
    real_estate_allocation = round(100 - stock_allocation - bond_allocation, 2)  # Ensure allocations sum to 100%

    # Step 1: Investment amounts for each category
    stock_investment = round(total_investment * (stock_allocation / 100),2)
    bond_investment = round(total_investment * (bond_allocation / 100),2)
    real_estate_investment = round(total_investment * (real_estate_allocation / 100),2)
    
    # Question
    question = (
        f"{person_name}, aged {age}, has ${total_investment} to invest. They decide to allocate {stock_allocation:.2f}% to stocks, "
        f"{bond_allocation:.2f}% to bonds, and the remaining {real_estate_allocation:.2f}% to real estate, ensuring the total allocation is 100%. "
        f"How much will be allocated to each investment category?"
    )
    
    # Solution
    solution = (
        f"Step 1: Calculate the amount allocated to stocks, bonds, and real estate:\n"
        f"  Stocks = ${total_investment} × {stock_allocation:.2f}% = ${stock_investment:.2f}\n"
        f"  Bonds = ${total_investment} × {bond_allocation:.2f}% = ${bond_investment:.2f}\n"
        f"  Real Estate = ${total_investment} × {real_estate_allocation:.2f}% = ${real_estate_investment:.2f}"
    )
    
    return question, solution

# Template 3
def template_investment_market_volatility():
    """3:Intermediate: Generates a market volatility scenario where an investor must calculate 
    potential upper and lower bounds of their portfolio value based on expected growth 
    rates and market fluctuations."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    current_age = random.randint(30, 50)
    current_investment = round(random.randint(500000, 3000000),2)  # Current stock investment
    growth_rate = round(random.uniform(5.0, 20.0), 2)  # Past year's growth (%)
    volatility = round(random.uniform(5.0, 15.0), 2)  # Expected fluctuation (%)
    
    # Step 1: Calculate future value after growth
    future_value_growth = round(current_investment * (1 + growth_rate / 100),2)
    
    # Step 2: Calculate upper and lower bounds based on volatility
    future_value_upper = round(future_value_growth * (1 + volatility / 100),2)
    future_value_lower = round(future_value_growth * (1 - volatility / 100),2)
    
    # Question
    question = (
        f"{person_name}, aged {current_age}, has ${current_investment} invested in stocks, which has grown by {growth_rate:.2f}% over the past year. "
        f"Due to market volatility, they expect their portfolio to fluctuate by ±{volatility:.2f}% based on the future value of their investment. "
        f"What is the potential range of their portfolio value by the end of the next year?"
    )
    
    # Solution
    solution = (
        f"Step 1: Calculate the future value of the investment after growth:\n"
        f"  Future Value (Growth) = ${current_investment} × (1 + {growth_rate:.2f}%) = ${future_value_growth:.2f}\n\n"
        f"Step 2: Calculate the upper bound of the portfolio value with volatility:\n"
        f"  Upper Bound = ${future_value_growth:.2f} × (1 + {volatility:.2f}%) = ${future_value_upper:.2f}\n\n"
        f"Step 3: Calculate the lower bound of the portfolio value with volatility:\n"
        f"  Lower Bound = ${future_value_growth:.2f} × (1 - {volatility:.2f}%) = ${future_value_lower:.2f}"
    )
    
    return question, solution


# Template 4
def template_investment_withdrawals():
    """4:Intermediate: Creates a retirement planning scenario where an investor must calculate 
    their remaining investment balance after regular withdrawals, considering both 
    annual growth and periodic distributions."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    current_age = random.randint(40, 60)
    initial_investment = round(random.randint(1000000, 5000000),2)  # Initial investment
    annual_growth_rate = round(random.uniform(4.0, 8.0), 2)  # Annual growth rate (%)
    annual_withdrawal = round(random.randint(100000, 500000),2) # Annual withdrawal
    years_of_withdrawal = random.randint(5, 20)  # Number of years for withdrawals
    
    # Calculate future value after each year of growth and withdrawal
    future_value = initial_investment
    for i in range(1, years_of_withdrawal + 1):
        future_value = round((future_value * (1 + annual_growth_rate / 100) - annual_withdrawal),2)
    
    # Question
    question = (
        f"{person_name}, aged {current_age}, has ${initial_investment} invested in a mutual fund that grows at an annual rate of {annual_growth_rate:.2f}%. "
        f"They plan to withdraw ${annual_withdrawal} annually for the next {years_of_withdrawal} years. "
        f"How much will they have left in their investment account after {years_of_withdrawal} years?"
    )
    
    # Solution
    solution = (
        f"Step 1: Calculate the remaining balance after {years_of_withdrawal} years of withdrawals and compounding growth:\n"
        f"  Starting with an initial investment of ${initial_investment}, apply the annual growth rate of {annual_growth_rate:.2f}%.\n"
        f"  After each year's growth, subtract the withdrawal of ${annual_withdrawal}.\n"
        f"  After {years_of_withdrawal} years, the remaining value will be ${future_value:.2f}."
    )
    
    return question, solution

# Template 5
def template_investment_diversification():
    """5:Advanced: Generates a portfolio diversification problem that analyzes the impact 
    of adding a new asset class to an existing portfolio, considering correlation 
    coefficients and overall risk assessment."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    current_age = random.randint(35, 55)
    stocks_investment = round(random.randint(1000000, 5000000),2)  # Stocks investment
    bonds_investment = round(random.randint(500000, 3000000),2)  # Bonds investment
    real_estate_investment = round(random.randint(1000000, 5000000),2) # Real estate investment
    new_investment = round(random.randint(500000, 2000000),2)  # New asset class investment
    correlation_with_existing_portfolio = round(random.uniform(-0.5, 0.5), 2)  # Correlation with existing portfolio
    
    total_existing_investment = round((stocks_investment + bonds_investment + real_estate_investment),2)
    total_investment_with_new_asset = round((total_existing_investment + new_investment),2)
    
    # Question
    question = (
        f"{person_name}, aged {current_age}, has a portfolio that includes ${stocks_investment} in stocks, ${bonds_investment} in bonds, and ${real_estate_investment} in real estate. "
        f"They are considering diversifying further by investing ${new_investment} in a new asset class, such as commodities or international equities. "
        f"The new asset class has a correlation of {correlation_with_existing_portfolio:.2f} with their existing investments. "
        f"How will this diversification likely impact their overall portfolio risk?"
    )
    
    # Solution
    solution = (
        f"Step 1: Calculate the total existing portfolio investment:\n"
        f"  Total Existing Investment = ${stocks_investment} + ${bonds_investment} + ${real_estate_investment} = ${total_existing_investment:.2f}\n\n"
        f"Step 2: Calculate the total investment after diversification:\n"
        f"  Total Investment with New Asset = ${total_existing_investment} + ${new_investment} = ${total_investment_with_new_asset:.2f}\n\n"
        f"Step 3: Assess the impact of correlation on portfolio risk:\n"
        f"  Diversification helps reduce risk by investing in assets that don’t move in the same direction. Correlation values range from -1 to +1, with:\n"
        f"  - A negative correlation (closer to -0.5) reducing risk significantly, as the new asset tends to perform well when existing investments decline.\n"
        f"  - A correlation near zero indicating that the new asset moves independently, offering some diversification benefits.\n"
        f"  - A high positive correlation (closer to +0.5) suggesting an increase in risk, as the new asset moves similarly to the existing portfolio."
    )
    
    return question, solution

def main():
    """
    Generate 10 instances of each template with different random seeds 
    and write the results to a JSON file.
    """
    import json
    # List of template functions
    templates = [
        template_investment_growth_simple,
        template_investment_portfolio_allocation,
        template_investment_market_volatility,
        template_investment_withdrawals,
        template_investment_diversification
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
    output_file = "../../testset/personal_finance/personalinvest.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == '__main__':
    main()