import random

# Named entities for companies and industries
company_names = ["Tesla Inc.", "Apple Inc.", "Amazon.com", "SpaceX", "Google LLC"]
industry_names = ["automotive", "technology", "e-commerce", "aerospace", "internet services"]

# Template 1
def template_retirement_savings_simple():
    """1:Basic: Creates a straightforward retirement savings problem where an individual 
    needs to calculate additional annual savings needed to reach a target amount, 
    considering current savings and years until retirement."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    age = random.randint(30, 50)
    retirement_age = random.randint(60, 65)
    years_left = retirement_age - age
    target_savings = round(random.randint(5000000, 20000000),2)  # Target savings
    current_savings = round(random.randint(500000, 5000000),2)  # Current savings
    annual_savings = round(random.randint(100000, 500000),2)  # Current annual savings
    total_savings_needed = round((target_savings - current_savings),2)
    additional_savings_needed = round((total_savings_needed / years_left),2)
    
    question = (
        f"{person_name}, aged {age}, plans to retire in {years_left} years. They aim to save ${target_savings} by the time they retire and currently have ${current_savings} saved. "
        f"Assuming they save ${annual_savings} every year, how much more do they need to save annually to reach their retirement goal?"
    )
    
    # Step 1: Calculate additional savings needed annually
    solution = (
        f"Step 1: Calculate how much more needs to be saved annually:\n"
        f"  Total Additional Savings Needed = ${target_savings} - ${current_savings} = ${total_savings_needed:.2f}\n"
        f"  Additional Annual Savings Needed = ${total_savings_needed:.2f} / {years_left} = ${additional_savings_needed:.2f} per year"
    )
    
    return question, solution


# Template 2
def template_retirement_investment_returns():
    """2:Basic: Generates a retirement planning problem that incorporates investment 
    returns, calculating future values of both current savings and ongoing contributions 
    using compound interest formulas."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    current_age = random.randint(30, 50)
    retirement_age = random.randint(60, 65)
    years_left = retirement_age - current_age
    current_savings = round(random.randint(500000, 3000000),2)  # Current savings
    annual_savings = round(random.randint(100000, 500000),2)  # Annual savings
    investment_return_rate = round(random.uniform(4.0, 10.0),2)  # Expected return rate (%)
    
    future_value_of_current_savings = round((current_savings * (1 + investment_return_rate / 100) ** years_left),2)
    future_value_of_savings = round((annual_savings * (((1 + investment_return_rate / 100) ** years_left - 1) / (investment_return_rate / 100))),2)
    total_future_savings = round((future_value_of_current_savings + future_value_of_savings),2)
    
    question = (
        f"{person_name}, aged {current_age}, has saved ${current_savings} so far and plans to retire in {years_left} years. "
        f"If they expect an average annual return of {investment_return_rate:.2f}% on their investments, how much will they have at retirement if they continue to save ${annual_savings} per year?"
    )
    # Step 1: Calculate future value of current savings
    # Step 2: Calculate future value of additional savings
    solution = (
        f"Step 1: Calculate the future value of current savings:\n"
        f"  Future Value (Current Savings) = ${current_savings} × (1 + {investment_return_rate:.2f}%)^{years_left} = ${future_value_of_current_savings:.2f}\n\n"
        f"Step 2: Calculate the future value of additional savings:\n"
        f"  Future Value (Additional Savings) = ${annual_savings} × [(1 + {investment_return_rate:.2f}%)^{years_left} - 1] / {investment_return_rate:.2f}% = ${future_value_of_savings:.2f}\n\n"
        f"  Total Savings at Retirement = ${future_value_of_current_savings:.2f} + ${future_value_of_savings:.2f} = ${total_future_savings:.2f}"
    )
    return question, solution

# Template 3
def template_retirement_inflation_adjustment():
    """3:Intermediate: Creates a retirement planning scenario that factors in inflation, 
    calculating how much future savings are needed to maintain the purchasing power 
    of a target amount in today's terms."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    current_age = random.randint(30, 50)
    retirement_age = random.randint(60, 65)
    years_left = retirement_age - current_age
    target_savings_in_today_terms = round(random.randint(5000000, 20000000),2)  # Savings target in today's terms
    inflation_rate = round(random.uniform(2.0, 5.0),2)  # Annual inflation rate (%)
    
    future_savings_needed = round((target_savings_in_today_terms * (1 + inflation_rate / 100) ** years_left),2)
    
    question = (
        f"{person_name}, aged {current_age}, plans to retire in {years_left} years and wants to have ${target_savings_in_today_terms} in today’s terms. "
        f"Assuming an annual inflation rate of {inflation_rate:.2f}%, how much will they actually need to save by the time they retire to match the purchasing power of ${target_savings_in_today_terms}?"
    )
    # Step 1: Calculate future value of savings required after adjusting for inflation
    solution = (
        f"Step 1: Adjust the target savings for inflation:\n"
        f"  Future Value of Target Savings = ${target_savings_in_today_terms} × (1 + {inflation_rate:.2f}%)^{years_left} = ${future_savings_needed:.2f}"
    )
    return question, solution

# Template 4
def template_retirement_shortfall():
    """4:Intermediate: Generates a retirement planning problem where an individual must 
    calculate additional annual savings needed to cover a shortfall between current 
    savings trajectory and an updated retirement goal."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    current_age = random.randint(35, 50)
    retirement_age = random.randint(60, 65)
    years_left = retirement_age - current_age
    current_savings = round(random.randint(500000, 3000000),2)  # Current savings
    target_savings = round(random.randint(10000000, 30000000),2)  # New retirement goal
    additional_savings_needed = round((target_savings - current_savings),2)
    annual_savings = round(random.randint(100000, 500000),2)  # Current annual savings
    annual_return_rate = round(random.uniform(4.0, 8.0),2)  # Expected annual return (%)
    
    # Step 1: Calculate future value of current savings
    future_value_current_savings = round(current_savings * (1 + annual_return_rate / 100) ** years_left, 2)
    
    # Step 2: Calculate future value of current annual savings
    future_value_annual_savings = round(
        annual_savings * (((1 + annual_return_rate / 100) ** years_left - 1) / (annual_return_rate / 100)),
        2
    )
    
    # Step 3: Calculate total projected savings at retirement
    total_projected_savings = round(future_value_current_savings + future_value_annual_savings, 2)
    
    # Step 4: Calculate the shortfall
    shortfall = round(target_savings - total_projected_savings, 2)
    
    # Step 5: Calculate additional annual savings needed to cover the shortfall
    if shortfall > 0:
        additional_annual_savings = round(
            shortfall / (((1 + annual_return_rate / 100) ** years_left - 1) / (annual_return_rate / 100)),
            2
        )
    else:
        additional_annual_savings = 0  # No additional savings needed
    
    question = (
        f"{person_name}, aged {current_age}, has saved ${current_savings} so far and currently saves ${annual_savings} annually. "
        f"They aim to have ${target_savings} at retirement in {years_left} years. "
        f"Assuming an annual investment return of {annual_return_rate:.2f}%, how much additional money will they need to save each year to meet their retirement goal?"
    )
    
    solution = (
        f"Step 1: Calculate the future value of current savings with investment returns:\n"
        f"  Future Value of Current Savings = ${current_savings} × (1 + {annual_return_rate:.2f}%)^{years_left} = ${future_value_current_savings:.2f}\n\n"
        
        f"Step 2: Calculate the future value of current annual savings:\n"
        f"  Future Value of Annual Savings = ${annual_savings} × [((1 + {annual_return_rate:.2f}%)^{years_left} - 1) / {annual_return_rate:.2f}%] = ${future_value_annual_savings:.2f}\n\n"
        
        f"Step 3: Calculate total projected savings at retirement:\n"
        f"  Total Projected Savings = ${future_value_current_savings:.2f} + ${future_value_annual_savings:.2f} = ${total_projected_savings:.2f}\n\n"
        
        f"Step 4: Calculate the shortfall between the target and projected savings:\n"
        f"  Shortfall = ${target_savings} - ${total_projected_savings:.2f} = ${shortfall:.2f}\n\n"
    )
    
    if shortfall > 0:
        solution += (
            f"Step 5: Calculate the additional annual savings needed to cover the shortfall:\n"
            f"  Additional Annual Savings = Shortfall / [((1 + r)^n - 1) / r]\n"
            f"                           = ${shortfall:.2f} / [((1 + {annual_return_rate:.2f}%)^{years_left} - 1) / {annual_return_rate:.2f}%]\n"
            f"                           = ${additional_annual_savings:.2f} per year"
        )
    else:
        solution += (
            f"Step 5: Since the projected savings (${total_projected_savings:.2f}) exceeds the target savings (${target_savings}),\n"
            f"  no additional savings are required. The current savings plan is sufficient to meet or exceed the retirement goal."
        )
    
    return question, solution


# Template 5
def template_retirement_early():
    """5:Advanced: Creates a complex early retirement scenario that considers multiple 
    factors including current savings, future contributions, investment returns, 
    annual withdrawals, and life expectancy to determine retirement feasibility."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    current_age = random.randint(30, 50)
    early_retirement_age = random.randint(50, 60)
    years_until_retirement = early_retirement_age - current_age
    current_savings = round(random.randint(500000, 3000000), 2)  # Current savings
    annual_savings = round(random.randint(100000, 500000), 2)  # Annual savings until retirement
    investment_return_rate = round(random.uniform(4.0, 8.0), 2)  # Expected annual return rate (%)
    retirement_return_rate = round(investment_return_rate * 0.75, 2)  # More conservative return during retirement
    annual_withdrawal = round(random.randint(300000, 1000000),2)  # Annual retirement withdrawal
    life_expectancy = random.randint(75, 90)  # Life expectancy (years)
    retirement_years = life_expectancy - early_retirement_age

    # Step 1: Calculate the future value of current savings
    future_value_of_current_savings = round((current_savings * (1 + investment_return_rate / 100) ** years_until_retirement),2)

    # Step 2: Calculate the future value of additional annual savings
    future_value_of_savings = round((annual_savings * (((1 + investment_return_rate / 100) ** years_until_retirement - 1) / (investment_return_rate / 100))),2)

    # Step 3: Calculate the total retirement savings at early retirement
    total_savings_at_retirement = round((future_value_of_current_savings + future_value_of_savings),2)

    # Step 4: Calculate if the total retirement savings will be enough for planned annual withdrawals
    # Using the present value of an annuity formula to determine the capital needed for withdrawals
    # PV = PMT × [1 - (1 + r)^-n] / r
    capital_needed_for_withdrawals = round(
        annual_withdrawal * (1 - (1 + retirement_return_rate / 100) ** -retirement_years) / (retirement_return_rate / 100),
        2
    )
    
    # Step 5: Determine if there's a shortfall or surplus
    shortfall_or_surplus = round((total_savings_at_retirement - capital_needed_for_withdrawals),2)
    
    # Step 6: Simulate year-by-year to verify (optional, for solution explanation)
    remaining_balance = total_savings_at_retirement
    years_sustainable = 0
    
    for year in range(1, retirement_years + 1):
        # Apply withdrawal at the beginning of the year
        remaining_balance -= annual_withdrawal
        
        if remaining_balance <= 0:
            break
            
        # Apply investment returns for the year
        remaining_balance = round(remaining_balance * (1 + retirement_return_rate / 100), 2)
        years_sustainable += 1

    question = (
        f"{person_name}, currently {current_age} years old, plans to retire early at age {early_retirement_age}. They currently have ${current_savings} saved and expect to save ${annual_savings} per year "
        f"until retirement. Assuming an average annual return of {investment_return_rate:.2f}% on their investments before retirement and {retirement_return_rate:.2f}% during retirement, how much will they have at retirement, and will it be enough if they want to withdraw "
        f"${annual_withdrawal} per year for the rest of their life (assumed life expectancy of {life_expectancy} years)?"
    )

    solution = (
        f"Step 1: Calculate the future value of current savings:\n"
        f"  Future Value (Current Savings) = ${current_savings} × (1 + {investment_return_rate:.2f}%)^{years_until_retirement} = ${future_value_of_current_savings:.2f}\n\n"
        f"Step 2: Calculate the future value of additional annual savings:\n"
        f"  Future Value (Additional Savings) = ${annual_savings} × [(1 + {investment_return_rate:.2f}%)^{years_until_retirement} - 1] / {investment_return_rate:.2f}% = ${future_value_of_savings:.2f}\n\n"
        f"Step 3: Calculate the total savings at retirement:\n"
        f"  Total Savings at Retirement = ${future_value_of_current_savings:.2f} + ${future_value_of_savings:.2f} = ${total_savings_at_retirement:.2f}\n\n"
        f"Step 4: Calculate the capital needed to sustain annual withdrawals of ${annual_withdrawal} for {retirement_years} years:\n"
        f"  Using the present value of an annuity formula:\n"
        f"  Capital Needed = Annual Withdrawal × [1 - (1 + r)^-n] / r\n"
        f"                 = ${annual_withdrawal} × [1 - (1 + {retirement_return_rate:.2f}%)^-{retirement_years}] / {retirement_return_rate:.2f}%\n"
        f"                 = ${capital_needed_for_withdrawals:.2f}\n\n"
        f"Step 5: Determine if there's a shortfall or surplus:\n"
        f"  Shortfall/Surplus = ${total_savings_at_retirement:.2f} - ${capital_needed_for_withdrawals:.2f} = ${shortfall_or_surplus:.2f}\n\n"
    )
    
    if shortfall_or_surplus >= 0:
        solution += (
            f"The retirement plan is feasible. {person_name} will have enough savings to withdraw ${annual_withdrawal} "
            f"per year until age {life_expectancy}, with a projected surplus of ${shortfall_or_surplus:.2f}."
        )
    else:
        solution += (
            f"The retirement plan is not feasible as currently structured. {person_name} will face a shortfall of ${-shortfall_or_surplus:.2f}. "
            f"Options to consider include: increasing current savings, delaying retirement, reducing planned withdrawals, "
            f"or seeking higher investment returns."
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
        template_retirement_savings_simple,
        template_retirement_investment_returns,
        template_retirement_inflation_adjustment,
        template_retirement_shortfall,
        template_retirement_early
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
    output_file = "../../testset/personal_finance/saveretire.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == '__main__':
    main()