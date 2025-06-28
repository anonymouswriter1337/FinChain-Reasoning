import random

def basic_interest_rate_hike():
    """1:Basic: Interest Rate Hike"""
    original_value = random.randint(500000, 2000000)
    rate_hike = round(random.uniform(1, 5), 1)
    decrease_percentage = round(random.uniform(3, 10), 1)
    question = (
        f"An interest rate hike of {rate_hike:.1f}% causes a bond portfolio to decrease in value by {decrease_percentage:.1f}%. "
        f"If the portfolio was originally worth ${original_value:,}, what is its new value?"
    )
    decrease_amount = round(original_value * (decrease_percentage / 100), 2)
    new_value = original_value - decrease_amount
    solution = (
        f"Step 1: Calculate the decrease in value:\n"
        f"  {original_value:,} * {decrease_percentage / 100:.2f} = {decrease_amount:,.2f}.\n"
        f"Step 2: Subtract the decrease from the original value:\n"
        f"  The new portfolio value = {original_value:,} - {decrease_amount:,.2f} = {new_value:,.2f}."
    )
    return question, solution

def basic_market_downturn():
    """2:Basic: Market Downturn"""
    original_value = random.randint(100000, 1000000)
    drop_percentage = round(random.uniform(10, 30), 1)
    question = (
        f"A market downturn causes stock prices to drop by {drop_percentage:.1f}%. "
        f"If an investor's portfolio was initially worth ${original_value:,}, what is its value after the downturn?"
    )
    drop_amount = round(original_value * (drop_percentage / 100), 2)
    new_value = original_value - drop_amount
    solution = (
        f"Step 1: Calculate the drop in value:\n"
        f"  {original_value:,} * {drop_percentage / 100:.2f} = {drop_amount:,.2f}.\n"
        f"Step 2: Subtract the drop from the initial value:\n"
        f"  The portfolio value = {original_value:,} - {drop_amount:,.2f} = {new_value:,.2f}."
    )
    return question, solution

def intermediate_var_increase():
    """3:Intermediate: Increase in VaR"""
    original_var = random.randint(50000, 200000)
    increase_percentage = round(random.uniform(5, 20), 1)
    question = (
        f"The value-at-risk (VaR) of a portfolio increases by {increase_percentage:.1f}%, from ${original_var:,} to a new value. "
        f"What is the new VaR?"
    )
    increase_amount = round(original_var * (increase_percentage / 100), 2)
    new_var = original_var + increase_amount
    solution = (
        f"Step 1: Calculate the increase in VaR:\n"
        f"  {original_var:,} * {increase_percentage / 100:.2f} = {increase_amount:,.2f}.\n"
        f"Step 2: Add the increase to the original VaR:\n"
        f"  The new VaR = {original_var:,} + {increase_amount:,.2f} = {new_var:,.2f}."
    )
    return question, solution

def intermediate_stress_test():
    """4:Intermediate: Stress Test"""
    portfolio_value = random.randint(500000, 3000000)
    equity_percentage = random.randint(50, 80)
    bond_percentage = 100 - equity_percentage
    equity_loss_percentage = round(random.uniform(20, 40), 1)
    bond_loss_percentage = round(random.uniform(5, 15), 1)
    equity_value = round(portfolio_value * (equity_percentage / 100), 2)
    bond_value = round(portfolio_value * (bond_percentage / 100), 2)
    equity_loss = round(equity_value * (equity_loss_percentage / 100), 2)
    bond_loss = round(bond_value * (bond_loss_percentage / 100), 2)
    question = (
        f"A stress test simulates a market crash where equities lose {equity_loss_percentage:.1f}% and bonds lose {bond_loss_percentage:.1f}%. "
        f"If a portfolio is {equity_percentage}% equities and {bond_percentage}% bonds worth ${portfolio_value:,}, what is the portfolio value after the crash?"
    )
    new_value = portfolio_value - equity_loss - bond_loss
    solution = (
        f"Step 1: Calculate equity value:\n"
        f"  {equity_percentage}% of {portfolio_value:,} = {equity_value:,.2f}.\n"
        f"Step 2: Calculate bond value:\n"
        f"  {bond_percentage}% of {portfolio_value:,} = {bond_value:,.2f}.\n"
        f"Step 3: Calculate losses:\n"
        f"  Equities lose {equity_value:,.2f} * {equity_loss_percentage / 100:.2f} = {equity_loss:,.2f}, "
        f"Bonds lose {bond_value:,.2f} * {bond_loss_percentage / 100:.2f} = {bond_loss:,.2f}.\n"
        f"Step 4: Subtract losses:\n"
        f"  The portfolio value = {portfolio_value:,} - {equity_loss:,.2f} - {bond_loss:,.2f} = {new_value:,.2f}."
    )
    return question, solution

def advanced_systemic_risk():
    """5:Advanced: Systemic Risk Event"""
    portfolio_value = random.randint(1000000, 5000000)
    equity_percentage = random.randint(60, 80)
    bond_percentage = 100 - equity_percentage
    equity_loss_percentage = round(random.uniform(25, 35), 1)
    bond_loss_percentage = round(random.uniform(15, 25), 1)
    equity_value = round(portfolio_value * (equity_percentage / 100), 2)
    bond_value = round(portfolio_value * (bond_percentage / 100), 2)
    equity_loss = round(equity_value * (equity_loss_percentage / 100), 2)
    bond_loss = round(bond_value * (bond_loss_percentage / 100), 2)
    question = (
        f"A systemic risk event causes equities to lose {equity_loss_percentage:.1f}% and bonds to lose {bond_loss_percentage:.1f}%. "
        f"If a portfolio is {equity_percentage}% equities and {bond_percentage}% bonds worth ${portfolio_value:,}, what is the portfolio value after the event?"
    )
    new_value = portfolio_value - equity_loss - bond_loss
    solution = (
        f"Step 1: Calculate equity value:\n"
        f"  {equity_percentage}% of {portfolio_value:,} = {equity_value:,.2f}.\n"
        f"Step 2: Calculate bond value:\n"
        f"  {bond_percentage}% of {portfolio_value:,} = {bond_value:,.2f}.\n"
        f"Step 3: Calculate losses:\n"
        f"  Equities lose {equity_value:,.2f} * {equity_loss_percentage / 100:.2f} = {equity_loss:,.2f}, "
        f"Bonds lose {bond_value:,.2f} * {bond_loss_percentage / 100:.2f} = {bond_loss:,.2f}.\n"
        f"Step 4: Subtract losses:\n"
        f"  The portfolio value = {portfolio_value:,} - {equity_loss:,.2f} - {bond_loss:,.2f} = {new_value:,.2f}."
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
        basic_interest_rate_hike,
        basic_market_downturn,
        intermediate_var_increase,
        intermediate_stress_test,
        advanced_systemic_risk
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
    output_file = "../../testset/risk_management/scenario_plan.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == '__main__':
    main()