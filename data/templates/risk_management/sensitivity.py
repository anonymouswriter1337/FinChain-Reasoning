import random

def generate_random_value(range_min, range_max):
    """Generate a random integer value within the specified range."""
    return random.randint(range_min, range_max)

# BASIC SCENARIOS
def basic_portfolio_single_factor():
    """1:Basic: Portfolio with single factor sensitivity analysis."""
    portfolio_value = generate_random_value(10000, 100000)
    weight = generate_random_value(10, 90)
    factor_change = round(random.uniform(-5, 5), 2)
    sensitivity = round(random.uniform(-2, 2), 2)

    question = f"A portfolio worth ${portfolio_value} has {weight}% allocated to an asset sensitive to interest rates. If the interest rate changes by {factor_change:.2f}% and the sensitivity is {sensitivity:.2f}, how does the portfolio value change?"
    
    solv = (portfolio_value * (weight / 100) * (factor_change / 100) * sensitivity)
    solution = (
        f"Step 1: Calculate asset's contribution to sensitivity:\n"
        f"  {portfolio_value} * {weight / 100:.2f} = ${portfolio_value * (weight / 100):.2f}.\n"
        f"Step 2: Calculate sensitivity impact:\n"
        f"  {factor_change / 100:.4f} * {sensitivity:.2f} = {factor_change / 100 * sensitivity:.4f}.\n"
        f"Step 3: Apply impact:\n"
        f"  ${portfolio_value * (weight / 100):.2f} * {factor_change / 100 * sensitivity:.4f} = ${solv:.2f}."
    )

    return question, solution

def basic_sensitivity_of_stock():
    """2:Basic: Stock price sensitivity to market index."""
    stock_value = generate_random_value(100, 1000)
    sensitivity = round(random.uniform(-1, 1), 2)
    market_change = round(random.uniform(-3, 3), 2)

    question = f"A stock priced at ${stock_value} has a sensitivity of {sensitivity:.2f} to the market index. If the market index changes by {market_change:.2f}%, how does the stock price change?"

    solv = (stock_value * (market_change / 100) * sensitivity)
    solution = (
        f"Step 1: Calculate market impact:\n"
        f"  {market_change / 100:.4f} * {sensitivity:.2f} = {market_change / 100 * sensitivity:.4f}.\n"
        f"Step 2: Apply impact to stock price:\n"
        f"  ${stock_value} * {market_change / 100 * sensitivity:.4f} = ${solv:.2f}."
    )

    return question, solution


# INTERMEDIATE SCENARIOS
def intermediate_multi_asset_portfolio():
    """3:Intermediate: Multi-asset portfolio with interest rate sensitivity."""
    portfolio_value = generate_random_value(50000, 200000)
    asset_weights = [random.randint(10, 50) for _ in range(3)]
    factor_change = round(random.uniform(-4, 4), 2)
    sensitivities = [round(random.uniform(-1.5, 1.5), 2) for _ in range(3)]

    question = f"A portfolio worth ${portfolio_value} has 3 asset classes with weights {asset_weights[0]}%, {asset_weights[1]}%, and {asset_weights[2]}%, respectively. Their sensitivities to interest rates are {sensitivities[0]:.2f}, {sensitivities[1]:.2f}, and {sensitivities[2]:.2f}. If interest rates change by {factor_change:.2f}%, how does the portfolio value change?"

    solv = sum((portfolio_value * (weight / 100) * (factor_change / 100) * sensitivity) for weight, sensitivity in zip(asset_weights, sensitivities))
    solution = f"Step 1: Calculate contribution for each asset:\n"
    for i in range(3):
        impact = round(portfolio_value * (asset_weights[i] / 100) * (factor_change / 100) * sensitivities[i], 2)
        solution += f"  Asset {i+1}: {portfolio_value} * {asset_weights[i] / 100:.2f} * {factor_change / 100:.4f} * {sensitivities[i]:.2f} = ${impact:.2f}.\n"
    solution += f"Step 2: Sum up contributions:\n"
    solution += f"  Total = ${solv:.2f}."

    return question, solution

def intermediate_volatility_sensitivity():
    """4:Intermediate: Value at Risk (VaR) sensitivity to volatility changes."""
    initial_var = generate_random_value(50000, 200000)
    delta_volatility = round(random.uniform(-10, 10), 2)
    sensitivity = round(random.uniform(1, 2), 2)

    question = f"A firm's Value at Risk (VaR) is ${initial_var} with a sensitivity of {sensitivity:.2f} to volatility changes. If volatility changes by {delta_volatility:.2f}%, how does the VaR change?"

    solv = initial_var * (delta_volatility / 100) * sensitivity
    solution = (
        f"Step 1: Calculate the volatility impact:\n"
        f"  {delta_volatility / 100:.4f} * {sensitivity:.2f} = {delta_volatility / 100 * sensitivity:.4f}.\n"
        f"Step 2: Apply impact:\n"
        f"  ${initial_var} * {delta_volatility / 100 * sensitivity:.4f} = ${solv:.2f}."
    )

    return question, solution


# ADVANCED SCENARIOS
def advanced_multi_factor_portfolio():
    """5:Advanced: Multi-factor portfolio with various sensitivity exposures."""
    portfolio_value = generate_random_value(500000, 1000000)
    weights = [random.randint(10, 30) for _ in range(4)]
    factors = ['interest rates', 'currency', 'inflation', 'market volatility']
    factor_changes = [round(random.uniform(-5, 5), 2) for _ in range(4)]
    sensitivities = [round(random.uniform(-1, 1), 2) for _ in range(4)]

    question = f"A portfolio worth ${portfolio_value} is exposed to {factors[0]}, {factors[1]}, {factors[2]}, and {factors[3]} with weights {weights}%. The sensitivities and respective changes are:\n"
    for i, factor in enumerate(factors):
        question += f"  {factor}: Sensitivity {sensitivities[i]:.2f}, Change {factor_changes[i]:.2f}%\n"
    question += "What is the total impact on the portfolio?"

    solv = sum((portfolio_value * (weights[i] / 100) * (factor_changes[i] / 100) * sensitivities[i]) for i in range(4))
    solution = f"Step 1: Calculate contribution for each factor:\n"
    for i, factor in enumerate(factors):
        impact = round(portfolio_value * (weights[i] / 100) * (factor_changes[i] / 100) * sensitivities[i], 2)
        solution += f"  {factor}: {portfolio_value} * {weights[i] / 100:.2f} * {factor_changes[i] / 100:.4f} * {sensitivities[i]:.2f} = ${impact:.2f}.\n"
    solution += f"Step 2: Sum up contributions:\n"
    solution += f"  Total = ${solv:.2f}."

    return question, solution

def main():
    """
    Generate 10 instances of each template with different random seeds 
    and write the results to a JSON file.
    """
    import json
    # List of template functions
    templates = [
        basic_portfolio_single_factor,
        basic_sensitivity_of_stock,
        intermediate_multi_asset_portfolio,
        intermediate_volatility_sensitivity,
        advanced_multi_factor_portfolio
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
    output_file = "../../testset/risk_management/sensitivity.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == '__main__':
    main()