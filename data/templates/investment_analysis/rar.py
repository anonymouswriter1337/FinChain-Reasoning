import random

# Named entities for investors and projects
investor_names = ["John Doe", "Susan Lee", "Emily White", "Mark Smith", "David Brown"]
project_names = [
    "Tesla Gigafactory", "Apple iPhone Launch", "Amazon Web Services Expansion", "SpaceX Starship Development",
    "Google Data Center Build", "Microsoft Azure", "Netflix Content Production", "Uber Autonomous Driving Initiative",
    "Facebook Metaverse", "Samsung Semiconductor Factory"
]

# Risk and Return Templates

# Template 1: Calculate Expected Return
def template_expected_return():
    """1:Basic: Expected Return Calculation"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    returns = [round(random.uniform(2, 12), 2) for _ in range(4)]  # Possible returns (%)
    probabilities = [round(random.uniform(0.1, 0.4), 2) for _ in range(4)]  # Probabilities
    probabilities = [round(p / sum(probabilities), 2) for p in probabilities]  # Normalize probabilities
    question = (
        f"{investor_name} is analyzing {project_name} with the following possible annual returns and probabilities:\n"
        f"Returns: {[f'{r:.2f}%' for r in returns]}\n"
        f"Probabilities: {[f'{p:.2f}' for p in probabilities]}.\n"
        f"What is the expected return for {project_name}?"
    )
    # Step 1: Calculate the expected return
    expected_return = sum(r * p for r, p in zip(returns, probabilities))
    # Step 2: Present the result
    solution = (
        f"Step 1: Compute the expected return using the formula:\n"
        f"  Expected Return = Σ(Return × Probability)\n"
        f"                   = {' + '.join([f'({r:.2f} × {p:.2f})' for r, p in zip(returns, probabilities)])}\n\n"

        f"Step 2: Add the results:\n"
        f"  Expected Return = {expected_return:.2f}%"
    )
    return question, solution

# Template 2: Risk (Variance) of Returns
def template_return_variance():
    """2:Basic: Variance of Returns Calculation"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    returns = [round(random.uniform(5, 15), 2) for _ in range(3)]  # Possible returns (%)
    probabilities = [round(random.uniform(0.2, 0.5), 2) for _ in range(3)]  # Probabilities
    probabilities = [round(p / sum(probabilities), 2) for p in probabilities]  # Normalize probabilities
    question = (
        f"{investor_name} is evaluating the risk of {project_name}. The annual returns and their probabilities are:\n"
        f"Returns: {[f'{r:.2f}%' for r in returns]}\n"
        f"Probabilities: {[f'{p:.2f}' for p in probabilities]}.\n"
        f"What is the variance of returns for {project_name}?"
    )
    # Step 1: Calculate the expected return
    expected_return = sum(r * p for r, p in zip(returns, probabilities))
    # Step 2: Calculate the variance
    variance = sum(p * ((r - expected_return) ** 2) for r, p in zip(returns, probabilities))
    solution = (
        f"Step 1: Compute the expected return:\n"
        f"  Expected Return = Σ(Return × Probability)\n"
       f"                   = {' + '.join([f'({r:.2f} × {p:.2f})' for r, p in zip(returns, probabilities)])}\n\n"
        f"Step 2: Compute the variance using the formula:\n"
        f"  Variance = Σ(Probability × (Return - Expected Return)^2)\n"
        f"           = {variance:.2f}"
    )
    return question, solution

# Template 3: Portfolio Expected Return
def template_portfolio_expected_return():
    """3:Intermediate: Portfolio Expected Return Calculation"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    weights = [round(random.uniform(0.1, 0.5), 2) for _ in range(3)]  # Weights of assets
    weights = [round(w / sum(weights), 2) for w in weights]  # Normalize weights
    returns = [round(random.uniform(5, 12), 2) for _ in range(3)]  # Returns of assets (%)
    question = (
        f"{investor_name} is analyzing the expected return for a portfolio containing {project_name}. "
        f"The asset weights are {[f'{w:.2f}' for w in weights]} and their respective returns are {[f'{r:.2f}%' for r in returns]}. "
        f"What is the portfolio's expected return?"
    )
    # Step 1: Multiply weights by returns
    weighted_returns = [w * r for w, r in zip(weights, returns)]
    # Step 2: Compute the expected return
    expected_return = sum(weighted_returns)
    # Step 3: Present the result
    solution = (
        f"Step 1: Multiply weights by returns:\n"
        f"  Weighted Returns = {[f'{w:.2f} × {r:.2f}%' for w, r in zip(weights, returns)]}\n\n"
        f"Step 2: Compute the sum of weighted returns:\n"
        f"  Expected Return = Σ(Weighted Returns)\n"
        f"                   = {expected_return:.2f}%\n\n"
        f"Step 3: Present the portfolio's expected return:\n"
        f"  Portfolio Expected Return = {expected_return:.2f}%"
    )
    return question, solution

# Template 4: Covariance of Returns
def template_covariance_of_returns():
    """4:Intermediate: Covariance of Returns Calculation"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    returns_asset1 = [round(random.uniform(5, 12), 2) for _ in range(3)]
    returns_asset2 = [round(random.uniform(4, 10), 2) for _ in range(3)]
    question = (
        f"{investor_name} is evaluating the covariance of returns for {project_name} between two assets.\n"
        f"The returns for Asset 1 over three periods are {[f'{r:.2f}%' for r in returns_asset1]}.\n"
        f"The returns for Asset 2 over the same periods are {[f'{r:.2f}%' for r in returns_asset2]}.\n"
        f"What is the covariance of returns?"
    )
    # Step 1: Compute means of both assets
    mean_asset1 = sum(returns_asset1) / len(returns_asset1)
    mean_asset2 = sum(returns_asset2) / len(returns_asset2)
    # Step 2: Calculate the covariance
    covariance = sum((r1 - mean_asset1) * (r2 - mean_asset2) for r1, r2 in zip(returns_asset1, returns_asset2)) / len(returns_asset1)
    # Step 3: Present the covariance
    solution = (
        f"Step 1: Compute the means of Asset 1 and Asset 2:\n"
        f"  Mean (Asset 1) = {mean_asset1:.2f}%, Mean (Asset 2) = {mean_asset2:.2f}%\n\n"
        f"Step 2: Compute the covariance:\n"
        f"  Covariance = Σ((Return Asset 1 - Mean 1) × (Return Asset 2 - Mean 2)) / n\n"
        f"             = {covariance:.2f}\n\n"
        f"Step 3: Present the covariance:\n"
        f"  Covariance of Returns = {covariance:.2f}"
    )
    return question, solution

# Template 5: Portfolio Risk (Standard Deviation)
def template_portfolio_risk():
    """5:Advanced: Portfolio Risk (Standard Deviation) Calculation"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    weights = [round(random.uniform(0.2, 0.6), 2), round(random.uniform(0.2, 0.6), 2)]
    weights = [round(w / sum(weights), 2) for w in weights]  # Normalize weights
    std_devs = [round(random.uniform(5, 10), 2), round(random.uniform(4, 8), 2)]  # Standard deviations of assets
    correlation = round(random.uniform(-1, 1), 2)  # Correlation coefficient
    question = (
        f"{investor_name} is analyzing the risk of their portfolio for {project_name}. "
        f"The asset weights are {[f'{w:.2f}' for w in weights]}, the standard deviations are {[f'{s:.2f}%' for s in std_devs]}, "
        f"and the correlation between the assets is {correlation:.2f}. Calculate the portfolio's standard deviation."
    )
    # Step 1: Compute the weighted variances
    variance = sum((w ** 2) * (s ** 2) for w, s in zip(weights, std_devs))
    # Step 2: Compute the covariance term
    covariance = 2 * weights[0] * weights[1] * std_devs[0] * std_devs[1] * correlation
    # Step 3: Compute the total portfolio variance
    portfolio_variance = variance + covariance
    # Step 4: Compute the standard deviation
    portfolio_std_dev = portfolio_variance ** 0.5
    solution = (
        f"Step 1: Compute the weighted variances:\n"
        f"  Variance = Σ(Weight^2 × StdDev^2)\n"
        f"           = {variance:.2f}\n\n"
        f"Step 2: Compute the covariance term:\n"
        f"  Covariance Term = 2 × Weight 1 × Weight 2 × StdDev 1 × StdDev 2 × Correlation\n"
        f"                  = {covariance:.2f}\n\n"
        f"Step 3: Compute the total portfolio variance:\n"
        f"  Portfolio Variance = Variance + Covariance\n"
        f"                    = {portfolio_variance:.2f}\n\n"
        f"Step 4: Compute the portfolio's standard deviation:\n"
        f"  Standard Deviation = √Portfolio Variance\n"
        f"                     = {portfolio_std_dev:.2f}%"
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
        template_expected_return,
        template_return_variance,
        template_portfolio_expected_return,
        template_covariance_of_returns,
        template_portfolio_risk
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
    output_file = "../../testset/investment_analysis/rar.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()