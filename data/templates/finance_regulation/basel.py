import random

investor_names = ["John Doe", "Alice Johnson", "Michael Chen", "Emily Davis", "Robert Lee"]
company_names = ["JP Morgan", "Goldman Sachs", "Bank of America", "Citigroup", "Wells Fargo", "Morgan Stanley"]

# ----------------------- BASIC QUESTIONS -----------------------

def basel_leverage_ratio_test():
    """1:Basic: Leverage Ratio Computation"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    tier1 = round(random.uniform(5, 10), 2)
    total_exposure = round(random.uniform(100, 200), 2)
    ratio = round((tier1 / total_exposure) * 100, 2)

    question = (
        f"{investor} is assessing {company}'s leverage ratio. "
        f"The bank has Tier 1 capital of ${tier1} million and total exposure of ${total_exposure} million. "
        f"What is the leverage ratio and does it meet the Basel III minimum of 3%?"
    )

    solution = (
        f"Step 1: Leverage Ratio = (Tier 1 / Total Exposure) × 100\n"
        f"        = ({tier1} / {total_exposure}) × 100 = {ratio:.2f}%\n"
        f"Step 2: Compare with Basel minimum of 3% → {'Yes' if ratio >= 3 else 'No'}"
    )

    return question, solution


def basel_tier_composition_analysis():
    """2:Basic: Capital Ratio Decomposition"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    tier1 = round(random.uniform(5, 10), 2)
    tier2 = round(random.uniform(2, 5), 2)
    rwa = round(random.uniform(70, 120), 2)
    total_ratio = round(((tier1 + tier2) / rwa) * 100, 2)
    tier1_ratio = round((tier1 / rwa) * 100, 2)

    question = (
        f"{investor} examines {company}'s capital composition. The Tier 1 capital is ${tier1} million, Tier 2 is ${tier2} million, "
        f"and risk-weighted assets are ${rwa} million. Calculate the total capital ratio and Tier 1 ratio."
    )

    solution = (
        f"Step 1: Tier 1 Ratio = ({tier1} / {rwa}) × 100 = {tier1_ratio:.2f}%\n"
        f"Step 2: Total Capital Ratio = ({tier1 + tier2} / {rwa}) × 100 = {total_ratio:.2f}%"
    )
    return question, solution

# ----------------------- INTERMEDIATE QUESTIONS -----------------------

def basel_capital_buffer_requirement():
    """3:Intermediate: Capital Conservation Buffer Check"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    total_ratio = round(random.uniform(8.5, 10.5), 2)
    min_ratio = 8
    buffer_required = 2.5

    question = (
        f"{investor} is examining whether {company} meets the capital conservation buffer requirements. "
        f"The bank's capital adequacy ratio is {total_ratio}%. "
        f"Under Basel III, the minimum ratio is {min_ratio}% and the buffer required is {buffer_required}%. "
        f"Does the bank meet the full requirement?"
    )

    solution = (
        f"Step 1: Total Required = Minimum Capital Ratio + Buffer = {min_ratio}% + {buffer_required}% = {min_ratio + buffer_required}%\n"
        f"Step 2: Compare Bank’s Ratio = {total_ratio}% → "
        f"{'Meets' if total_ratio >= (min_ratio + buffer_required) else 'Does not meet'} the requirement"
    )

    return question, solution

def basel_rwa_increase_effect():
    """4:Intermediate: Impact of RWA Increase on Capital Ratio"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    capital = round(random.uniform(10, 20), 2)
    rwa_initial = round(random.uniform(80, 100), 2)
    rwa_new = rwa_initial + random.randint(10, 30)
    ratio_old = round((capital / rwa_initial) * 100, 2)
    ratio_new = round((capital / rwa_new) * 100, 2)

    question = (
        f"{investor} analyzes the impact of risk-weighted asset increase on {company}'s capital adequacy. "
        f"The bank has capital of ${capital} million. RWA increased from ${rwa_initial} million to ${rwa_new} million. "
        f"How did the capital adequacy ratio change?"
    )

    solution = (
        f"Step 1: Old Ratio = ({capital} / {rwa_initial}) × 100 = {ratio_old:.2f}%\n"
        f"Step 2: New Ratio = ({capital} / {rwa_new}) × 100 = {ratio_new:.2f}%\n"
        f"Step 3: Interpretation → Ratio decreased due to RWA increase."
    )

    return question, solution

# ----------------------- ADVANCED QUESTIONS -----------------------

def basel_multiple_requirements_check():
    """5:Advanced: Evaluate All Basel III Requirements"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    tier1 = round(random.uniform(5, 10), 2)
    tier2 = round(random.uniform(1, 4), 2)
    rwa = round(random.uniform(70, 100), 2)
    exposure = round(random.uniform(150, 250), 2)
    hqla = round(random.uniform(100, 160), 2)
    outflows = round(random.uniform(90, 140), 2)

    cap_ratio = round((tier1 + tier2) / rwa * 100, 2)
    leverage_ratio = round(tier1 / exposure * 100, 2)
    lcr = round(hqla / outflows * 100, 2)

    question = (
        f"{investor} assesses whether {company} meets all major Basel III thresholds:\n"
        f"- Capital Ratio: Tier 1 = ${tier1}M, Tier 2 = ${tier2}M, RWA = ${rwa}M\n"
        f"- Leverage Ratio: Total Exposure = ${exposure}M\n"
        f"- Liquidity Coverage Ratio: HQLA = ${hqla}M, Net Outflows = ${outflows}M\n"
        f"Evaluate if the bank meets: Capital Ratio ≥ 10.5%, Leverage ≥ 3%, LCR ≥ 100%."
    )

    solution = (
        f"Step 1: Capital Ratio = ({tier1 + tier2} / {rwa}) × 100 = {cap_ratio:.2f}% → {'✓' if cap_ratio >= 10.5 else '✗'}\n"
        f"Step 2: Leverage Ratio = ({tier1} / {exposure}) × 100 = {leverage_ratio:.2f}% → {'✓' if leverage_ratio >= 3 else '✗'}\n"
        f"Step 3: LCR = ({hqla} / {outflows}) × 100 = {lcr:.2f}% → {'✓' if lcr >= 100 else '✗'}\n"
        f"Conclusion: {'All requirements met' if (cap_ratio >= 10.5 and leverage_ratio >= 3 and lcr >= 100) else 'Some requirements not met'}"
    )

    return question, solution


def main():
    """
    Generate 10 instances of each template with different random seeds 
    and write the results to a JSON file.
    """
    import json
    # ----------- Export All to JSONL -----------

    # List of template functions
    templates = [
        basel_leverage_ratio_test,
        basel_tier_composition_analysis,
        basel_capital_buffer_requirement,
        basel_rwa_increase_effect,
        basel_multiple_requirements_check
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
    output_file = "../../testset/finance_regulation/basel.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")

    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")


if __name__ == "__main__":
   main()