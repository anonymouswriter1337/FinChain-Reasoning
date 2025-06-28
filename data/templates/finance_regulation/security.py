import random

# Sample named entities
investor_names = ["Alice Johnson", "Robert Smith", "Maria Garcia", "James Wang", "Linda Davis"]
company_names = [
    "Tesla", "Amazon", "Apple", "Google", "Microsoft",
    "Meta", "Netflix", "JPMorgan Chase", "Goldman Sachs", "Pfizer"
]

def random_entities():
    return random.choice(investor_names), random.choice(company_names)

### BASIC LEVEL (1-2 steps) ###

def basic_disclosure_threshold():
    """1:Basic: Determine if disclosure is required based on threshold"""
    investor, company = random_entities()
    threshold = 5  # 5% SEC reporting threshold
    ownership = round(random.uniform(1, 10), 2)
    
    question = (
        f"{investor} has recently acquired a stake in {company}. They now own {ownership:.2f}% of the company's shares.\n"
        f"SEC rules require filing a Schedule 13D form if ownership exceeds {threshold}%. "
        f"Does {investor} need to file a disclosure?"
    )

    solution = (
        f"Step 1: Compare the ownership percentage to the disclosure threshold.\n"
        f"{ownership:.2f}% {'>' if ownership > threshold else '<='} {threshold}%\n\n"
        f"Step 2: {'Yes, disclosure is required.' if ownership > threshold else 'No, disclosure is not required.'}"
    )
    return question, solution

def basic_insider_trading_gain():
    """2:Basic: Calculate gain from insider trading"""
    investor, company = random_entities()
    buy_price = round(random.uniform(80, 100), 2)
    sell_price = round(random.uniform(110, 140), 2)
    shares = random.randint(100, 1000)
    
    gain = (sell_price - buy_price) * shares
    question = (
        f"{investor}, an executive at {company}, bought {shares} shares at ${buy_price:.2f} based on non-public information.\n"
        f"They sold them later at ${sell_price:.2f}. What is the total gain made through insider trading?"
    )
    solution = (
        f"Step 1: Calculate the gain per share: ${sell_price:.2f} - ${buy_price:.2f} = ${sell_price - buy_price:.2f}\n"
        f"Step 2: Multiply by number of shares: {shares} × ${sell_price - buy_price:.2f} = ${gain:.2f}"
    )
    return question, solution

### INTERMEDIATE LEVEL (3-4 steps) ###

def intermediate_reg_a_investment_limit():
    """3:Intermediate: Check if investment exceeds Reg A+ Tier 2 limit"""
    investor, company = random_entities()
    annual_income = random.randint(50000, 150000)
    investment = random.randint(20000, 70000)
    limit_ratio = 0.1
    limit = round(annual_income * limit_ratio, 2)

    question = (
        f"{investor} wants to invest ${investment} in a Regulation A+ Tier 2 offering from {company}.\n"
        f"The SEC limits investments to 10% of the investor's annual income if unaudited. "
        f"Their income is ${annual_income}. Can {investor} legally make this investment?"
    )

    solution = (
        f"Step 1: Calculate 10% of the investor's income: 10% × ${annual_income} = ${limit:.2f}\n"
        f"Step 2: Compare investment with the limit: ${investment} {'>' if investment > limit else '<='} ${limit:.2f}\n\n"
        f"Step 3: {'No, the investment exceeds the limit.' if investment > limit else 'Yes, the investment is within the limit.'}"
    )
    return question, solution

def intermediate_multiple_reporting_thresholds():
    """4:Intermediate: Calculate combined ownership across entities for reporting requirement"""
    investor, company = random_entities()
    direct = round(random.uniform(2, 4), 2)
    trust = round(random.uniform(1, 3), 2)
    spouse = round(random.uniform(1, 4), 2)
    total = direct + trust + spouse

    question = (
        f"{investor} holds shares in {company} through three sources:\n"
        f"- Direct ownership: {direct:.2f}%\n"
        f"- A family trust: {trust:.2f}%\n"
        f"- Their spouse: {spouse:.2f}%\n"
        f"The SEC requires disclosure when beneficial ownership exceeds 5%.\n"
        f"Does {investor} need to file a report?"
    )

    solution = (
        f"Step 1: Total beneficial ownership = {direct:.2f}% + {trust:.2f}% + {spouse:.2f}% = {total:.2f}%\n"
        f"Step 2: Compare with threshold (5%) → {total:.2f}% {'>' if total > 5 else '<='} 5%\n\n"
        f"Step 3: {'Yes, disclosure is required.' if total > 5 else 'No, disclosure is not required.'}"
    )
    return question, solution


### ADVANCED LEVEL (>4 steps) ###
def advanced_merger_announcement_timing():
    """5:Advanced: Evaluate gains from timed trades around a merger announcement with SEC penalty logic"""
    investor, company = random_entities()
    pre_price = round(random.uniform(50, 70), 2)
    post_price = round(pre_price * random.uniform(1.2, 1.5), 2)
    shares = random.randint(500, 2000)

    question = (
        f"{investor} traded {shares} shares of {company} the day before a confidential merger announcement.\n"
        f"The price went from ${pre_price:.2f} to ${post_price:.2f} after the announcement. "
        f"The SEC is investigating for insider trading. What is the illicit gain and possible penalty?"
    )

    gain_per_share = post_price - pre_price
    total_gain = gain_per_share * shares
    penalty = 3 * total_gain

    solution = (
        f"Step 1: Gain per share = ${post_price:.2f} - ${pre_price:.2f} = ${gain_per_share:.2f}\n"
        f"Step 2: Total gain = {shares} × ${gain_per_share:.2f} = ${total_gain:.2f}\n"
        f"Step 3: Under SEC rules, the penalty can be up to 3× the illicit gain.\n"
        f"Step 4: Penalty = 3 × ${total_gain:.2f} = ${penalty:.2f}"
    )

    return question, solution

# def advanced_accredited_investor_test():
#     """Advanced: Determine if multiple criteria qualify for accredited investor status"""
#     investor, company = random_entities()
#     income = random.randint(180000, 300000)
#     net_worth = random.randint(500000, 2000000)
#     joint_income = random.randint(250000, 400000)

#     question = (
#         f"{investor} wants to invest in a private placement offered by {company}.\n"
#         f"To qualify as an accredited investor, one must meet at least one of these:\n"
#         f"- Annual income over $200,000 ($300,000 jointly), or\n"
#         f"- Net worth over $1,000,000 excluding primary residence.\n"
#         f"{investor} reports ${income} in income, ${joint_income} jointly, and ${net_worth} in net worth.\n"
#         f"Do they qualify as an accredited investor?"
#     )

#     qualifies = income > 200000 or joint_income > 300000 or net_worth > 1_000_000
#     solution = (
#         f"Step 1: Check individual income > $200,000 → {'Yes' if income > 200000 else 'No'}\n"
#         f"Step 2: Check joint income > $300,000 → {'Yes' if joint_income > 300000 else 'No'}\n"
#         f"Step 3: Check net worth > $1,000,000 → {'Yes' if net_worth > 1_000_000 else 'No'}\n\n"
#         f"Step 4: If any are true → Accredited investor: {'Yes' if qualifies else 'No'}"
#     )
#     return question, solution




def main():
    """
    Generate 10 instances of each template with different random seeds 
    and write the results to a JSON file.
    """
    import json
    # ----------- Export All to JSONL -----------

    # List of template functions
    templates = [
        basic_disclosure_threshold,
        basic_insider_trading_gain,
        intermediate_reg_a_investment_limit,
        intermediate_multiple_reporting_thresholds,
        advanced_merger_announcement_timing
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
    output_file = "../../testset/finance_regulation/security.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")

    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")


if __name__ == "__main__":
   main()