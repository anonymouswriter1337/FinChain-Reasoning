import random

# Named Entities
investor_names = ["Alice Johnson", "Michael Chen", "Nina Patel", "Carlos Rivera", "Olivia Thompson"]
company_names = [
    "JP Morgan Chase", "Goldman Sachs", "Wells Fargo", "Citibank", "Bank of America",
    "Morgan Stanley", "Capital One", "US Bancorp", "PNC Financial", "American Express"
]


def aml_fine_calculation():
    """1:Basic: Fine for non-compliance"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    violations = random.randint(2, 5)
    fine_per_violation = random.choice([50000, 75000, 100000])
    question = (
        f"{company} was found to have {violations} AML violations. Each violation incurs a fine of ${fine_per_violation}. "
        f"What is the total fine?"
    )
    total_fine = violations * fine_per_violation
    solution = (
        f"Step 1: Multiply number of violations by fine per violation:\n"
        f"  {violations} × ${fine_per_violation} = ${total_fine}\n"
        f"Answer: Total fine = ${total_fine}"
    )
    return question, solution


def compliance_risk_score():
    """2:Basic: Risk score calculation"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    likelihood = random.randint(1, 5)
    impact = random.randint(1, 5)
    detectability = random.randint(1, 5)
    question = (
        f"{investor} assesses {company}'s compliance risk using a Risk Priority Number (RPN), calculated as:\n"
        f"  RPN = Likelihood × Impact × Detectability\n"
        f"Given: Likelihood = {likelihood}, Impact = {impact}, Detectability = {detectability}\n"
        f"What is the RPN?"
    )
    rpn = likelihood * impact * detectability
    solution = (
        f"Step 1: Multiply all risk factors:\n"
        f"  {likelihood} × {impact} × {detectability} = {rpn}\n"
        f"Answer: RPN = {rpn}"
    )
    return question, solution

# def suspicious_transaction_flag():
#     """Basic: Suspicious Transaction Flagging"""
#     investor = random.choice(investor_names)
#     company = random.choice(company_names)
#     transactions = [random.randint(8000, 15000) for _ in range(4)]
#     threshold = 10000
#     flagged = [amt for amt in transactions if amt > threshold]
#     question = (
#         f"{company} flagged suspicious transactions above ${threshold}. "
#         f"The amounts in a single day were: {transactions}. "
#         f"Which transactions should be flagged?"
#     )
#     solution = (
#         f"Step 1: Identify amounts above ${threshold}:\n"
#         f"  Flagged: {flagged}\n"
#         f"Answer: {len(flagged)} transaction(s) flagged: {flagged}"
#     )
#     return question, solution


def delayed_compliance_fine():
    """3:Intermediate: Fines with time escalation"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    base_fine = 20000
    delay_days = random.randint(3, 7)
    fine_per_day = 5000
    total_fine = base_fine + delay_days * fine_per_day
    question = (
        f"{company} was fined ${base_fine} for late AML reporting. Additionally, ${fine_per_day} is charged for each of the {delay_days} delay days. "
        f"What is the total fine?"
    )
    solution = (
        f"Step 1: Calculate additional fine = {delay_days} × ${fine_per_day} = ${delay_days * fine_per_day}\n"
        f"Step 2: Total fine = ${base_fine} + additional = ${total_fine}\n"
        f"Answer: ${total_fine}"
    )
    return question, solution

# Weighted risk of regions
def regional_risk_weighted_score():
    """4:Intermediate: Weighted risk of regions"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    regions = ["North America", "Europe", "Asia"]
    weights = [round(random.uniform(0.2, 0.5), 2) for _ in regions]
    weights = [round(w / sum(weights), 2) for w in weights]
    risks = [random.randint(3, 9) for _ in regions]
    weighted_risk = round(sum(w * r for w, r in zip(weights, risks)), 2)
    question = (
        f"{company} evaluates compliance risk in three regions: {regions}. "
        f"Assigned weights: {weights}, Risk scores: {risks}. "
        f"What is the weighted average risk score?"
    )
    solution = (
        f"Step 1: Multiply weight × risk for each region:\n"
        + "\n".join([f"  {w} × {r} = {round(w*r,2)}" for w, r in zip(weights, risks)]) + "\n"
        f"Step 2: Sum all weighted risks: {weighted_risk}\n"
        f"Answer: Weighted Risk Score = {weighted_risk}"
    )
    return question, solution


def composite_risk_rating():
    """5:Advanced: Composite risk rating"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    categories = ["Customer", "Transaction", "Geography"]
    scores = [random.randint(1, 5) for _ in categories]
    weights = [0.4, 0.3, 0.3]
    composite = round(sum(s * w for s, w in zip(scores, weights)), 2)
    question = (
        f"{investor} assigns risk scores to {company} as follows:\n"
        f"  Customer Risk = {scores[0]}, Transaction Risk = {scores[1]}, Geography Risk = {scores[2]}\n"
        f"With weights {weights}, what is the composite risk rating?"
    )
    solution = (
        f"Step 1: Multiply scores by weights:\n"
        + "\n".join([f"  {scores[i]} × {weights[i]} = {round(scores[i]*weights[i],2)}" for i in range(3)]) + "\n"
        f"Step 2: Add all values = {composite}\n"
        f"Answer: Composite Risk Rating = {composite}"
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
        aml_fine_calculation,
        compliance_risk_score,
        delayed_compliance_fine,
        regional_risk_weighted_score,
        composite_risk_rating
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
    output_file = "../../testset/finance_regulation/compliance.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")

    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")


if __name__ == "__main__":
   main()