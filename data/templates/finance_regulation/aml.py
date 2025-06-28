import random

investor_names = ["John Doe", "Sarah Nguyen", "Emily White", "Carlos Alvarez", "Ava Patel"]
company_names = [
    "Goldman Sachs", "Wells Fargo", "Bank of America", "JPMorgan Chase", "Morgan Stanley",
    "Citibank", "Charles Schwab", "American Express", "Fidelity Investments", "Robinhood"
]

# BASIC SCENARIOS (1–2 steps)
def aml_basic_threshold_violation():
    """1:Basic: Detect cash transaction above AML threshold"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    amount = random.randint(10001, 25000)
    threshold = 10000
    question = (
        f"{investor} deposited ${amount} in cash into their account at {company}. "
        f"Given the AML threshold for cash reporting is ${threshold}, should this transaction be reported?"
    )
    solution = (
        f"Step 1: Compare the cash deposit (${amount}) with the AML threshold (${threshold}).\n"
        f"Since ${amount} > ${threshold}, the transaction exceeds the threshold.\n\n"
        f"Step 2: Conclusion:\n"
        f"This transaction should be reported under AML regulations."
    )
    return question, solution


def aml_basic_structuring_detection():
    """2:Basic: Detect structuring by splitting deposits"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    amounts = [random.randint(4000, 6000) for _ in range(2)]
    total = sum(amounts)
    threshold = 10000
    question = (
        f"{investor} made two consecutive cash deposits of ${amounts[0]} and ${amounts[1]} into an account at {company}. "
        f"The AML reporting threshold is ${threshold}. Is this an example of structuring?"
    )
    solution = (
        f"Step 1: Add the two deposits: ${amounts[0]} + ${amounts[1]} = ${total}\n"
        f"Step 2: Since ${total} > ${threshold}, and the deposits are split to avoid detection, this indicates structuring.\n\n"
        f"Conclusion: Yes, this is likely an attempt to evade reporting and constitutes structuring."
    )
    return question, solution

# INTERMEDIATE SCENARIOS (3–4 steps)
def aml_intermediate_beneficial_ownership_check():
    """3:Intermediate: Verify disclosure of beneficial ownership"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    ownerships = [random.randint(20, 45) for _ in range(2)]
    final_ownership = sum(ownerships)
    threshold = 25
    question = (
        f"{investor} owns {ownerships[0]}% of a shell company and an additional {ownerships[1]}% via a trust. "
        f"The shell company is opening an account at {company}. Must the beneficial ownership be disclosed?"
    )
    solution = (
        f"Step 1: Add ownership percentages: {ownerships[0]}% + {ownerships[1]}% = {final_ownership}%\n"
        f"Step 2: Compare with AML disclosure threshold of {threshold}%.\n"
        f"Since {final_ownership}% > {threshold}%, beneficial ownership disclosure is required.\n\n"
        f"Conclusion: Yes, the beneficial ownership must be disclosed to comply with AML regulations."
    )
    return question, solution


# def aml_intermediate_transaction_layering_analysis():
#     """Intermediate: Identify layering across international transfers"""
#     investor = random.choice(investor_names)
#     company = random.choice(company_names)
#     steps = random.randint(3, 4)
#     question = (
#         f"{investor} transferred funds through {steps} different international entities before finally depositing them in an account at {company}. "
#         f"Each step obscured the origin of the funds. What AML stage does this represent?"
#     )
#     solution = (
#         f"Step 1: Review the behavior — multiple transfers to obscure origin.\n"
#         f"Step 2: AML processes define 'layering' as the stage involving movement of funds to hide origin.\n"
#         f"Step 3: {steps} transfers indicates classic layering tactics.\n\n"
#         f"Conclusion: This represents the 'Layering' stage of money laundering."
#     )
#     return question, solution

def aml_intermediate_multiple_entities_with_ownership():
    """4:Intermediate: Trace ownership across multiple shell companies"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    layers = [
        {"layer": 1, "ownership": 60},
        {"layer": 2, "ownership": 50},
        {"layer": 3, "ownership": 40},
    ]
    effective_ownership = round(1.0, 2)
    for l in layers:
        effective_ownership *= (l["ownership"] / 100)
    effective_ownership_percent = round(effective_ownership * 100, 2)
    threshold = 25
    question = (
        f"{investor} indirectly owns shares in an entity opening an account at {company} via 3 layered shell companies with "
        f"{layers[0]['ownership']}%, {layers[1]['ownership']}%, and {layers[2]['ownership']}% ownership at each level.\n"
        f"Does this exceed the beneficial ownership threshold of {threshold}%?"
    )
    solution = (
        f"Step 1: Multiply ownership at each layer:\n"
        f"  Effective Ownership = ({layers[0]['ownership']}% × {layers[1]['ownership']}% × {layers[2]['ownership']}%)\n"
        f"                     = {effective_ownership_percent:.2f}%\n\n"
        f"Step 2: Compare with threshold: {effective_ownership_percent:.2f}% {'>' if effective_ownership_percent > threshold else '<='} {threshold}%\n"
        f"Step 3: Conclusion: {'Yes' if effective_ownership_percent > threshold else 'No'}, disclosure {'is' if effective_ownership_percent > threshold else 'is not'} required."
    )
    return question, solution

# def aml_intermediate_transaction_monitoring_volume():
#     """Intermediate: Trigger flag based on transaction volume in short time"""
#     investor = random.choice(investor_names)
#     company = random.choice(company_names)
#     txns = [random.randint(1500, 2500) for _ in range(5)]
#     total = sum(txns)
#     threshold = 10000
#     question = (
#         f"{investor} made the following 5 transfers to an account at {company} within 3 days: {', '.join(f'${t}' for t in txns)}. "
#         f"If the total exceeds ${threshold}, a Suspicious Activity Report (SAR) is triggered. Should a SAR be filed?"
#     )
#     solution = (
#         f"Step 1: Add all transfers: {' + '.join(str(t) for t in txns)} = ${total}\n"
#         f"Step 2: Compare with the threshold: ${total} > ${threshold}\n\n"
#         f"Conclusion: A SAR should be filed due to unusual activity exceeding the reportable threshold in a short period."
#     )
#     return question, solution

# DIFFICULT SCENARIOS (>4 steps)
# def aml_Advanced_risk_weighting_customer_risk_score():
#     """Advanced: Calculate customer risk score using weighted attributes"""
#     investor = random.choice(investor_names)
#     company = random.choice(company_names)
#     weights = {"jurisdiction": 0.4, "transaction_volume": 0.3, "business_type": 0.3}
#     scores = {
#         "jurisdiction": random.randint(1, 5),
#         "transaction_volume": random.randint(1, 5),
#         "business_type": random.randint(1, 5)
#     }
#     risk_score = sum(weights[k] * scores[k] for k in weights)
#     question = (
#         f"{investor} is being onboarded at {company}. The AML risk factors are:\n"
#         f"Jurisdiction Risk: {scores['jurisdiction']}/5\n"
#         f"Transaction Volume Risk: {scores['transaction_volume']}/5\n"
#         f"Business Type Risk: {scores['business_type']}/5\n"
#         f"The weights are 40%, 30%, and 30% respectively.\n"
#         f"What is the overall AML risk score for the customer?"
#     )
#     solution = (
#         f"Step 1: Multiply each risk factor by its weight:\n"
#         f"  Jurisdiction: {weights['jurisdiction']} × {scores['jurisdiction']} = {weights['jurisdiction'] * scores['jurisdiction']:.2f}\n"
#         f"  Transaction Volume: {weights['transaction_volume']} × {scores['transaction_volume']} = {weights['transaction_volume'] * scores['transaction_volume']:.2f}\n"
#         f"  Business Type: {weights['business_type']} × {scores['business_type']} = {weights['business_type'] * scores['business_type']:.2f}\n\n"
#         f"Step 2: Add the results:\n"
#         f"  Risk Score = {risk_score:.2f}"
#     )
#     return question, solution



def aml_Advanced_sar_decision_based_on_risk_score_and_flags():
    """5:Advanced: File SAR based on score and high-risk flags"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    score = random.randint(70, 95)
    flags = random.sample(["PEP", "Unusual transaction", "Offshore entity"], k=2)
    threshold = 75
    question = (
        f"{investor} has a risk score of {score} at {company}. The system also flagged the customer as: {', '.join(flags)}.\n"
        f"If a SAR is triggered when risk score exceeds {threshold} and more than one red flag is present, should a SAR be filed?"
    )
    decision = score > threshold and len(flags) > 1
    solution = (
        f"Step 1: Evaluate risk score: {score} {'>' if score > threshold else '<='} {threshold}\n"
        f"Step 2: Count red flags: {len(flags)} {'>' if len(flags) > 1 else '<='} 1\n"
        f"Step 3: Both conditions are {'met' if decision else 'not met'}.\n\n"
        f"Step 4: Conclusion: {'File' if decision else 'Do not file'} a Suspicious Activity Report."
    )
    return question, solution


# def aml_Advanced_compare_transactions_across_time():
#     """Advanced: Detect trend in transaction volume increase"""
#     investor = random.choice(investor_names)
#     company = random.choice(company_names)
#     txns_week1 = random.randint(2000, 4000)
#     txns_week2 = random.randint(5000, 8000)
#     increase = txns_week2 - txns_week1
#     question = (
#         f"{investor} at {company} had total transfers of ${txns_week1} in Week 1 and ${txns_week2} in Week 2. "
#         f"Calculate the increase and determine if this rapid change is an AML red flag."
#     )
#     solution = (
#         f"Step 1: Subtract to find increase: ${txns_week2} - ${txns_week1} = ${increase}\n"
#         f"Step 2: A sharp increase in short time suggests potential structuring or laundering.\n\n"
#         f"Conclusion: Yes, this is a red flag and should trigger enhanced monitoring."
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
        aml_basic_threshold_violation,
        aml_basic_structuring_detection,
        aml_intermediate_beneficial_ownership_check,
        aml_intermediate_multiple_entities_with_ownership,
        aml_Advanced_sar_decision_based_on_risk_score_and_flags
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
    output_file = "../../testset/finance_regulation/aml.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")

    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")


if __name__ == "__main__":
   main()