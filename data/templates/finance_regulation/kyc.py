import random

# Sample named entities
investor_names = ["Alice Johnson", "Brian Thompson", "Catherine Lee", "Daniel Wright", "Evelyn Clark"]
company_names = ["Wells Fargo", "JPMorgan Chase", "Goldman Sachs", "Citibank", "Bank of America"]

# Helper to choose random entities
def get_random_entities():
    return random.choice(investor_names), random.choice(company_names)

### ========== BASIC QUESTIONS (1-2 step) ==========
def kyc_basic_customer_type():
    """1:Basic: Identify customer type for KYC"""
    investor, company = get_random_entities()
    account_type = random.choice(["personal savings", "joint", "business"])
    question = (
        f"{investor} is opening a {account_type} account at {company}. "
        f"According to KYC regulations, what type of customer classification does this account fall under?"
    )
    if account_type == "business":
        classification = "corporate customer"
    else:
        classification = "individual customer"

    solution = (
        f"Step 1: Determine the type of account: {account_type}.\n"
        f"Step 2: Match with KYC classifications:\n"
        f"  - Business → Corporate customer\n"
        f"  - Personal/Joint → Individual customer\n"
        f"Answer: {classification}"
    )
    return question, solution


def kyc_basic_id_required():
    """2:Basic: Check ID document type"""
    investor, company = get_random_entities()
    document = random.choice(["passport", "utility bill", "driver’s license"])
    question = (
        f"{investor} submitted a {document} to {company} for KYC. "
        f"Is this considered a valid form of identity proof under standard KYC regulations?"
    )
    is_valid = document in ["passport", "driver’s license"]
    solution = (
        f"Step 1: Check the submitted document: {document}.\n"
        f"Step 2: Valid ID documents include passport, driver's license, etc.\n"
        f"Answer: {'Yes' if is_valid else 'No'}"
    )
    return question, solution

### ========== INTERMEDIATE QUESTIONS (3-4 steps) ==========

def kyc_risk_level_by_country():
    """3:Intermediate: Determine risk level based on country"""
    investor, company = get_random_entities()
    country = random.choice(["Germany", "Iran", "USA", "North Korea", "France"])
    high_risk_countries = ["Iran", "North Korea"]
    question = (
        f"{investor} is opening an account at {company}, and their nationality is {country}. "
        f"Based on international KYC standards, how should the customer's risk level be classified?"
    )
    risk_level = "High Risk" if country in high_risk_countries else "Standard Risk"
    solution = (
        f"Step 1: Check the country of nationality: {country}.\n"
        f"Step 2: Compare with FATF high-risk list: {high_risk_countries}.\n"
        f"Step 3: {'Country is' if country in high_risk_countries else 'Country is not'} on the list.\n"
        f"Answer: {risk_level}"
    )
    return question, solution



def kyc_source_of_funds_analysis():
    """4:Intermediate: Trace and analyze source of funds"""
    investor, company = get_random_entities()
    income_sources = ["inheritance", "cryptocurrency mining", "salary", "lottery winnings"]
    source = random.choice(income_sources)
    high_risk = ["cryptocurrency mining", "lottery winnings"]
    question = (
        f"{investor} is depositing $50,000 into a new account at {company}. "
        f"The declared source of funds is {source}. According to KYC rules, how should this be handled?"
    )
    solution = (
        f"Step 1: Declared source of funds = {source}\n"
        f"Step 2: Check risk level: {'High' if source in high_risk else 'Low'} risk source\n"
        f"Step 3: {'Unconventional source' if source in high_risk else 'Conventional employment/inheritance'}\n"
        f"Step 4: {'Require additional verification and documentation' if source in high_risk else 'Standard verification is sufficient'}\n"
        f"Answer: {'High-risk, enhanced verification required' if source in high_risk else 'Low-risk, standard KYC applies'}"
    )
    return question, solution


### ========== ADVANCED QUESTIONS (5+ steps) ==========


def kyc_customer_risk_rating():
    """5:Advanced: Compute a customer risk rating using 'highest risk prevails' approach"""
    investor, company = get_random_entities()
    factors = {
        "residency": random.choice(["low-risk country", "high-risk country"]),
        "account_type": random.choice(["savings", "offshore", "business"]),
        "transaction_volume": random.randint(5000, 30000)
    }

    # Determine risk level for each factor
    residency_risk = "High" if factors["residency"] == "high-risk country" else "Low"
    account_risk = "High" if factors["account_type"] in ["offshore", "business"] else "Low"
    volume_risk = "High" if factors["transaction_volume"] > 10000 else "Low"

    # Apply "highest risk prevails" logic
    if "High" in [residency_risk, account_risk, volume_risk]:
        risk_rating = "High"
    else:
        risk_rating = "Low"

    question = (
        f"{investor} is onboarding at {company} with the following profile:\n"
        f"- Residency: {factors['residency']}\n"
        f"- Account Type: {factors['account_type']}\n"
        f"- Expected Transaction Volume: ${factors['transaction_volume']}/month\n"
        f"What is their overall KYC risk rating?"
    )

    solution = (
        f"Step 1: Evaluate residency → {residency_risk} risk\n"
        f"Step 2: Evaluate account type → {account_risk} risk\n"
        f"Step 3: Evaluate transaction volume → {volume_risk} risk\n"
        f"Step 4: Apply 'highest risk prevails' → Final Rating: {risk_rating}\n"
        f"Answer: {risk_rating} Risk"
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
        kyc_basic_customer_type,
        kyc_basic_id_required,
        kyc_risk_level_by_country,
        kyc_source_of_funds_analysis,
        kyc_customer_risk_rating
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
    output_file = "../../testset/finance_regulation/kyc.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")

    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")


if __name__ == "__main__":
   main()