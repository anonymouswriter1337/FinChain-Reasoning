import random

company_names = ["Microsoft", "Apple", "NVIDIA", "Amazon", "Alphabet", "Meta Platforms", "Berkshire Hathaway", "Eli Lilly", "Broadcom", "Visa", "JPMorgan Chase", "Tesla", "Walmart", "Mastercard", "UnitedHealth"]


bank_names = ["JPMorgan Chase", "Bank of America", "Wells Fargo", "Goldman Sachs", 
              "Morgan Stanley", "PNC Financial Services", "Capital One"]

def validate_inputs(cost, useful_life, months_used=None, salvage_value=None):
    """

    Validates input parameters for depreciation calculations.

    Scenario: Ensuring that the inputs for asset depreciation calculations meet necessary constraints. 
    The cost and useful life must be positive. If provided, months_used should be between 1 and 12, 
    and salvage_value must be non-negative and less than cost.
    
    Raises:
        ValueError: If any input does not meet the specified requirements.
    """
    if cost <= 0:
        raise ValueError("Cost must be positive")
    if useful_life <= 0:
        raise ValueError("Useful life must be positive")
    if months_used is not None and not (0 < months_used <= 12):
        raise ValueError("Months used must be between 1 and 12")
    if salvage_value is not None and (salvage_value < 0 or salvage_value >= cost):
        raise ValueError("Salvage value must be non-negative and less than cost")

# Template 1 (basic)
def template_depreciation_straight_line():
    """
    1:Basic: Straight-Line Depreciation

    Scenario: A company purchases a machine for a certain cost and uses the straight-line depreciation method
    over a defined useful life with a given salvage value. The goal is to determine the annual depreciation expense.
    
    Returns:
        tuple: (question, solution) where:
            - question (str): A formatted problem statement describing the depreciation scenario.
            - solution (str): Step-by-step solution with calculations.
    """
    company_name = random.choice(company_names)
    cost = random.randint(50000, 100000)
    useful_life = random.randint(5, 15)
    salvage_value = random.randint(5000, 20000)
    
    question = f"""{company_name} purchases a new machine for ${cost}. The estimated useful life of the machine is {useful_life} years,
    and it is expected to have a salvage value of ${salvage_value} at the end of its useful life. {company_name} follows the
    straight-line depreciation method. What is the annual depreciation expense for this machine?
    """
    
    step1_formula = "Depreciable Amount = Cost - Salvage Value"
    step1 = cost - salvage_value
    step2_formula = "Annual Depreciation Expense = Depreciable Amount / Useful Life"
    depreciation_expense = step1 / useful_life
    
    solution = f"""
    Step 1: Compute the depreciable amount using the formula:
            {step1_formula}
            {cost} - {salvage_value} = {step1}
    
    Step 2: Compute the annual depreciation expense:
            {step2_formula}
            {step1} / {useful_life} = {depreciation_expense:.2f} per year.
    """
    
    return question, solution

# Template 2 (basic)
def template_sum_of_years_digits():
    """
    2:Basic: Sum-of-the-Years-Digits Depreciation

    Scenario: A company applies the sum-of-the-years-digits (SYD) method to depreciate an asset.
    The objective is to determine the depreciation expense for a specific year.
    
    Returns:
        tuple: (question, solution) where:
            - question (str): A formatted problem statement describing the depreciation scenario.
            - solution (str): Step-by-step solution with calculations.
    """
    company_name = random.choice(company_names)
    cost = random.randint(60000, 120000)
    useful_life = random.randint(5, 15)
    year = random.randint(1, useful_life)
    
    question = f"""{company_name} purchases an asset for ${cost} and applies the sum-of-the-years-digits (SYD) depreciation method.
    The asset has a useful life of {useful_life} years. What is the depreciation expense for year {year}?
    """
    
    sum_of_years = sum(range(1, useful_life + 1))
    step1_formula = "Depreciation Factor = Remaining Life / Sum of Years"
    depreciation_factor = round((useful_life - year + 1) / sum_of_years, 2)
    step2_formula = "Depreciation Expense = Depreciable Amount * Depreciation Factor"
    depreciable_amount = cost
    depreciation_expense = depreciable_amount * depreciation_factor
    
    solution = f"""
    Step 1: Compute the depreciation factor using the formula:
            {step1_formula}
            ({useful_life} - {year} + 1) / {sum_of_years} = {depreciation_factor:.2f}
    
    Step 2: Compute the depreciation expense:
            {step2_formula}
            {depreciable_amount} * {depreciation_factor:.2f} = {depreciation_expense:.2f}
    """
      
    return question, solution


# Template 3 (intermediate)
def template_revaluation_model():
    """
    3:Intermediate: Revaluation Model for Asset Depreciation

    Generates a financial reasoning question and solution related to the revaluation model for asset depreciation.
    
    Scenario:
    - A company reassesses the fair value of an asset and applies the revaluation model for depreciation calculation.
    - The company follows IFRS guidelines requiring recalculating depreciation after revaluation.
    
    Returns:
        tuple: (question, solution) where:
            - question (str): The problem statement.
            - solution (str): The step-by-step solution explanation.
    """
    company_name = random.choice(company_names)
    cost = random.randint(80000, 150000)
    useful_life = random.randint(5, 20)
    revalued_amount = round(cost * random.uniform(1.1, 1.5), 2)
    new_useful_life = useful_life - random.randint(1, 3)

    # Validate inputs
    validate_inputs(cost, useful_life)
    validate_inputs(revalued_amount, new_useful_life)  # Validate revalued amount separately

    annual_depreciation_old = round(cost / useful_life, 2)
    annual_depreciation_new = round(revalued_amount / new_useful_life, 2)
    
    question = f"""{company_name} initially records a commercial property at ${cost} under the cost model, assigning it a useful life of {useful_life} years.
    After a market reassessment, the company applies the revaluation model, recognizing a fair value increase to ${revalued_amount:.2f}.
    Consequently, the remaining useful life of the asset is reassessed to {new_useful_life} years. Given these changes, calculate the
    new annual depreciation expense the company should report under the revaluation model.
    """
    
    solution = f"""
    Step 1: Compute the initial depreciation expense using the cost model:
            Formula: Depreciation Expense = Cost / Useful Life
            Computation: {cost} / {useful_life} = {annual_depreciation_old:.2f}
    
    Step 2: Recalculate the depreciation expense after revaluation using the new fair value:
            Formula: New Depreciation Expense = Revalued Amount / New Useful Life
            Computation: {revalued_amount:.2f} / {new_useful_life} = {annual_depreciation_new:.2f}
    
    Step 3: Compare and analyze the impact of revaluation on depreciation expense, and its effect on financial statements.
    """
    
    return question, solution


# Template 4 (intermediate)
def template_impairment_loss():
    """
    4:Intermediate: Asset Impairment Loss Calculation

    Generates a financial reasoning question and solution related to asset impairment loss.
    
    Scenario:
    - A company evaluates the recoverability of an asset and records an impairment loss when its carrying amount exceeds the recoverable amount.
    
    Returns:
        tuple: (question, solution) where:
            - question (str): The problem statement.
            - solution (str): The step-by-step solution explanation.
    """
    company_name = random.choice(company_names)
    cost = random.randint(90000, 160000)
    accumulated_depreciation = random.randint(20000, 60000)

    # Validate inputs
    validate_inputs(cost, 1)  # 1 is minimum useful life
    if accumulated_depreciation >= cost:
        raise ValueError("Accumulated depreciation cannot exceed cost")
    
    carrying_amount = cost - accumulated_depreciation
    recoverable_amount = round(carrying_amount * random.uniform(0.7, 0.9), 2)
    impairment_loss = max(0, carrying_amount - recoverable_amount)
    
    question = f"""{company_name} owns an asset originally purchased for ${cost}. Over time, accumulated depreciation has reduced its book value to ${carrying_amount:.2f}.
    Due to changing market conditions, the company performs an impairment test and determines the recoverable amount to be ${recoverable_amount:.2f}.
    If the carrying amount exceeds the recoverable amount, {company_name} must recognize an impairment loss. Calculate the impairment loss, if any, that
    should be reported.
    """
    
    solution = f"""
    Step 1: Determine the carrying amount of the asset:
            Formula: Carrying Amount = Cost - Accumulated Depreciation
            Computation: {cost} - {accumulated_depreciation} = {carrying_amount:.2f}
    
    Step 2: Compare the carrying amount to the recoverable amount:
            Recoverable Amount = {recoverable_amount:.2f}
            If Carrying Amount > Recoverable Amount, then Impairment Loss = Carrying Amount - Recoverable Amount
    
    Step 3: Compute and recognize the impairment loss:
            Formula: Impairment Loss = max(0, Carrying Amount - Recoverable Amount)
            Computation: max(0, {carrying_amount:.2f} - {recoverable_amount:.2f}) = {impairment_loss:.2f}
    """

    
    return question, solution

# Template 5 (advanced)
def template_asset_impairment_reversal():
    """
    5:Advanced: Asset Impairment Loss Reversal

    Generates a financial reasoning question and solution related to the reversal of an asset impairment loss.
    
    Scenario:
    - A company evaluates an impaired asset and subsequently reverses part of the impairment due to improved market conditions.
    - The impairment test follows IFRS guidelines, ensuring that the reversal does not exceed the asset's original book value before impairment.
    
    Returns:
        tuple: (question, solution) where:
            - question (str): The problem statement.
            - solution (str): The step-by-step solution explanation.
    """
    company_name = random.choice(company_names)
    original_cost = random.randint(100000, 200000)
    accumulated_depreciation = random.randint(30000, 80000)
    carrying_amount = round(original_cost - accumulated_depreciation, 2)
    impairment_loss = round(carrying_amount * random.uniform(0.2, 0.4), 2)
    new_carrying_amount = round(carrying_amount - impairment_loss, 2)
    reversal_amount = round(impairment_loss * random.uniform(0.3, 0.6), 2)
    adjusted_carrying_amount = round(new_carrying_amount + reversal_amount, 2)
    
    question = f"""{company_name} initially acquired an industrial asset for ${original_cost}. 
    Over time, depreciation has reduced its book value to ${carrying_amount:.2f}. 
    Due to adverse market conditions, an impairment test results in an impairment loss of ${impairment_loss:.2f}, further decreasing its value.
    However, after a year, the market stabilizes, allowing {company_name} to reverse ${reversal_amount:.2f} of the impairment loss. 
    As per IFRS guidelines, {company_name} must ensure that the adjusted carrying amount does not exceed the book value before impairment.
    
    Determine:
    1. The new carrying amount after impairment.
    2. The adjusted carrying amount after reversal.
    3. Verify whether the adjusted value remains within IFRS limits.
    """
    
    solution = f"""
    Step 1: Compute the carrying amount before impairment:
            Carrying Amount = Cost - Accumulated Depreciation
                            = {original_cost} - {accumulated_depreciation} = {carrying_amount:.2f}
    
    Step 2: Compute the new carrying amount after impairment loss:
            New Carrying Amount = Carrying Amount - Impairment Loss
                                = {carrying_amount:.2f} - {impairment_loss:.2f} = {new_carrying_amount:.2f}
    
    Step 3: Compute the adjusted carrying amount after impairment reversal:
            Adjusted Carrying Amount = New Carrying Amount + Reversal Amount
                                     = {new_carrying_amount:.2f} + {reversal_amount:.2f} = {adjusted_carrying_amount:.2f}
    
    Step 4: Verify that the adjusted carrying amount does not exceed the pre-impairment book value:
            Adjusted Carrying Amount ≤ Carrying Amount before Impairment
            {adjusted_carrying_amount:.2f} ≤ {carrying_amount:.2f} (Valid: {adjusted_carrying_amount <= carrying_amount})
    """

    return question, solution

def main():
    """
    Generate 10 instances of each template with different random seeds 
    and write the results to a JSON file.
    """
    import json
    # List of template functions
    templates = [
        template_depreciation_straight_line,
        template_sum_of_years_digits,
        template_revaluation_model,
        template_impairment_loss,
        template_asset_impairment_reversal
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
    output_file = "../../testset/accounting_and_financial_reporting/depreciation_and_amortization.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()