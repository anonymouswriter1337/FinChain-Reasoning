import random


company_names = ["Microsoft", "Apple", "NVIDIA", "Amazon", "Alphabet", "Meta Platforms", "Berkshire Hathaway", "Eli Lilly", "Broadcom", "Visa", "JPMorgan Chase", "Tesla", "Walmart", "Mastercard", "UnitedHealth"]

bank_names = ["JPMorgan Chase", "Bank of America", "Wells Fargo", "Goldman Sachs", 
              "Morgan Stanley", "PNC Financial Services", "Capital One"]

def generate_random_value(low, high):
    return random.randint(low, high)


def validate_financials(revenue, cogs, operating_expenses=0, depreciation=0):
    """
    Validates basic financial metrics to ensure they make business sense.
    
    Parameters:
        revenue (float): Total revenue, must be positive.
        cogs (float): Cost of Goods Sold (COGS), cannot be negative and must not exceed revenue.
        operating_expenses (float, optional): Operating expenses, default is 0, cannot be negative.
        depreciation (float, optional): Depreciation expense, default is 0, cannot be negative.
    
    Raises:
        ValueError: If any of the financial constraints are violated.
    """

    if revenue <= 0:
        raise ValueError("Revenue must be positive")
    if cogs < 0:
        raise ValueError("COGS cannot be negative")
    if operating_expenses < 0:
        raise ValueError("Operating expenses cannot be negative")
    if depreciation < 0:
        raise ValueError("Depreciation cannot be negative")
    if cogs > revenue:
        raise ValueError("COGS cannot exceed revenue")
    if (cogs + operating_expenses + depreciation) > revenue:
        raise ValueError("Total costs cannot exceed revenue")
    
# Template 1 (basic)
def template_revenue_vs_cogs():
    """
    1:Basic: Revenue and Cost of Goods Sold (COGS)

    Generates a financial reasoning question for calculating Gross Profit.
    
    Scenario:
    A company wants to determine its Gross Profit by subtracting its Cost of Goods Sold (COGS) from its total revenue.
    
    Returns:
    tuple: (question, solution) where question is a formatted financial scenario,
           and solution provides a step-by-step breakdown of the answer.
    """
    company_name = random.choice(company_names)
    revenue = random.randint(50000, 100000)
    cogs = random.randint(10000, 50000)
    question = f"{company_name} has a total revenue of ${revenue:,} and its Cost of Goods Sold (COGS) is ${cogs:,}. What is its Gross Profit?"
    solution = (
        f"Step-1: Identify the values given:\n"
        f"  Revenue = ${revenue:,}\n"
        f"  COGS = ${cogs:,}\n\n"
        f"Step-2: Use the formula for Gross Profit:\n"
        f"  Gross Profit = Revenue - COGS\n"
        f"               = ${revenue:,} - ${cogs:,} = ${revenue - cogs:,}"
    )
    
    return question, solution

# Template 2 (basic)
def template_operating_expenses_effect():
    """
    2:Basic: Operating Expenses Effect

    Generates a financial reasoning question for calculating Operating Profit.
    
    Scenario:
    A company wants to determine its Operating Profit by deducting Operating Expenses from its Gross Profit.
    
    Returns:
    tuple: (question, solution) where question is a formatted financial scenario,
           and solution provides a step-by-step breakdown of the answer.
    """
    company_name = random.choice(company_names)
    gross_profit = random.randint(40000, 80000)
    operating_expenses = random.randint(10000, 30000)
    question = (f"{company_name}'s Gross Profit is ${gross_profit:,}, and its Operating Expenses amount to "
                f"${operating_expenses:,}. What is its Operating Profit?")
    solution = (
        f"Step-1: Identify the values given:\n"
        f"  Gross Profit = ${gross_profit:,}\n"
        f"  Operating Expenses = ${operating_expenses:,}\n\n"
        f"Step-2: Use the formula for Operating Profit:\n"
        f"  Operating Profit = Gross Profit - Operating Expenses\n"
        f"                   = ${gross_profit:,} - ${operating_expenses:,} = ${gross_profit - operating_expenses:,}"
    )

    return question, solution


# Template 3 (intermediate)
def template_break_even_analysis():
    """
    3:Intermediate: Break-Even Analysis

    Generates a financial reasoning question for Break-Even Analysis.
    
    Scenario:
    A manufacturing company wants to determine its Break-Even Point and the number of units required to achieve a target profit.
    
    Returns:
    tuple: (question, solution) where question is a formatted financial scenario,
           and solution provides a step-by-step breakdown of the answer.
    """
    try: 
        # Generate random values
        company_name = random.choice(company_names)
        sales_price_per_unit = generate_random_value(50, 150)
        variable_cost_per_unit = generate_random_value(20, 50)
        fixed_costs = generate_random_value(50000, 200000)
        target_profit = generate_random_value(20000, 100000)

        # Validate inputs
        if sales_price_per_unit <= 0:
            raise ValueError("Sales price per unit must be positive")
        if variable_cost_per_unit <= 0:
            raise ValueError("Variable cost per unit must be positive")
        if fixed_costs <= 0:
            raise ValueError("Fixed costs must be positive")
        
        # Check for negative contribution margin
        contribution_margin = sales_price_per_unit - variable_cost_per_unit
        if contribution_margin <= 0:
            raise ValueError("Contribution margin is negative or zero. Price must exceed variable cost.")

        # Calculate break-even point
        break_even_units = fixed_costs / contribution_margin
        if break_even_units < 0:
            raise ValueError("Break-even point calculation resulted in negative units.")

        # Calculate target profit units
        target_units = (fixed_costs + target_profit) / contribution_margin
        if target_units < break_even_units:
            raise ValueError("Target profit calculation error: Required units less than break-even point.")

        question = (
            f"{company_name} manufactures a product that sells for ${sales_price_per_unit} per unit. The production cost includes "
            f"a variable cost of ${variable_cost_per_unit} per unit and fixed costs totaling ${fixed_costs} annually. The management "
            f"team is analyzing how many units need to be sold to cover costs and achieve profitability."
            f"Determine the Contribution Margin per unit, the Break-Even Point in units, and the Sales Volume required to meet "
            f"a target profit of ${target_profit}. Provide your answers step by step."
        )

        solution = (
            f"Step 1: Calculate the Contribution Margin per unit.\n"
            f"Contribution Margin = Sales Price per Unit - Variable Cost per Unit\n"
            f"Contribution Margin = {sales_price_per_unit} - {variable_cost_per_unit} = {sales_price_per_unit - variable_cost_per_unit}\n\n"
            f"Step 2: Calculate the Break-Even Point in units.\n"
            f"Break-Even Point = Fixed Costs / Contribution Margin\n"
            f"Break-Even Point = {fixed_costs} / {sales_price_per_unit - variable_cost_per_unit} = "
            f"{fixed_costs / (sales_price_per_unit - variable_cost_per_unit):.2f} units\n\n"
            f"Step 3: Calculate the Sales Volume required to achieve the target profit.\n"
            f"Sales Volume = (Fixed Costs + Target Profit) / Contribution Margin\n"
            f"Sales Volume = ({fixed_costs} + {target_profit}) / {sales_price_per_unit - variable_cost_per_unit} = "
            f"{(fixed_costs + target_profit) / (sales_price_per_unit - variable_cost_per_unit):.2f} units"
        )

        return question, solution

    except ValueError as e:
        return f"Error generating question: {str(e)}", None


# Template 4 (intermediate)
def template_tax_shield_analysis():
    """
    4:Intermediate: Tax Shield Analysis

    Generates a financial reasoning question for Tax Shield Analysis.
    
    Scenario:
    A company wants to determine the tax-saving benefits of depreciation expenses on taxable income.
    
    Returns:
    tuple: (question, solution) where question is a formatted financial scenario,
           and solution provides a step-by-step breakdown of the answer.
    """
    try:
        # Generate random values
        company_name = random.choice(company_names)
        revenue = generate_random_value(700000, 1200000)
        cogs = generate_random_value(300000, 600000)
        operating_expenses = generate_random_value(100000, 300000)
        depreciation = generate_random_value(50000, 100000)
        tax_rate = random.randint(20, 35)  # as a percentage

        # Validate inputs
        validate_financials(revenue, cogs, operating_expenses, depreciation)
        
        if tax_rate <= 0 or tax_rate >= 100:
            raise ValueError("Tax rate must be between 0 and 100 percent")

        # Calculate taxable income before depreciation
        taxable_income_before = revenue - cogs - operating_expenses
        if taxable_income_before < 0:
            raise ValueError("Business is operating at a loss before depreciation")

        # Calculate tax shield
        tax_shield = depreciation * (tax_rate / 100)
        if tax_shield > taxable_income_before * (tax_rate / 100):
            raise ValueError("Tax shield exceeds maximum possible tax benefit")


        question = (
            f"{company_name} earned ${revenue} in revenue, with a Cost of Goods Sold (COGS) of ${cogs} and operating expenses totaling ${operating_expenses}. "
            f"They also recorded a depreciation expense of ${depreciation}. The corporate tax rate is {tax_rate}%. Calculate the tax shield effect "
            f"of depreciation and determine the taxable income with and without the depreciation expense applied."
        )

        solution = (
            f"Step 1: Calculate Taxable Income without considering Depreciation.\n"
            f"Taxable Income (Before Depreciation) = Revenue - COGS - Operating Expenses\n"
            f"                                     = {revenue} - {cogs} - {operating_expenses} = {revenue - cogs - operating_expenses}\n\n"
            f"Step 2: Calculate Taxable Income after considering Depreciation.\n"
            f"Taxable Income (After Depreciation) = Taxable Income (Before Depreciation) - Depreciation\n"
            f"                                    = {revenue - cogs - operating_expenses} - {depreciation} = {revenue - cogs - operating_expenses - depreciation}\n\n"
            f"Step 3: Calculate the Tax Shield effect of Depreciation.\n"
            f"Tax Shield = Depreciation × (Tax Rate / 100)\n"
            f"           = {depreciation} × ({tax_rate} / 100) = {depreciation * (tax_rate / 100):.2f}"
        )

        return question, solution
    except ValueError as e:
        return f"Error generating question: {str(e)}", None
    


# Template 5 (advanced)
def template_goodwill_impairment():
    """
    5:Advanced: Goodwill Impairment Analysis

    Generates a financial reasoning question evaluating the impact of goodwill impairment on the income statement.
    
    Scenario:
    A company acquires a subsidiary, leading to the recognition of goodwill. This scenario examines how goodwill impairment affects operating income and net income after taxes.
    
    Returns:
    tuple: (question, solution) where question is a formatted financial scenario,
           and solution provides a step-by-step breakdown of the answer.
    """
    # Generate random values
    year = generate_random_value(2000, 2025)
    company_name1, company_name2 = random.sample(company_names, 2)
    acquisition_price = generate_random_value(1000000, 5000000)
    fair_value_of_assets = generate_random_value(800000, 4500000)
    goodwill_impairment = generate_random_value(50000, 200000)
    revenue = generate_random_value(2000000, 6000000)
    operating_expenses = generate_random_value(1000000, 3000000)
    tax_rate = random.randint(20, 35)

    question = (
        f"{company_name1} acquired {company_name2} for ${acquisition_price}, and the fair value of the acquired net assets was ${fair_value_of_assets}. "
        f"The difference, recorded as goodwill, was later subject to an impairment test, resulting in a goodwill impairment loss of ${goodwill_impairment}. "
        f"In the same fiscal year {year}, the company reported revenue of ${revenue} and incurred operating expenses of ${operating_expenses}. With a corporate tax rate "
        f"of {tax_rate}%, calculate the following: \n1. Goodwill recorded at acquisition.\n2. Operating income after accounting for goodwill impairment.\n3. Net income after considering taxes. Provide detailed calculations."
    )

    solution = (
        f"Step 1. Calculate the goodwill recorded at acquisition.\n"
        f"Goodwill = Acquisition Price - Fair Value of Assets\n"
        f"         = {acquisition_price} - {fair_value_of_assets} = {acquisition_price - fair_value_of_assets}\n\n"
        f"Step 2. Calculate Operating Income after goodwill impairment.\n"
        f"Operating Income = Revenue - Operating Expenses - Goodwill Impairment\n"
        f"                 = {revenue} - {operating_expenses} - {goodwill_impairment} = {revenue - operating_expenses - goodwill_impairment}\n\n"
        f"Step 3. Calculate Net Income before taxes.\n"
        f"Net Income (Before Tax) = Operating Income\n"
        f"                        = {revenue - operating_expenses - goodwill_impairment}\n\n"
        f"Step 4. Calculate taxes and Net Income after taxes.\n"
        f"Taxes = Net Income (Before Tax) × (Tax Rate / 100)\n"
        f"      = {revenue - operating_expenses - goodwill_impairment} × ({tax_rate} / 100) = "
        f"{((revenue - operating_expenses - goodwill_impairment) * (tax_rate / 100)):.2f}\n"
        f"Net Income (After Tax) = Net Income (Before Tax) - Taxes\n"
        f"                       = {revenue - operating_expenses - goodwill_impairment} - "
        f"{((revenue - operating_expenses - goodwill_impairment) * (tax_rate / 100)):.2f} = "
        f"{((revenue - operating_expenses - goodwill_impairment) - ((revenue - operating_expenses - goodwill_impairment) * (tax_rate / 100))):.2f}"
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
        template_revenue_vs_cogs,
        template_operating_expenses_effect,
        template_break_even_analysis,
        template_tax_shield_analysis,
        template_goodwill_impairment
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
    output_file = "../../testset/accounting_and_financial_reporting/income_statements.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()
