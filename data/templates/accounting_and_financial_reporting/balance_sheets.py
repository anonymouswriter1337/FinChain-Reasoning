import random

company_names = ["Microsoft", "Apple", "NVIDIA", "Amazon", "Alphabet", "Meta Platforms", "Berkshire Hathaway", "Eli Lilly", "Broadcom", "Visa", "JPMorgan Chase", "Tesla", "Walmart", "Mastercard", "UnitedHealth"]


bank_names = ["JPMorgan Chase", "Bank of America", "Wells Fargo", "Goldman Sachs", 
              "Morgan Stanley", "PNC Financial Services", "Capital One"]

# Template 1 (basic)
def template_cash_accounts_payable():
    """
    1:Basic: Cash and Accounts Payable

    Scenario:
        A company needs to assess its net cash position after settling its accounts payable.
        Given the company's available cash and outstanding accounts payable, the goal is 
        to compute the net cash position using:

            Net Cash Position = Cash - Accounts Payable

    Returns:
        tuple: A tuple containing:
            - str: A question asking to compute the net cash position after paying off accounts payable.
            - str: A step-by-step solution explaining the calculation.
    """
    company_name = random.choice(company_names)
    cash = random.randint(5000, 20000)
    accounts_payable = random.randint(2000, 10000)
    net_cash = cash - accounts_payable

    question = (
        f"{company_name} has ${cash} in cash and ${accounts_payable} in accounts payable. Calculate the net cash position "
        f"of the company after paying off all its accounts payable."
    )

    solution = (
        f"Step 1. Net Cash Position = Cash - Accounts Payable\n"
        f"                          = {cash} - {accounts_payable} = ${net_cash}"
    )

    return question, solution

# Template 2 (basic)
def template_balance_sheet_equation():
    """
    2:Basic: Balance Sheet Equation

    Scenario:
        A company’s financial position is assessed using the balance sheet equation.
        Given the company’s assets and liabilities, the goal is to compute the equity 
        using the fundamental accounting equation: 

            Equity = Assets - Liabilities

    Returns:
        tuple: A tuple containing:
            - str: A question asking to compute the company's equity.
            - str: A step-by-step solution explaining the calculation.
    """
    company_name = random.choice(company_names)
    assets = random.randint(10000, 50000)
    liabilities = random.randint(5000, 30000)
    equity = assets - liabilities

    question = (
        f"{company_name} has ${assets} in assets and ${liabilities} in liabilities. Using the balance sheet equation, "
        f"calculate the company’s equity."
    )

    solution = (
        f"Step 1. Equity = Assets - Liabilities\n"
        f"               = {assets} - {liabilities} = ${equity}"
    )

    return question, solution


# Template 6 (intermediate)
def template_inventory_valuation():
    """
    3:Intermediate: Inventory Valuation

    Scenario:
        A company is evaluating its inventory using different valuation methods.
        Given the starting inventory, purchases, and cost of goods sold (COGS) under 
        FIFO, LIFO, and WAC methods, the goal is to compute the closing inventory 
        using the formula:

            Closing Inventory = Starting Inventory + Purchases - COGS

    Returns:
        tuple: A tuple containing:
            - str: A question asking to compute the closing inventory under different methods.
            - str: A step-by-step solution explaining the calculation.
    """
    # Data generation
    company_name = random.choice(company_names)
    inventory_start = random.randint(500, 1000)
    purchases = random.randint(2000, 5000)
    sales = random.randint(1500, 3000)
    fifo_cogs = random.randint(1000, 2000)
    lifo_cogs = fifo_cogs + random.randint(-200, 200)
    wac_cogs = fifo_cogs + random.randint(-100, 100)

    question = (
        f"{company_name} started the year with an inventory worth ${inventory_start}. Over the year, it made purchases worth "
        f"${purchases} and sold goods costing ${sales} using different valuation methods. Under FIFO, the cost of goods sold "
        f"was ${fifo_cogs}, under LIFO it was ${lifo_cogs}, and under WAC it was ${wac_cogs}. Calculate the closing inventory "
        f"for each method."
    )

    solution = (
        f"Using the formula: Closing Inventory = Starting Inventory + Purchases - COGS\n"
        f"Step 1. FIFO: {inventory_start} + {purchases} - {fifo_cogs} = ${inventory_start + purchases - fifo_cogs}\n"
        f"Step 2. LIFO: {inventory_start} + {purchases} - {lifo_cogs} = ${inventory_start + purchases - lifo_cogs}\n"
        f"Step 3. WAC: {inventory_start} + {purchases} - {wac_cogs} = ${inventory_start + purchases - wac_cogs}"
    )

    return question, solution

# Template 7 (intermediate)
def template_deferred_tax():
    """
    4:Intermediate: Deferred Tax Calculation

    Scenario:
        A company reports a temporary difference between its tax base and 
        carrying amount of an asset. Given the profit before tax, the temporary 
        difference, and the tax rate, the goal is to compute the deferred tax 
        asset/liability and the taxable profit using:

            Taxable Profit = Profit Before Tax - Temporary Difference
            Deferred Tax = Temporary Difference * Tax Rate

    Returns:
        tuple: A tuple containing:
            - str: A question asking to compute the deferred tax asset/liability and taxable profit.
            - str: A step-by-step solution explaining the calculation.
    """
    company_name = random.choice(company_names)
    profit_before_tax = random.randint(10000, 50000)
    temporary_difference = random.randint(2000, 8000)
    tax_rate = random.randint(20, 30) / 100
    tax_base = profit_before_tax - temporary_difference
    deferred_tax = temporary_difference * tax_rate

    question = (
        f"{company_name} reported a profit before tax of ${profit_before_tax}. There was a temporary difference of "
        f"${temporary_difference} between the tax base and the carrying amount of an asset. If the tax rate is "
        f"{int(tax_rate * 100)}%, calculate the deferred tax asset or liability and the taxable profit."
    )

    solution = (
        f"Step 1: Taxable profit = Profit Before Tax - Temporary Difference\n"
        f"                       = {profit_before_tax} - {temporary_difference} = ${tax_base}\n"
        f"Step 2: Deferred Tax = Temporary Difference * Tax Rate\n"
        f"                     = {temporary_difference} * {tax_rate} = ${deferred_tax}"
    )

    return question, solution

# Template 15 (advanced)
def template_business_combination():
    """
    5:Advanced: Business Combination Analysis

    Scenario:
        A company acquires a controlling stake in another company. 
        Given the purchase price, the net assets value, and the fair value of 
        the non-controlling interest (NCI), the goal is to compute goodwill. 
        Additionally, the net profit is allocated between the parent company 
        and the NCI using:

            Goodwill = Purchase Price + NCI Value - Net Assets Value
            Parent’s Share of Net Profit = Net Profit * Parent’s Ownership Percentage
            NCI’s Share of Net Profit = Net Profit * (1 - Parent’s Ownership Percentage)

    Returns:
        tuple: A tuple containing:
            - str: A question asking to compute goodwill, parent company's share of net profit, and NCI's share.
            - str: A step-by-step solution explaining the calculations.
    """
    company_name1, company_name2 = random.sample(company_names, 2)
    purchase_price = 20000000  # Price paid for the acquisition
    net_assets_value = 15000000  # Fair value of the net assets
    nci_value = 5000000  # Fair value of the non-controlling interest (NCI)
    goodwill = purchase_price + nci_value - net_assets_value
    net_profit = 2000000  # Net profit of the subsidiary
    parent_share = 0.8  # Parent's ownership percentage
    parent_profit_share = net_profit * parent_share
    nci_profit_share = net_profit * (1 - parent_share)

    question = (
        f"{company_name1} acquires an 80% stake in {company_name2} for ${purchase_price}. The fair value of {company_name2}’s net identifiable assets "
        f"is ${net_assets_value}, and the fair value of the non-controlling interest (NCI) is ${nci_value}. During the year, {company_name2} "
        f"reported a net profit of ${net_profit}. Calculate the goodwill arising from the acquisition, the parent company’s share "
        f"of net profit, and the share of net profit attributable to NCI."
    )

    solution = (
        f"Step 1. Goodwill = Purchase Price + NCI Value - Net Assets Value\n"
        f"   = {purchase_price} + {nci_value} - {net_assets_value} = ${goodwill}\n"
        f"Step 2. Parent Company’s Share of Net Profit = Net Profit * Parent’s Ownership Percentage\n"
        f"   = {net_profit} * {parent_share} = ${parent_profit_share}\n"
        f"Step 3. NCI’s Share of Net Profit = Net Profit * (1 - Parent’s Ownership Percentage)\n"
        f"   = {net_profit} * {1 - parent_share} = ${nci_profit_share}"
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
        template_cash_accounts_payable,
        template_balance_sheet_equation,
        template_inventory_valuation,
        template_deferred_tax,
        template_business_combination
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
    output_file = "../../testset/accounting_and_financial_reporting/balance_sheets.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()