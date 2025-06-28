import random
import json

# Named entities for investors and companies
investor_names = ["John Doe", "Susan Lee", "Emily White", "Mark Smith", "David Brown"]
company_names = ["Apple", "Google", "Microsoft", "Amazon", "Facebook", "Tesla", "Netflix", "Walmart", "JP Morgan", "Berkshire Hathaway"]

# Basic Level Question 1: Synergy Cost Reduction via Percentage Savings
def synergy_basic_cost_reduction():
    """1:Basic: Calculate annual cost savings from percentage cost reduction"""
    # Select investor and two distinct companies
    investor = random.choice(investor_names)
    company_a, company_b = random.sample(company_names, 2)
    # Randomly generate the annual cost bases (in millions)
    cost_base_a = random.randint(50, 200)
    cost_base_b = random.randint(50, 200)
    # Synergy percentage savings (in %)
    synergy_pct = random.randint(5, 20)
    
    question = (
        f"{investor} is evaluating a merger between {company_a} and {company_b}. "
        f"{company_a} has an annual operating cost base of ${cost_base_a} million, and {company_b} has ${cost_base_b} million. "
        f"It is expected that redundant operations will be eliminated, achieving a cost saving of {synergy_pct}% on the combined cost base. "
        f"Calculate the annual cost savings from the merger."
    )
    
    # Step-by-step solution
    total_cost = cost_base_a + cost_base_b
    savings = round(total_cost * (synergy_pct / 100), 2)
    solution = (
        f"Step 1: Compute the total cost base:\n"
        f"         Total Cost = {cost_base_a} + {cost_base_b} = {total_cost} million\n\n"
        f"Step 2: Compute the cost savings:\n"
        f"         Savings = Total Cost × (Synergy Percentage / 100)\n"
        f"                 = {total_cost} × ({synergy_pct} / 100) = {savings} million\n"
        f"Hence, the annual cost savings from the merger is {savings} million dollars."
    )
    
    return question, solution

# Basic Level Question 2: Fixed Cost Savings from Eliminating Redundant Expenses
def synergy_basic_fixed_savings():
    """2:Basic: Calculate total fixed annual cost savings by summing individual savings"""
    investor = random.choice(investor_names)
    company_a, company_b = random.sample(company_names, 2)
    # Each company saves a fixed amount (in millions) by reducing overlapping administrative costs
    saving_a = random.randint(5, 20)
    saving_b = random.randint(5, 20)
    
    question = (
        f"{investor} expects that the merger between {company_a} and {company_b} will cut duplicate administrative costs. "
        f"{company_a} can save ${saving_a} million annually and {company_b} can save ${saving_b} million annually. "
        f"Calculate the total fixed annual cost savings from the merger."
    )
    
    total_fixed_savings = saving_a + saving_b
    solution = (
        f"Step 1: Identify the savings for each company:\n"
        f"         Savings for {company_a} = {saving_a} million\n"
        f"         Savings for {company_b} = {saving_b} million\n\n"
        f"Step 2: Total fixed savings = {saving_a} + {saving_b} = {total_fixed_savings} million\n"
        f"Thus, the merger yields a total annual fixed cost savings of {total_fixed_savings} million dollars."
    )
    
    return question, solution

# Intermediate Level Question 1: Operational Expense Synergy with Integration Cost
def synergy_intermediate_operational_improvement():
    """3:Intermediate: Calculate net annual savings from variable expense reduction and integration cost amortization"""
    investor = random.choice(investor_names)
    company_a, company_b = random.sample(company_names, 2)
    # Variable operating expenses (in millions)
    op_exp_a = random.randint(100, 300)
    op_exp_b = random.randint(100, 300)
    # Synergy percentage for cost reduction (in %)
    synergy_pct = random.randint(10, 25)
    # Integration cost and amortization period
    integration_cost = random.randint(20, 50)
    amortization_years = random.randint(3, 7)
    
    question = (
        f"{investor} plans a merger between {company_a} and {company_b}. "
        f"{company_a} and {company_b} have variable operating expenses of ${op_exp_a} million and ${op_exp_b} million respectively. "
        f"The merger is expected to reduce these expenses by {synergy_pct}% due to operational improvements. "
        f"However, an integration cost of ${integration_cost} million is incurred, amortized over {amortization_years} years. "
        f"Calculate the net annual savings from the merger."
    )
    
    # Compute total operating expense
    total_op_exp = op_exp_a + op_exp_b
    # Synergy savings calculation
    synergy_savings = round(total_op_exp * (synergy_pct / 100), 2)
    # Amortized integration cost per year
    annual_integration_cost = round(integration_cost / amortization_years, 2)
    # Net annual savings
    net_savings = round(synergy_savings - annual_integration_cost, 2)
    
    solution = (
        f"Step 1: Compute the total operating expenses:\n"
        f"         Total Expenses = {op_exp_a} + {op_exp_b} = {total_op_exp} million\n\n"
        f"Step 2: Calculate the synergy savings:\n"
        f"         Synergy Savings = Total Expenses × ({synergy_pct} / 100) = {total_op_exp} × {synergy_pct/100:.2f} = {synergy_savings} million\n\n"
        f"Step 3: Compute the annual integration cost:\n"
        f"         Annual Integration Cost = Integration Cost / Amortization Years = {integration_cost} / {amortization_years} = {annual_integration_cost} million\n\n"
        f"Step 4: Determine net annual savings:\n"
        f"         Net Savings = Synergy Savings - Annual Integration Cost = {synergy_savings} - {annual_integration_cost} = {net_savings} million\n"
        f"Therefore, the net annual savings from the merger is {net_savings} million dollars."
    )
    
    return question, solution

# Intermediate Level Question 2: Supply Chain Efficiency Synergy with Training Cost
def synergy_intermediate_supply_chain_efficiency():
    """4:Intermediate: Calculate net annual supply chain savings after accounting for training cost amortization"""
    investor = random.choice(investor_names)
    company_a, company_b = random.sample(company_names, 2)
    # Procurement costs (in millions)
    procurement_a = random.randint(50, 150)
    procurement_b = random.randint(50, 150)
    # Synergy percentage for procurement cost reduction (in %)
    synergy_pct = random.randint(5, 15)
    # Training cost for new supply chain system and amortization period
    training_cost = random.randint(5, 15)
    amortization_years = random.randint(2, 5)
    
    question = (
        f"{investor} anticipates that merging {company_a} and {company_b} will improve supply chain efficiency. "
        f"{company_a} has procurement costs of ${procurement_a} million and {company_b} has ${procurement_b} million. "
        f"The consolidation is expected to reduce procurement costs by {synergy_pct}% through better vendor negotiations. "
        f"A training cost of ${training_cost} million for a new supply chain system will be incurred, amortized over {amortization_years} years. "
        f"Calculate the net annual savings from the supply chain synergy."
    )
    
    total_procurement = procurement_a + procurement_b
    synergy_savings = round(total_procurement * (synergy_pct / 100), 2)
    annual_training_cost = round(training_cost / amortization_years, 2)
    net_savings = round(synergy_savings - annual_training_cost, 2)
    
    solution = (
        f"Step 1: Compute the total procurement cost:\n"
        f"         Total Procurement Cost = {procurement_a} + {procurement_b} = {total_procurement} million\n\n"
        f"Step 2: Calculate the procurement savings:\n"
        f"         Savings = Total Procurement Cost × ({synergy_pct} / 100) = {total_procurement} × {synergy_pct/100:.2f} = {synergy_savings} million\n\n"
        f"Step 3: Determine the annual training cost:\n"
        f"         Annual Training Cost = Training Cost / Amortization Years = {training_cost} / {amortization_years} = {annual_training_cost} million\n\n"
        f"Step 4: Net annual savings = Procurement Savings - Annual Training Cost = {synergy_savings} - {annual_training_cost} = {net_savings} million\n"
        f"Hence, the net annual supply chain savings is {net_savings} million dollars."
    )
    
    return question, solution

# Advanced Level Question: Full Financial Adjustment with Multiple Synergies and Tax Impact
def synergy_advanced_full_financial_adjustment():
    """5:Advanced: Calculate net annual synergy savings after cost synergy, revenue synergy, restructuring cost, and tax adjustment (min 5 steps)"""
    investor = random.choice(investor_names)
    company_a, company_b = random.sample(company_names, 2)
    # Combined operating expenses (in millions)
    operating_expense = random.randint(200, 500)
    # Cost synergy percentage (in %)
    synergy_pct = random.randint(10, 20)
    # Revenue synergy bonus (in millions)
    revenue_synergy = random.randint(10, 50)
    # Restructuring cost (in millions) and its amortization period (in years)
    restructuring_cost = random.randint(50, 100)
    amortization_years = random.randint(3, 7)
    # Tax rate (in %)
    tax_rate = random.randint(20, 35)
    
    question = (
        f"{investor} is evaluating a complex merger between {company_a} and {company_b}. "
        f"The combined operating expenses total ${operating_expense} million, and the merger is projected to achieve a cost synergy of {synergy_pct}% on these expenses. "
        f"In addition, revenue synergies are expected to add an extra ${revenue_synergy} million annually. "
        f"However, the merger incurs restructuring costs of ${restructuring_cost} million, amortized over {amortization_years} years. "
        f"Given an effective tax rate of {tax_rate}%, calculate the net annual synergy savings after accounting for integration costs and tax adjustments."
    )
    
    # Step 1: Compute cost synergy savings.
    cost_synergy = round(operating_expense * (synergy_pct / 100), 2)
    
    # Step 2: Identify the revenue synergy bonus.
    # (This is given directly as revenue_synergy)
    
    # Step 3: Calculate the total synergy savings before integration costs.
    total_synergy = round(cost_synergy + revenue_synergy, 2)
    
    # Step 4: Compute the annual restructuring (integration) cost.
    annual_restructuring = round(restructuring_cost / amortization_years, 2)
    
    # Step 5: Determine the preliminary net synergy savings after subtracting integration costs.
    preliminary_net = round(total_synergy - annual_restructuring, 2)
    
    # Step 6: Adjust for the tax impact to obtain the final net annual savings.
    net_annual_savings = round(preliminary_net * (1 - tax_rate / 100), 2)
    
    solution = (
        f"Step 1: Calculate cost synergy savings:\n"
        f"         Cost Synergy = Operating Expense × (Synergy Percentage / 100)\n"
        f"                      = {operating_expense} × ({synergy_pct} / 100) = {cost_synergy} million\n\n"
        f"Step 2: Identify the revenue synergy bonus:\n"
        f"         Revenue Synergy = {revenue_synergy} million\n\n"
        f"Step 3: Compute total synergy savings before integration costs:\n"
        f"         Total Synergy = Cost Synergy + Revenue Synergy = {cost_synergy} + {revenue_synergy} = {total_synergy} million\n\n"
        f"Step 4: Determine the annual restructuring cost:\n"
        f"         Annual Restructuring Cost = Restructuring Cost / Amortization Years\n"
        f"                                = {restructuring_cost} / {amortization_years} = {annual_restructuring} million\n\n"
        f"Step 5: Compute preliminary net savings (before tax adjustment):\n"
        f"         Preliminary Net Savings = Total Synergy - Annual Restructuring Cost = {total_synergy} - {annual_restructuring} = {preliminary_net} million\n\n"
        f"Step 6: Adjust for tax impact:\n"
        f"         Net Annual Savings = Preliminary Net Savings × (1 - Tax Rate / 100)\n"
        f"                           = {preliminary_net} × (1 - {tax_rate} / 100) = {net_annual_savings} million\n\n"
        f"Thus, the net annual synergy savings after tax adjustments is {net_annual_savings} million dollars."
    )
    
    return question, solution


def main():
    """
    Generate one instance for each of the five merger synergy cost saving templates and write the results to a JSONL file.
    """
    # List of template functions
    templates = [
        synergy_basic_cost_reduction,
        synergy_basic_fixed_savings,
        synergy_intermediate_operational_improvement,
        synergy_intermediate_supply_chain_efficiency,
        synergy_advanced_full_financial_adjustment
    ]
    
    # List to store all generated problems
    all_problems = []
    
    # Generate one instance per template
    for template_func in templates:
        id = template_func.__doc__.split(':')[0].strip()
        level = template_func.__doc__.split(':')[1].strip()
        
        # Set a unique random seed for reproducibility
        for i in range(10):
        # Generate a unique seed for each problem
            seed = random.randint(1000000000, 4000000000)
            random.seed(seed)
            
            # Generate the question and solution
            question, solution = template_func()
            
            # Create a JSON entry for the problem
            problem_entry = {
                "seed": seed,
                "id": id,
                "level": level,
                "question": question,
                "solution": solution
            }
            
            all_problems.append(problem_entry)
            
            # Reset random seed after each instance
            random.seed()
    
    random.shuffle(all_problems)
    # Write all problems to a JSONL file
    output_file = "../../testset/mergers_and_acquisitions/synergies_and_cost_savings.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
    main()