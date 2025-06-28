# https://en.wikipedia.org/wiki/Return_on_investment

import random

# Named entities for investors and projects
investor_names = ["John Doe", "Susan Lee", "Emily White", "Mark Smith", "David Brown"]
project_names = [
    "Tesla Gigafactory", "Apple iPhone Launch", "Amazon Web Services Expansion", "SpaceX Starship Development",
    "Google Data Center Build", "Microsoft Azure", "Netflix Content Production", "Uber Autonomous Driving Initiative",
    "Facebook Metaverse", "Samsung Semiconductor Factory"
]

# Template 1: ROI (Return on Investment) with Revenue and Fixed Investment
def template_revenue_roi():
    """1:Basic: ROI with revenue and fixed investment"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    x = random.randint(20000, 60000)  # Initial investment
    R = random.randint(50000, 100000)  # Revenue
    question = (
        f"{investor_name} invested ${x} in {project_name}, which generated ${R} in revenue. "
        f"What is the ROI for {project_name}?"
    )
    net_profit = R - x
    ROI = (net_profit / x) * 100
    solution = (
        f"Step 1: Compute the net profit:\n"
        f"  Net Profit = Revenue - Investment Cost = {R} - {x} = {net_profit}\n\n"
        f"Step 2: Compute the ROI:\n"
        f"  ROI = (Net Profit / Investment Cost) × 100 = ({net_profit} / {x}) × 100 = {ROI:.2f}%"
    )
    return question, solution

# Template 2: ROI with Tax Implications
def template_tax_implications_roi():
    """2:Basic: ROI with tax implications"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    x = random.randint(30000, 70000)  # Initial investment
    R = random.randint(50000, 150000)  # Revenue before tax
    t = round(random.uniform(10, 30), 2)         # Tax rate
    question = (
        f"{investor_name} invested ${x} in {project_name}, which generated ${R} in revenue. "
        f"The tax rate is {t:.2f}%. What is the after-tax ROI for {project_name}?"
    )
    tax = R * (t / 100)
    R_after_tax = R - tax
    net_profit = R_after_tax - x
    ROI = (net_profit / x) * 100
    solution = (
        f"Step 1: Compute the after-tax revenue and net profit:\n"
        f"  Revenue_after_tax = Revenue - Tax = {R} - ({R} × {t / 100:.4f}) = {R_after_tax:.2f}\n"
        f"  Net Profit = Revenue_after_tax - Investment Cost = {R_after_tax:.2f} - {x} = {net_profit}\n\n"
        f"Step 2: Compute the ROI:\n"
        f"  ROI = (Net Profit / Investment Cost) × 100 = ({net_profit} / {x}) × 100 = {ROI:.2f}%"
    )
    return question, solution

# Medium level: three steps in the solution

# Template 3: ROI with Multi-Year Revenue
def template_multi_year_revenue():
    """3:Intermediate: ROI with multi-year revenue"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    x = random.randint(30000, 70000)  # Initial investment
    R1 = random.randint(10000, 30000)  # Revenue Year 1
    R2 = random.randint(10000, 30000)  # Revenue Year 2
    question = (
        f"{investor_name} invested ${x} in {project_name}, which generated ${R1} in Year 1 and ${R2} in Year 2. "
        f"What is the ROI for {project_name} after 2 years?"
    )
    total_revenue = R1 + R2
    net_profit = total_revenue - x
    ROI = (net_profit / x) * 100
    solution = (
        f"Step 1: Compute the total revenue:\n"
        f"  Total Revenue = Year 1 + Year 2 = {R1} + {R2} = {total_revenue}\n\n"
        f"Step 2: Compute the net profit:\n"
        f"  Net Profit = Total Revenue - Initial Investment = {total_revenue} - {x} = {net_profit}\n\n"
        f"Step 3: Compute the ROI:\n"
        f"  ROI = (Net Profit / Initial Investment) × 100 = ({net_profit} / {x}) × 100 = {ROI:.2f}%"
    )
    return question, solution

# Template 4: ROI with Tax and Additional Costs
def template_tax_and_costs_roi():
    """4:Intermediate: ROI with tax and additional costs"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    x = random.randint(30000, 70000)  # Initial investment
    R = random.randint(50000, 150000)  # Revenue
    C = random.randint(5000, 15000)   # Additional cost
    t = round(random.uniform(10, 30), 2)        # Tax rate
    question = (
        f"{investor_name} invested ${x} in {project_name}, which generated ${R} in revenue and incurred ${C} in additional costs. "
        f"The corporate tax rate is {t:.2f}%. What is the ROI for {project_name}?"
    )
    revenue_after_tax = round(R - (R * (t / 100)), 2)
    total_cost = x + C
    net_profit = revenue_after_tax - total_cost
    ROI = (net_profit / x) * 100
    solution = (
        f"Step 1: Compute the after-tax revenue:\n"
        f"  Revenue_after_tax = Revenue - (Revenue × Tax Rate) = {R} - ({R} × {t / 100:.4f}) = {revenue_after_tax:.2f}\n\n"
        f"Step 2: Compute the total cost and net profit:\n"
        f"  Total Cost = Initial Investment + Additional Costs = {x} + {C} = {total_cost}\n"
        f"  Net Profit = Revenue_after_tax - Total Cost = {revenue_after_tax:.2f} - {total_cost} = {net_profit}\n\n"
        f"Step 3: Compute the ROI:\n"
        f"  ROI = (Net Profit / Initial Investment) × 100 = ({net_profit} / {x}) × 100 = {ROI:.2f}%"
    )
    return question, solution

# Harder ones with four steps in the solution

# Template 5: ROI with Multi-Year Revenue, Salvage Value, and Tax
def template_multi_year_salvage_tax_roi():
    """5:Advanced: ROI with multi-year revenue, salvage value, and tax"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    x = random.randint(30000, 70000)  # Initial investment
    R1 = random.randint(10000, 30000)  # Revenue Year 1
    R2 = random.randint(10000, 30000)  # Revenue Year 2
    SV = random.randint(5000, 15000)   # Salvage value
    t = round(random.uniform(10, 30), 2)         # Tax rate
    
    question = (
        f"{investor_name} invested ${x} in {project_name}, which generated ${R1} in Year 1 and ${R2} in Year 2. "
        f"The project had a salvage value of ${SV}, and the corporate tax rate is {t:.2f}%. "
        f"What is the after-tax ROI for {project_name}?"
    )
    
    total_cash_inflow = R1 + R2 + SV
    profit_before_tax = total_cash_inflow - x
    tax = profit_before_tax * (t / 100)
    profit_after_tax = profit_before_tax - tax
    ROI = (profit_after_tax / x) * 100
    
    solution = (
        f"Step 1: Compute the total cash inflow:\n"
        f"  Total Cash Inflow = Year 1 + Year 2 + Salvage Value = ${R1} + ${R2} + ${SV} = ${total_cash_inflow}\n\n"
        f"Step 2: Compute the profit before tax:\n"
        f"  Profit Before Tax = Total Cash Inflow - Initial Investment = ${total_cash_inflow} - ${x} = ${profit_before_tax}\n\n"
        f"Step 3: Compute the tax amount:\n"
        f"  Tax = Profit Before Tax × Tax Rate = ${profit_before_tax} × {t / 100:.4f} = ${tax:.2f}\n\n"
        f"Step 4: Compute the profit after tax:\n"
        f"  Profit After Tax = Profit Before Tax - Tax = ${profit_before_tax} - ${tax:.2f} = ${profit_after_tax:.2f}\n\n"
        f"Step 5: Compute the after-tax ROI:\n"
        f"  After-tax ROI = (Profit After Tax / Initial Investment) × 100\n"
        f"                = (${profit_after_tax:.2f} / ${x}) × 100 = {ROI:.2f}%"
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
        template_revenue_roi,
        template_tax_implications_roi,
        template_multi_year_revenue,
        template_tax_and_costs_roi,
        template_multi_year_salvage_tax_roi
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
    output_file = "../../testset/investment_analysis/roi.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()