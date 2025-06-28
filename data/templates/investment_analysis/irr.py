import random
import numpy as np

# Named entities for investors and projects
investor_names = ["John Doe", "Susan Lee", "Emily White", "Mark Smith", "David Brown"]
project_names = [
    "Tesla Gigafactory", "Apple iPhone Launch", "Amazon Web Services Expansion", "SpaceX Starship Development",
    "Google Data Center Build", "Microsoft Azure", "Netflix Content Production", "Uber Autonomous Driving Initiative",
    "Facebook Metaverse", "Samsung Semiconductor Factory"
]

# Template 1: IRR (Internal Rate of Return) for Single Cash Flow
def template_irr_single_cash_flow():
    """1:Basic: IRR for a single cash flow"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    investment = random.randint(10000, 50000)  # Initial investment
    cash_flow = random.randint(20000, 60000)   # Single cash flow after 1 year
    question = (
        f"{investor_name} invested ${investment} in {project_name}, which returns ${cash_flow} after 1 year. "
        f"What is the IRR for {project_name}?"
    )
    irr = ((cash_flow / investment) - 1) * 100
    solution = (
        f"Step 1: Use the IRR formula for a single cash flow:\n"
        f"  NPV = 0 = -Investment + Cash Flow / (1 + IRR)\n"
        f"  Rearranging: IRR = (Cash Flow / Investment) - 1\n\n"
        f"Step 2: Compute the IRR:\n"
        f"  IRR = ({cash_flow} / {investment}) - 1 = {irr:.2f}%"
    )
    return question, solution

# Template 2: IRR for Single Cash Flow (Multiple Years)
def template_irr_single_cash_flow_multiple_years():
    """2:Basic: IRR for a single cash flow over multiple years"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    investment = random.randint(10000, 50000)  # Initial investment
    cash_flow = random.randint(30000, 100000)  # Single cash flow
    years = random.randint(2, 5)               # Number of years
    question = (
        f"{investor_name} invested ${investment} in {project_name}, which returns ${cash_flow} after {years} years. "
        f"What is the IRR for {project_name}?"
    )
    irr = ((cash_flow / investment) ** (1 / years) - 1) * 100
    solution = (
        f"Step 1: Use the IRR formula for a single cash flow over multiple years:\n"
        f"  NPV = 0 = -Investment + Cash Flow / (1 + IRR)^t\n"
        f"  Rearranging: IRR = (Cash Flow / Investment)^(1/t) - 1\n\n"
        f"Step 2: Compute the IRR:\n"
        f"  IRR = ({cash_flow} / {investment})^(1/{years}) - 1 = {irr:.2f}%"
    )
    return question, solution

# Template 3: IRR for Single Cash Flow with Additional Costs
def template_irr_additional_costs():
    """3:Intermediate: IRR for a single cash flow with additional costs"""
    investor_name = random.choice(investor_names)
    project_name  = random.choice(project_names)

    investment        = random.randint(30_000, 70_000)   # t = 0 outflow
    additional_cost   = random.randint(5_000, 15_000)    # t = 1 extra outflow
    gross_cash_inflow = random.randint(40_000, 100_000)  # t = 1 inflow

    net_cash_inflow = gross_cash_inflow - additional_cost  # what the investor actually receives
    irr = (net_cash_inflow / investment - 1) * 100         # one-period IRR as a %

    question = (
        f"{investor_name} invested ${investment:,} in {project_name}.\n"
        f"One year later the project generated ${gross_cash_inflow:,}, "
        f"but required an additional cost of ${additional_cost:,} before payout.\n"
        f"What is the IRR for {project_name}?"
    )

    solution = (
        "Step 1  Compute the net cash inflow at t = 1:\n"
        f"  Net Cash Inflow = Gross Cash Inflow − Additional Cost\n"
        f"                  = {gross_cash_inflow:,} − {additional_cost:,}\n"
        f"                  = {net_cash_inflow:,}\n\n"
        "Step 2  Apply the one-period IRR formula (same as ROI for a single period):\n"
        f"  IRR = (Net Cash Inflow ÷ Initial Investment) − 1\n"
        f"      = ({net_cash_inflow:,} ÷ {investment:,}) − 1\n"
        f"      = {net_cash_inflow / investment:.4f} − 1\n\n"
        f"Step 3  Convert to a percentage:\n"
        f"  IRR = {(irr):.2f}%"
    )

    return question, solution

# Template 4: IRR with Salvage Value and Multiple Cash Flows
def template_irr_salvage_value():
    """4:Intermediate: IRR with salvage value and multiple cash flows"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    investment = random.randint(40000, 80000)  # Initial investment
    cash_flows = [random.randint(10000, 20000), random.randint(10000, 20000)]  # Two-year cash flows
    salvage_value = random.randint(5000, 15000)  # Salvage value at the end
    question = (
        f"{investor_name} invested ${investment} in {project_name}, which generated ${cash_flows[0]} in Year 1, "
        f"${cash_flows[1]} in Year 2, and had a salvage value of ${salvage_value} at the end of Year 2. "
        f"What is the IRR for {project_name}?"
    )
    total_cash_flow = cash_flows[0] + cash_flows[1] + salvage_value
    irr = ((total_cash_flow / investment) - 1) * 100
    solution = (
        f"Step 1: Compute the total cash flow including salvage value:\n"
        f"  Total Cash Flow = Year 1 + Year 2 + Salvage Value = {cash_flows[0]} + {cash_flows[1]} + {salvage_value} = {total_cash_flow}\n\n"
        f"Step 2: Compute the ratio of total cash flow to investment:\n"
        f"  Ratio = Total Cash Flow / Investment = {total_cash_flow} / {investment} = {total_cash_flow / investment:.2f}\n\n"
        f"Step 3: Compute the IRR:\n"
        f"  IRR = (Ratio - 1) × 100 = ({total_cash_flow / investment:.2f} - 1) × 100 = {irr:.2f}%"
    )
    return question, solution

# Template 5: IRR with Multiple Cash Flows and Tax Rate
def template_irr_single_cash_flow_taxes_refined():
    """5:Advanced: IRR with single cash flow and tax rate"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    investment = random.randint(30000, 70000)  # Initial investment
    cash_flow = random.randint(40000, 100000)  # Single cash flow
    tax_rate = round(random.uniform(10, 30), 2)      # Tax rate
    question = (
        f"{investor_name} invested ${investment} in {project_name}, which generates a single cash flow of ${cash_flow} after 1 year. "
        f"The corporate tax rate is {tax_rate:.2f}%. What is the IRR for {project_name}?"
    )
    # Step 1: Compute after-tax cash flow
    after_tax_cash_flow = cash_flow * (1 - tax_rate / 100)
    # Step 2: Compute the ratio of after-tax cash flow to investment
    ratio = after_tax_cash_flow / investment
    # Step 3: Calculate IRR
    irr = (ratio - 1) * 100
    # Step 4: Interpret and validate IRR
    solution = (
        f"Step 1: Adjust the cash flow for taxes:\n"
        f"  After-Tax Cash Flow = Cash Flow × (1 - Tax Rate)\n"
        f"                      = {cash_flow} × (1 - {tax_rate / 100:.4f}) = {after_tax_cash_flow:.2f}\n\n"
        f"Step 2: Compute the ratio of after-tax cash flow to investment:\n"
        f"  Ratio = After-Tax Cash Flow / Investment\n"
        f"        = {after_tax_cash_flow:.2f} / {investment} = {ratio:.2f}\n\n"
        f"Step 3: Calculate the IRR:\n"
        f"  IRR = (Ratio - 1) × 100\n"
        f"      = ({ratio:.2f} - 1) × 100 = {irr:.2f}%\n\n"
        f"Step 4: Validate the IRR:\n"
        f"  The IRR of {project_name} is {irr:.2f}%. This value confirms that the project's return "
        f"accounts for the adjusted cash flow after taxes."
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
        template_irr_single_cash_flow,
        template_irr_single_cash_flow_multiple_years,
        template_irr_additional_costs,
        template_irr_salvage_value,
        template_irr_single_cash_flow_taxes_refined
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
    output_file = "../../testset/investment_analysis/irr.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem, ensure_ascii=False))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()