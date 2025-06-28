# https://en.wikipedia.org/wiki/Net_present_value
import random

# Named entities for investors and projects
investor_names = ["John Doe", "Susan Lee", "Emily White", "Mark Smith", "David Brown"]
project_names = [
    "Tesla Gigafactory", "Apple iPhone Launch", "Amazon Web Services Expansion", "SpaceX Starship Development",
    "Google Data Center Build", "Microsoft Azure", "Netflix Content Production", "Uber Autonomous Driving Initiative",
    "Facebook Metaverse", "Samsung Semiconductor Factory"
]

# Simple ones with two steps in the solution:

# Template 1: Single-Year NPV (net present value)
def template_single_year_npv():
    """1:Basic: NPV with one cash flow"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    x = random.randint(10000, 50000)  # Initial investment
    C = random.randint(15000, 60000)  # Cash flow after 1 year
    r = round(random.uniform(5, 15), 2)         # Discount rate
    question = (
        f"{investor_name} spends ${x} on {project_name}, which returns ${C} after 1 year. "
        f"If the discount rate is {r:.2f}%, what is the NPV of {project_name}?"
    )
    PV = C / (1 + r / 100)
    NPV = PV - x
    solution = (
        f"Step 1: Compute the present value (PV):\n"
        f"  PV = {C} / (1 + {r / 100:.4f}) = {PV:.2f}\n\n"
        f"Step 2: Compute the NPV:\n"
        f"  NPV = PV - Initial Investment = {PV:.2f} - {x} = {NPV:.2f}"
    )
    return question, solution

# Template 2: Multi-Year NPV with Fixed Cash Flows
def template_multi_year_fixed_npv():
    """2:Basic: NPV with fixed cash flows"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    x = random.randint(10000, 50000)  # Initial investment
    C = random.randint(5000, 20000)  # Annual cash flow
    n = random.randint(3, 5)         # Number of years
    r = round(random.uniform(5, 15), 2)        # Discount rate
    question = (
        f"{investor_name} is evaluating {project_name}, which generates ${C} annually for {n} years. "
        f"If the discount rate is {r:.2f}%, what is the NPV of {project_name}?"
    )
    r_frac = r / 100
    PV = C * (1 - (1 + r_frac)**-n) / r_frac
    NPV = PV - x
    solution = (
        f"Step 1: Compute the present value of an annuity:\n"
        f"  PV = {C} × (1 - (1 + {r_frac:.4f})^{-n}) / {r_frac:.4f}\n"
        f"     = {PV:.2f}\n\n"
        f"Step 2: Compute the NPV:\n"
        f"  NPV = PV - Initial Investment = {PV:.2f} - {x} = {NPV:.2f}"
    )
    return question, solution

# Medium level: three steps in the solution

# Template 3: Multi-Year NPV with Variable Cash Flows
def template_multi_year_variable_npv():
    """3:Intermediate: NPV with variable cash flows"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    x = random.randint(20000, 60000)  # Initial investment
    C1 = random.randint(5000, 20000)  # Cash flow Year 1
    C2 = random.randint(5000, 20000)  # Cash flow Year 2
    C3 = random.randint(5000, 20000)  # Cash flow Year 3
    r = round(random.uniform(5, 15), 2)   # Discount rate
    question = (
        f"{investor_name} is considering {project_name}, which generates cash flows of ${C1} in Year 1, "
        f"${C2} in Year 2, and ${C3} in Year 3. If the discount rate is {r:.2f}%, what is the NPV of {project_name}?"
    )
    r_frac = r / 100
    PV1 = C1 / (1 + r_frac)**1
    PV2 = C2 / (1 + r_frac)**2
    PV3 = C3 / (1 + r_frac)**3
    TotalPV = PV1 + PV2 + PV3
    NPV = TotalPV - x
    solution = (
        f"Step 1: Compute the present value for each year:\n"
        f"  PV1 = {C1} / (1 + {r_frac:.4f})^1 = {PV1:.2f}\n"
        f"  PV2 = {C2} / (1 + {r_frac:.4f})^2 = {PV2:.2f}\n"
        f"  PV3 = {C3} / (1 + {r_frac:.4f})^3 = {PV3:.2f}\n\n"
        f"Step 2: Compute the total present value:\n"
        f"  TotalPV = PV1 + PV2 + PV3 = {TotalPV:.2f}\n\n"
        f"Step 3: Compute the NPV:\n"
        f"  NPV = TotalPV - Initial Investment = {TotalPV:.2f} - {x} = {NPV:.2f}"
    )
    return question, solution

# Template 4: NPV with Variable Discount Rates
def template_variable_discount_npv():
    """4:Intermediate: NPV with variable discount rates"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    x = random.randint(20000, 60000)  # Initial investment
    C1 = random.randint(5000, 20000)  # Cash flow Year 1
    C2 = random.randint(5000, 20000)  # Cash flow Year 2
    C3 = random.randint(5000, 20000)  # Cash flow Year 3
    r1 = round(random.uniform(5, 10), 2)        # Discount rate Year 1
    r2 = round(random.uniform(10, 15), 2)       # Discount rate Year 2
    r3 = random.uniform(15, 20)       # Discount rate Year 3
    
    question = (
        f"{investor_name} is analyzing {project_name}, which has cash flows of ${C1} in Year 1, ${C2} in Year 2, "
        f"and ${C3} in Year 3. The discount rates are {r1:.2f}% for Year 1, {r2:.2f}% for Year 2, and {r3:.2f}% for Year 3. "
        f"What is the NPV of {project_name}?"
    )
    
    # Corrected PV calculations using cumulative discount factors
    PV1 = C1 / (1 + r1/100)
    PV2 = C2 / ((1 + r1/100) * (1 + r2/100))
    PV3 = C3 / ((1 + r1/100) * (1 + r2/100) * (1 + r3/100))
    
    TotalPV = PV1 + PV2 + PV3
    NPV = TotalPV - x
    
    solution = (
        f"Step 1: Compute the present value for each year with cumulative discount rates:\n"
        f"  PV1 = {C1} / (1 + {r1/100:.4f}) = {PV1:.2f}\n"
        f"  PV2 = {C2} / ((1 + {r1/100:.4f}) × (1 + {r2/100:.4f})) = {PV2:.2f}\n"
        f"  PV3 = {C3} / ((1 + {r1/100:.4f}) × (1 + {r2/100:.4f}) × (1 + {r3/100:.4f})) = {PV3:.2f}\n\n"
        f"Step 2: Compute the total present value:\n"
        f"  TotalPV = PV1 + PV2 + PV3 = {TotalPV:.2f}\n\n"
        f"Step 3: Compute the NPV:\n"
        f"  NPV = TotalPV - Initial Investment = {TotalPV:.2f} - {x} = {NPV:.2f}"
    )
    return question, solution

# Template 5: NPV with Salvage Value
def template_npv_with_salvage_value():
    """5:Advanced: NPV with salvage value"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    x = random.randint(30000, 70000)  # Initial investment
    C = random.randint(10000, 30000)  # Annual cash flow
    n = random.randint(3, 5)          # Number of years
    SV = random.randint(10000, 30000) # Salvage value at the end of the project
    r = round(random.uniform(5, 15), 2)         # Discount rate
    question = (
        f"{investor_name} is considering {project_name}, which generates ${C} annually for {n} years and has a salvage "
        f"value of ${SV} at the end of Year {n}. If the discount rate is {r:.2f}%, what is the NPV of {project_name}?"
    )
    r_frac = r / 100
    PV_annuity = C * (1 - (1 + r_frac)**-n) / r_frac
    PV_salvage = SV / (1 + r_frac)**n
    TotalPV = PV_annuity + PV_salvage
    NPV = TotalPV - x
    solution = (
        f"Step 1: Compute the present value of the cash flows (annuity):\n"
        f"  PV_annuity = {C} × (1 - (1 + {r_frac:.4f})^{-n}) / {r_frac:.4f} = {PV_annuity:.2f}\n\n"
        f"Step 2: Compute the present value of the salvage value:\n"
        f"  PV_salvage = {SV} / (1 + {r_frac:.4f})^{n} = {PV_salvage:.2f}\n\n"
        f"Step 3: Compute the total present value:\n"
        f"  TotalPV = PV_annuity + PV_salvage = {PV_annuity:.2f} + {PV_salvage:.2f} = {TotalPV:.2f}\n\n"
        f"Step 4: Compute the NPV:\n"
        f"  NPV = TotalPV - Initial Investment = {TotalPV:.2f} - {x} = {NPV:.2f}"
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
        template_single_year_npv,
        template_multi_year_fixed_npv,
        template_multi_year_variable_npv,
        template_variable_discount_npv,
        template_npv_with_salvage_value
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
    output_file = "../../testset/investment_analysis/npv.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem, ensure_ascii=False))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()