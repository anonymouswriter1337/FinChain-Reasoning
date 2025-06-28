import random

# Named entities for investors and projects
investor_names = ["John Doe", "Susan Lee", "Emily White", "Mark Smith", "David Brown"]
project_names = [
    "Tesla Gigafactory", "Apple iPhone Launch", "Amazon Web Services Expansion", "SpaceX Starship Development",
    "Google Data Center Build", "Microsoft Azure", "Netflix Content Production", "Uber Autonomous Driving Initiative",
    "Facebook Metaverse", "Samsung Semiconductor Factory"
]

# Template 1: Simple CI Calculation
def template_ci_simple_calculation():
    """1:Basic: Simple Compound Interest Calculation"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    principal = random.randint(1000, 5000)  # Principal amount
    rate = round(random.uniform(2, 10), 2)           # Annual interest rate (%)
    time = random.randint(1, 5)             # Time in years
    question = (
        f"{investor_name} invested ${principal} in {project_name}. The investment grows at an annual interest rate of {rate:.2f}% "
        f"compounded annually over {time} years. Calculate the compound interest."
    )
    # Step 1: Calculate the compound amount
    compound_amount = round(principal * (1 + rate / 100) ** time, 2)
    # Step 2: Calculate the compound interest
    ci = compound_amount - principal
    solution = (
        f"Step 1: Compute the compound amount:\n"
        f"  Compound Amount = Principal × (1 + Rate / 100)^Time\n"
        f"                  = {principal} × (1 + {rate / 100:.4f})^{time} = {compound_amount:.2f}\n\n"
        f"Step 2: Compute the compound interest:\n"
        f"  Compound Interest = Compound Amount - Principal\n"
        f"                   = {compound_amount:.2f} - {principal} = {ci:.2f}"
    )
    return question, solution

# Template 2: CI with Quarterly Compounding
def template_ci_quarterly_compounding():
    """2:Basic: Compound Interest with Quarterly Compounding"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    principal = random.randint(1000, 7000)  # Principal amount
    rate = round(random.uniform(2, 8), 2)             # Annual interest rate (%)
    time = random.randint(1, 3)             # Time in years
    question = (
        f"{investor_name} invested ${principal} in {project_name}, which grows at an annual interest rate of {rate:.2f}% "
        f"compounded quarterly for {time} years. Calculate the compound interest."
    )
    # Step 1: Calculate the compound amount
    n = 4  # Compounding frequency (quarterly)
    compound_amount = round(principal * (1 + rate / (100 * n)) ** (n * time), 2)
    # Step 2: Calculate the compound interest
    ci = compound_amount - principal
    solution = (
        f"Step 1: Compute the compound amount with quarterly compounding:\n"
        f"  Compound Amount = Principal × (1 + Rate / (100 × n))^(n × Time)\n"
        f"                  = {principal} × (1 + {rate / (100 * n):.4f})^{n * time} = {compound_amount:.2f}\n\n"
        f"Step 2: Compute the compound interest:\n"
        f"  Compound Interest = Compound Amount - Principal\n"
        f"                   = {compound_amount:.2f} - {principal} = {ci:.2f}"
    )
    return question, solution

# Template 3: CI with Rate and Total Amount Known
def template_ci_rate_and_total_known():
    """3:Intermediate: Compound Interest with Rate and Total Amount Known"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    total_amount = random.randint(5000, 15000)  # Total amount after compounding
    rate = round(random.uniform(2, 10), 2)               # Annual interest rate (%)
    time = random.randint(1, 5)                # Time in years
    question = (
        f"{investor_name} received a total amount of ${total_amount} from their investment in {project_name}. "
        f"The investment grew at an annual interest rate of {rate:.2f}% compounded annually over {time} years. "
        f"Calculate the compound interest."
    )
    # Step 1: Use the compound interest formula to find the principal
    principal = round(total_amount / (1 + rate / 100) ** time, 2)
    # Step 2: Compute the compound amount using the derived principal
    compound_amount = round(principal * (1 + rate / 100) ** time, 2)
    # Step 3: Calculate the compound interest
    ci = compound_amount - principal
    solution = (
        f"Step 1: Compute the initial principal using the formula:\n"
        f"  Principal = Total Amount / (1 + Rate / 100)^Time\n"
        f"           = {total_amount} / (1 + {rate / 100:.4f})^{time} = {principal:.2f}\n\n"
        f"Step 2: Confirm the compound amount:\n"
        f"  Compound Amount = Principal × (1 + Rate / 100)^Time\n"
        f"                  = {principal:.2f} × (1 + {rate / 100:.4f})^{time} = {compound_amount:.2f}\n\n"
        f"Step 3: Compute the compound interest:\n"
        f"  Compound Interest = Compound Amount - Principal\n"
        f"                   = {compound_amount:.2f} - {principal:.2f} = {ci:.2f}"
    )
    return question, solution

# Template 4: CI with Half-Yearly Compounding
def template_ci_half_yearly_compounding():
    """4:Intermediate: compound interest with half-yearly compounding"""
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    principal = random.randint(2000, 8000)  # Principal amount
    rate = round(random.uniform(3, 9), 2)             # Annual interest rate (%)
    time = random.randint(2, 5)             # Time in years
    question = (
        f"{investor_name} invested ${principal} in {project_name}, which grew at an annual interest rate of {rate:.2f}% "
        f"compounded half-yearly over {time} years. Calculate the compound interest."
    )
    # Step 1: Compute the compound amount
    n = 2  # Compounding frequency (half-yearly)
    compound_amount = round(principal * (1 + rate / (100 * n)) ** (n * time), 2)
    # Step 2: Compute the compound interest
    ci = compound_amount - principal
    # Step 3: Verify the total amount
    solution = (
        f"Step 1: Compute the compound amount with half-yearly compounding:\n"
        f"  Compound Amount = Principal × (1 + Rate / (100 × n))^(n × Time)\n"
        f"                  = {principal} × (1 + {rate / (100 * n):.4f})^{n * time} = {compound_amount:.2f}\n\n"
        f"Step 2: Compute the compound interest:\n"
        f"  Compound Interest = Compound Amount - Principal\n"
        f"                   = {compound_amount:.2f} - {principal} = {ci:.2f}\n\n"
        f"Step 3: Verify the result by confirming the compound amount matches the formula."
    )
    return question, solution

# Template 5: CI with Varying Compounding Frequencies
def template_ci_varying_frequencies():
    '''5:Advanced: Compound Interest with Varying Compounding Frequencies'''
    investor_name = random.choice(investor_names)
    project_name = random.choice(project_names)
    principal = random.randint(1000, 8000)  # Principal amount
    rate = round(random.uniform(3, 10), 2)            # Annual interest rate (%)
    time = random.randint(1, 5)             # Time in years
    n = random.choice([1, 2, 4, 12])        # Compounding frequency
    question = (
        f"{investor_name} invested ${principal} in {project_name}, which grew at an annual interest rate of {rate:.2f}% "
        f"compounded {n} times a year for {time} years. Calculate the compound interest."
    )
    # Step 1: Compute the periodic rate
    periodic_rate = round(rate / (100 * n), 4)
    # Step 2: Compute the total number of compounding periods
    periods = n * time
    # Step 3: Compute the compound amount
    compound_amount = round(principal * (1 + periodic_rate) ** periods, 2)
    # Step 4: Compute the compound interest
    ci = compound_amount - principal
    solution = (
        f"Step 1: Compute the periodic rate:\n"
        f"  Periodic Rate = Rate / (100 × n)\n"
        f"                = {rate} / (100 × {n}) = {periodic_rate:.4f}\n\n"
        f"Step 2: Compute the total number of compounding periods:\n"
        f"  Periods = n × Time = {n} × {time} = {periods}\n\n"
        f"Step 3: Compute the compound amount:\n"
        f"  Compound Amount = Principal × (1 + Periodic Rate)^Periods\n"
        f"                  = {principal} × (1 + {periodic_rate:.4f})^{periods} = {compound_amount:.2f}\n\n"
        f"Step 4: Compute the compound interest:\n"
        f"  Compound Interest = Compound Amount - Principal\n"
        f"                   = {compound_amount:.2f} - {principal} = {ci:.2f}"
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
        template_ci_simple_calculation,
        template_ci_quarterly_compounding,
        template_ci_rate_and_total_known,
        template_ci_half_yearly_compounding,
        template_ci_varying_frequencies
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
    output_file = "../../testset/investment_analysis/ci.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem, ensure_ascii=False))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()