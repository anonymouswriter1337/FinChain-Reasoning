import random

# Named entities for companies and industries
company_names = ["Tesla Inc.", "Apple Inc.", "Amazon.com", "SpaceX", "Google LLC"]
industry_names = ["automotive", "technology", "e-commerce", "aerospace", "internet services"]

# Template 1: Budget Allocation Simple
def template_budget_allocation_simple():
    """1:Basic: Creates a simple budget allocation problem where a company divides its total budget between marketing and operations using a specified percentage split."""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    total_budget = round(random.randint(500000, 5000000),2)  # Total budget
    marketing_percentage = round(random.uniform(10, 50),2) # Percentage allocated to marketing

    # Step 1: Calculate marketing and operations allocations
    marketing_allocation = round(total_budget * (marketing_percentage / 100), 2)
    operations_allocation = round(total_budget - marketing_allocation, 2)

    # Question formulation
    question = (
        f"{company_name}, operating in the {industry} industry, has a total budget of ${total_budget} for the upcoming quarter. "
        f"The company plans to allocate {marketing_percentage:.2f}% of the budget to marketing and the remaining amount to operations. "
        f"How much of the budget will be allocated to marketing and how much to operations?"
    )

    # Solution formulation
    solution = (
        f"Step 1: Calculate the budget allocated to marketing and operations:\n"
        f"  Marketing Allocation = {marketing_percentage:.2f}% of ${total_budget} = ${marketing_allocation:.2f}\n"
        f"  Operations Allocation = ${total_budget} - ${marketing_allocation:.2f} = ${operations_allocation:.2f}"
    )

    return question, solution


# Template 2: Budget Reallocation Due to Costs
def template_budget_reallocation_due_to_costs():
    """2:Basic: Generates a problem where a company must reallocate a portion of its marketing budget to operations due to increased costs, requiring calculations of new budgets after the shift."""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    marketing_budget = round(random.randint(500000, 3000000),2) # Initial marketing budget
    operations_budget = round(random.randint(1000000, 5000000),2)  # Initial operations budget
    reallocation_percentage = round(random.uniform(5, 25),2)  # Percentage of marketing budget reallocated to operations

    # Step 1: Calculate reallocated amount
    reallocated_amount = round(marketing_budget * (reallocation_percentage / 100), 2)
    new_marketing_budget = round(marketing_budget - reallocated_amount, 2)
    new_operations_budget = round(operations_budget + reallocated_amount, 2)

    # Question formulation
    question = (
        f"{company_name}, a major player in the {industry} industry, had originally allocated ${marketing_budget} to marketing and ${operations_budget} to operations. "
        f"However, due to increased production costs, the company needs to shift {reallocation_percentage:.2f}% of the marketing budget to operations. "
        f"How much will the new marketing and operations budgets be after the reallocation?"
    )

    # Solution formulation
    solution = (
        f"Step 1: Calculate the amount reallocated from marketing to operations:\n"
        f"  Reallocated Amount = {reallocation_percentage:.2f}% of ${marketing_budget} = ${reallocated_amount:.2f}\n\n"
        f"Step 2: Calculate the new budgets:\n"
        f"  New Marketing Budget = ${marketing_budget} - ${reallocated_amount} = ${new_marketing_budget:.2f}\n"
        f"  New Operations Budget = ${operations_budget} + ${reallocated_amount} = ${new_operations_budget:.2f}"
    )

    return question, solution

# Template 3: Budget Adjustment for New Project
def template_budget_adjustment_new_project():
    """3:Intermediate: Creates a complex budget adjustment scenario where a company must reallocate funds 
    between R&D, marketing, and operations to accommodate a new project, involving multiple budget 
    categories and calculations."""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    total_budget = round(random.randint(3000000, 10000000),2)  # Total budget
    r_and_d_percentage = round(random.uniform(15, 30),2)  # Percentage allocated to R&D
    marketing_percentage = round(random.uniform(20, 40),2)  # Percentage allocated to marketing
    additional_r_and_d = round(random.randint(500000, 1500000),2)  # Additional R&D allocation

    # Step 1: Calculate original R&D, marketing, and operations budgets
    r_and_d_budget = round(total_budget * (r_and_d_percentage / 100), 2)
    marketing_budget = round(total_budget * (marketing_percentage / 100), 2)
    operations_budget = round(total_budget - r_and_d_budget - marketing_budget, 2)

    # Step 2: Adjust the R&D budget for the new project
    new_r_and_d_budget = round(r_and_d_budget + additional_r_and_d, 2)
    new_operations_budget = round(operations_budget - additional_r_and_d, 2)

    # Question formulation
    question = (
        f"{company_name}, in the {industry} industry, has a budget of ${total_budget} for the quarter. The company allocated {r_and_d_percentage:.2f}% to R&D, "
        f"{marketing_percentage:.2f}% to marketing, and the remainder to operations. Midway through the quarter, the company decides to allocate an additional ${additional_r_and_d} "
        f"to R&D for a new project, reducing the operations budget. How much will the new operations and R&D budgets be after the adjustment?"
    )

    # Solution formulation
    solution = (
        f"Step 1: Calculate the original R&D, marketing, and operations budgets:\n"
        f"  R&D Budget = ${r_and_d_budget:.2f}, Marketing Budget = ${marketing_budget:.2f}, Operations Budget = ${operations_budget:.2f}\n\n"
        f"Step 2: Adjust the R&D budget for the new project:\n"
        f"  New R&D Budget = ${r_and_d_budget} + ${additional_r_and_d} = ${new_r_and_d_budget:.2f}\n\n"
        f"Step 3: Calculate the new operations budget:\n"
        f"  New Operations Budget = ${operations_budget} - ${additional_r_and_d} = ${new_operations_budget:.2f}"
    )

    return question, solution

# Template 4: Budget Profit Maximization
def template_budget_profit_maximization():
    """4:Intermediate: Generates a profit maximization problem that involves analyzing how increased R&D 
    investment affects overall profits and future budget availability, incorporating projected growth 
    calculations."""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    total_budget = round(random.randint(5000000, 20000000),2)  # Total budget
    r_and_d_percentage = round(random.uniform(10, 25),2)  # Percentage allocated to R&D
    marketing_percentage = round(random.uniform(20, 40),2)  # Percentage allocated to marketing
    additional_r_and_d = round(random.randint(1000000, 3000000),2)  # Increase in R&D allocation
    projected_profit_increase = round(random.uniform(5, 15),2)  # Projected profit increase (%)
    
    # Step 1: Calculate current allocations
    r_and_d_budget = round(total_budget * (r_and_d_percentage / 100), 2)
    marketing_budget = round(total_budget * (marketing_percentage / 100), 2)
    operations_budget = round(total_budget - r_and_d_budget - marketing_budget, 2)
    
    # Step 2: Calculate the projected increase in profit
    original_profit = round(total_budget * 0.20, 2)  # Assuming 20% profit margin
    increased_profit = round(original_profit * (1 + projected_profit_increase / 100), 2)
    
    # Step 3: Calculate the new total budget for the next quarter
    new_total_budget = round(total_budget + increased_profit, 2)
    
    # Question formulation
    question = (
        f"{company_name}, a company in the {industry} sector, has a total quarterly budget of ${total_budget}. The company allocates {r_and_d_percentage:.2f}% to R&D, "
        f"{marketing_percentage:.2f}% to marketing, and the remainder to operations. If R&D investment is increased by ${additional_r_and_d}, the company's quarterly profits are projected "
        f"to increase by {projected_profit_increase:.2f}%. How will the increase in profits affect the overall budget, and what is the new total available budget for the next quarter?"
    )

    # Solution formulation
    solution = (
        f"Step 1: Calculate the current allocations:\n"
        f"  R&D Budget = ${r_and_d_budget:.2f}, Marketing Budget = ${marketing_budget:.2f}, Operations Budget = ${operations_budget:.2f}\n\n"
        f"Step 2: Calculate the projected increase in profit:\n"
        f"  Original Profit = ${original_profit:.2f}, Projected Profit Increase = ${increased_profit:.2f}\n\n"
        f"Step 3: Calculate the new total budget for the next quarter:\n"
        f"  New Total Budget = ${total_budget} + ${increased_profit} = ${new_total_budget:.2f}\n\n"
        f"Step 4: The new budget available for the next quarter includes the projected profit increase."
    )

    return question, solution

# Template 5: Budget Long-Term Growth
def template_budget_long_term_growth():
    """5:Advanced: Creates a long-term budget planning problem that factors in annual growth rates 
    over multiple years, calculating future budgets for R&D, marketing, and operations with compound 
    growth considerations."""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    current_quarterly_budget = round(random.randint(10000000, 50000000),2)  # Quarterly budget
    annual_growth_rate = round(random.uniform(5, 15),2)  # Expected annual revenue growth (%)
    r_and_d_percentage = round(random.uniform(15, 30),2)  # Percentage allocated to R&D
    marketing_percentage = round(random.uniform(20, 40),2)  # Percentage allocated to marketing
    number_of_years = 3  # Long-term planning period

    # Step 1: Calculate the future quarterly budget after revenue growth
    future_quarterly_budget = round(current_quarterly_budget * (1 + annual_growth_rate / 100) ** number_of_years, 2)

    # Step 2: Calculate the future R&D budget
    future_r_and_d_budget = round(future_quarterly_budget * (r_and_d_percentage / 100), 2)
    
    # Step 3: Calculate the future marketing budget
    future_marketing_budget = round(future_quarterly_budget * (marketing_percentage / 100), 2)
    
    # Step 4: Calculate the future operations budget
    future_operations_budget = round(future_quarterly_budget - future_r_and_d_budget - future_marketing_budget, 2)

    # Question formulation
    question = (
        f"{company_name}, operating in the {industry} industry, has created a 3-year budget plan. The company currently has a quarterly budget of ${current_quarterly_budget} "
        f"and expects its revenue to grow by {annual_growth_rate:.2f}% annually. The budget is allocated as follows: {r_and_d_percentage:.2f}% to R&D, {marketing_percentage:.2f}% to marketing, "
        f"and the remainder to operations. How will the budget for each category (R&D, marketing, and operations) change after 3 years, assuming the expected revenue growth?"
    )
    
    # Solution formulation
    solution = (
        f"Step 1: Calculate the future quarterly budget after revenue growth:\n"
        f"  Future Quarterly Budget = ${current_quarterly_budget} × (1 + {annual_growth_rate:.2f}%)^{number_of_years} = ${future_quarterly_budget:.2f}\n\n"
        f"Step 2: Calculate the future R&D budget:\n"
        f"  Future R&D Budget = ${future_quarterly_budget:.2f} × {r_and_d_percentage:.2f}% = ${future_r_and_d_budget:.2f}\n\n"
        f"Step 3: Calculate the future marketing budget:\n"
        f"  Future Marketing Budget = ${future_quarterly_budget:.2f} × {marketing_percentage:.2f}% = ${future_marketing_budget:.2f}\n\n"
        f"Step 4: Calculate the future operations budget:\n"
        f"  Future Operations Budget = ${future_quarterly_budget:.2f} - ${future_r_and_d_budget:.2f} - ${future_marketing_budget:.2f} = ${future_operations_budget:.2f}"
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
        template_budget_allocation_simple,
        template_budget_reallocation_due_to_costs,
        template_budget_adjustment_new_project,
        template_budget_profit_maximization,
        template_budget_long_term_growth
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
    output_file = "../../testset/personal_finance/budgeting.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == '__main__':
    main()