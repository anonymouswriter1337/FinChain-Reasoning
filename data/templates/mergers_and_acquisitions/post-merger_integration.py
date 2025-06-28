import random
import json

# Named entities for investors and companies (US companies)
investor_names = ["John Doe", "Susan Lee", "Emily White", "Mark Smith", "David Brown"]
company_names = ["Apple", "Microsoft", "Amazon", "Google", "Facebook", "Tesla", "JPMorgan", "Walmart", "Boeing", "Coca-Cola"]

# ---------------------- Basic Questions ---------------------- #
def basic_integration_synergy_calculation():
    """1:Basic: Integration Synergy Savings Calculation"""
    investor = random.choice(investor_names)
    comp_list = random.sample(company_names, 2)
    company_a, company_b = comp_list[0], comp_list[1]
    # Duplicate operational costs (in million dollars)
    cost_a = random.randint(50, 200)
    cost_b = random.randint(50, 200)
    # Synergy reduction rate (percentage)
    synergy_rate = round(random.uniform(25, 35), 2)
    
    question = (
        f"{investor} oversaw the merger of {company_a} and {company_b}. Before integration, "
        f"{company_a} incurred duplicate operational costs of ${cost_a} million and {company_b} incurred "
        f"duplicate costs of ${cost_b} million. Post-integration, it is expected that these costs are reduced "
        f"by {synergy_rate}%. Calculate the total cost savings achieved by the merger."
    )
    total_cost = cost_a + cost_b
    savings = round(total_cost * synergy_rate / 100, 2)
    solution = (
        f"Step 1: Compute the total duplicate cost:\n"
        f"  Total Cost = {cost_a} + {cost_b} = {total_cost} million\n\n"
        f"Step 2: Calculate the cost savings using the synergy reduction rate:\n"
        f"  Savings = Total Cost × (Synergy Rate / 100)\n"
        f"          = {total_cost} × ({synergy_rate} / 100) = {savings} million\n\n"
        f"Therefore, the total cost savings is {savings} million dollars."
    )
    return question, solution

def basic_valuation_adjustment_after_integration():
    """2:Basic: Valuation Adjustment After Integration"""
    investor = random.choice(investor_names)
    comp_list = random.sample(company_names, 2)
    company_a, company_b = comp_list[0], comp_list[1]
    # Pre-merger valuations (in million dollars)
    valuation_a = random.randint(500, 3000)
    valuation_b = random.randint(500, 3000)
    # Integration premium percentage
    premium = round(random.uniform(5, 15), 2)
    
    combined = valuation_a + valuation_b
    premium_value = round(combined * premium / 100, 2)
    post_valuation = combined + premium_value
    question = (
        f"{investor} oversaw the merger of {company_a} and {company_b}. Before the merger, {company_a} was valued at "
        f"${valuation_a} million and {company_b} at ${valuation_b} million. With an expected integration premium of "
        f"{premium}%, calculate the post-merger valuation."
    )
    solution = (
        f"Step 1: Compute the combined pre-merger valuation:\n"
        f"  Combined Valuation = {valuation_a} + {valuation_b} = {combined} million\n\n"
        f"Step 2: Calculate the premium value:\n"
        f"  Premium Value = Combined Valuation × (Premium / 100)\n"
        f"               = {combined} × ({premium} / 100) = {premium_value} million\n\n"
        f"Step 3: Compute the post-merger valuation:\n"
        f"  Post-Merger Valuation = Combined Valuation + Premium Value\n"
        f"                        = {combined} + {premium_value} = {post_valuation} million\n"
    )
    return question, solution

# -------------------- Intermediate Questions -------------------- #
def intermediate_integration_cost_reduction():
    """3:Intermediate: Integration Cost Reduction Analysis (4 steps)"""
    investor = random.choice(investor_names)
    comp_list = random.sample(company_names, 2)
    company_a, company_b = comp_list[0], comp_list[1]
    # Overlapping operating costs (in million dollars)
    op_cost_a = random.randint(50, 300)
    op_cost_b = random.randint(50, 300)
    reduction_percent = round(random.uniform(20, 40), 2)
    integration_cost = random.randint(20, 100)
    
    total_overlap = op_cost_a + op_cost_b
    potential_saving = round(total_overlap * reduction_percent / 100, 2)
    net_saving = round(potential_saving - integration_cost, 2)
    saving_percentage = round((net_saving / total_overlap) * 100, 2)
    
    question = (
        f"{investor} oversaw the merger of {company_a} and {company_b}. The overlapping operating costs were "
        f"${op_cost_a} million for {company_a} and ${op_cost_b} million for {company_b}. The merger is expected to "
        f"reduce these costs by {reduction_percent}%, but it incurs a one-time integration cost of ${integration_cost} million. "
        f"Calculate the net cost reduction achieved by the merger and its percentage relative to the total overlapping cost."
    )
    solution = (
        f"Step 1: Compute the total overlapping cost:\n"
        f"  Total Overlap = {op_cost_a} + {op_cost_b} = {total_overlap} million\n\n"
        f"Step 2: Calculate the potential cost saving:\n"
        f"  Potential Saving = Total Overlap × (Reduction Percent / 100)\n"
        f"                   = {total_overlap} × ({reduction_percent} / 100) = {potential_saving} million\n\n"
        f"Step 3: Subtract the integration cost to obtain net savings:\n"
        f"  Net Saving = Potential Saving - Integration Cost\n"
        f"             = {potential_saving} - {integration_cost} = {net_saving} million\n\n"
        f"Step 4: Calculate the net saving percentage relative to the total overlapping cost:\n"
        f"  Net Saving Percentage = (Net Saving / Total Overlap) × 100\n"
        f"                         = ({net_saving} / {total_overlap}) × 100 = {saving_percentage}%\n\n"
        f"Thus, the net cost reduction is {net_saving} million, which is {saving_percentage}% of the total overlapping cost."
    )
    return question, solution

def intermediate_cultural_integration_index_adjustment():
    """4:Intermediate: Cultural Integration Index Adjustment (4 steps)"""
    investor = random.choice(investor_names)
    comp_list = random.sample(company_names, 2)
    company_a, company_b = comp_list[0], comp_list[1]
    # Cultural indices (in percentages) for both companies
    index_a = random.randint(50, 90)
    index_b = random.randint(50, 90)
    # Weight factors for the companies (in percentages)
    weight_a = random.randint(40, 70)
    weight_b = 100 - weight_a
    bonus = random.randint(1, 5)  # bonus percentage points due to integration efficiencies
    
    weighted_avg = round((index_a * weight_a + index_b * weight_b) / 100, 2)
    post_merger_index = round(weighted_avg + bonus, 2)
    improvement = round(post_merger_index - weighted_avg, 2)
    
    question = (
        f"{investor} oversaw the merger of {company_a} and {company_b}. The cultural integration indices for "
        f"{company_a} and {company_b} were {index_a}% and {index_b}% respectively. With {company_a} contributing "
        f"{weight_a}% and {company_b} contributing {weight_b}% to the new culture, and expecting an integration bonus of "
        f"{bonus} percentage points, calculate the adjusted post-merger cultural index and the improvement over the baseline."
    )
    solution = (
        f"Step 1: Compute the weighted average of the cultural indices:\n"
        f"  Weighted Average = (Index_A × Weight_A + Index_B × Weight_B) / 100\n"
        f"                   = ({index_a} × {weight_a} + {index_b} × {weight_b}) / 100 = {weighted_avg}%\n\n"
        f"Step 2: Identify the bonus integration factor: {bonus}%\n\n"
        f"Step 3: Compute the final post-merger cultural index by adding the bonus:\n"
        f"  Post-Merger Index = Weighted Average + Bonus\n"
        f"                    = {weighted_avg}% + {bonus}% = {post_merger_index}%\n\n"
        f"Step 4: Calculate the improvement over the baseline weighted average:\n"
        f"  Improvement = Post-Merger Index - Weighted Average\n"
        f"              = {post_merger_index}% - {weighted_avg}% = {improvement}%\n\n"
        f"Thus, the adjusted post-merger cultural index is {post_merger_index}%, reflecting a {improvement}% improvement."
    )
    return question, solution

# ---------------------- Advanced Question ---------------------- #
def advanced_integration_synergy_and_costs_balancing():
    """5:Advanced: Net Financial Impact of Integration Over Multiple Years (5 steps)"""
    investor = random.choice(investor_names)
    comp_list = random.sample(company_names, 2)
    company_a, company_b = comp_list[0], comp_list[1]
    # Annual synergy savings, incremental integration expense, one-time integration cost (in million dollars)
    S = random.randint(50, 300)
    E = random.randint(10, 50)
    I = random.randint(20, 100)
    # Pre-merger revenue (in million dollars) and revenue boost percentage
    P = random.randint(200, 2000)
    R = round(random.uniform(3, 10), 2)
    # Time period in years
    T = random.randint(3, 7)
    
    total_synergy = S * T
    total_expense = E * T
    net_synergy = round(total_synergy - total_expense - I, 2)
    revenue_increase = round(P * (R / 100) * T, 2)
    overall_net_impact = round(net_synergy + revenue_increase, 2)
    
    question = (
        f"{investor} is managing the merger of {company_a} and {company_b}. The merger is projected to yield annual "
        f"synergy savings of ${S} million, incur an annual incremental integration expense of ${E} million, and a one-time "
        f"integration cost of ${I} million. Additionally, the merger is expected to boost revenue by {R}% over the pre-merger "
        f"revenue of ${P} million. Calculate the overall net financial impact of the merger over {T} years."
    )
    solution = (
        f"Step 1: Calculate total synergy savings over {T} years:\n"
        f"  Total Synergy Savings = S × T = {S} × {T} = {total_synergy} million\n\n"
        f"Step 2: Calculate total incremental integration expenses over {T} years:\n"
        f"  Total Expenses = E × T = {E} × {T} = {total_expense} million\n\n"
        f"Step 3: Compute net synergy savings by subtracting expenses and the one-time cost:\n"
        f"  Net Synergy Saving = Total Synergy Savings - Total Expenses - I\n"
        f"                    = {total_synergy} - {total_expense} - {I} = {net_synergy} million\n\n"
        f"Step 4: Calculate the additional revenue increase over {T} years:\n"
        f"  Revenue Increase = Pre-merger Revenue × (R / 100) × T\n"
        f"                   = {P} × ({R} / 100) × {T} = {revenue_increase} million\n\n"
        f"Step 5: Determine the overall net impact:\n"
        f"  Overall Net Impact = Net Synergy Saving + Revenue Increase\n"
        f"                     = {net_synergy} + {revenue_increase} = {overall_net_impact} million\n\n"
        f"Thus, the overall net financial impact is {overall_net_impact} million dollars."
    )
    return question, solution

# ----------------------------- Main ----------------------------- #
def main():
    """
    Generate multiple instances for each financial integration template and write the results to a JSONL file.
    """
    # List of template functions
    templates = [
        basic_integration_synergy_calculation,
        basic_valuation_adjustment_after_integration,
        intermediate_integration_cost_reduction,
        intermediate_cultural_integration_index_adjustment,
        advanced_integration_synergy_and_costs_balancing
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
    output_file = "../../testset/mergers_and_acquisitions/post-merger_integration.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
    main()
