import random
import json

# Named entities for investors and companies
investor_names = ["Alice Johnson", "Bob Smith", "Catherine Lee", "Daniel Brown", "Eva Green"]
company_names = ["Apple Inc.", "Google LLC", "Microsoft Corp.", "Amazon.com Inc.", "Tesla Inc."]

# ------------------ Basic Due Diligence Questions ------------------

def basic_due_diligence_working_capital():
    """
    1:Basic: Calculate Working Capital
    An investor evaluates a company's liquidity by computing its working capital.
    """
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    # Generate current assets and liabilities ensuring assets > liabilities for positive working capital
    current_assets = random.randint(20000, 50000)
    current_liabilities = random.randint(10000, current_assets - 5000)
    
    question = (
        f"{investor} is conducting due diligence on {company}. The company reports current assets of ${current_assets} "
        f"and current liabilities of ${current_liabilities}. Calculate the working capital."
    )
    
    # Working capital is the difference between current assets and current liabilities.
    working_capital = current_assets - current_liabilities
    solution = (
        "Step 1: Recall the formula for Working Capital:\n"
        "        Working Capital = Current Assets - Current Liabilities.\n"
        f"Step 2: Substitute the given values: {current_assets} - {current_liabilities} = {working_capital}.\n"
        f"Therefore, the working capital is ${working_capital}."
    )
    return question, solution

def basic_due_diligence_pe_valuation():
    """
    2:Basic: Calculate Market Capitalization using P/E Ratio
    An investor assesses a company's valuation by computing its market capitalization from its net income and P/E ratio.
    """
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    net_income = random.randint(500, 2000)  # in millions
    pe_ratio = round(random.uniform(10, 25), 1)
    
    question = (
        f"{investor} is reviewing {company} as a potential acquisition target. The company reported a net income of "
        f"${net_income} million and has a Price/Earnings (P/E) ratio of {pe_ratio}. Calculate the market capitalization."
    )
    
    # Market Capitalization = Net Income * P/E Ratio.
    market_cap = round(net_income * pe_ratio, 2)
    solution = (
        "Step 1: Recall the market cap calculation formula:\n"
        "        Market Capitalization = Net Income × P/E Ratio.\n"
        f"Step 2: Substitute the values: {net_income} × {pe_ratio} = {market_cap}.\n"
        f"Therefore, the market capitalization is ${market_cap} million."
    )
    return question, solution

# ------------------ Intermediate Due Diligence Questions ------------------

# def intermediate_due_diligence_synergy_estimation():
#     """
#     Intermediate: Estimate Annual Synergy Benefit from a Merger
#     An investor evaluates a merger between two companies by estimating revenue synergies.
#     """
#     investor = random.choice(investor_names)
#     company1 = random.choice(company_names)
#     # Ensure the two companies are different
#     company2 = random.choice([c for c in company_names if c != company1])
#     revenue1 = random.randint(100000, 500000)
#     revenue2 = random.randint(100000, 500000)
#     synergy_rate = round(random.uniform(0.03, 0.07), 4)  # e.g., 3% to 7%
#     integration_cost = random.randint(50000, 150000)  # one-time cost
    
#     question = (
#         f"{investor} is considering the merger of {company1} and {company2}. {company1} has an annual revenue of ${revenue1}, "
#         f"and {company2} has an annual revenue of ${revenue2}. The merger is expected to realize synergies at a rate of "
#         f"{synergy_rate*100:.1f}% of the combined revenue, but a one-time integration cost of ${integration_cost} will be incurred. "
#         f"Calculate the net annual synergy benefit (assume integration cost is amortized over one year)."
#     )
    
#     # Step 1: Compute combined annual revenue.
#     combined_revenue = revenue1 + revenue2
#     # Step 2: Compute the potential synergy benefit.
#     potential_synergy = round(combined_revenue * synergy_rate, 2)
#     # Step 3: Adjust for the integration cost.
#     net_synergy = round(potential_synergy - integration_cost, 2)
    
#     solution = (
#         "Step 1: Calculate combined revenue:\n"
#         f"        {revenue1} + {revenue2} = {combined_revenue}.\n"
#         "Step 2: Compute the potential synergy benefit:\n"
#         f"        Combined Revenue × Synergy Rate = {combined_revenue} × {synergy_rate:.4f} = {potential_synergy}.\n"
#         "Step 3: Subtract the integration cost to get the net synergy benefit:\n"
#         f"        {potential_synergy} - {integration_cost} = {net_synergy}.\n"
#         f"Therefore, the net annual synergy benefit is ${net_synergy}."
#     )
#     return question, solution

def intermediate_due_diligence_synergy_estimation():
    """
    3:Intermediate: Estimate Annual Synergy Benefit from a Merger
    An investor evaluates a merger between two companies by estimating revenue synergies.
    """
    investor = random.choice(investor_names)
    company1 = random.choice(company_names)
    # Ensure the two companies are different
    company2 = random.choice([c for c in company_names if c != company1])
    revenue1 = random.randint(100000, 500000)
    revenue2 = random.randint(100000, 500000)
    synergy_rate = round(random.uniform(0.03, 0.07), 4)  # e.g., 3% to 7%
    integration_cost = random.randint(50000, 150000)  # one-time cost
    
    question = (
        f"{investor} is considering the merger of {company1} and {company2}. {company1} has an annual revenue of ${revenue1}, "
        f"and {company2} has an annual revenue of ${revenue2}. The merger is expected to realize synergies at a rate of "
        f"{synergy_rate*100:.1f}% of the combined revenue, but a one-time integration cost of ${integration_cost} will be incurred. "
        f"Calculate the net annual synergy benefit (assume integration cost is amortized over one year)."
    )
    
    # Step 1: Compute combined annual revenue.
    combined_revenue = revenue1 + revenue2
    # Step 2: Compute the potential synergy benefit.
    annual_synergy = round(combined_revenue * synergy_rate, 2)
    # Step 3: Adjust for the integration cost (amortized over one year).
    net_first_year_synergy = round(annual_synergy - integration_cost, 2)
    
    solution = (
        "Step 1: Calculate combined revenue:\n"
        f"        {revenue1} + {revenue2} = {combined_revenue}.\n"
        "Step 2: Compute the annual synergy benefit:\n"
        f"        Combined Revenue × Synergy Rate = {combined_revenue} × {synergy_rate:.4f} = {annual_synergy}.\n"
        "Step 3: Subtract the amortized integration cost to get the net first-year synergy benefit:\n"
        f"        {annual_synergy} - {integration_cost} = {net_first_year_synergy}.\n"
        f"Therefore, the net annual synergy benefit is ${net_first_year_synergy} for the first year, and ${annual_synergy} for subsequent years (without the integration cost)."
    )
    
    return question, solution

def intermediate_due_diligence_liability_adjustment():
    """
    4:Intermediate: Calculate Expected Contingent Liability from a Pending Lawsuit
    An investor assesses potential risks by calculating the expected liability given a probability of a loss.
    """
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    potential_liability = random.randint(100000, 500000)
    probability = round(random.uniform(0.3, 0.7), 2)  # probability between 30% and 70%
    
    question = (
        f"{investor} is performing due diligence on {company}. The company faces a pending lawsuit that could cost up to "
        f"${potential_liability} if lost. With a {probability*100:.0f}% chance of incurring this cost, calculate the expected liability."
    )
    
    # Expected Liability = Potential Liability * Probability.
    expected_liability = round(potential_liability * probability, 2)
    solution = (
        "Step 1: Recall that Expected Liability = Potential Liability × Probability of Occurrence.\n"
        f"Step 2: Calculate: {potential_liability} × {probability} = {expected_liability}.\n"
        f"Therefore, the expected contingent liability is ${expected_liability}."
    )
    return question, solution

# ------------------ Advanced Due Diligence Question ------------------

def advanced_due_diligence_dcf_valuation():
    """
    5:Advanced: Perform a 5-year DCF valuation of enterprise value.

    Key correction:
        • Year-1 FCF is given; growth begins **after** Year-1.
          Therefore FCF_t = FCF_1 · (1+g)^(t-1),  t = 1…N.
    """
    investor = random.choice(investor_names)
    company  = random.choice(company_names)

    fcf_year1           = random.randint(200_000, 1_000_000)           # Year-1 free cash flow
    growth_rate         = round(random.uniform(0.03, 0.07), 4)         # 3-7 %
    discount_rate       = round(random.uniform(0.10, 0.15), 4)         # 10-15 %
    terminal_growth     = round(random.uniform(0.02, 0.04), 4)         # 2-4 %
    years               = 5

    question = (
        f"{investor} is evaluating {company} for acquisition and wants to perform a DCF valuation. "
        f"The company's Year-1 free cash flow (FCF) is ${fcf_year1:,}. It is projected to grow at "
        f"{growth_rate*100:.2f}% per year for the next {years-1} years. The discount rate is "
        f"{discount_rate*100:.2f}%, and the terminal growth rate beyond Year-{years} is "
        f"{terminal_growth*100:.2f}%. Calculate the enterprise value by discounting the projected "
        f"FCFs and the terminal value."
    )

    pv_sum, pv_lines = 0.0, []
    for t in range(1, years + 1):
        # Apply growth starting **after** Year-1
        fcf_t = fcf_year1 * (1 + growth_rate) ** (t - 1)
        pv_t  = fcf_t / (1 + discount_rate) ** t
        pv_sum += pv_t

        pv_lines.append(
            f"Year {t}:  FCF = {fcf_year1} × (1+{growth_rate:.4f})^{t-1} = {fcf_t:,.2f};   "
            f"PV = {fcf_t:,.2f} / (1+{discount_rate:.4f})^{t} = {pv_t:,.2f}"
        )

    # Terminal value uses the Year-5 FCF as the next-year cash flow (FCF_6)
    fcf_yearN   = fcf_year1 * (1 + growth_rate) ** (years - 1)
    terminal_tv = (fcf_yearN * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_tv       = terminal_tv / (1 + discount_rate) ** years
    enterprise_value = pv_sum + pv_tv

    solution = (
        "Step 1 – Discount forecast-period FCFs:\n"
        + "\n".join(pv_lines) +
        f"\n\nStep 2 – Terminal value at end of Year {years}:\n"
        f"        TV = [FCF_{years} × (1 + g_T)] / (r – g_T)\n"
        f"           = [{fcf_yearN:,.2f} × (1 + {terminal_growth:.4f})] / "
        f"({discount_rate:.4f} – {terminal_growth:.4f}) = {terminal_tv:,.2f}\n"
        f"Step 3 – Present value of TV: {terminal_tv:,.2f} / (1+{discount_rate:.4f})^{years} = {pv_tv:,.2f}\n"
        f"Step 4 – Enterprise value: ΣPV_FCF ({pv_sum:,.2f}) + PV_TV ({pv_tv:,.2f}) "
        f"= {enterprise_value:,.2f}\n\n"
        f"Therefore, the estimated enterprise value is **${enterprise_value:,.2f}**."
    )

    return question, solution

# def advanced_due_diligence_dcf_valuation():
#     """
#     Advanced: Perform DCF Valuation for Enterprise Value Estimation
#     An investor uses a discounted cash flow (DCF) analysis to estimate a company's enterprise value.
#     """
#     investor = random.choice(investor_names)
#     company = random.choice(company_names)
#     fcf_initial = random.randint(200000, 1000000)  # Free Cash Flow in dollars for Year 1
#     growth_rate = round(random.uniform(0.03, 0.07), 4)  # Annual growth rate (3%-7%)
#     discount_rate = round(random.uniform(0.10, 0.15), 4)  # Discount rate (10%-15%)
#     terminal_growth_rate = round(random.uniform(0.02, 0.04), 4)  # Terminal growth rate (2%-4%)
#     years = 5

#     question = (
#         f"{investor} is evaluating {company} for acquisition and wants to perform a DCF valuation. The company's Year 1 free cash flow (FCF) "
#         f"is ${fcf_initial}. It is projected to grow at {growth_rate*100:.1f}% per year over the next {years} years. The discount rate is "
#         f"{discount_rate*100:.1f}% and the terminal growth rate beyond year {years} is {terminal_growth_rate*100:.1f}%. "
#         f"Calculate the enterprise value by determining the present value of the projected FCFs and the terminal value."
#     )
    
#     pv_sum = 0
#     pv_details = ""
#     fcf_current = fcf_initial
    
#     for year in range(1, years+1):
#         # For Year 1, use the initial FCF without growth
#         # For subsequent years, apply growth from previous year
#         if year == 1:
#             fcf_year = fcf_current
#         else:
#             fcf_current = round(fcf_current * (1 + growth_rate), 2)
#             fcf_year = fcf_current
            
#         # Discount each year's FCF back to present value
#         pv = round(fcf_year / ((1 + discount_rate) ** year), 2)
#         pv_sum += pv
        
#         if year == 1:
#             pv_details += f"Year {year}: FCF = {fcf_year} (initial), PV = {fcf_year}/(1+{discount_rate:.4f})^{year} = {pv}\n"
#         else:
#             pv_details += f"Year {year}: FCF = {fcf_year} (previous year's FCF × (1+{growth_rate:.4f})), PV = {fcf_year}/(1+{discount_rate:.4f})^{year} = {pv}\n"

#     # Calculate terminal value using the perpetuity formula with growth
#     # For Year 6 (after the forecast period), apply growth to Year 5 FCF
#     fcf_year6 = round(fcf_current * (1 + growth_rate), 2)
#     terminal_value = round(fcf_year6 / (discount_rate - terminal_growth_rate), 2)
#     pv_terminal = round(terminal_value / ((1 + discount_rate) ** years), 2)
#     enterprise_value = round(pv_sum + pv_terminal, 2)

#     solution = (
#         "Step 1: Calculate the present value of each year's free cash flow (FCF):\n" +
#         pv_details +
#         f"\nStep 2: Calculate the terminal value (TV) at the end of year {years}:\n"
#         f"        First, calculate Year {years+1} FCF: {fcf_current} × (1 + {growth_rate:.4f}) = {fcf_year6}\n"
#         f"        TV = FCF in Year {years+1} / (Discount Rate - Terminal Growth Rate)\n"
#         f"        TV = {fcf_year6} / ({discount_rate:.4f} - {terminal_growth_rate:.4f}) = {terminal_value}\n"
#         "Step 3: Discount the terminal value to present value:\n"
#         f"        PV of TV = {terminal_value} / (1 + {discount_rate:.4f})^{years} = {pv_terminal}\n"
#         "Step 4: Sum the present values of the FCFs and the discounted terminal value:\n"
#         f"        Enterprise Value = {pv_sum} + {pv_terminal} = {enterprise_value}\n"
#         f"Therefore, the estimated enterprise value is ${enterprise_value}."
#     )
    
#     return question, solution

# ------------------ Main Method ------------------

def main():
    """
    Generate 5 due diligence QA pairs and save the results to a JSON file.
    There are two basic, two intermediate, and one advanced question.
    """
    templates = [
        basic_due_diligence_working_capital,
        basic_due_diligence_pe_valuation,
        intermediate_due_diligence_synergy_estimation,
        intermediate_due_diligence_liability_adjustment,
        advanced_due_diligence_dcf_valuation
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
    output_file = "../../testset/mergers_and_acquisitions/due_diligence.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
    main()