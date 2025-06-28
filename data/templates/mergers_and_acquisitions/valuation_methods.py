import random
import json

# Named entities for investors and US companies
investor_names = ["John Doe", "Susan Lee", "Emily White", "Mark Smith", "David Brown"]
us_company_names = [
    "Apple", "Microsoft", "Amazon", "Google", "Facebook",
    "Tesla", "Netflix", "Walmart", "Boeing", "Coca-Cola"
]

# Template 1: Basic Comparable Company Analysis using Revenue Multiple
def basic_comparable_revenue_valuation():
    """1:Basic: Comparable Company Analysis using Revenue Multiple"""
    investor = random.choice(investor_names)
    company = random.choice(us_company_names)
    revenue = random.randint(50, 200)  # Revenue in million dollars
    revenue_multiple = round(random.uniform(2, 5), 2)
    net_debt = random.randint(5, 20)  # Net debt in million dollars
    question = (
        f"{investor} is evaluating the acquisition of {company}. The company generated revenue of ${revenue} million. "
        f"Comparable companies trade at a revenue multiple of {revenue_multiple}x. Given the company's net debt is ${net_debt} million, "
        f"calculate the estimated equity value using comparable company analysis."
    )
    # Step 1: Calculate Enterprise Value (EV)
    ev = round(revenue * revenue_multiple, 2)
    # Step 2: Calculate Equity Value (EV minus net debt)
    equity_value = round(ev - net_debt, 2)
    solution = (
        f"Step 1: Calculate Enterprise Value (EV):\n"
        f"  EV = Revenue × Revenue Multiple = {revenue} × {revenue_multiple} = {ev} million\n\n"
        f"Step 2: Calculate Equity Value:\n"
        f"  Equity Value = EV - Net Debt = {ev} - {net_debt} = {equity_value} million"
    )
    return question, solution

# Template 2: Basic Valuation using EBITDA Multiple
def basic_ebitda_multiple_valuation():
    """2:Basic: Valuation using EBITDA Multiple"""
    investor = random.choice(investor_names)
    company = random.choice(us_company_names)
    ebitda = random.randint(10, 100)  # EBITDA in million dollars
    ebitda_multiple = round(random.uniform(5, 10), 2)
    net_debt = random.randint(10, 30)  # Net debt in million dollars
    question = (
        f"{investor} is considering acquiring {company}. The company's EBITDA is ${ebitda} million, and the industry average EBITDA "
        f"multiple is {ebitda_multiple}x. With a net debt of ${net_debt} million, calculate the implied equity value."
    )
    # Step 1: Compute Enterprise Value
    ev = round(ebitda * ebitda_multiple, 2)
    # Step 2: Compute Equity Value
    equity_value = round(ev - net_debt, 2)
    solution = (
        f"Step 1: Calculate Enterprise Value (EV):\n"
        f"  EV = EBITDA × EBITDA Multiple = {ebitda} × {ebitda_multiple} = {ev} million\n\n"
        f"Step 2: Calculate Equity Value:\n"
        f"  Equity Value = EV - Net Debt = {ev} - {net_debt} = {equity_value} million"
    )
    return question, solution

# Template 3: Intermediate Discounted Cash Flow (DCF) Valuation
def intermediate_dcf_valuation():
    """3:Intermediate: Discounted Cash Flow (DCF) Valuation"""
    investor = random.choice(investor_names)
    company = random.choice(us_company_names)
    fcf0 = random.randint(5, 30)  # Initial free cash flow (in million dollars)
    growth_rate = round(random.uniform(0.02, 0.07), 4)  # Annual growth rate (2%-7%)
    discount_rate = round(random.uniform(0.08, 0.12), 4)  # Discount rate (8%-12%)
    years = random.randint(5, 7)  # Projection period in years
    terminal_growth = round(random.uniform(0.01, 0.03), 4)  # Terminal growth rate (1%-3%)
    net_debt = random.randint(10, 40)
    
    question = (
        f"{investor} is evaluating the acquisition of {company} using a discounted cash flow (DCF) analysis. The company's current free cash flow "
        f"is ${fcf0} million, and it is expected to grow at an annual rate of {growth_rate*100:.2f}% for {years} years. The discount rate is "
        f"{discount_rate*100:.2f}%, and the terminal growth rate is {terminal_growth*100:.2f}%. Given a net debt of ${net_debt} million, "
        f"calculate the estimated equity value."
    )
    
    pv_sum = 0
    pv_detail_list = []
    for t in range(1, years + 1):
        fcf_t = round(fcf0 * ((1 + growth_rate) ** t), 2)
        pv_t = round(fcf_t / ((1 + discount_rate) ** t), 2)
        pv_detail_list.append(
            f"  Year {t}: FCF = {fcf0}×(1+{growth_rate:.4f})^{t} = {fcf_t} million, PV = {fcf_t}/(1+{discount_rate:.4f})^{t} = {pv_t} million"
        )
        pv_sum += pv_t
    pv_details = "\n".join(pv_detail_list)
    
    fcf_final = round(fcf0 * ((1 + growth_rate) ** years), 2)
    terminal_value = round(fcf_final * (1 + terminal_growth) / (discount_rate - terminal_growth), 2)
    discounted_terminal = round(terminal_value / ((1 + discount_rate) ** years), 2)
    
    enterprise_value = round(pv_sum + discounted_terminal, 2)
    equity_value = round(enterprise_value - net_debt, 2)
    
    solution = (
        f"Step 1: Calculate the present value (PV) of projected free cash flows for each year:\n"
        f"{pv_details}\n\n"
        f"Total PV of free cash flows = {pv_sum} million\n\n"
        f"Step 2: Calculate the terminal value using the perpetuity growth model:\n"
        f"  Terminal FCF (in year {years}) = {fcf_final} million\n"
        f"  Terminal Value = {fcf_final}×(1+{terminal_growth:.4f})/( {discount_rate:.4f} - {terminal_growth:.4f} ) = {terminal_value} million\n"
        f"  Discounted Terminal Value = {terminal_value}/(1+{discount_rate:.4f})^{years} = {discounted_terminal} million\n\n"
        f"Step 3: Calculate Enterprise Value (EV):\n"
        f"  EV = Total PV + Discounted Terminal Value = {pv_sum} + {discounted_terminal} = {enterprise_value} million\n\n"
        f"Step 4: Calculate Equity Value:\n"
        f"  Equity Value = EV - Net Debt = {enterprise_value} - {net_debt} = {equity_value} million"
    )
    
    return question, solution

# Template 4: Intermediate Precedent Transaction Valuation
def intermediate_precedent_transaction_valuation():
    """4:Intermediate: Precedent Transaction Valuation"""
    investor = random.choice(investor_names)
    company = random.choice(us_company_names)
    precedent_revenue = random.randint(100, 300)  # Revenue of precedent company (in million dollars)
    precedent_transaction_value = random.randint(500, 1500)  # Transaction value (in million dollars)
    target_revenue = random.randint(50, 200)  # Target company's revenue (in million dollars)
    net_debt = random.randint(5, 30)  # Net debt of target (in million dollars)
    
    question = (
        f"{investor} is evaluating the acquisition of {company} based on precedent transactions. In a similar past transaction, a company with a revenue "
        f"of ${precedent_revenue} million was acquired for ${precedent_transaction_value} million. Assuming this precedent sets the revenue multiple, "
        f"estimate the equity value of {company} with a revenue of ${target_revenue} million and net debt of ${net_debt} million."
    )
    
    # Step 1: Calculate the precedent revenue multiple.
    revenue_multiple = round(precedent_transaction_value / precedent_revenue, 2)
    # Step 2: Estimate the target's Enterprise Value.
    ev = round(target_revenue * revenue_multiple, 2)
    # Step 3: Calculate Equity Value.
    equity_value = round(ev - net_debt, 2)
    solution = (
        f"Step 1: Calculate Revenue Multiple:\n"
        f"  Revenue Multiple = Precedent Transaction Value / Precedent Revenue = {precedent_transaction_value} / {precedent_revenue} = {revenue_multiple}x\n\n"
        f"Step 2: Estimate Enterprise Value (EV):\n"
        f"  EV = Target Revenue × Revenue Multiple = {target_revenue} × {revenue_multiple} = {ev} million\n\n"
        f"Step 3: Calculate Equity Value:\n"
        f"  Equity Value = EV - Net Debt = {ev} - {net_debt} = {equity_value} million"
    )
    return question, solution

# Template 5: Advanced Synergy Valuation in M&A
def advanced_synergy_valuation():
    """5:Advanced: Synergy Valuation in Mergers and Acquisitions"""
    investor = random.choice(investor_names)
    # Select two distinct companies
    company_a = random.choice(us_company_names)
    company_b = random.choice([c for c in us_company_names if c != company_a])
    ebitda_a = random.randint(50, 150)  # EBITDA of company A (in million dollars)
    ebitda_b = random.randint(30, 100)  # EBITDA of company B (in million dollars)
    ebitda_multiple = round(random.uniform(7, 10), 2)
    synergy_percentage = round(random.uniform(0.05, 0.15), 4)  # Synergy benefit percentage (5%-15%)
    net_debt_a = random.randint(10, 40)
    net_debt_b = random.randint(10, 40)
    
    question = (
        f"{investor} is considering a merger between {company_a} and {company_b}. {company_a} has an EBITDA of ${ebitda_a} million and "
        f"{company_b} has an EBITDA of ${ebitda_b} million. Both companies are valued using an EBITDA multiple of {ebitda_multiple}x. "
        f"Additionally, the merger is expected to yield synergy benefits that increase the combined EBITDA by {synergy_percentage*100:.2f}%. "
        f"If the net debts of the two companies are ${net_debt_a} million and ${net_debt_b} million respectively, calculate the combined equity value."
    )
    
    # Step 1: Calculate the combined standalone EBITDA.
    combined_ebitda = ebitda_a + ebitda_b
    # Step 2: Calculate the synergy benefit.
    synergy_benefit = round(combined_ebitda * synergy_percentage, 2)
    # Step 3: Calculate the adjusted combined EBITDA.
    adjusted_ebitda = combined_ebitda + synergy_benefit
    # Step 4: Compute the combined Enterprise Value (EV).
    ev = round(adjusted_ebitda * ebitda_multiple, 2)
    # Step 5: Calculate the total net debt.
    total_net_debt = net_debt_a + net_debt_b
    # Step 6: Compute the combined Equity Value.
    equity_value = round(ev - total_net_debt, 2)
    
    solution = (
        f"Step 1: Calculate combined standalone EBITDA:\n"
        f"  Combined EBITDA = EBITDA_{company_a} + EBITDA_{company_b} = {ebitda_a} + {ebitda_b} = {combined_ebitda} million\n\n"
        f"Step 2: Calculate the synergy benefit:\n"
        f"  Synergy Benefit = Combined EBITDA × Synergy Percentage = {combined_ebitda} × {synergy_percentage:.4f} = {synergy_benefit} million\n\n"
        f"Step 3: Calculate adjusted combined EBITDA:\n"
        f"  Adjusted EBITDA = Combined EBITDA + Synergy Benefit = {combined_ebitda} + {synergy_benefit} = {adjusted_ebitda} million\n\n"
        f"Step 4: Compute Enterprise Value (EV):\n"
        f"  EV = Adjusted EBITDA × EBITDA Multiple = {adjusted_ebitda} × {ebitda_multiple} = {ev} million\n\n"
        f"Step 5: Calculate total net debt:\n"
        f"  Total Net Debt = Net Debt_{company_a} + Net Debt_{company_b} = {net_debt_a} + {net_debt_b} = {total_net_debt} million\n\n"
        f"Step 6: Calculate combined Equity Value:\n"
        f"  Equity Value = EV - Total Net Debt = {ev} - {total_net_debt} = {equity_value} million"
    )
    return question, solution

def main():
    """
    Generate 10 instances of each valuation template (5 functions) with different random seeds
    and save the generated QA pairs along with step-by-step solutions to a JSONL file.
    """
    templates = [
        basic_comparable_revenue_valuation,
        basic_ebitda_multiple_valuation,
        intermediate_dcf_valuation,
        intermediate_precedent_transaction_valuation,
        advanced_synergy_valuation
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
    output_file = "../../testset/mergers_and_acquisitions/valuation_methods.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
    main()
