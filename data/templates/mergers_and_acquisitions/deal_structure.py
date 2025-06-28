import random
import json

# Named entities for investors and companies
investor_names = ["John Doe", "Susan Lee", "Emily White", "Mark Smith", "David Brown"]
company_names = ["Apple", "Google", "Microsoft", "Amazon", "Tesla", "Facebook", "Netflix", "Walmart"]

# ------------------------------------------------------------------
# Basic Scenario 1: Cash Payment Calculation in a Deal Structure
def basic_cash_payment_calculation():
    """1:Basic: Cash Payment Calculation in a Deal Structure"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    total_deal = random.randint(50, 200)  # Total deal value in million dollars
    cash_percent = random.randint(30, 70)  # Cash payment percentage
    question = (
        f"{investor} is acquiring {company} for ${total_deal} million. The deal is structured so that "
        f"{cash_percent}% of the payment is made in cash while the rest is in stock. "
        f"Calculate the cash component of the deal."
    )
    # Step 1: Calculate cash payment = total_deal * (cash_percent / 100)
    cash_payment = round(total_deal * cash_percent / 100, 2)
    solution = (
        f"Step 1: Identify the total deal value and cash percentage:\n"
        f"  Total Deal Value = ${total_deal} million\n"
        f"  Cash Percentage = {cash_percent}%\n\n"
        f"Step 2: Calculate the cash payment:\n"
        f"  Cash Payment = Total Deal Value × (Cash Percentage / 100)\n"
        f"               = {total_deal} × ({cash_percent} / 100) = {cash_payment} million dollars"
    )
    return question, solution

# ------------------------------------------------------------------
# Basic Scenario 2: Stock Swap Calculation in a Deal Structure
def basic_stock_swap_calculation():
    """2:Basic: Stock Swap Calculation in a Deal Structure"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    total_deal = round(random.uniform(10, 50), 2)  # Total deal value in million dollars
    share_price = round(random.uniform(0.5, 5.0), 2)  # Share price (in million dollars per share)
    question = (
        f"{investor} is acquiring {company} in a stock swap deal valued at ${total_deal} million. "
        f"If each share is valued at ${share_price} million, how many shares will be issued?"
    )
    # Step 1: Calculate shares = total_deal / share_price
    shares = round(total_deal / share_price, 2)
    solution = (
        f"Step 1: Identify the deal value and share price:\n"
        f"  Total Deal Value = ${total_deal} million\n"
        f"  Share Price = ${share_price} million per share\n\n"
        f"Step 2: Compute the number of shares issued:\n"
        f"  Shares Issued = Total Deal Value / Share Price\n"
        f"                = {total_deal} / {share_price} = {shares} shares"
    )
    return question, solution

# ------------------------------------------------------------------
# Intermediate Scenario 1: Earnout Payment with Revenue Bonus Calculation
def intermediate_earnout_with_bonus_calculation():
    """3:Intermediate: Earnout Payment Calculation with Revenue Bonus in a Deal Structure"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    base_price = random.randint(20, 100)           # Base price in million dollars
    earnout_max = random.randint(5, 20)              # Maximum earnout in million dollars
    target_revenue = random.randint(50, 150)         # Target revenue in million dollars
    # Set actual revenue between the target and up to 130% of the target
    actual_revenue = random.randint(target_revenue, int(target_revenue * 1.30))
    bonus_rate = round(random.uniform(0.05, 0.20), 2)  # Bonus rate: additional % (in decimal)
    max_bonus = random.randint(2, 10)                # Maximum bonus in million dollars
    
    question = (
        f"{investor} acquired {company} for a base price of ${base_price} million. The deal includes an earnout clause "
        f"that pays up to ${earnout_max} million if {company} meets a target revenue of ${target_revenue} million. "
        f"If {company} achieves an actual revenue of ${actual_revenue} million, the earnout is prorated. Moreover, "
        f"if the actual revenue exceeds 110% of the target, an additional bonus is paid at a rate of {bonus_rate*100:.0f}% "
        f"on the excess revenue, capped at ${max_bonus} million. Calculate the total deal value (base price plus total earnout)."
    )
    # Step 1: Calculate basic prorated earnout
    earnout_basic = round(earnout_max * actual_revenue / target_revenue, 2)
    # Step 2: Determine bonus threshold (110% of target revenue)
    bonus_threshold = round(1.10 * target_revenue, 2)
    # Step 3: Compute bonus (if actual_revenue exceeds bonus_threshold)
    bonus_unclipped = round((actual_revenue - bonus_threshold) * bonus_rate, 2) if actual_revenue > bonus_threshold else 0
    bonus = min(bonus_unclipped, max_bonus)
    # Step 4: Total earnout is sum of basic earnout and bonus
    total_earnout = round(earnout_basic + bonus, 2)
    # Step 5: Total deal value is base price plus total earnout
    total_deal_value = round(base_price + total_earnout, 2)
    solution = (
        f"Step 1: Compute the Basic Earnout Payment:\n"
        f"  Basic Earnout = Earnout Max × (Actual Revenue / Target Revenue)\n"
        f"                = {earnout_max} × ({actual_revenue} / {target_revenue}) = {earnout_basic} million dollars\n\n"
        f"Step 2: Determine the Bonus Threshold:\n"
        f"  Bonus Threshold = 110% of Target Revenue = 1.10 × {target_revenue} = {bonus_threshold} million dollars\n\n"
        f"Step 3: Compute the Bonus Payment (if applicable):\n"
        f"  Unclipped Bonus = (Actual Revenue - Bonus Threshold) × Bonus Rate\n"
        f"                  = ({actual_revenue} - {bonus_threshold}) × {bonus_rate} = {bonus_unclipped} million dollars\n"
        f"  Bonus = min(Unclipped Bonus, Maximum Bonus)\n"
        f"        = min({bonus_unclipped}, {max_bonus}) = {bonus} million dollars\n\n"
        f"Step 4: Total Earnout Payment = Basic Earnout + Bonus = {earnout_basic} + {bonus} = {total_earnout} million dollars\n\n"
        f"Step 5: Total Deal Value = Base Price + Total Earnout\n"
        f"          = {base_price} + {total_earnout} = {total_deal_value} million dollars"
    )
    return question, solution

# ------------------------------------------------------------------
# Intermediate Scenario 2: Adjusted Equity Shares Calculation in a Leveraged Buyout with Bonus Shares
def intermediate_leveraged_buyout_adjusted_shares_calculation():
    """4:Intermediate: Adjusted Equity Shares Calculation in a Leveraged Buyout with Bonus Shares for Low Share Price"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    total_deal = random.randint(50, 200)  # Total deal value in million dollars
    debt_percent = random.randint(20, 60)   # Debt financing percentage
    equity_percent = 100 - debt_percent     # Equity financing percentage
    share_price = round(random.uniform(0.5, 5.0), 2)  # Share price (in million dollars per share)
    # Define a threshold such that if share_price < threshold, bonus shares are added
    threshold_value = 2.0
    bonus_factor = 0.05  # Additional bonus shares equal to 5% of basic shares if condition met
    
    question = (
        f"In a leveraged buyout, {investor} acquires {company} for a total deal value of ${total_deal} million. "
        f"The deal is financed with {debt_percent}% debt and the remaining {equity_percent}% is raised through equity. "
        f"Equity is raised by issuing shares at a price of ${share_price} million per share. If the share price is below "
        f"${threshold_value} million, an additional 5% bonus in shares is provided. Calculate the adjusted number of shares issued."
    )
    # Step 1: Compute the equity value
    equity_value = round(total_deal * equity_percent / 100, 2)
    # Step 2: Compute basic number of shares (without bonus)
    basic_shares = round(equity_value / share_price, 2)
    # Step 3: Determine bonus shares if share_price < threshold_value
    bonus_shares = round(basic_shares * bonus_factor, 2) if share_price < threshold_value else 0
    # Step 4: Adjusted number of shares = basic shares + bonus shares
    adjusted_shares = round(basic_shares + bonus_shares, 2)
    solution = (
        f"Step 1: Compute the Equity Value:\n"
        f"  Equity Value = Total Deal Value × (Equity Percentage / 100)\n"
        f"               = {total_deal} × ({equity_percent} / 100) = {equity_value} million dollars\n\n"
        f"Step 2: Compute the Basic Number of Shares:\n"
        f"  Basic Shares = Equity Value / Share Price\n"
        f"               = {equity_value} / {share_price} = {basic_shares} shares\n\n"
        f"Step 3: Determine Bonus Shares:\n"
        f"  Since Share Price ({share_price}) < Threshold (${threshold_value}), bonus shares = 5% of basic shares\n"
        f"              = {basic_shares} × {bonus_factor} = {bonus_shares} shares\n\n"
        f"Step 4: Adjusted Number of Shares = Basic Shares + Bonus Shares\n"
        f"              = {basic_shares} + {bonus_shares} = {adjusted_shares} shares"
    )
    return question, solution

# ------------------------------------------------------------------
# Advanced Scenario: Multi-step Calculation Involving Control Premium, Debt, and Earnout
def advanced_control_premium_debt_earnout_calculation():
    """5:Advanced: Multi-step Deal Structure with Control Premium, Earnout, and Debt Financing"""
    investor = random.choice(investor_names)
    company = random.choice(company_names)
    base_price = random.randint(50, 150)                  # Base price in million dollars
    control_premium_percent = random.randint(10, 30)        # Control premium percentage
    earnout_max = random.randint(5, 20)                     # Maximum earnout in million dollars
    achievement_percent = random.randint(50, 100)           # Performance achievement percentage
    debt_percent = random.randint(20, 50)                   # Percentage of deal financed with debt
    question = (
        f"{investor} is acquiring {company} with a multifaceted deal structure. The base price is ${base_price} million. "
        f"A control premium of {control_premium_percent}% is applied to the base price. Additionally, there is an earnout clause "
        f"that pays up to ${earnout_max} million, prorated by performance (currently at {achievement_percent}%). "
        f"Finally, the deal is financed with {debt_percent}% debt. Calculate the following:\n"
        f"  (a) The control premium amount.\n"
        f"  (b) The prorated earnout payment.\n"
        f"  (c) The total deal value (base price + control premium + earnout payment).\n"
        f"  (d) The equity value after subtracting the debt financing."
    )
    # Step 1: Compute the control premium amount
    control_premium = round(base_price * control_premium_percent / 100, 2)
    # Step 2: Compute the prorated earnout payment
    prorated_earnout = round(earnout_max * achievement_percent / 100, 2)
    # Step 3: Calculate the total deal value
    total_deal_value = round(base_price + control_premium + prorated_earnout, 2)
    # Step 4: Determine the debt financing amount
    debt_amount = round(total_deal_value * debt_percent / 100, 2)
    # Step 5: Calculate the equity value
    equity_value = round(total_deal_value - debt_amount, 2)
    solution = (
        f"Step 1: Compute the Control Premium Amount:\n"
        f"  Control Premium = Base Price × (Control Premium % / 100)\n"
        f"                  = {base_price} × ({control_premium_percent} / 100) = {control_premium} million dollars\n\n"
        f"Step 2: Compute the Prorated Earnout Payment:\n"
        f"  Prorated Earnout = Earnout Max × (Achievement % / 100)\n"
        f"                   = {earnout_max} × ({achievement_percent} / 100) = {prorated_earnout} million dollars\n\n"
        f"Step 3: Calculate the Total Deal Value:\n"
        f"  Total Deal Value = Base Price + Control Premium + Prorated Earnout\n"
        f"                   = {base_price} + {control_premium} + {prorated_earnout} = {total_deal_value} million dollars\n\n"
        f"Step 4: Compute the Debt Amount:\n"
        f"  Debt Amount = Total Deal Value × (Debt % / 100)\n"
        f"              = {total_deal_value} × ({debt_percent} / 100) = {debt_amount} million dollars\n\n"
        f"Step 5: Compute the Equity Value:\n"
        f"  Equity Value = Total Deal Value - Debt Amount\n"
        f"               = {total_deal_value} - {debt_amount} = {equity_value} million dollars"
    )
    return question, solution

# ------------------------------------------------------------------
# Main method to generate problems and write to a JSONL file
def main():
    """
    Generate financial reasoning QA pairs on Deal Structure scenarios and save the results to a JSONL file.
    """
    # List of template functions
    templates = [
        basic_cash_payment_calculation,             # Basic
        basic_stock_swap_calculation,               # Basic
        intermediate_earnout_with_bonus_calculation,  # Intermediate (Enhanced)
        intermediate_leveraged_buyout_adjusted_shares_calculation,  # Intermediate (Enhanced)
        advanced_control_premium_debt_earnout_calculation  # Advanced
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
    output_file = "../../testset/mergers_and_acquisitions/deal_structure.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
    main()
