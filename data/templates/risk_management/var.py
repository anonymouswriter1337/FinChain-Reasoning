import math
import random
import scipy.stats as stats

def generate_random_value(range_min, range_max):
    return random.randint(range_min, range_max)

def basic_portfolio_var():
    """1:Basic: Calculate VaR for a portfolio with given parameters."""
    portfolio_value = generate_random_value(1000000, 10000000)
    confidence_level = round(random.uniform(0.8, 0.99), 2)
    z_score = round(random.uniform(1.5, 2.5), 2)
    volatility = round(random.uniform(0.01, 0.05), 3)  # 1-5%

    var = z_score * volatility * portfolio_value
    question = (
        f"A portfolio is valued at ${portfolio_value:,}. It has a daily volatility of {volatility*100:.1f}% and is evaluated at a {confidence_level*100:.0f}% confidence level. "
        f"Assuming a Z-score of {z_score}, "
        "what is the Value at Risk (VaR)?"
    )
    solution = (
        f"Step 1: VaR is calculated as Z-score * volatility * portfolio value.\n"
        f"  VaR = {z_score:.2f} * {volatility*100:.1f}% * ${portfolio_value:,} = ${var:,.2f}"
    )
    return question, solution

def basic_var_with_time_horizon():
    """2:Basic: Calculate VaR for a portfolio over a multi-day horizon."""
    portfolio_value = generate_random_value(500000, 5000000)
    confidence_level = round(random.uniform(0.8, 0.99), 2)
    z_score = round(random.uniform(1.5, 2.5), 2)
    daily_volatility = round(random.uniform(0.01, 0.02), 3)  # 1-2%
    time_horizon = generate_random_value(1, 10)

    var = z_score * daily_volatility * math.sqrt(time_horizon) * portfolio_value
    question = (
        f"A portfolio is valued at ${portfolio_value:,}. It has a daily volatility of {daily_volatility*100:.1f}% "
        f"and is evaluated at a {confidence_level*100:.0f}% confidence level over {time_horizon} days. "
        f"Assuming a Z-score of {z_score}, "
        "what is the Value at Risk (VaR) for this time horizon?"
    )
    solution = (
        f"Step 1: VaR is calculated as Z-score * daily volatility * sqrt(time horizon) * portfolio value.\n"
        f"  VaR = {z_score:.2f} * {daily_volatility*100:.1f}% * sqrt({time_horizon}) * ${portfolio_value:,} = ${var:,.2f}"
    )
    return question, solution


def intermediate_weighted_portfolio_var():
    """3:Intermediate: Calculate VaR for a weighted portfolio of two assets."""
    # Generate realistic portfolio parameters
    portfolio_value = round(random.uniform(2000000, 5000000), -3)  # Round to nearest thousand
    weight_a = round(random.uniform(0.3, 0.7), 2)
    weight_b = round(1 - weight_a, 2)  # Ensure weights sum to 1 exactly
    
    # Annual volatilities (standard deviations)
    annual_volatility_a = round(random.uniform(0.12, 0.25), 2)  # 12-25% annually
    annual_volatility_b = round(random.uniform(0.08, 0.20), 2)  # 8-20% annually
    
    # Convert to daily volatilities (approximate using sqrt of time)
    trading_days = 252
    daily_volatility_a = annual_volatility_a / math.sqrt(trading_days)
    daily_volatility_b = annual_volatility_b / math.sqrt(trading_days)
    
    # Realistic correlation between -0.2 and 0.8 (assets often have positive correlation)
    correlation = round(random.uniform(-0.2, 0.8), 2)
    
    # Calculate portfolio volatility (daily)
    portfolio_daily_volatility = math.sqrt(
        (weight_a * daily_volatility_a) ** 2 +
        (weight_b * daily_volatility_b) ** 2 +
        2 * weight_a * weight_b * daily_volatility_a * daily_volatility_b * correlation
    )
    
    # Different confidence levels for more varied questions
    confidence_levels = {
        0.95: 1.645,  # 95% confidence
        0.99: 2.326,  # 99% confidence
        0.975: 1.96   # 97.5% confidence
    }
    
    # Randomly select a confidence level
    confidence = random.choice([0.95, 0.99, 0.975])
    z_score = confidence_levels[confidence]
    
    # Calculate 1-day VaR
    daily_var = z_score * portfolio_daily_volatility * portfolio_value
    
    # Generate time horizon (1, 5, or 10 days)
    time_horizon = random.choice([1, 5, 10])
    
    # Scale VaR to the appropriate time horizon (using square root of time rule)
    var = daily_var * math.sqrt(time_horizon) if time_horizon > 1 else daily_var
    
    question = (
        f"A portfolio worth ${portfolio_value:,.0f} consists of two assets with weights {weight_a:.2f} and {weight_b:.2f}. "
        f"Asset A has an annual volatility of {annual_volatility_a:.2%} and Asset B has an annual volatility of {annual_volatility_b:.2%}. "
        f"The correlation between these assets is {correlation:.2f}. "
        f"Calculate the {confidence:.1%} Value at Risk (VaR) for a {time_horizon}-day time horizon."
    )
    
    solution = (
        f"Step 1: Convert annual volatilities to {time_horizon}-day volatilities.\n"
        f"  Daily volatility of Asset A = {annual_volatility_a:.2%} / √252 = {daily_volatility_a:.4%}\n"
        f"  Daily volatility of Asset B = {annual_volatility_b:.2%} / √252 = {daily_volatility_b:.4%}\n\n"
        
        f"Step 2: Calculate the portfolio's daily volatility using the formula:\n"
        f"  σp = √(w₁²σ₁² + w₂²σ₂² + 2w₁w₂σ₁σ₂ρ₁₂)\n"
        f"  σp = √({weight_a:.2f}² × {daily_volatility_a:.4%}² + {weight_b:.2f}² × {daily_volatility_b:.4%}² + "
        f"2 × {weight_a:.2f} × {weight_b:.2f} × {daily_volatility_a:.4%} × {daily_volatility_b:.4%} × {correlation:.2f})\n"
        f"  Daily portfolio volatility = {portfolio_daily_volatility:.4%}\n\n"
        
        f"Step 3: Calculate the {time_horizon}-day portfolio volatility.\n"
        f"  {time_horizon}-day volatility = Daily volatility × √{time_horizon} = {portfolio_daily_volatility:.4%} × √{time_horizon} = {portfolio_daily_volatility * math.sqrt(time_horizon):.4%}\n\n"
        
        f"Step 4: Calculate the {confidence:.1%} VaR.\n"
        f"  VaR = Z-score × {time_horizon}-day volatility × Portfolio Value\n"
        f"  VaR = {z_score:.3f} × {portfolio_daily_volatility * math.sqrt(time_horizon):.4%} × ${portfolio_value:,.0f}\n"
        f"  VaR = ${var:,.2f}\n\n"
        
        f"This means there is a {(1-confidence):.1%} probability that the portfolio will lose more than ${var:,.2f} over the next {time_horizon} {'day' if time_horizon == 1 else 'days'}."
    )
    
    return question, solution

def intermediate_var_with_interest_rate():
    """4:Intermediate: Impact of interest rates on VaR."""
    portfolio_value = generate_random_value(3000000, 8000000)
    rate_change = random.uniform(0.005, 0.02)  # 0.5-2% change
    duration = generate_random_value(3, 10)
    confidence_level = random.uniform(0.8, 0.99)
    z_score = round(random.uniform(1.5, 2.5), 2)

    sensitivity = portfolio_value * duration * rate_change
    var = z_score * sensitivity

    question = (
        f"A bond portfolio worth ${portfolio_value:,} has a duration of {duration} years. A 1% interest rate change is expected, "
        f"and z-score is {z_score:.2f} at a {confidence_level*100:.0f}% confidence level. What is the Value at Risk (VaR)?"
    )
    solution = (
        f"Step 1: Sensitivity is calculated as portfolio value * duration * rate change.\n"
        f"  Sensitivity = ${portfolio_value:,} * {duration} * {rate_change*100:.1f}% = ${sensitivity:,.2f}.\n"
        f"Step 2: VaR = Z-score * sensitivity.\n"
        f"VaR = {z_score} * ${sensitivity:,.2f} = ${var:,.2f}"
    )
    return question, solution


# Original code deemed incorrect by financial advisor. A new code by Claude 3.7 Sonet has been provided.
def advanced_var_with_discounted_cash_flows():
    """
    5:Advanced:Calculate Δ-normal VaR for a portfolio of future cash flows.

    Returns
    -------
    question : str  – formatted exam-style question
    solution : str  – step-by-step worked answer
    """
    # --------------------------
    # 1. Scenario assumptions
    # --------------------------
    num_flows      = random.randint(3, 6)
    cash_flows     = [(round(random.uniform(1e6, 3e6), -3), t + 1)
                      for t in range(num_flows)]                 # (amount, year)

    discount_rate  = round(random.uniform(0.03, 0.07), 3)        # 3–7 %
    annual_vol     = round(random.uniform(0.10, 0.25), 3)        # 10–25 %
    trading_days   = 252
    daily_vol      = annual_vol / math.sqrt(trading_days)

    confidence_map = {0.95: 1.645, 0.99: 2.326}
    confidence     = random.choice(list(confidence_map.keys()))
    z_score        = confidence_map[confidence]

    horizon_days   = random.choice([1, 10, 20])
    rho            = round(random.uniform(0.30, 0.80), 2)        # correlation

    # --------------------------
    # 2. Present values
    # --------------------------
    pvs = [amt / (1 + discount_rate) ** yr for amt, yr in cash_flows]

    # --------------------------
    # 3. VaR of each cash flow (FIXED)
    # --------------------------
    sigma_h        = daily_vol * math.sqrt(horizon_days)
    vars_i         = [pv * z_score * sigma_h for pv in pvs]      # <-- no duration factor

    # --------------------------
    # 4. Aggregate VaR with correlation
    # --------------------------
    sumsq   = sum(v**2 for v in vars_i)
    cross   = sum(2 * rho * vars_i[i] * vars_i[j]
                  for i in range(len(vars_i))
                  for j in range(i + 1, len(vars_i)))
    portfolio_var = math.sqrt(sumsq + cross)

    # --------------------------
    # 5. Build Q&A strings
    # --------------------------
    cf_text = ", ".join(
        f"${amt:,.0f} at the end of year {yr}"
        for amt, yr in cash_flows[:-1]
    ) + (f", and ${cash_flows[-1][0]:,.0f} at the end of year {cash_flows[-1][1]}"
         if len(cash_flows) > 1 else
         f"${cash_flows[0][0]:,.0f} at the end of year {cash_flows[0][1]}")

    question = (
        f"A portfolio consists of future cash flows of {cf_text}. "
        f"The annual discount rate is {discount_rate:.2%} and the annual return volatility is {annual_vol:.2%}. "
        f"Assuming the cash flows have a pairwise return correlation of {rho:.2f}, "
        f"what is the {int(confidence*100)}% Value at Risk over a {horizon_days}-day horizon "
        f"for the present value of the portfolio?"
    )

    # --- Worked answer ---
    solution  = "Step 1 – Present values:\n"
    for (amt, yr), pv in zip(cash_flows, pvs):
        solution += f"  PV = ${amt:,.0f} / (1 + {discount_rate:.2%})^{yr} = ${pv:,.2f}\n"
    solution += f"  Total PV = ${sum(pvs):,.2f}\n\n"

    solution += ("Step 2 – Convert annual to daily volatility:\n"
                 f"  σ₁d = {annual_vol:.2%} / √252 = {daily_vol:.4%}\n\n")

    solution += (f"Step 3 – VaR of each cash flow over {horizon_days} day(s):\n"
                 f"  σ_h = σ₁d × √h = {daily_vol:.4%} × √{horizon_days} = {sigma_h:.4%}\n"
                 f"  VaR_i = PV_i × z × σ_h\n")
    for idx, (pv, v) in enumerate(zip(pvs, vars_i), 1):
        solution += f"    CF {idx}: ${pv:,.2f} × {z_score:.3f} × {sigma_h:.4%} = ${v:,.2f}\n"

    solution += ("\nStep 4 – Aggregate VaR with correlation ρ = "
                 f"{rho:.2f} (Δ-normal portfolio formula):\n"
                 "  VaR_p = √(Σ VaR_i² + ΣΣ 2ρ VaR_i VaR_j)\n"
                 f"        = √({sumsq:,.2f} + {cross:,.2f})\n"
                 f"        = ${portfolio_var:,.2f}\n\n")

    tail = "day" if horizon_days == 1 else "days"
    solution += (
        f"Therefore, there is a {(1-confidence):.0%} chance that the portfolio’s present value "
        f"will fall by more than **${portfolio_var:,.2f}** over the next {horizon_days} {tail}."
    )

    return question, solution
'''
def advanced_var_with_discounted_cash_flows():
    """Advanced: Calculate VaR for a portfolio of future cash flows."""
    # Number of future cash flows (3-6)
    num_flows = random.randint(3, 6)
    
    # Generate cash flow amounts and timing
    cash_flows = []
    for i in range(num_flows):
        amount = round(random.uniform(1000000, 3000000), -3)
        year = i + 1  # Cash flows occurring at the end of years 1, 2, 3, etc.
        cash_flows.append((amount, year))
    
    # Annual discount rate
    discount_rate = round(random.uniform(0.03, 0.07), 3)  # 3-7%
    
    # Portfolio volatility (annual)
    annual_volatility = round(random.uniform(0.10, 0.25), 3)  # 10-25%
    
    # Convert to daily volatility
    trading_days = 252
    daily_volatility = annual_volatility / math.sqrt(trading_days)
    
    # VaR parameters
    confidence_levels = {0.95: 1.645, 0.99: 2.326}
    confidence = random.choice([0.95, 0.99])
    z_score = confidence_levels[confidence]
    var_horizon_days = random.choice([1, 10, 20])
    
    # Calculate present values and VaR for each cash flow
    present_values = []
    var_values = []
    
    for amount, year in cash_flows:
        # Calculate present value of this cash flow
        pv = amount / ((1 + discount_rate) ** year)
        present_values.append(pv)
        
        # Calculate VaR for this cash flow
        cf_volatility = daily_volatility * math.sqrt(var_horizon_days)
        
        # Duration effect: longer-term cash flows are more sensitive to interest rate changes
        duration_factor = year
        
        # VaR for this cash flow
        cf_var = pv * z_score * cf_volatility * duration_factor
        var_values.append(cf_var)
    
    # Total present value and VaR
    total_pv = sum(present_values)
    
    # For VaR aggregation, we need to account for correlation between cash flows
    correlation = round(random.uniform(0.3, 0.8), 2)  # Moderate to high correlation
    
    # Calculate VaR with correlation
    sum_squared_vars = sum([var**2 for var in var_values])
    cross_vars = 0
    for i in range(len(var_values)):
        for j in range(i+1, len(var_values)):
            cross_vars += 2 * correlation * var_values[i] * var_values[j]
    
    total_var = math.sqrt(sum_squared_vars + cross_vars)
    
    # Format cash flows for question
    cf_text = ""
    for idx, (amount, year) in enumerate(cash_flows):
        cf_text += f"${amount:,.0f} at the end of year {year}"
        if idx < len(cash_flows) - 2:
            cf_text += ", "
        elif idx == len(cash_flows) - 2:
            cf_text += ", and "
    
    question = (
        f"A portfolio consists of future cash flows of {cf_text}. "
        f"The annual discount rate is {discount_rate:.2%} and the annual volatility is {annual_volatility:.2%}. "
        f"Assuming the cash flows have a correlation of {correlation:.2f}, calculate the {confidence:.0%} "
        f"Value at Risk (VaR) over a {var_horizon_days}-day horizon for the present value of this portfolio."
    )
    
    solution = (
        f"Step 1: Calculate the present value of each cash flow.\n"
        "  PV = CF / (1 + r)^t\n"
    )
    
    # Add calculation for each cash flow
    for idx, ((amount, year), pv) in enumerate(zip(cash_flows, present_values)):
        solution += f"  Cash Flow {idx+1}: ${amount:,.0f} / (1 + {discount_rate:.2%})^{year} = ${pv:,.2f}\n"
    
    solution += f"  Total Present Value = ${total_pv:,.2f}\n\n"
    
    solution += (
        f"Step 2: Convert annual volatility to daily volatility.\n"
        f"  Daily volatility = Annual volatility / √252\n"
        f"  Daily volatility = {annual_volatility:.2%} / √{trading_days} = {daily_volatility:.4%}\n\n"
        
        f"Step 3: Calculate VaR for each cash flow over a {var_horizon_days}-day horizon.\n"
        f"  VaR formula: VaR = PV × z-score × volatility × duration factor × √horizon days\n"
    )
    
    # Add VaR calculation for each cash flow
    for idx, ((amount, year), pv, var) in enumerate(zip(cash_flows, present_values, var_values)):
        solution += (
            f"  Cash Flow {idx+1} VaR: ${pv:,.2f} × {z_score:.3f} × {daily_volatility:.4%} × "
            f"{year} × √{var_horizon_days} = ${var:,.2f}\n"
        )
    
    solution += (
        f"\nStep 4: Aggregate the VaRs accounting for correlation ({correlation:.2f}).\n"
        f"  Total VaR = √(Σ VaR_i² + Σ Σ 2 × ρ × VaR_i × VaR_j) where i≠j\n"
        f"  Total VaR = √(${sum_squared_vars:,.2f} + ${cross_vars:,.2f})\n"
        f"  Total VaR = ${total_var:,.2f}\n\n"
        
        f"Therefore, there is a {(1-confidence):.0%} probability that the present value of this portfolio of cash flows "
        f"will decrease by more than ${total_var:,.2f} over the next {var_horizon_days} {'day' if var_horizon_days == 1 else 'days'}."
    )
    
    return question, solution
'''

def main():
    """
    Generate 10 instances of each template with different random seeds 
    and write the results to a JSON file.
    """
    import json
    # List of template functions
    templates = [
        basic_portfolio_var,
        basic_var_with_time_horizon,
        intermediate_weighted_portfolio_var,
        intermediate_var_with_interest_rate,
        advanced_var_with_discounted_cash_flows
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
    output_file = "../../testset/risk_management/var.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == '__main__':
    main()