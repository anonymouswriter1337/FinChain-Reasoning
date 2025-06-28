import random

###############################################################################
# Data Pools
###############################################################################
investor_names = ["Alice Wu", "Brad Johnson", "Carla Simmons", "Daniel Craig", "Eva Gonzalez"]
bond_issuers = ["U.S. Treasury", "XYZ Corporation", "ABC Bank", "City of Metropolis", "Acme Corp"]

###############################################################################
# BASIC TEMPLATES (2 Steps)
###############################################################################

# Template 1 (Basic): Zero-Coupon Bond Price
def template_bp_easy1():
    """
    1:Basic: Zero-Coupon Bond Price
    Zero-coupon bond pricing:
    - Face Value (F)
    - Years to maturity (t)
    - Yield per year (r)
    2 steps:
      1) Apply present value formula
      2) Conclude the bond price
    """
    issuer = random.choice(bond_issuers)
    investor = random.choice(investor_names)
    face_value = random.randint(1000, 2000)
    years_to_maturity = random.randint(1, 5)
    yield_rate = round(random.uniform(0.02, 0.1), 3)

    question = (
        f"{investor} is looking to purchase a zero-coupon bond issued by {issuer}. "
        f"The bond has a face value of ${face_value} and will mature in {years_to_maturity} years. "
        f"If the annual yield is {yield_rate*100:.2f}%, what is the price of the bond today?"
    )

    bond_price = face_value / ((1 + yield_rate) ** years_to_maturity)

    solution = (
        f"Step 1: Apply the present value formula for a zero-coupon bond:\n"
        f"  Price = Face Value / (1 + r)^t\n"
        f"        = ${face_value} / (1 + {yield_rate:.3f})^{years_to_maturity}\n\n"
        f"Step 2: Calculate the bond price:\n"
        f"  Price = ${bond_price:,.2f}"
    )

    return question, solution


# Template 2 (Basic): 1-Year Coupon Bond Price
def template_bp_easy2():
    """
    2:Basic: 1-Year Coupon Bond Price
    1-year coupon bond:
    - Face Value (F)
    - Annual coupon (C)
    - Annual yield (r)
    2 steps:
      1) Calculate present value of (coupon + principal)
      2) Conclude the bond price
    """
    issuer = random.choice(bond_issuers)
    investor = random.choice(investor_names)
    face_value = random.randint(1000, 2000)
    coupon_rate = round(random.uniform(0.02, 0.08), 3)
    yield_rate = round(random.uniform(0.02, 0.1), 3)

    coupon_payment = round(face_value * coupon_rate, 2)

    question = (
        f"{investor} is considering a 1-year bond issued by {issuer}. It has a face value of ${face_value}, "
        f"and pays an annual coupon at {coupon_rate*100:.2f}%. If the required yield is {yield_rate*100:.2f}%, "
        f"calculate the bond's price."
    )

    future_cash_flow = round(coupon_payment + face_value, 2)
    bond_price = future_cash_flow / (1 + yield_rate)

    solution = (
        f"Step 1: Sum the coupon and face value, then discount at the required yield:\n"
        f"  Cash Flow in 1 year = Face Value + Coupon = ${face_value} + ${coupon_payment:,.2f} = ${future_cash_flow:,.2f}\n"
        f"  Bond Price = Cash Flow / (1 + r)\n"
        f"            = ${future_cash_flow:,.2f} / (1 + {yield_rate:.3f})\n\n"
        f"Step 2: Calculate the price:\n"
        f"  Bond Price = ${bond_price:,.2f}"
    )

    return question, solution

###############################################################################
# INTERMEDIATE TEMPLATES (3 Steps)
###############################################################################

# Template 3 (Intermediate): Multi-Year Coupon Bond Price
def template_bp_medium1():
    """
    3:Intermediate: Multi-Year Coupon Bond Price
    Multi-year annual coupon bond:
    - Face Value (F)
    - Annual coupon (C)
    - Years to maturity (n)
    - Yield (r)
    3 steps:
      1) Present value of annual coupon payments
      2) Present value of face value
      3) Sum them up
    """
    issuer = random.choice(bond_issuers)
    investor = random.choice(investor_names)
    face_value = random.choice([1000, 2000])
    coupon_rate = round(random.uniform(0.03, 0.09), 3)
    years_to_maturity = random.randint(2, 5)
    yield_rate = round(random.uniform(0.02, 0.1), 3)

    coupon_payment = round(face_value * coupon_rate, 2)

    question = (
        f"{investor} wants to find the fair price of a bond issued by {issuer}. The bond has:\n"
        f"- Face Value = ${face_value}\n"
        f"- Annual Coupon Rate = {coupon_rate*100:.2f}% (i.e., ${coupon_payment:.2f} per year)\n"
        f"- Maturity in {years_to_maturity} years\n"
        f"- Required annual yield = {yield_rate*100:.2f}%\n\n"
        f"Calculate the fair price of this multi-year coupon bond."
    )

    pv_coupons = 0
    for t in range(1, years_to_maturity + 1):
        pv_coupons += coupon_payment / ((1 + yield_rate) ** t)

    pv_face_value = face_value / ((1 + yield_rate) ** years_to_maturity)
    bond_price = pv_coupons + pv_face_value

    solution = (
        f"Step 1: Calculate the present value of each annual coupon and sum them:\n"
        f"  PV(Coupons) = Σ [Coupon / (1 + r)^t] for t=1..{years_to_maturity}\n\n"
        f"Step 2: Calculate the present value of the face value:\n"
        f"  PV(Face Value) = Face Value / (1 + r)^{years_to_maturity}\n\n"
        f"Step 3: Sum the present values:\n"
        f"  Bond Price = PV(Coupons) + PV(Face Value)\n"
        f"  => ≈ ${bond_price:,.2f}"
    )

    return question, solution


# Template 4 (Intermediate): Solve Yield for a 2-Year Bond
def template_bp_medium2():
    """
    4:Intermediate: Solve Yield for a 2-Year Bond
    2-year bond, given Price, Face Value, and annual Coupon, solve for yield (r).
    3 steps:
      1) Write the 2-year bond pricing formula
      2) Rearrange or apply numeric approach
      3) Solve for r and interpret
    """
    issuer = random.choice(bond_issuers)
    investor = random.choice(investor_names)
    face_value = 1000
    coupon_rate = round(random.uniform(0.03, 0.08), 3)
    coupon_payment = round(face_value * coupon_rate, 2)
    true_yield = round(random.uniform(0.02, 0.1), 3)  # hidden yield for generating price

    p1 = coupon_payment / (1 + true_yield)
    p2 = (coupon_payment + face_value) / ((1 + true_yield) ** 2)
    bond_price = round(p1 + p2, 2)

    question = (
        f"{investor} is analyzing a 2-year bond issued by {issuer}, face value ${face_value}, "
        f"annual coupon rate {coupon_rate*100:.2f}% (${coupon_payment:.2f} per year). "
        f"The bond trades at ${bond_price:,.2f}. Find the yield to maturity."
    )

    solution = (
        f"Step 1: Pricing formula for a 2-year bond:\n"
        f"  P = C/(1+r) + (C + F)/(1+r)^2\n\n"
        f"Step 2: Plug in the values:\n"
        f"  {bond_price:,.2f} = {coupon_payment:,.2f}/(1+r) + "
        f"({coupon_payment:,.2f} + {face_value})/(1+r)^2\n\n"
        f"Step 3: Solve for r (using a numeric or quadratic formula).\n"
        f"  The approximate yield is {true_yield*100:.2f}%."
    )
    return question, solution

###############################################################################
# ADVANCED TEMPLATE (4 Steps)
###############################################################################

# Template 5 (Advanced): Semiannual Coupon Bond with Accrued Interest
def template_bp_hard1():
    """
    5:Advanced: Semiannual Coupon Bond with Accrued Interest
    This problem has 4 steps:
      1) Compute clean price (discount future semiannual coupons + face value)
      2) Compute accrued interest
      3) Add accrued interest to clean price
      4) Summarize final bond price
    """
    issuer = random.choice(bond_issuers)
    investor = random.choice(investor_names)
    face_value = 1000
    annual_coupon_rate = round(random.uniform(0.03, 0.08), 3)
    semiannual_coupon = round(face_value * annual_coupon_rate / 2, 2)
    years_to_maturity = random.randint(2, 5)
    semiannual_yield = round(random.uniform(0.015, 0.05), 3)
    periods = years_to_maturity * 2

    # Step 1: Clean price = PV of future semiannual coupons + PV of face value
    pv_coupons = 0
    for t in range(1, periods + 1):
        pv_coupons += semiannual_coupon / ((1 + semiannual_yield) ** t)
    pv_face_value = face_value / ((1 + semiannual_yield) ** periods)
    clean_price = round(pv_coupons + pv_face_value, 2)

    # Step 2: Accrued interest
    # Assume we've gone 'months_since_coupon' months into the current 6-month period
    months_since_coupon = random.randint(1, 5)
    accrued_interest = round(semiannual_coupon * (months_since_coupon / 6), 2)

    # Step 3: Dirty price
    dirty_price = clean_price + accrued_interest

    question = (
        f"{investor} is evaluating a bond issued by {issuer} with:\n"
        f" - Face Value = ${face_value}\n"
        f" - Annual Coupon Rate = {annual_coupon_rate*100:.2f}% (paid semiannually)\n"
        f" - {years_to_maturity} years to maturity\n"
        f" - Current semiannual yield = {semiannual_yield*100:.2f}%\n"
        f" - {months_since_coupon} months since last coupon.\n\n"
        f"1) Calculate the clean price.\n"
        f"2) Compute accrued interest.\n"
        f"3) Determine the dirty price.\n"
        f"4) Summarize the final bond price."
    )

    solution = (
        f"Step 1: Clean Price (PV of all future coupon payments + PV of face value)\n"
        f"   Computed ≈ ${clean_price:,.2f}\n\n"
        f"Step 2: Accrued interest = (semiannual coupon) × (months since last coupon / 6)\n"
        f"   = ${accrued_interest:,.2f}\n\n"
        f"Step 3: Dirty Price = Clean Price + Accrued Interest\n"
        f"   = ${dirty_price:,.2f}\n\n"
        f"Step 4: The invoice price is ${dirty_price:,.2f}"
    )

    return question, solution

###############################################################################
# MAIN
###############################################################################

def main():
    """
    Generate 10 instances of each template with different random seeds 
    and write the results to a JSONL file.
    """
    import json

    # Collect our bond-pricing templates
    templates = [
        template_bp_easy1,
        template_bp_easy2,
        template_bp_medium1,
        template_bp_medium2,
        template_bp_hard1
    ]

    all_problems = []
    
    # For each template, generate 10 problem instances
    for template_func in templates:
        # Extract level from docstring's first line (e.g. "Basic", "Intermediate", "Advanced")
        id = template_func.__doc__.split(':')[0].strip()
        level = template_func.__doc__.split(':')[1].strip()

        for i in range(10):
            seed = random.randint(1000000000, 4000000000)
            random.seed(seed)

            question, solution = template_func()

            problem_entry = {
                "seed": seed,
                "id": id,
                "level": level,
                "question": question,
                "solution": solution
            }
            all_problems.append(problem_entry)

            # Reset to a system-based seed after each generation
            random.seed()

    # Shuffle all generated problems
    random.shuffle(all_problems)

    # Write to JSON Lines (.jsonl) format
    output_file = "../../testset/financial_markets/bond_pricing.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")

    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")


if __name__ == "__main__":
    main()
