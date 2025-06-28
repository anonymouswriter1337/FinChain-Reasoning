import random
import json
import math   # used for a few percentage‑rate calculations

###############################################################################
# Data Pools
###############################################################################
user_names      = ["Alice Wu", "Brad Johnson", "Carla Simmons", "Daniel Craig", "Eva Gonzalez"]
merchant_names  = ["Cafe Aroma", "TechZone", "FashionHub", "BookNook", "GreenGrocer"]
payment_platforms = ["PayWave", "QuickPay", "TapX", "FlexiPay", "ZenoPay"]

###############################################################################
# BASIC TEMPLATES (2 Steps)
###############################################################################

def template_pt_easy1():
    """
    1:Basic: Card Processing Fee for a Single Sale
    Variables:
      • Sale amount (A)
      • Merchant discount rate (m)
    Steps (2):
      1) Fee = A × m
      2) Net amount merchant receives = A − Fee
    """
    merchant = random.choice(merchant_names)
    platform = random.choice(payment_platforms)
    amount   = round(random.uniform(20, 500), 2)
    mdr      = round(random.uniform(0.015, 0.03), 4)   # 1.5 %–3 %

    fee = amount * mdr
    net = amount - fee

    question = (
        f"{merchant} processes a ${amount:.2f} card payment through {platform}. "
        f"The platform charges a merchant discount rate of {mdr*100:.2f}%. "
        f"How much is the fee, and how much does {merchant} receive net of the fee?"
    )

    solution = (
        f"Step 1: Fee = ${amount:.2f} × {mdr:.4f} = ${fee:.2f}\n\n"
        f"Step 2: Net amount = ${amount:.2f} − ${fee:.2f} = ${net:.2f}"
    )

    return question, solution


def template_pt_easy2():
    """
    2:Basic: Buy‑Now‑Pay‑Later Equal Installments
    Variables:
      • Purchase price (P)
      • Number of installments (n)
      • Service fee rate (f) on total purchase
    Steps (2):
      1) Total due = P × (1 + f)
      2) Installment payment = Total due ÷ n
    """
    user      = random.choice(user_names)
    merchant  = random.choice(merchant_names)
    platform  = random.choice(payment_platforms)
    purchase  = round(random.uniform(100, 2000), 2)
    n_inst    = random.choice([4, 6, 12])
    fee_rate  = round(random.uniform(0.00, 0.08), 3)   # 0 %–8 % service fee

    total_due = purchase * (1 + fee_rate)
    installment = total_due / n_inst

    question = (
        f"{user} uses {platform}'s Buy‑Now‑Pay‑Later option at {merchant} to purchase "
        f"items totaling ${purchase:.2f}. The service fee is {fee_rate*100:.2f}% of the purchase price, "
        f"spread evenly over {n_inst} installments. What is each installment amount?"
    )

    solution = (
        f"Step 1: Total amount owed = ${purchase:.2f} × (1 + {fee_rate:.3f}) = ${total_due:.2f}\n\n"
        f"Step 2: Installment = ${total_due:.2f} ÷ {n_inst} = ${installment:.2f}"
    )

    return question, solution

###############################################################################
# INTERMEDIATE TEMPLATES (3 Steps)
###############################################################################

def template_pt_medium1():
    """
    3:Intermediate: Total Fees on Multiple Small Transactions
    Scenario:
      • Platform charges fixed fee (c) per transaction + percentage fee (p)
      • Several transaction amounts provided
    Steps (3):
      1) Compute fee for each transaction
      2) Sum all fees and total sales
      3) Calculate effective fee rate = Total fees ÷ Total sales
    """
    merchant = random.choice(merchant_names)
    platform = random.choice(payment_platforms)
    per_tx_fee = 0.30                                 # flat $0.30
    pct_fee    = 0.025                                # 2.5 %
    num_tx = random.randint(4, 7)
    tx_amounts = [round(random.uniform(5, 120), 2) for _ in range(num_tx)]

    fees = [per_tx_fee + amt * pct_fee for amt in tx_amounts]
    total_fees  = sum(fees)
    total_sales = sum(tx_amounts)
    eff_rate = total_fees / total_sales

    tx_lines = "\n".join([f"  • ${amt:.2f}" for amt in tx_amounts])
    question = (
        f"{merchant} processes these transactions via {platform} (flat fee $0.30 + 2.5% of amount):\n"
        f"{tx_lines}\n\n"
        f"1) What are the fees for each transaction?\n"
        f"2) What is the total fee paid?\n"
        f"3) What is the effective fee rate on the batch?"
    )

    fee_lines = "\n".join([f"  • Fee on ${amt:.2f} = $0.30 + 2.5% × ${amt:.2f} = ${fee:.2f}"
                           for amt, fee in zip(tx_amounts, fees)])
    solution = (
        f"Step 1:\n{fee_lines}\n\n"
        f"Step 2: Total fees = ${total_fees:.2f}; Total sales = ${total_sales:.2f}\n\n"
        f"Step 3: Effective rate = ${total_fees:.2f} ÷ ${total_sales:.2f} = {eff_rate*100:.2f}%"
    )

    return question, solution


def template_pt_medium2():
    """
    4:Intermediate: Cross‑Border Payment Conversion and Fees
    Variables:
      • Sale amount in EUR (E)
      • Mid‑market EUR/USD rate (R)
      • Platform spread (s)
      • Fixed processing fee (f)
    Steps (3):
      1) Platform conversion rate = R × (1 − s)
      2) Convert amount to USD
      3) Subtract fixed fee to get net USD received
    """
    merchant = random.choice(merchant_names)
    platform = random.choice(payment_platforms)
    amount_eur = round(random.uniform(50, 1000), 2)
    mid_rate   = round(random.uniform(1.10, 1.20), 4)
    spread     = round(random.uniform(0.015, 0.03), 4)   # 1.5 %–3 %
    fixed_fee  = round(random.uniform(0, 5), 2)

    conv_rate = mid_rate * (1 - spread)
    usd_before_fee = amount_eur * conv_rate
    net_usd = usd_before_fee - fixed_fee

    question = (
        f"{merchant} receives a €{amount_eur:.2f} payment through {platform}. "
        f"The mid‑market EUR/USD rate is {mid_rate}. {platform} applies a "
        f"{spread*100:.2f}% spread and charges a ${fixed_fee:.2f} fixed fee. "
        f"How many USD will {merchant} receive after conversion and fees?"
    )

    solution = (
        f"Step 1: Conversion rate = {mid_rate} × (1 − {spread:.4f}) = {conv_rate:.4f}\n\n"
        f"Step 2: USD before fee = €{amount_eur:.2f} × {conv_rate:.4f} = ${usd_before_fee:.2f}\n\n"
        f"Step 3: Net USD = ${usd_before_fee:.2f} − ${fixed_fee:.2f} = ${net_usd:.2f}"
    )

    return question, solution

###############################################################################
# ADVANCED TEMPLATE (4 Steps)
###############################################################################

def template_pt_hard1():
    """
    5:Advanced: Tiered Interchange + Subscription Model
    Scenario:
      • Platform charges monthly subscription (S)
      • Variable percentage fee: 1.7% on first $10,000 sales, then 1.4% on excess
      • Flat per‑transaction fee (c)
      • Merchant has N transactions, each averaging A dollars
    Steps (4):
      1) Compute total sales and split into tiers
      2) Calculate variable percentage fees for each tier
      3) Add flat per‑transaction fees and subscription
      4) Derive effective blended fee rate on total sales
    """
    merchant = random.choice(merchant_names)
    platform = random.choice(payment_platforms)
    n_tx   = random.randint(300, 800)
    avg_tx = round(random.uniform(20, 100), 2)
    sub_fee = 50.00
    flat_fee = 0.08
    tier_cap = 10_000
    tier1_rate = 0.017
    tier2_rate = 0.014

    total_sales = n_tx * avg_tx
    tier1_sales = min(total_sales, tier_cap)
    tier2_sales = max(0, total_sales - tier_cap)

    var_fee = tier1_sales * tier1_rate + tier2_sales * tier2_rate
    flat_total = n_tx * flat_fee
    total_fees = sub_fee + var_fee + flat_total
    eff_rate = total_fees / total_sales

    question = (
        f"{merchant} uses {platform}'s tiered pricing plan:\n"
        f"  • Monthly subscription: ${sub_fee:.2f}\n"
        f"  • 1.7% on first $10,000 in monthly sales, 1.4% thereafter\n"
        f"  • $0.08 per transaction\n\n"
        f"This month they processed {n_tx} transactions averaging ${avg_tx:.2f} each.\n"
        f"1) What are total monthly sales?\n"
        f"2) How much variable percentage fee is owed?\n"
        f"3) What are total fees including flat and subscription charges?\n"
        f"4) What is the effective fee rate for the month?"
    )

    solution = (
        f"Step 1: Total sales = {n_tx} × ${avg_tx:.2f} = ${total_sales:,.2f}\n"
        f"        Tier split → First ${tier_cap:,}: ${tier1_sales:,.2f}; Excess: ${tier2_sales:,.2f}\n\n"
        f"Step 2: Variable fees = 1.7% × ${tier1_sales:,.2f} + 1.4% × ${tier2_sales:,.2f}\n"
        f"        = ${tier1_sales * tier1_rate:,.2f} + ${tier2_sales * tier2_rate:,.2f} "
        f"= ${var_fee:,.2f}\n\n"
        f"Step 3: Flat fees = {n_tx} × $0.08 = ${flat_total:,.2f}\n"
        f"        Total fees = Subscription ${sub_fee:.2f} + Variable ${var_fee:,.2f} + Flat ${flat_total:,.2f} "
        f"= ${total_fees:,.2f}\n\n"
        f"Step 4: Effective fee rate = ${total_fees:,.2f} ÷ ${total_sales:,.2f} "
        f"= {eff_rate*100:.2f}%"
    )

    return question, solution

###############################################################################
# MAIN
###############################################################################

def main():
    """
    Generate 10 instances of each Payment‑Technologies template and save to JSONL.
    """
    templates = [
        template_pt_easy1,
        template_pt_easy2,
        template_pt_medium1,
        template_pt_medium2,
        template_pt_hard1
    ]

    problems = []
    for template_func in templates:
        id = template_func.__doc__.split(':')[0].strip()
        level = template_func.__doc__.split(':')[1].strip()
        for _ in range(10):
            seed = random.randint(1_000_000_000, 4_000_000_000)
            random.seed(seed)
            q, a = template_func()
            problems.append({
                "seed": seed,
                "id": id,
                "level": level,
                "question": q,
                "solution": a
            })
            random.seed()  # reset

    random.shuffle(problems)
    outfile = "../../testset/fintech/payment_technologies.jsonl"
    with open(outfile, "w") as f:
        for p in problems:
            f.write(json.dumps(p))
            f.write("\n")

    print(f"Successfully generated {len(problems)} problems and saved to {outfile}")


if __name__ == "__main__":
    main()
