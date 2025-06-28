import random
import json
import math   # used in some calculations

###############################################################################
# Data Pools
###############################################################################
investor_names = ["Alice Wu", "Brad Johnson", "Carla Simmons", "Daniel Craig", "Eva Gonzalez"]
robo_advisors = ["BetterWealth", "SmartInvest", "RoboMax", "WealthBot", "AlgoAdvisor"]

###############################################################################
# BASIC TEMPLATES (2 Steps)
###############################################################################

def template_ra_easy1():
    """
    1:Basic: Advisory Fee on Portfolio Value
    Variables:
      • Portfolio value (V)
      • Annual advisory fee rate (f)
    Steps (2):
      1) Fee = V × f
      2) Net portfolio value after fee = V − Fee
    """
    investor = random.choice(investor_names)
    advisor  = random.choice(robo_advisors)
    value = random.randint(5_000, 50_000)
    fee_rate = round(random.uniform(0.0025, 0.01), 4)   # 0.25 %–1 %

    fee = value * fee_rate
    net_value = value - fee

    question = (
        f"{investor} has ${value:,} managed by {advisor}, which charges an annual advisory fee "
        f"of {fee_rate*100:.2f}%. What is the fee for the year, and what is the net value after the fee?"
    )

    solution = (
        f"Step 1: Fee = ${value:,} × {fee_rate:.4f} = ${fee:,.2f}\n\n"
        f"Step 2: Net value = ${value:,} − ${fee:,.2f} = ${net_value:,.2f}"
    )

    return question, solution


def template_ra_easy2():
    """
    2:Basic: One‑Year Return Net of Advisory Fee
    Variables:
      • Portfolio value (V)
      • Gross return (g)
      • Advisory fee rate (f)
    Steps (2):
      1) Growth: V × (1 + g)
      2) Subtract advisory fee = ending value × f
    """
    investor = random.choice(investor_names)
    advisor  = random.choice(robo_advisors)
    value = random.randint(10_000, 100_000)
    gross_return = round(random.uniform(-0.05, 0.12), 3)   # −5 % to 12 %
    fee_rate = round(random.uniform(0.003, 0.01), 4)

    end_value_before_fee = value * (1 + gross_return)
    fee = end_value_before_fee * fee_rate
    end_value_after_fee = end_value_before_fee - fee

    question = (
        f"{investor}'s portfolio at {advisor} is worth ${value:,}. Over the next year it earns a "
        f"gross return of {gross_return*100:.2f}%. {advisor} charges {fee_rate*100:.2f}% of assets as its fee. "
        f"What will the portfolio be worth after fees?"
    )

    solution = (
        f"Step 1: Ending value before fee = ${value:,} × (1 + {gross_return:.3f}) "
        f"= ${end_value_before_fee:,.2f}\n\n"
        f"Step 2: Fee = {fee_rate*100:.2f}% of ${end_value_before_fee:,.2f} = ${fee:,.2f}\n"
        f"        Ending value after fee = ${end_value_before_fee:,.2f} − ${fee:,.2f} "
        f"= ${end_value_after_fee:,.2f}"
    )

    return question, solution

###############################################################################
# INTERMEDIATE TEMPLATES (3 Steps)
###############################################################################

def template_ra_medium1():
    """
    3:Intermediate: Rebalancing to Target Allocation
    Scenario:
      • Two‑asset portfolio (Equities / Bonds)
      • Current values given
      • Target allocation given
    Steps (3):
      1) Compute total portfolio value
      2) Calculate target dollar amounts
      3) Determine buy/sell amounts to rebalance
    """
    investor = random.choice(investor_names)
    advisor  = random.choice(robo_advisors)
    equities_val = random.randint(20_000, 60_000)
    bonds_val    = random.randint(10_000, 40_000)
    target_equity_pct = random.choice([0.6, 0.7, 0.8])   # 60 %, 70 %, 80 %

    total_val = equities_val + bonds_val
    target_equity_val = total_val * target_equity_pct
    target_bond_val   = total_val - target_equity_val

    rebalance_equity = target_equity_val - equities_val   # positive → buy, negative → sell
    rebalance_bonds  = target_bond_val - bonds_val

    question = (
        f"{investor}'s portfolio managed by {advisor} currently holds:\n"
        f"  • Equities: ${equities_val:,}\n"
        f"  • Bonds:    ${bonds_val:,}\n"
        f"The target allocation is {target_equity_pct*100:.0f}% equities and "
        f"{(1-target_equity_pct)*100:.0f}% bonds.\n"
        f"How much of each asset should be bought or sold to rebalance exactly to the target?"
    )

    solution = (
        f"Step 1: Total value = ${equities_val:,} + ${bonds_val:,} = ${total_val:,}\n\n"
        f"Step 2: Targets → Equities = {target_equity_pct*100:.0f}% × ${total_val:,} "
        f"= ${target_equity_val:,.2f}; Bonds = ${target_bond_val:,.2f}\n\n"
        f"Step 3: Rebalance amounts:\n"
        f"  • Equities: ${target_equity_val:,.2f} − ${equities_val:,} = "
        f"{'Buy' if rebalance_equity>0 else 'Sell'} ${abs(rebalance_equity):,.2f}\n"
        f"  • Bonds:    ${target_bond_val:,.2f} − ${bonds_val:,} = "
        f"{'Buy' if rebalance_bonds>0 else 'Sell'} ${abs(rebalance_bonds):,.2f}"
    )

    return question, solution


def template_ra_medium2():
    """
    4:Intermediate: Projected Portfolio Value with Monthly Contributions
    Variables:
      • Monthly contribution (C)
      • Annual expected return (r)
      • Number of years (t)
      • Advisory fee (f) charged annually on ending balance
    Steps (3):
      1) Compute future value of an ordinary annuity with monthly compounding
      2) Apply advisory fee for each year (approximate: one fee at end)
      3) Report net projected value
    """
    investor = random.choice(investor_names)
    advisor  = random.choice(robo_advisors)
    monthly_contrib = random.randint(200, 600)
    years = random.choice([5, 10, 15])
    annual_return = round(random.uniform(0.04, 0.08), 3)
    fee_rate = round(random.uniform(0.0025, 0.0075), 4)  # 0.25 %–0.75 %

    months = years * 12
    monthly_rate = annual_return / 12
    fv_before_fee = monthly_contrib * (((1 + monthly_rate) ** months - 1) / monthly_rate)
    net_value = fv_before_fee * (1 - fee_rate)  # one fee at the end for simplicity

    question = (
        f"{investor} plans to contribute ${monthly_contrib} each month into a portfolio at {advisor} for "
        f"{years} years. The expected annual return is {annual_return*100:.2f}% (compounded monthly). "
        f"{advisor} charges an annual advisory fee of {fee_rate*100:.2f}% of assets. "
        f"What is the projected portfolio value after fees at the end of {years} years?"
    )

    solution = (
        f"Step 1: Future value before fees (ordinary annuity, monthly compounding):\n"
        f"        FV = C × [((1 + r)^n − 1) / r]\n"
        f"        = ${monthly_contrib} × [((1 + {monthly_rate:.5f})^{months} − 1) / {monthly_rate:.5f}] "
        f"= ${fv_before_fee:,.2f}\n\n"
        f"Step 2: Advisory fee at end = {fee_rate*100:.2f}% × ${fv_before_fee:,.2f} "
        f"= ${fv_before_fee*fee_rate:,.2f}\n\n"
        f"Step 3: Net projected value = ${fv_before_fee:,.2f} − fee "
        f"= ${net_value:,.2f}"
    )

    return question, solution

###############################################################################
# ADVANCED TEMPLATE (4 Steps)
###############################################################################

def template_ra_hard1():
    """
    5:Advanced: Tax‑Loss Harvesting Impact
    Scenario:
      • Investor holds an ETF purchased at cost basis B
      • Current market value lower than basis
      • Sells ETF, realises capital loss, immediately buys similar ETF
      • Uses loss to offset ordinary income at marginal tax rate (m)
    Steps (4):
      1) Calculate realised loss
      2) Compute tax savings = loss × m
      3) Determine new cost basis (purchase price of replacement)
      4) Summarise net economic benefit (tax savings) and unchanged market exposure
    """
    investor = random.choice(investor_names)
    advisor  = random.choice(robo_advisors)
    shares = random.randint(50, 200)
    cost_basis = round(random.uniform(50, 120), 2)        # purchase price
    current_price = round(cost_basis * random.uniform(0.6, 0.95), 2)  # lower price
    marginal_rate = random.choice([0.22, 0.24, 0.32])     # 22 %, 24 %, 32 %

    realised_loss = (current_price - cost_basis) * shares
    tax_savings = -realised_loss * marginal_rate          # loss is negative
    new_basis = current_price                             # assume repurchase at market

    question = (
        f"{investor}'s robo‑advisor {advisor} recommends tax‑loss harvesting. "
        f"The investor owns {shares} shares of ETF X bought at ${cost_basis:.2f} per share. "
        f"The market price has fallen to ${current_price:.2f}. "
        f"{investor}'s marginal income‑tax rate is {marginal_rate*100:.0f}%. "
        f"1) What capital loss will be realised if the ETF is sold?\n"
        f"2) How much tax can be saved this year?\n"
        f"3) What is the cost basis of the replacement ETF after repurchase?\n"
        f"4) Briefly explain the net benefit of this strategy."
    )

    solution = (
        f"Step 1: Realised loss = ({current_price:.2f} − {cost_basis:.2f}) × {shares} "
        f"= ${realised_loss:,.2f}\n\n"
        f"Step 2: Tax savings = −Loss × marginal rate "
        f"= ${-realised_loss:,.2f} × {marginal_rate:.2f} = ${tax_savings:,.2f}\n\n"
        f"Step 3: New cost basis equals repurchase price = ${new_basis:.2f} per share\n\n"
        f"Step 4: The investor keeps market exposure (switches to a similar ETF) "
        f"while pocketing ${tax_savings:,.2f} in tax savings, improving after‑tax returns."
    )

    return question, solution

###############################################################################
# MAIN
###############################################################################

def main():
    """
    Generate 10 instances of each Robo‑Advisor template and save to JSONL.
    """
    templates = [
        template_ra_easy1,
        template_ra_easy2,
        template_ra_medium1,
        template_ra_medium2,
        template_ra_hard1
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
                "level": level,
                "question": q,
                "solution": a
            })
            random.seed()  # reset to system seed

    random.shuffle(problems)

    outfile = "../../testset/fintech/robo_advisors.jsonl"
    with open(outfile, "w") as f:
        for p in problems:
            f.write(json.dumps(p))
            f.write("\n")

    print(f"Successfully generated {len(problems)} problems and saved to {outfile}")


if __name__ == "__main__":
    main()
