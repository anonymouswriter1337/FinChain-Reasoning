import random
import json
import math          # needed for the amortisation math

###############################################################################
# Data Pools
###############################################################################
user_names = ["Alice Wu", "Brad Johnson", "Carla Simmons", "Daniel Craig", "Eva Gonzalez"]
digital_banks = ["NeoBank", "FinEdge", "CloudBank", "BrightPay", "MobileMoney"]

###############################################################################
# BASIC TEMPLATES (2 Steps)
###############################################################################

def template_db_easy1():
    """1:Basic: Simple-Interest Savings Balance
    Variables:
      • Principal (P)
      • Annual simple interest rate (r)
      • Time in years (t)
    Steps (2):
      1) Interest = P × r × t
      2) Ending balance = P + interest
    """
    user = random.choice(user_names)
    bank = random.choice(digital_banks)
    principal = random.randint(500, 5000)
    rate = round(random.uniform(0.01, 0.05), 3)
    years = random.randint(1, 3)

    interest = principal * rate * years
    ending_balance = principal + interest

    question = (
        f"{user} deposits ${principal} into a simple‑interest savings account at {bank}. "
        f"The account pays {rate*100:.2f}% per year. "
        f"How much will be in the account after {years} year(s)?"
    )

    solution = (
        f"Step 1: Interest = P × r × t = ${principal} × {rate:.3f} × {years} = ${interest:.2f}\n\n"
        f"Step 2: Ending balance = ${principal} + ${interest:.2f} = ${ending_balance:.2f}"
    )

    return question, solution


def template_db_easy2():
    """2:Basic: Mobile Transfer Fee
    Variables:
      • Transfer amount (A)
      • Fee rate (f)
    Steps (2):
      1) Fee = A × f
      2) Total deduction = A + fee
    """
    user = random.choice(user_names)
    bank = random.choice(digital_banks)
    amount = round(random.uniform(50, 500), 2)
    fee_rate = round(random.uniform(0.005, 0.02), 3)

    fee = amount * fee_rate
    total = amount + fee

    question = (
        f"{user} uses {bank}'s app to transfer ${amount:.2f}. "
        f"The platform charges a {fee_rate*100:.2f}% fee. "
        f"What is the fee and total deduction?"
    )

    solution = (
        f"Step 1: Fee = ${amount:.2f} × {fee_rate:.3f} = ${fee:.2f}\n\n"
        f"Step 2: Total deduction = ${amount:.2f} + ${fee:.2f} = ${total:.2f}"
    )

    return question, solution

###############################################################################
# INTERMEDIATE TEMPLATES (3 Steps)
###############################################################################

def template_db_medium1():
    """3:Intermediate: Monthly Compound Interest with Recurring Deposits
    Variables:
      • Monthly deposit (D)
      • Annual nominal rate (r)
      • Number of months (n)
    Steps (3):
      1) Monthly rate = r / 12
      2) Future value of annuity
      3) Report ending balance
    """
    user = random.choice(user_names)
    bank = random.choice(digital_banks)
    monthly_deposit = random.randint(50, 300)
    annual_rate = round(random.uniform(0.02, 0.06), 3)
    months = random.choice([12, 24, 36])

    monthly_rate = annual_rate / 12
    fv = monthly_deposit * (((1 + monthly_rate) ** months - 1) / monthly_rate)

    question = (
        f"{user} sets up a ${monthly_deposit} monthly deposit at {bank}, "
        f"earning {annual_rate*100:.2f}% APR compounded monthly. "
        f"What will the balance be after {months} months?"
    )

    solution = (
        f"Step 1: Monthly rate = {annual_rate:.3f} / 12 = {monthly_rate:.5f}\n\n"
        f"Step 2: FV = D × [((1 + r)^n − 1) / r]\n"
        f"        = ${monthly_deposit} × [((1 + {monthly_rate:.5f})^{months} − 1) / {monthly_rate:.5f}]\n\n"
        f"Step 3: Balance ≈ ${fv:,.2f}"
    )

    return question, solution


def template_db_medium2():
    """4:Intermediate: Digital Wallet Spending Analysis
    Steps (3):
      1) Sum per category
      2) Total spend
      3) Identify top category
    """
    user = random.choice(user_names)
    categories = ["Food Delivery", "Streaming", "Ride‑Hailing", "Online Shopping", "Subscriptions"]
    num_tx = random.randint(4, 7)
    tx = [(random.choice(categories), round(random.uniform(5, 120), 2)) for _ in range(num_tx)]

    spend = {}
    for c, a in tx:
        spend[c] = spend.get(c, 0) + a
    total_spend = sum(spend.values())
    top_cat = max(spend, key=spend.get)

    tx_lines = "\n".join([f"  • {c}: ${a:.2f}" for c, a in tx])
    question = (
        f"{user}'s wallet shows:\n{tx_lines}\n\n"
        f"1) Total spent?\n2) Highest‑spend category?"
    )

    spend_lines = "\n".join([f"  {c}: ${spend[c]:.2f}" for c in spend])
    solution = (
        f"Step 1: Category totals:\n{spend_lines}\n\n"
        f"Step 2: Total spend = ${total_spend:.2f}\n\n"
        f"Step 3: Highest category = {top_cat} (${spend[top_cat]:.2f})"
    )

    return question, solution

###############################################################################
# ADVANCED TEMPLATE (4 Steps)
###############################################################################

def template_db_hard1():
    """5:Advanced: Personal Loan with Extra Payment
    Steps (4):
      1) Monthly payment via amortisation formula
      2) Amortise to extra‑payment month and apply lump sum
      3) Re‑solve for remaining months
      4) Summarise new payoff timeline
    """
    user = random.choice(user_names)
    bank = random.choice(digital_banks)
    principal = random.randint(5000, 15000)
    annual_rate = round(random.uniform(0.04, 0.09), 3)
    years = random.choice([2, 3, 4])
    months_total = years * 12
    extra_month = random.randint(6, months_total // 2)
    extra_payment = random.randint(500, 2000)

    monthly_rate = annual_rate / 12
    payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** (-months_total))

    balance = principal
    for _ in range(extra_month):
        interest = balance * monthly_rate
        principal_paid = payment - interest
        balance -= principal_paid

    balance -= extra_payment

    if balance > 0:
        new_months = math.ceil(
            math.log(payment / (payment - balance * monthly_rate), 1 + monthly_rate)
        )
    else:
        new_months = 0  # paid off immediately

    question = (
        f"{user} borrows ${principal} from {bank} at {annual_rate*100:.2f}% APR for {years} years "
        f"(equal monthly payments). After {extra_month} months an extra payment of ${extra_payment} is made.\n"
        f"1) What is the original monthly payment?\n"
        f"2) What balance remains right after the extra payment?\n"
        f"3) How many additional months will it take to repay the loan with the same monthly payment?"
    )

    solution = (
        f"Step 1: Monthly payment = P·i / (1 − (1+i)^−n) "
        f"= ${payment:,.2f}\n\n"
        f"Step 2: Balance after {extra_month} months and extra payment ≈ ${balance:,.2f}\n\n"
        f"Step 3: Remaining months ≈ {new_months}\n\n"
        f"Step 4: New total payoff time = {extra_month + new_months} months."
    )

    return question, solution

###############################################################################
# MAIN
###############################################################################

def main():
    templates = [
        template_db_easy1,
        template_db_easy2,
        template_db_medium1,
        template_db_medium2,
        template_db_hard1,
    ]

    all_problems = []
    for template_func in templates:
        id = template_func.__doc__.split(':')[0].strip()
        level = template_func.__doc__.split(':')[1].strip()

        for _ in range(10):
            seed = random.randint(1_000_000_000, 4_000_000_000)
            random.seed(seed)
            question, solution = template_func()
            all_problems.append({
                "seed": seed,
                "id": id,
                "level": level,
                "question": question,
                "solution": solution
            })
            random.seed()

    random.shuffle(all_problems)
    outfile = "../../testset/fintech/digital_banking.jsonl"
    with open(outfile, "w") as f:
        for p in all_problems:
            f.write(json.dumps(p))
            f.write("\n")

    print(f"Created {len(all_problems)} problems → {outfile}")


if __name__ == "__main__":
    main()
