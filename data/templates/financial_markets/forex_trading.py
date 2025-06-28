import random
import json

###############################################################################
# Data Pools
###############################################################################
investor_names = ["Alice Wu", "Brad Johnson", "Carla Simmons", "Daniel Craig", "Eva Gonzalez"]
currency_pairs = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD",
    "USD/CHF", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY"
]

###############################################################################
# BASIC TEMPLATES (2 Steps)
###############################################################################

def template_fx_easy1():
    """
    1:Basic: Simple Currency Conversion
    Simple currency conversion:
    - From currency A to currency B
    - Given an exchange rate and an amount
    Steps (2):
      1) Identify the correct operation (multiply or divide by the rate)
      2) Perform the calculation to find the converted amount
    """
    investor = random.choice(investor_names)
    pair = random.choice(["EUR/USD", "GBP/USD", "USD/JPY"])  # Keep it simple
    amount = round(random.uniform(500, 5000), 2)

    if pair == "EUR/USD":
        rate = round(random.uniform(1.05, 1.20), 4)
        question = (
            f"{investor} wants to convert €{amount} to US dollars. The current {pair} exchange rate is "
            f"{rate} (meaning 1 EUR = {rate} USD). How many USD will {investor} receive?"
        )
        converted_amount = amount * rate
        solution = (
            f"Step 1: Multiply EUR by the EUR/USD rate to get USD.\n"
            f"  €{amount:.2f} × {rate} = ${converted_amount:.2f}\n\n"
            f"Step 2: Conclude the conversion:\n"
            f"  {investor} will receive ${converted_amount:.2f}."
        )

    elif pair == "GBP/USD":
        rate = round(random.uniform(1.15, 1.40), 4)
        question = (
            f"{investor} wants to convert £{amount} to US dollars. The current {pair} exchange rate is "
            f"{rate} (meaning 1 GBP = {rate} USD). How many USD will {investor} receive?"
        )
        converted_amount = amount * rate
        solution = (
            f"Step 1: Multiply the amount in GBP by the GBP/USD rate.\n"
            f"  £{amount:.2f} × {rate} = ${converted_amount:.2f}\n\n"
            f"Step 2: Conclude the conversion:\n"
            f"  {investor} will receive ${converted_amount:.2f}."
        )

    else:  # pair == "USD/JPY"
        rate = round(random.uniform(100, 150), 2)
        question = (
            f"{investor} wants to convert ${amount} to Japanese yen. The current {pair} exchange rate is "
            f"{rate} (meaning 1 USD = {rate} JPY). How many JPY will {investor} receive?"
        )
        converted_amount = amount * rate
        solution = (
            f"Step 1: Multiply the amount in USD by the USD/JPY rate.\n"
            f"  ${amount:.2f} × {rate} = ¥{converted_amount:.2f}\n\n"
            f"Step 2: Conclude the conversion:\n"
            f"  {investor} will receive ¥{converted_amount:.2f}."
        )

    return question, solution


def template_fx_easy2():
    """
    2:Basic: Pip Value Calculation
    Pip value calculation for a currency pair:
    - Typical for pairs like EUR/USD, GBP/USD (1 pip = 0.0001), or USD/JPY (1 pip = 0.01)
    Steps (2):
      1) Determine pip size and total pip value
      2) Multiply pip value by position size
    """
    investor = random.choice(investor_names)
    pair = random.choice(["EUR/USD", "GBP/USD", "USD/JPY"])
    lot_size = random.choice([10000, 50000, 100000])  # e.g., mini-lots, standard lots

    if pair in ["EUR/USD", "GBP/USD"]:
        pip_size = 0.0001
        question = (
            f"{investor} is trading {pair} with a position size of {lot_size} units. One pip for this pair is 0.0001. "
            f"Calculate how much one pip movement is worth in the quote currency (USD) for {lot_size} units."
        )
        pip_value = pip_size * lot_size
        solution = (
            f"Step 1: Identify that 1 pip = 0.0001 for {pair}.\n"
            f"Step 2: Multiply by the position size:\n"
            f"  Pip Value = 0.0001 × {lot_size} = ${pip_value:.2f} per pip."
        )

    else:  # USD/JPY
        pip_size = 0.01
        question = (
            f"{investor} is trading {pair} with a position size of {lot_size} units. "
            f"For USD/JPY, one pip is typically 0.01. Calculate how much one pip movement "
            f"is worth in JPY for {lot_size} units."
        )
        pip_value = pip_size * lot_size
        solution = (
            f"Step 1: Identify that 1 pip = 0.01 for JPY pairs.\n"
            f"Step 2: Multiply by the position size:\n"
            f"  Pip Value = 0.01 × {lot_size} = ¥{pip_value:.2f} per pip."
        )

    return question, solution

###############################################################################
# INTERMEDIATE TEMPLATES (3 Steps)
###############################################################################

def template_fx_medium1():
    """
    3:Intermediate: Calculating Profit/Loss on a Forex Trade
    Calculate profit/loss from opening a position and closing it later:
    - Pair
    - Opening price
    - Closing price
    - Position size (lots)
    Steps (3):
      1) Find the pip difference
      2) Multiply by position size to get total pips
      3) Convert total pips to money
    """
    investor = random.choice(investor_names)
    pair = random.choice(["EUR/USD", "GBP/USD", "USD/JPY"])
    lot_size = random.choice([10000, 100000])  # e.g. a mini lot or standard lot

    if pair in ["EUR/USD", "GBP/USD"]:
        open_price = round(random.uniform(1.05, 1.20), 4)
        close_price = round(open_price + random.uniform(-0.01, 0.02), 4)
        pip_size = 0.0001
        pip_diff = round((close_price - open_price) / pip_size, 1)
        pip_value = round(lot_size * pip_size, 2)
        profit_loss = pip_diff * pip_value

        question = (
            f"{investor} opened a {pair} position (size: {lot_size} units) at {open_price} and closed it at "
            f"{close_price}. For {pair}, 1 pip = 0.0001. Calculate the profit or loss in USD."
        )

        solution = (
            f"Step 1: Difference in pips = (Close - Open) / 0.0001 = {pip_diff:.1f} pips.\n\n"
            f"Step 2: Pip value = {lot_size} × 0.0001 = ${pip_value:.2f} per pip.\n\n"
            f"Step 3: Total P/L = {pip_diff:.1f} × ${pip_value:.2f} = ${profit_loss:.2f}.\n\n"
            f"Result: {investor} has a {'profit' if profit_loss >= 0 else 'loss'} of ${profit_loss:.2f}."
        )

    else:  # USD/JPY
        open_price = round(random.uniform(105, 120), 2)
        close_price = round(open_price + random.uniform(-1, 2), 2)
        pip_size = 0.01
        pip_diff = round((close_price - open_price) / pip_size, 1)
        pip_value = round(lot_size * pip_size, 2)
        profit_loss = pip_diff * pip_value

        question = (
            f"{investor} opened a {pair} position (size: {lot_size} units) at {open_price} and closed it at "
            f"{close_price}. For {pair}, 1 pip = 0.01. Calculate the profit or loss in JPY."
        )

        solution = (
            f"Step 1: Difference in pips = (Close - Open) / 0.01 = {pip_diff:.1f} pips.\n\n"
            f"Step 2: Pip value = {lot_size} × 0.01 = ¥{pip_value:.2f} per pip.\n\n"
            f"Step 3: Total P/L = {pip_diff:.1f} × ¥{pip_value:.2f} = ¥{profit_loss:.2f}.\n\n"
            f"Result: {investor} has a {'profit' if profit_loss >= 0 else 'loss'} of ¥{profit_loss:.2f}."
        )

    return question, solution


def template_fx_medium2():
    """
    4:Intermediate: Cross Currency Conversion
    Calculate a cross rate:
    - We know e.g. EUR/USD and GBP/USD, we want EUR/GBP
    Steps (3):
      1) Identify the given pair rates
      2) Use cross rate formula
      3) Convert an amount in the new pair
    """
    investor = random.choice(investor_names)
    eur_usd = round(random.uniform(1.05, 1.20), 4)
    gbp_usd = round(random.uniform(1.15, 1.35), 4)
    amount_eur = round(random.uniform(500, 2000), 2)
    cross_eur_gbp = round(eur_usd / gbp_usd, 4)

    question = (
        f"{investor} knows the following exchange rates:\n"
        f"  EUR/USD = {eur_usd}\n"
        f"  GBP/USD = {gbp_usd}\n"
        f"Using these, find the cross rate EUR/GBP. Then convert €{amount_eur:.2f} into GBP using this cross rate."
    )

    converted_gbp = amount_eur * cross_eur_gbp

    solution = (
        f"Step 1: Identify the given rates.\n"
        f"  EUR/USD = {eur_usd}, GBP/USD = {gbp_usd}\n\n"
        f"Step 2: Use cross rate formula: EUR/GBP = (EUR/USD) ÷ (GBP/USD)\n"
        f"          = {eur_usd} ÷ {gbp_usd} = {cross_eur_gbp:.4f}\n\n"
        f"Step 3: Convert €{amount_eur:.2f} to GBP.\n"
        f"  €{amount_eur:.2f} × {cross_eur_gbp:.4f} = £{converted_gbp:.2f}"
    )

    return question, solution

###############################################################################
# ADVANCED TEMPLATE (4 Steps)
###############################################################################

def template_fx_hard1():
    """
    5:Advanced: Triangular Arbitrage
    4-step Triangular Arbitrage Example:
      1) Convert initial currency (e.g. USD) to second currency (e.g. EUR) using rate
      2) Convert second currency (EUR) to third currency (e.g. GBP) using cross rate
      3) Convert third currency (GBP) back to initial currency (USD)
      4) Check if there's a profit or loss from the initial amount
    """
    investor = random.choice(investor_names)
    usd_eur = round(random.uniform(0.80, 0.95), 4)  
    eur_gbp = round(random.uniform(0.80, 0.90), 4)  
    gbp_usd = round(random.uniform(1.25, 1.40), 4)  
    initial_usd = round(random.uniform(1000, 3000), 2)

    eur_received = round(initial_usd * usd_eur, 2)
    gbp_received = round(eur_received * eur_gbp, 2)
    final_usd = round(gbp_received * gbp_usd, 2)
    net_result = final_usd - initial_usd

    question = (
        f"{investor} observes the following exchange rates:\n"
        f"  USD/EUR = {usd_eur}\n"
        f"  EUR/GBP = {eur_gbp}\n"
        f"  GBP/USD = {gbp_usd}\n\n"
        f"Starting with ${initial_usd:.2f}, perform a triangular arbitrage:\n"
        f"1) Convert USD to EUR\n"
        f"2) Convert EUR to GBP\n"
        f"3) Convert GBP back to USD\n"
        f"4) Determine if there's a profit or loss in USD after these conversions."
    )

    solution = (
        f"Step 1: USD → EUR\n"
        f"  €{eur_received:.2f} = ${initial_usd:.2f} × {usd_eur}\n\n"
        f"Step 2: EUR → GBP\n"
        f"  £{gbp_received:.2f} = €{eur_received:.2f} × {eur_gbp}\n\n"
        f"Step 3: GBP → USD\n"
        f"  ${final_usd:.2f} = £{gbp_received:.2f} × {gbp_usd}\n\n"
        f"Step 4: Net result = ${final_usd:.2f} - ${initial_usd:.2f} = ${net_result:.2f}\n"
        f"  If this is positive, we have an arbitrage profit; if negative, a loss."
    )

    return question, solution

###############################################################################
# MAIN FUNCTION
###############################################################################

def main():
    """
    Generate 10 instances of each template with different random seeds 
    and write the results to a JSON Lines file.
    """
    # Collect our FX question templates
    templates = [
        template_fx_easy1,
        template_fx_easy2,
        template_fx_medium1,
        template_fx_medium2,
        template_fx_hard1
    ]

    all_problems = []
    
    # For each template, generate 10 problem instances
    for template_func in templates:
        # Grab the "Basic", "Intermediate", or "Advanced" label from the docstring
        id = template_func.__doc__.split(':')[0].strip()
        level = template_func.__doc__.split(':')[1].strip()

        for _ in range(10):
            seed = random.randint(1000000000, 4000000000)
            random.seed(seed)

            question, solution = template_func()

            problem_entry = {
                "seed": seed,
                "id": id,
                "level": level,  # e.g. "Basic", "Intermediate", or "Advanced"
                "question": question,
                "solution": solution
            }
            all_problems.append(problem_entry)

            # Reset to a system-based seed after each generation
            random.seed()

    # Shuffle all generated problems
    random.shuffle(all_problems)

    # Write to JSON Lines (.jsonl) format
    output_file = "../../testset/financial_markets/forex_trading.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")

    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")


if __name__ == "__main__":
    main()
