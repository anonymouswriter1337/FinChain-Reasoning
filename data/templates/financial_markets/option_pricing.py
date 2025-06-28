import random
import math
import json

###############################################################################
# Data Pools
###############################################################################
investor_names = ["Alice Wu", "Brad Johnson", "Carla Simmons", "Daniel Craig", "Eva Gonzalez"]
underlying_assets = [
    "Apple Inc.", "Tesla Inc.", "Amazon.com Inc.", "Microsoft Corp.", "Netflix Inc.",
    "Google LLC", "Meta Platforms", "Nvidia Corp.", "Disney Co.", "Coca-Cola Co."
]

###############################################################################
# BASIC TEMPLATES (2 Steps)
###############################################################################

def template_op_easy1():
    """
    1:Basic: Intrinsic Value of a Call Option
    Intrinsic Value of a Call Option:
    - Stock Price (S)
    - Strike Price (K)
    2 steps:
      1) Compare S vs. K
      2) Intrinsic Value = max(S - K, 0)
    """
    investor = random.choice(investor_names)
    asset = random.choice(underlying_assets)
    S = round(random.uniform(80, 150), 2)  # Current stock price
    K = round(random.uniform(70, 160), 2)  # Strike price

    question = (
        f"{investor} holds a call option on {asset} with a strike price of ${K:.2f}. "
        f"The current stock price is ${S:.2f}. Calculate the intrinsic value of the call option."
    )

    # Step 1 & 2: Intrinsic Value = max(S - K, 0)
    in_the_money = (S > K)
    intrinsic_value = max(S - K, 0)

    solution = (
        f"Step 1: Compare stock price (S) to strike (K):\n"
        f"  S = ${S:.2f}, K = ${K:.2f} -> "
        f"{'in the money' if in_the_money else 'out of the money'}.\n\n"
        f"Step 2: Intrinsic Value = max(S - K, 0) = max(${S:.2f} - ${K:.2f}, 0) = ${intrinsic_value:.2f}."
    )

    return question, solution


def template_op_easy2():
    """
    2:Basic: Intrinsic Value of a Put Option
    Intrinsic Value of a Put Option:
    - Stock Price (S)
    - Strike Price (K)
    2 steps:
      1) Compare S vs. K
      2) Intrinsic Value = max(K - S, 0)
    """
    investor = random.choice(investor_names)
    asset = random.choice(underlying_assets)
    S = round(random.uniform(80, 150), 2)  # Current stock price
    K = round(random.uniform(70, 160), 2)  # Strike price

    question = (
        f"{investor} holds a put option on {asset} with a strike price of ${K:.2f}. "
        f"The current stock price is ${S:.2f}. Calculate the intrinsic value of the put option."
    )

    # Step 1 & 2: Intrinsic Value = max(K - S, 0)
    in_the_money = (S < K)
    intrinsic_value = max(K - S, 0)

    solution = (
        f"Step 1: Compare stock price (S) to strike (K):\n"
        f"  S = ${S:.2f}, K = ${K:.2f} -> "
        f"{'in the money' if in_the_money else 'out of the money'}.\n\n"
        f"Step 2: Intrinsic Value = max(K - S, 0) = max(${K:.2f} - ${S:.2f}, 0) = ${intrinsic_value:.2f}."
    )

    return question, solution


###############################################################################
# INTERMEDIATE TEMPLATES (3 Steps)
###############################################################################

def template_op_medium1():
    """
    3:Intermediate: One-Period Binomial Model (Call)
    1-period binomial model for a call option:
    - Current stock price (S0)
    - Up factor (u) -> S_u
    - Down factor (d) -> S_d
    - Strike price (K)
    - Risk-free rate (r) for 1 period
    3 steps:
      1) Compute payoff in up/down states
      2) Compute risk-neutral probabilities
      3) Discount expected payoff to get option price
    """
    investor = random.choice(investor_names)
    asset = random.choice(underlying_assets)

    S0 = round(random.uniform(90, 120), 2)    # Current stock price
    K = round(random.uniform(80, 130), 2)     # Strike
    u = round(random.uniform(1.1, 1.3), 2)    # up factor
    d = round(random.uniform(0.7, 0.9), 2)    # down factor
    r = round(random.uniform(0.02, 0.08), 3)  # risk-free rate (2% to 8%)

    S_u = round(S0 * u, 2)
    S_d = round(S0 * d, 2)
    payoff_u = round(max(S_u - K, 0), 2)
    payoff_d = round(max(S_d - K, 0), 2)

    # risk-neutral probability p = ((1+r) - d) / (u - d)
    p = round(((1 + r) - d) / (u - d), 3)

    expected_payoff = round(p * payoff_u + (1 - p) * payoff_d, 2)
    option_price = round(expected_payoff / (1 + r), 2)

    question = (
        f"{investor} uses a 1-period binomial model to price a call on {asset}. "
        f"S0 = ${S0:.2f}, K = ${K:.2f}, u = {u}, d = {d}, r = {r*100:.2f}%. "
        f"Find the fair price of this call."
    )

    solution = (
        f"Step 1: Payoffs up/down:\n"
        f"  S_u = ${S_u:.2f}, payoff_u = max(S_u - K, 0) = ${payoff_u:.2f}\n"
        f"  S_d = ${S_d:.2f}, payoff_d = max(S_d - K, 0) = ${payoff_d:.2f}\n\n"
        f"Step 2: Risk-neutral probability:\n"
        f"  p = [ (1+r) - d ] / (u - d ) = {p:.3f}\n\n"
        f"Step 3: Discount expected payoff:\n"
        f"  E[Payoff] = p × {payoff_u:.2f} + (1-p) × {payoff_d:.2f} = ${expected_payoff:.2f}\n"
        f"  Option Price = E[Payoff]/(1 + r) = ${option_price:.2f}"
    )

    return question, solution


def template_op_medium2():
    """
    4:Intermediate: Profit/Loss from Buying a Call
    Calculate net profit/loss from buying a call:
    - Call premium (C)
    - Strike (K)
    - Stock price at expiration (S_T)
    3 steps:
      1) Compute intrinsic payoff
      2) Subtract premium
      3) Summarize net profit or loss
    """
    investor = random.choice(investor_names)
    asset = random.choice(underlying_assets)

    K = round(random.uniform(90, 110), 2)    # strike
    C = round(random.uniform(2, 10), 2)      # call premium
    S_T = round(random.uniform(80, 130), 2)  # stock price at expiration

    payoff = round(max(S_T - K, 0), 2)
    net_profit = round(payoff - C, 2)

    question = (
        f"{investor} bought a call on {asset} with strike ${K:.2f}, paying a premium of ${C:.2f}. "
        f"At expiration, the stock price is ${S_T:.2f}. What is the net profit or loss?"
    )

    solution = (
        f"Step 1: Intrinsic Payoff = max(S_T - K, 0) = ${payoff:.2f}\n\n"
        f"Step 2: Subtract premium:\n"
        f"  Net Profit = ${payoff:.2f} - ${C:.2f} = ${net_profit:.2f}\n\n"
        f"Step 3: {'Profit' if net_profit >= 0 else 'Loss'} of ${net_profit:.2f}."
    )

    return question, solution


###############################################################################
# ADVANCED TEMPLATE (4 Steps)
###############################################################################

def template_op_hard1():
    """
    5:Advanced: Black-Scholes Call Option Pricing
    Use the Black-Scholes formula for a call option:
      C = S0 * N(d1) - K * e^(-rT) * N(d2)
    where:
      d1 = [ln(S0/K) + (r + σ^2/2)*T ] / (σ * sqrt(T))
      d2 = d1 - σ*sqrt(T)
    4 steps:
      1) Compute d1
      2) Compute d2
      3) Compute N(d1) and N(d2)
      4) Calculate call price
    We'll do a rough numeric approach for demonstration.
    """
    investor = random.choice(investor_names)
    asset = random.choice(underlying_assets)

    S0 = round(random.uniform(90, 150), 2)
    K = round(random.uniform(80, 140), 2)
    r = round(random.uniform(0.01, 0.05), 3)   # risk-free rate
    sigma = round(random.uniform(0.1, 0.4), 3) # volatility
    T = round(random.uniform(0.5, 2), 2)       # time in years

    # Calculate d1
    d1_num = math.log(S0 / K) + (r + 0.5 * sigma**2) * T
    d1_den = sigma * math.sqrt(T)
    d1 = round(d1_num / d1_den, 4)

    # Calculate d2
    d2 = round(d1 - sigma * math.sqrt(T), 4)

    # Simple standard normal CDF approximation
    def cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    Nd1 = round(cdf(d1), 4)
    Nd2 = round(cdf(d2), 4)

    # Black-Scholes call price
    call_price = S0 * Nd1 - K * math.exp(-r * T) * Nd2

    question = (
        f"{investor} wants to price a call on {asset} using Black-Scholes.\n"
        f"S0=${S0:.2f}, K=${K:.2f}, r={r:.3f}, σ={sigma:.3f}, T={T:.2f}.\n"
        f"Estimate the call price."
    )

    solution = (
        f"Step 1: d1 = (ln(S0/K) + (r + σ²/2)*T) / (σ√T) ≈ {d1}\n"
        f"Step 2: d2 = d1 - σ√T ≈ {d2}\n"
        f"Step 3: N(d1) ≈ {Nd1}, N(d2) ≈ {Nd2}\n"
        f"Step 4: Call = S0×N(d1) - K×e^(-rT)×N(d2) ≈ ${call_price:.2f}"
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
    # 1) Gather all option-pricing templates in a list
    templates = [
        template_op_easy1,
        template_op_easy2,
        template_op_medium1,
        template_op_medium2,
        template_op_hard1
    ]

    all_problems = []

    # 2) For each template, generate 10 question/solution pairs
    for template_func in templates:
        # Extract the first line in the docstring up to ':' => e.g. "Basic", "Intermediate", "Advanced"
        id = template_func.__doc__.split(':')[0].strip()
        level = template_func.__doc__.split(':')[1].strip()

        for _ in range(10):
            # Create a unique seed for each problem
            seed = random.randint(1000000000, 4000000000)
            random.seed(seed)

            # Generate question & solution
            question, solution = template_func()

            # Store the problem
            problem_entry = {
                "seed": seed,
                "id": id,
                "level": level,
                "question": question,
                "solution": solution
            }
            all_problems.append(problem_entry)

            # Reset random seed to system-based
            random.seed()

    # 3) Shuffle all generated problems
    random.shuffle(all_problems)

    # 4) Write everything to a JSONL file
    output_file = "../../testset/financial_markets/option_pricing.jsonl"
    with open(output_file, "w") as f:
        for problem in all_problems:
            f.write(json.dumps(problem))
            f.write("\n")

    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")


# If running this file directly, just call main()
if __name__ == "__main__":
    main()
