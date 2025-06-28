import random

"""
Generates questions and step-by-step solutions for Decentralized Finance (DeFi) in Crypto Finance.
Includes randomization of inputs for each method and structured docstrings.
"""

# ========================
# 🟢 Basic Level
# ========================

def basic_yield_farming_apy():
    """
    1:Basic: Yield Farming with APY
    
    Simulates a deposit into a yield farming pool with a known APY and asks for the amount
    after 6 months using simple interest.
    
    Returns:
        tuple: (question_text, solution_text)
    """
    principal = random.randint(500, 2000)
    apy = random.uniform(4, 12)
    months = random.randint(1, 12)
    rate = apy / 100
    amount = round(principal * (1 + rate * (months / 12)), 2)
    question = (
        f"You deposit {principal} DAI into a yield farming pool offering {apy:.2f}% APY. "
        f"How much DAI will you have after {months} months?"
    )
    solution = (
        f"Step 1:\n"
        f"  APY means annual percentage yield. We're calculating simple interest over {months} months.\n"
        f"Step 2:\n"
        f"  Interest = Principal × Rate × Time = {principal} × {rate:.4f} × {months}/12\n"
        f"  Final Amount = {amount} DAI\n"
        f"Step 3:\n"
        f"  You will have approximately {amount} DAI after {months} months."
    )
    return question, solution

def basic_lending_collateral():
    """
    2:Basic: Collateral Calculation for Lending
    
    Creates a lending scenario where a user borrows USDC using ETH collateral.
    Calculates how much ETH is required based on a 150-250% collateral ratio.
    
    Returns:
        tuple: (question_text, solution_text)
    """
    borrow = random.randint(500, 3000)
    collateral_ratio = random.uniform(1.5, 2.5)  # 150% to 250%
    collateral_percentage = f"{collateral_ratio * 100:.2f}"
    eth_price = random.randint(1500, 3500)
    required_eth = round((borrow * collateral_ratio) / eth_price, 4)
    question = (
        f"You want to borrow {borrow} USDC from a DeFi protocol that requires {collateral_percentage}% collateral in ETH. "
        f"If ETH is trading at ${eth_price}, how much ETH must you deposit?"
    )
    usd_collateral = round(borrow * collateral_ratio, 2)
    solution = (
        f"Step 1:\n"
        f"  Required Collateral = Borrow × {collateral_percentage}% = {borrow} × {collateral_ratio:.2f} = {usd_collateral} USD worth of ETH\n"
        f"Step 2:\n"
        f"  ETH needed = {usd_collateral} / {eth_price} = {required_eth} ETH\n"
        f"Step 3:\n"
        f"  You must deposit approximately {required_eth} ETH."
    )
    return question, solution

# ========================
# 🟡 Intermediate Level
# ========================

def intermediate_apy_vs_apr():
    """
    3:Intermediate: APY vs APR Conversion
    
    Presents an APR rate and asks to compute the equivalent APY using compounding for X months.
    
    Returns:
        tuple: (question_text, solution_text)
    """
    apr = random.uniform(8, 15)
    compounding = random.randint(1, 12) # Compounding frequency (monthly)
    rate = apr / 100
    apy = round((1 + rate / compounding) ** compounding - 1, 4) * 100
    question = (
        f"A DeFi protocol offers {apr:.2f}% APR with compounding frequency of {compounding} months. What is the equivalent APY?"
    )
    solution = (
        f"Step 1:\n"
        f"  Use APY = (1 + r/n)^n - 1\n"
        f"  Where r = {rate:.4f}, n = {compounding}\n"
        f"Step 2:\n"
        f"  APY = (1 + {rate / compounding:.4f})^{compounding} - 1 ≈ {apy:.2f}%\n"
        f"Step 3:\n"
        f"  Equivalent APY is approximately {apy:.2f}%."
    )
    return question, solution

def intermediate_price_impact_swap():
    """
    4:Intermediate: AMM Price Impact Estimation
    
    Calculates the estimated price impact in a decentralized exchange swap based on
    trade size relative to liquidity pool size.
    
    Returns:
        tuple: (question_text, solution_text)
    """
    liquidity = random.randint(500000, 1000000)
    trade = random.randint(10000, 50000)
    price_impact = round((trade / liquidity) * 100, 2)
    question = (
        f"You are swapping {trade} USDC in a DeFi AMM with a liquidity pool of {liquidity} USDC. "
        f"Estimate the price impact as a percentage."
    )
    solution = (
        f"Step 1:\n"
        f"  Price impact ≈ Trade Size / Pool Size × 100 = {trade} / {liquidity} × 100\n"
        f"Step 2:\n"
        f"  Price impact ≈ {price_impact}%\n"
        f"Step 3:\n"
        f"  Your trade will move the price by about {price_impact}%."
    )
    return question, solution

# ========================
# 🔴 Advanced Level
# ========================

def advanced_flash_loan_arbitrage():
    """
    5:Advanced: Flash Loan Arbitrage Risk Analysis
    
    Presents a flash loan scenario with a net profit and explores risks of atomic arbitrage
    in DeFi protocols.
    
    Returns:
        tuple: (question_text, solution_text)
    """
    loan = random.randint(100000, 500000)
    profit = random.randint(500, 5000)
    question = (
        f"A trader takes a flash loan of {loan} USDC to perform arbitrage across two DEXs. "
        f"They return the loan in the same transaction and net a profit of {profit} USDC. "
        f"What are the risks associated with this strategy?"
    )
    solution = (
        f"Step 1:\n"
        f"  Flash loans must be repaid in the same transaction. If any step fails, the entire transaction reverts.\n"
        f"Step 2:\n"
        f"  Risks include slippage, MEV (Miner Extractable Value), smart contract bugs, and latency.\n"
        f"Step 3:\n"
        f"  Though profitable with {profit} USDC gain here, flash loan arbitrage is high-risk due to atomic execution."
    )
    return question, solution

def main():
    """
    Generate 10 instances of each template with different random seeds 
    and write the results to a JSON file.
    """
    import json
    # List of template functions
    templates = [
        basic_yield_farming_apy,
        basic_lending_collateral,
        intermediate_apy_vs_apr,
        intermediate_price_impact_swap,
        advanced_flash_loan_arbitrage
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
    output_file = "../../testset/crypto_finance/defi.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()