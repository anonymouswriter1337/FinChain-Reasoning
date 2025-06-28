import random

# Basic Questions
def basic_gas_fee_calculation():
    """
    1:Basic: Calculation of Ethereum gas fees for a transaction.
    
    Generates a problem to calculate the total gas fee in ETH for a transaction,
    given the gas price in gwei, gas units used, and the amount being transferred.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    gas_price = random.randint(10, 300)  # gwei
    gas_used = random.randint(21000, 100000)  # units
    amount_eth = round(random.uniform(0.1, 10), 2)  # ETH
    conversion_rate = 1e9  # gwei to ETH
    total_fee_eth = (gas_price * gas_used) / conversion_rate
    question = (
        f"A transaction of {amount_eth:.2f} ETH uses {gas_used} gas units at a gas price of {gas_price} gwei. "
        "What is the total gas fee in ETH?"
    )
    solution = (
        f"Step 1: Calculate the total fee in gwei:\n"
        f"  {gas_price} gwei * {gas_used} units = {gas_price * gas_used} gwei.\n"
        f"Step 2: Convert gwei to ETH:\n"
        f"  Final Answer: {gas_price * gas_used} gwei / {conversion_rate} = {total_fee_eth:.2f} ETH."
    )
    return question, solution

def basic_balance_after_transaction():
    """
    2:Basic: Tracking wallet balance after cryptocurrency transactions.
    
    Generates a problem to determine the final balance of a Bitcoin wallet
    after receiving and sending transactions.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    initial_balance = round(random.uniform(1, 10), 2)  # BTC
    received = round(random.uniform(0.1, 5), 2)  # BTC
    sent = round(random.uniform(0.1, 5), 2)  # BTC
    final_balance = initial_balance + received - sent
    question = (
        f"A wallet starts with {initial_balance:.2f} BTC, receives {received:.2f} BTC, and sends {sent:.2f} BTC. "
        "What is the final balance of the wallet?"
    )
    solution = (
        f"Step 1: Add received amount:\n"
        f"  {initial_balance:.2f} + {received:.2f} = {initial_balance + received:.2f}.\n"
        f"Step 2: Subtract sent amount:\n"
        f"  Final Answer: {initial_balance + received:.2f} - {sent:.2f} = {final_balance:.2f}."
    )
    return question, solution



def intermediate_staking_rewards():
    """
    3:Intermediate: Calculation of cryptocurrency staking rewards over time.
    
    Generates a problem to calculate the total staking rewards earned from
    Ethereum staking over a multi-year period at a specified interest rate.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    staked_amount = round(random.uniform(1, 50), 2)  # ETH
    annual_percentage = round(random.uniform(5, 20), 2)  # %
    duration_years = random.randint(1, 5)
    total_rewards = staked_amount * (annual_percentage / 100) * duration_years
    question = (
        f"An investor stakes {staked_amount:.2f} ETH at an annual interest rate of {annual_percentage:.2f}% for {duration_years} years. "
        "What are the total staking rewards?"
    )
    solution = (
        f"Step 1: Calculate annual rewards:\n"
        f"  {staked_amount:.2f} * {annual_percentage:.2f} / 100 = {staked_amount * (annual_percentage / 100):.2f}.\n"
        f"Step 2: Multiply by the duration:\n"
        f"  Final Answer: {staked_amount * (annual_percentage / 100):.2f} * {duration_years} = {total_rewards:.2f} ETH."
    )
    return question, solution

def intermediate_multiple_transactions():
    """
    4:Intermediate: Analysis of multiple cryptocurrency transactions including fees.
    
    Generates a problem to determine the final wallet balance after multiple
    transactions including network fees.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    initial_balance = round(random.uniform(1, 20), 2)  # BTC
    tx1 = round(random.uniform(0.1, 5), 2)  # BTC
    tx2 = round(random.uniform(0.1, 5), 2)  # BTC
    fee = round(random.uniform(0.001, 0.05), 4)  # BTC
    final_balance = initial_balance + tx1 - tx2 - fee
    question = (
        f"A wallet starts with {initial_balance:.2f} BTC, receives {tx1:.2f} BTC, and sends {tx2:.2f} BTC, with a network fee of {fee:.4f} BTC. "
        "What is the final wallet balance?"
    )
    solution = (
        f"Step 1: Add received amount:\n"
        f"  {initial_balance:.2f} + {tx1:.2f} = {initial_balance + tx1:.2f}.\n"
        f"Step 2: Subtract sent amount:\n"
        f"  {initial_balance + tx1:.2f} - {tx2:.2f} = {initial_balance + tx1 - tx2:.2f}.\n"
        f"Step 3: Subtract fee:\n"
        f"  Final Answer: {initial_balance + tx1 - tx2:.2f} - {fee:.4f} = {final_balance:.4f} BTC."
    )
    return question, solution


def advanced_multi_exchange_analysis():
    """
    5:Advanced: Complex cryptocurrency trading scenario across multiple exchanges.
    
    Generates a problem to calculate the final cryptocurrency amount after
    multiple exchanges with different rates and fees.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    btc_initial = round(random.uniform(0.1, 5), 2)  # BTC
    exchange_rate_1 = round(random.uniform(30000, 40000), 2)  # USD/BTC
    exchange_rate_2 = round(random.uniform(25000, 35000), 2)  # USD/BTC
    fee_percentage = round(random.uniform(0.1, 2), 2)  # %
    usd_value = round(btc_initial * exchange_rate_1, 2)
    fee = round((usd_value * fee_percentage) / 100, 2)
    btc_final = round((usd_value - fee) / exchange_rate_2, 2)
    question = (
        f"A trader exchanges {btc_initial:.2f} BTC at an initial rate of ${exchange_rate_1:.2f}/BTC, then incurs a fee of {fee_percentage:.2f}%, and finally exchanges the remaining amount at ${exchange_rate_2:.2f}/BTC. "
        "How much BTC does the trader have after all transactions?"
    )
    solution = (
        f"Step 1: Calculate USD value after first exchange:\n"
        f"  {btc_initial:.2f} BTC * ${exchange_rate_1:.2f} = ${usd_value:.2f}.\n"
        f"Step 2: Calculate fee:\n"
        f"  ${usd_value:.2f} * {fee_percentage:.2f} / 100 = ${fee:.2f}.\n"
        f"Step 3: Subtract fee and convert back to BTC:\n"
        f"  Final Answer: (${usd_value:.2f} - ${fee:.2f}) / ${exchange_rate_2:.2f} = {btc_final:.2f} BTC."
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
        basic_gas_fee_calculation,
        basic_balance_after_transaction,
        intermediate_staking_rewards,
        intermediate_multiple_transactions,
        advanced_multi_exchange_analysis,
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
    output_file = "../../testset/crypto_finance/tx_analysis.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()