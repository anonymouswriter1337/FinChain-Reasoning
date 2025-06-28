import random

def basic_large_transaction_to_exchange():
    """
    1:Basic: Analysis of a large crypto transaction to an exchange.
    
    Generates a problem about a significant Bitcoin transfer to Binance and calculates
    what percentage of the daily trading volume this single transaction represents.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    daily_volume = random.randint(10000, 50000)
    transaction_amount = random.randint(100, daily_volume // 10)
    question = (
        f"A whale moves {transaction_amount} BTC from a private wallet to Binance. The daily trading volume "
        f"on Binance is {daily_volume} BTC. What percentage of the daily trading volume does this transaction constitute?"
    )
    solution = (
        f"Step 1: Calculate the percentage of the transaction relative to the daily volume.\n"
        f"  Percentage = (Transaction Amount / Daily Volume) * 100 = ({transaction_amount} / {daily_volume}) * 100 = "
        f"{(transaction_amount / daily_volume) * 100:.2f}%.\n"
        f"  The transaction constitutes {(transaction_amount / daily_volume) * 100:.2f}% of the daily trading volume."
    )
    return question, solution

def basic_multiple_transactions_analysis():
    """
    2:Basic: Analysis of multiple crypto transactions from a whale.
    
    Generates a problem about a series of Bitcoin transfers to multiple wallets and calculates
    the total amount transferred and its proportion to the daily trading volume.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    transactions = [random.randint(100, 500) for _ in range(3)]
    daily_volume = random.randint(10000, 50000)
    total_transaction = sum(transactions)
    question = (
        f"A whale performs 3 transactions of {transactions[0]} BTC, {transactions[1]} BTC, and {transactions[2]} BTC to three separate wallets. "
        f"What is the total amount transferred, and how does it compare to a daily trading volume of {daily_volume} BTC?"
    )
    solution = (
        f"Step 1: Calculate the total transaction amount.\n"
        f"  Total = {transactions[0]} + {transactions[1]} + {transactions[2]} = {total_transaction} BTC.\n"
        f"Step 2: Calculate the percentage of the daily volume.\n"
        f"  Percentage = (Total Amount / Daily Volume) * 100 = ({total_transaction} / {daily_volume}) * 100 = "
        f"{(total_transaction / daily_volume) * 100:.2f}%.\n"
        f"  The total amount transferred is {total_transaction} BTC, which constitutes {(total_transaction / daily_volume) * 100:.2f}% of the daily trading volume."
    )
    return question, solution


def intermediate_market_impact_estimation():
    """
    3:Intermediate: Estimating market impact of a whale transaction.
    
    Generates a problem about the price impact of a large Bitcoin transfer to an exchange,
    calculating the new price after a given percentage drop following the transaction.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    transaction_amount = random.randint(1000, 2000)
    daily_volume = random.randint(5000, 20000)
    price_drop_percentage = round(random.uniform(1.0, 5.0), 2)
    initial_price = random.randint(30000, 60000)
    new_price = initial_price * (1 - price_drop_percentage / 100)
    question = (
        f"A whale transfers {transaction_amount} BTC to Binance, representing {(transaction_amount / daily_volume) * 100:.2f}% of the exchange's daily trading volume. "
        f"If the price of BTC drops by {price_drop_percentage:.2f}% after the transaction, what is the new price of BTC if it was initially ${initial_price}?"
    )
    solution = (
        f"Step 1: Calculate the price decrease.\n"
        f"  Decrease = Initial Price * Percentage Drop = {initial_price} * {price_drop_percentage / 100:.4f} = {initial_price * (price_drop_percentage / 100):.2f}.\n"
        f"Step 2: Subtract the decrease from the initial price.\n"
        f"  New Price = Initial Price - Decrease = {initial_price} - {initial_price * (price_drop_percentage / 100):.2f} = {new_price:.2f}.\n"
        f"  The new price of BTC is ${new_price:.2f}."
    )
    return question, solution

def intermediate_liquidity_pool_effect():
    """
    4:Intermediate: Analysis of liquidity pool changes due to whale activity.
    
    Generates a problem about a large Ethereum withdrawal from a liquidity pool,
    calculating the percentage of the pool that was withdrawn and the remaining liquidity.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    total_liquidity = random.randint(50000, 100000)
    withdrawal_amount = random.randint(1000, total_liquidity // 10)
    remaining_liquidity = total_liquidity - withdrawal_amount
    percentage = (withdrawal_amount / total_liquidity) * 100
    question = (
        f"A whale withdraws {withdrawal_amount} ETH from a liquidity pool with a total of {total_liquidity} ETH. "
        f"What percentage of the liquidity pool does this withdrawal represent, and how much ETH remains?"
    )
    solution = (
        f"Step 1: Calculate the percentage of the withdrawal.\n"
        f"  Percentage = (Withdrawal Amount / Total Liquidity) * 100 = ({withdrawal_amount} / {total_liquidity}) * 100 = {percentage:.2f}%.\n"
        f"Step 2: Subtract the withdrawal from the total liquidity.\n"
        f"  Remaining Liquidity = Total Liquidity - Withdrawal = {total_liquidity} - {withdrawal_amount} = {remaining_liquidity} ETH.\n"
        f"  The withdrawal represents {percentage:.2f}% of the liquidity pool, leaving {remaining_liquidity} ETH."
    )
    return question, solution


def advanced_hidden_wallet_tracking():
    """
    5:Advanced: Tracking complex transaction patterns across multiple wallets.
    
    Generates a problem about a multi-step Bitcoin transfer from one wallet to multiple
    destination wallets, calculating the distribution and percentages in each wallet.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    original_transfer = random.randint(500, 1000)
    split_wallets = 2
    per_wallet = original_transfer / split_wallets
    percentage = (per_wallet / original_transfer) * 100
    question = (
        f"A whale moves {original_transfer} BTC from Wallet A to Wallet B, then splits the amount evenly into Wallets C and D. "
        f"How much BTC is in Wallets C and D, and what is the percentage of the total original transfer in each?"
    )
    solution = (
        f"Step 1: Calculate the amount in Wallets C and D.\n"
        f"  Amount per Wallet = Transferred Amount / 2 = {original_transfer} / {split_wallets} = {per_wallet:.2f} BTC.\n"
        f"Step 2: Calculate the percentage of the original transfer in each wallet.\n"
        f"  Percentage = (Amount per Wallet / Original Transfer) * 100 = ({per_wallet:.2f} / {original_transfer}) * 100 = {percentage:.2f}%.\n"
        f"  Wallets C and D each hold {per_wallet:.2f} BTC, which is {percentage:.2f}% of the original transfer."
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
        basic_large_transaction_to_exchange,
        basic_multiple_transactions_analysis,
        intermediate_market_impact_estimation,
        intermediate_liquidity_pool_effect,
        advanced_hidden_wallet_tracking
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
    output_file = "../../testset/crypto_finance/whale_det.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()