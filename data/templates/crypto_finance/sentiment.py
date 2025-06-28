import random

def basic_trading_volume_increase():
    """1:Basic: Trading volume increased scenario"""
    coin_name = random.choice(["Bitcoin", "Ethereum", "Solana"])
    increase = random.randint(10, 500)
    question = (
        f"The 24-hour trading volume for {coin_name} increased by {increase}%. "
        f"What does this suggest about the market sentiment?"
    )
    solution = (
        f"Step 1:\n"
        f"  When the trading volume increases significantly, it often indicates heightened interest in {coin_name}. "
        f"This could be due to positive news or events, suggesting a bullish sentiment."
    )
    return question, solution

def basic_large_trades():
    """2:Basic: Large trades scenario"""
    coin_name = random.choice(["Bitcoin", "Ethereum", "Cardano"])
    trades = random.randint(100, 1000)
    question = (
        f"The number of large trades for {coin_name} rose by {trades} in the past 24 hours. "
        f"What can this imply about the sentiment?"
    )
    solution = (
        f"Step 1:\n"
        f"  A rise in large trades indicates institutional or high-net-worth investor activity. "
        f"This usually reflects a bullish sentiment as significant players are entering the market."
    )
    return question, solution


def intermediate_mixed_social_volume():
    """3:Intermediate: Mixed social sentiment with high trading volume"""
    coin_name = random.choice(["Bitcoin", "Ethereum"])
    volume_increase = random.randint(100, 300)
    positive_sentiment = random.randint(50, 70)
    negative_sentiment = 100 - positive_sentiment
    question = (
        f"The trading volume for {coin_name} rose by {volume_increase}%, but social sentiment is mixed with "
        f"{positive_sentiment}% positive and {negative_sentiment}% negative. What can this indicate?"
    )
    solution = (
        f"Step 1:\n"
        f"  The high trading volume indicates strong market activity. However, the mixed sentiment suggests "
        f"uncertainty, with traders divided between bullish and bearish views."
    )
    return question, solution

def intermediate_large_trades_social_mismatch():
    """4:Intermediate: Large trades with contradictory social sentiment"""
    coin_name = random.choice(["Ethereum", "Solana"])
    large_trades = random.randint(200, 500)
    positive_sentiment = random.randint(30, 50)
    question = (
        f"Large trades for {coin_name} have increased by {large_trades}, but social sentiment is only "
        f"{positive_sentiment}% positive. How should this be interpreted?"
    )
    solution = (
        f"Step 1:\n"
        f"  The increase in large trades suggests institutional interest, potentially bullish. However, the low social "
        f"sentiment indicates retail traders are cautious or bearish."
    )
    return question, solution


def advanced_unusual_activity():
    """5:Advanced: Unusual market activity with diverging factors"""
    coin_name = random.choice(["Bitcoin", "Dogecoin"])
    volume_increase = random.randint(200, 500)
    sentiment_decline = random.randint(10, 30)
    question = (
        f"The trading volume for {coin_name} surged by {volume_increase}%, but positive sentiment declined by {sentiment_decline}%. "
        f"What does this unusual activity suggest?"
    )
    solution = (
        f"Step 1:\n"
        f"  The volume surge with declining sentiment may indicate profit-taking or panic selling. This could suggest "
        f"short-term bearishness despite heightened market activity."
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
        basic_trading_volume_increase,
        basic_large_trades,
        intermediate_mixed_social_volume,
        intermediate_large_trades_social_mismatch,
        advanced_unusual_activity
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
    output_file = "../../testset/crypto_finance/sentiment.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()