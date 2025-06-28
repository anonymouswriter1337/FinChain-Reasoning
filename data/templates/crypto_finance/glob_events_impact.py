import random

"""
Generates questions and step-by-step solutions for Global and Macro-economic Events Impact Prediction in Crypto Finance.
Includes randomization of inputs for each method.
"""
# Basic Level
def basic_interest_rate_increase():
    """
    1:Basic: Analysis of central bank interest rate increases on Bitcoin prices.
    
    Generates a problem examining how an interest rate hike by the U.S. central bank
    might impact Bitcoin prices through reduced market liquidity.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    rate = round(random.uniform(0.25, 1.0), 2)
    question = (
        f"The central bank of the USA raises its interest rates by {rate}%. Historically, rate hikes reduce liquidity in the market. "
        "How might this event impact Bitcoin prices?"
    )
    solution = (
        f"Step 1:\n"
        f"  Higher interest rates increase borrowing costs, reducing liquidity in the market.\n"
        f"Step 2:\n"
        f"  Risk assets like Bitcoin often decline as investors seek safer options.\n"
        f"Step 3:\n"
        f"  Bitcoin prices may likely decrease in response to the rate hike of {rate}%."
    )
    return question, solution

def basic_inflation_rise():
    """
    2:Basic: Evaluation of inflation's effect on Bitcoin as a potential hedge.
    
    Generates a problem analyzing how rising inflation in the Eurozone might
    impact Bitcoin prices considering its potential role as an inflation hedge.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    inflation = random.randint(5, 15)
    question = (
        f"Inflation in the Eurozone rises to {inflation}%. Gold and Bitcoin are often viewed as inflation hedges. "
        "What impact might this have on Bitcoin prices?"
    )
    solution = (
        f"Step 1:\n"
        f"  High inflation of {inflation}% leads to a loss in purchasing power, increasing demand for inflation hedges.\n"
        f"Step 2:\n"
        f"  Bitcoin might see increased interest as a store of value.\n"
        f"Step 3:\n"
        f"  Bitcoin prices could potentially rise."
    )
    return question, solution

# Intermediate Level
def intermediate_trade_war():
    """
    3:Intermediate: Assessment of geopolitical trade tensions on cryptocurrency markets.
    
    Generates a problem examining how escalating trade conflicts between major economies
    and resulting stock market declines might affect Bitcoin prices.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    stock_decline = random.randint(5, 15)
    question = (
        f"A trade war escalates between the USA and China, causing global stock markets to decline by {stock_decline}%. "
        "What impact might this have on Bitcoin prices?"
    )
    solution = (
        f"Step 1:\n"
        f"  Stock market declines of {stock_decline}% often lead to increased risk aversion.\n"
        f"Step 2:\n"
        f"  Bitcoin's role as a speculative asset could lead to short-term price drops.\n"
        f"Step 3:\n"
        f"  However, in prolonged market declines, Bitcoin may be seen as a non-sovereign store of value, potentially increasing demand."
    )
    return question, solution

def intermediate_crypto_regulation():
    """
    4:Intermediate: Evaluation of regulatory changes on cryptocurrency markets.
    
    Generates a problem analyzing how new taxation policies on cryptocurrency
    transactions in a major market might impact trading volumes and prices.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    tax = random.randint(10, 50)
    question = (
        f"India announces a {tax}% tax on all cryptocurrency transactions, including gains. "
        "What impact might this regulation have on crypto trading volumes and prices in India?"
    )
    solution = (
        f"Step 1:\n"
        f"  High taxes of {tax}% reduce the profitability of crypto trading, discouraging participation.\n"
        f"Step 2:\n"
        f"  Trading volumes in India are likely to decrease.\n"
        f"Step 3:\n"
        f"  Prices might face downward pressure locally but remain stable globally."
    )
    return question, solution

# Advanced Level
def advanced_opec_cut():
    """
    5:Advanced: Analysis of energy market disruptions on cryptocurrency valuations.
    
    Generates a problem examining how significant changes in oil production and
    resulting price increases might indirectly affect cryptocurrency markets.
    
    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    oil_increase = random.randint(10, 20)
    question = (
        f"OPEC announces a significant cut in oil production, leading to a {oil_increase}% increase in oil prices. "
        "What impact might this have on cryptocurrency markets?"
    )
    solution = (
        f"Step 1:\n"
        f"  Higher oil prices of {oil_increase}% increase costs across global economies, potentially reducing disposable income.\n"
        f"Step 2:\n"
        f"  Reduced disposable income may lead to lower investments in risk assets like crypto.\n"
        f"Step 3:\n"
        f"  Crypto prices may face downward pressure in the short term."
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
        basic_interest_rate_increase,
        basic_inflation_rise,
        intermediate_trade_war,
        intermediate_crypto_regulation,
        advanced_opec_cut
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
    output_file = "../../testset/crypto_finance/glob_events_impact.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()