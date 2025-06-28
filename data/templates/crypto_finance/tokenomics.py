import random

"""
Generates questions and step-by-step solutions for Tokenomics scenarios in Crypto Finance.
Includes randomization of inputs for each method.
"""

# ========================
# 🟢 Basic Level
# ========================

def basic_staking_reward_calculation():
    """
    1:Basic: Staking rewards estimation.

    Computes staking yield after one year given a principal and APR reward rate.

    Returns:
        tuple: (question_text, solution_text)
    """
    staked_amount = random.randint(10_000, 100_000)
    reward_rate = random.randint(5, 15)
    question = (
        f"An investor stakes {staked_amount} tokens in a protocol offering a {reward_rate}% annual reward. "
        "How many tokens will the investor earn after one year?"
    )
    reward = int(staked_amount * reward_rate / 100)
    solution = (
        f"Step 1:\n"
        f"  Annual reward = {reward_rate}% of {staked_amount} = {reward} tokens.\n"
        f"Final Answer: {reward} tokens earned after one year."
    )
    return question, solution

def basic_airdrop_distribution():
    """
    2:Basic: Equal airdrop distribution.

    Given a total token airdrop and a number of recipients, calculates per-user allocation.

    Returns:
        tuple: (question_text, solution_text)
    """
    total_airdrop = random.randint(1_000_000, 10_000_000)
    eligible_users = random.randint(1000, 5000)
    question = (
        f"A protocol distributes {total_airdrop} tokens as an airdrop to {eligible_users} users. "
        "How many tokens does each user receive?"
    )
    per_user = total_airdrop // eligible_users
    solution = (
        f"Step 1:\n"
        f"  Tokens per user = {total_airdrop} / {eligible_users} = {per_user} tokens.\n"
        f"Final Answer: Each user receives {per_user} tokens."
    )
    return question, solution

# ========================
# 🟡 Intermediate Level
# ========================

def intermediate_token_vesting_schedule():
    """
    3:Intermediate: Linear vesting calculation over multiple years.

    Models a token vesting schedule with uniform unlocks across a defined time period.

    Returns:
        tuple: (question_text, solution_text)
    """
    total_allocated = random.randint(100_000_000, 500_000_000)
    vesting_years = random.choice([2, 3, 4, 5, 8, 10])
    years_passed = random.randint(1, vesting_years)
    question = (
        f"A project allocated {total_allocated} tokens to its team with a {vesting_years}-year linear vesting schedule. "
        f"How many tokens have vested after {years_passed} years?"
    )
    vested = int(total_allocated * years_passed / vesting_years)
    solution = (
        f"Step 1:\n"
        f"  Vesting is linear over {vesting_years} years.\n"
        f"Step 2:\n"
        f"  Vested tokens = ({years_passed}/{vesting_years}) × {total_allocated} = {vested} tokens.\n"
        f"Final Answer: {vested} tokens vested."
    )
    return question, solution

def intermediate_liquidity_mining_incentives():
    """
    4:Intermediate: Calculating liquidity mining rewards.

    Evaluates how much a user earns based on their share of liquidity in a mining program.

    Returns:
        tuple: (question_text, solution_text)
    """
    reward_pool = random.randint(5_000_000, 20_000_000)
    user_share = round(random.uniform(0.001, 0.05), 4)
    question = (
        f"A liquidity mining program offers a reward pool of {reward_pool} tokens. "
        f"A user contributes {user_share*100:.2f}% of total liquidity. "
        "How many tokens does the user receive as a reward?"
    )
    reward = int(reward_pool * user_share)
    solution = (
        f"Step 1:\n"
        f"  User's share = {user_share*100:.2f}% of {reward_pool} = {reward} tokens.\n"
        f"Final Answer: {reward} tokens earned."
    )
    return question, solution

# ========================
# 🔴 Advanced Level
# ========================

def advanced_multi_phase_vesting():
    """
    5:Advanced: Multi-phase vesting with randomized cliff and vesting parameters.

    Simulates a vesting schedule that includes a randomized cliff duration and percentage,
    followed by linear vesting for the remaining allocation.

    Returns:
        tuple: (question_text, solution_text) with the problem statement and detailed solution
    """
    total_allocation = random.randint(50_000_000, 500_000_000)
    vest_years = random.choice([3, 4, 5, 6])
    cliff_years = random.choice([1, 2])
    cliff_percent = random.choice([0.1, 0.2, 0.25, 0.3])
    years_elapsed = random.randint(0, vest_years)

    cliff_tokens = int(total_allocation * cliff_percent)
    linear_years = vest_years - cliff_years
    remaining_tokens = total_allocation - cliff_tokens

    if years_elapsed < cliff_years:
        vested = 0
        vest_note = "Cliff period not completed, no tokens vested yet."
    elif years_elapsed == cliff_years:
        vested = cliff_tokens
        vest_note = f"Only cliff tokens vested: {cliff_percent*100:.0f}% of {total_allocation} = {vested} tokens."
    else:
        post_cliff_years = years_elapsed - cliff_years
        linear_vested = int(remaining_tokens * (post_cliff_years / linear_years))
        vested = cliff_tokens + linear_vested
        vest_note = (
            f"Cliff tokens = {cliff_tokens}. "
            f"Linear vested = {linear_vested} tokens over {post_cliff_years} years after the cliff."
        )

    question = (
        f"A project allocates {total_allocation} tokens to its advisors with a "
        f"{cliff_years}-year cliff at {int(cliff_percent * 100)}%, followed by linear vesting over the next "
        f"{linear_years} years. How many tokens will be vested after {years_elapsed} years?"
    )

    solution = (
        f"Step 1:\n"
        f"  Total Allocation = {total_allocation} tokens.\n"
        f"  Cliff: {int(cliff_percent * 100)}% over {cliff_years} years = {cliff_tokens} tokens.\n"
        f"Step 2:\n"
        f"  Remaining {remaining_tokens} tokens vest linearly over {linear_years} years.\n"
        f"Step 3:\n"
        f"  Elapsed years = {years_elapsed} → {vest_note}\n"
        f"Final Answer: {vested} tokens vested after {years_elapsed} years."
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
        basic_staking_reward_calculation,
        basic_airdrop_distribution,
        intermediate_token_vesting_schedule,
        intermediate_liquidity_mining_incentives,
        advanced_multi_phase_vesting
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
    output_file = "../../testset/crypto_finance/tokenomics.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()