import random

# Named entities for companies and industries
company_names = ["Tesla Inc.", "Apple Inc.", "Amazon.com", "SpaceX", "Google LLC"]
industry_names = ["automotive", "technology", "e-commerce", "aerospace", "internet services"]

# Template 1: Loan Repayment with Simple Interest
def template_loan_repayment_simple():
    """1:Basic: Generates a simple loan repayment problem using simple interest calculation, 
    where a company needs to determine the total amount to be repaid over a fixed term."""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    loan_amount = round(random.randint(500000, 5000000),2)  # Loan amount
    interest_rate = round(random.uniform(3.0, 12.0),2)  # Interest rate (%)
    loan_term = random.randint(3, 15)  # Loan term (years)
    
    # Question formulation
    question = (
        f"{company_name}, operating in the {industry} industry, has taken a loan of ${loan_amount} at an annual interest rate of {interest_rate:.2f}%. "
        f"The loan is to be repaid in {loan_term} years using simple interest. Calculate the total amount the company will repay by the end of the loan term."
    )
    
    # Step 1: Calculate the total repayment amount using simple interest formula
    total_repayment = round(loan_amount * (1 + (interest_rate / 100) * loan_term), 2)
    
    # Solution formulation
    solution = (
        f"Step 1: Calculate the total repayment using the simple interest formula:\n"
        f"  Total Repayment = Loan Amount × (1 + Interest Rate × Loan Term)\n"
        f"                 = ${loan_amount} × (1 + {interest_rate:.2f}% × {loan_term})\n"
        f"                 = ${total_repayment:.2f}"
    )
    
    return question, solution

# Template 2: Loan Repayment with Monthly Installments
def template_loan_repayment_monthly_installments():
    """2:Basic: Creates a loan amortization problem where a company must calculate 
    monthly installments using compound interest, incorporating the time value of money 
    through the amortization formula."""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    loan_amount = round(random.randint(1000000, 10000000),2)  # Loan amount
    interest_rate = random.uniform(4.0, 10.0) # Interest rate (%)
    loan_term = random.randint(5, 20)  # Loan term (years)
    
    monthly_interest_rate = interest_rate / 12 / 100 # Monthly interest rate
    number_of_installments = loan_term * 12  # Total number of monthly installments
    
    # Question formulation
    question = (
        f"{company_name}, a leading company in the {industry} industry, has taken a loan of ${loan_amount} at an annual interest rate of {interest_rate:.2f}%. "
        f"The loan is to be repaid over {loan_term} years through equal monthly installments. Calculate the company’s monthly repayment amount."
    )
    
    # Step 1: Calculate monthly installment using the loan amortization formula
    monthly_installment = round(
        loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate) ** number_of_installments) / 
        ((1 + monthly_interest_rate) ** number_of_installments - 1), 2
    )
    
    # Solution formulation
    solution = (
        f"Step 1: Calculate the monthly installment using the amortization formula:\n"
        f"  Monthly Payment = [Loan Amount × Monthly Interest Rate × (1 + Monthly Interest Rate)^N] / [(1 + Monthly Interest Rate)^N - 1]\n"
        f"                 = ${loan_amount} × {monthly_interest_rate:.2f} × (1 + {monthly_interest_rate:.2f})^{number_of_installments} / "
        f"[(1 + {monthly_interest_rate:.2f})^{number_of_installments} - 1]\n"
        f"                 = ${monthly_installment:.2f}"
    )
    
    return question, solution

# Template 3: Loan Repayment with Extra Annual Payments
def template_loan_repayment_with_extra_payments():
    """3:Intermediate: Generates a complex loan scenario where a company makes additional annual 
    payments towards the principal, requiring calculations to determine the new loan term 
    and the impact of extra payments."""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    loan_amount = round(random.randint(2000000, 10000000),2)  # Loan amount
    interest_rate = round(random.uniform(5.0, 10.0),2)  # Interest rate (%)
    loan_term = random.randint(5, 25)  # Loan term (years)
    extra_payment = round(random.randint(100000, 500000),2)  # Extra annual payment towards principal
    
    monthly_interest_rate = interest_rate / 12 / 100  # Monthly interest rate
    number_of_installments = loan_term * 12  # Total number of installments
    
    # Question
    question = (
        f"{company_name}, operating in the {industry} industry, took a loan of ${loan_amount} at an annual interest rate of {interest_rate:.5f}%. "
        f"The loan is to be repaid in {loan_term} years through equal monthly installments. "
        f"However, the company decides to make an extra payment of ${extra_payment} every year towards the principal. "
        f"Calculate the new loan term after accounting for these extra annual payments."
    )
    
    # Step 1: Calculate the original monthly payment using the loan amortization formula
    monthly_payment = round(
        loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate) ** number_of_installments) /
        ((1 + monthly_interest_rate) ** number_of_installments - 1), 2
    )
    
    # Step 2: Simulate the effect of extra annual payments towards the principal each year
    remaining_balance = loan_amount
    months = 0
    extra_payment_years = 0
    
    while remaining_balance > 0:
        # Apply monthly payment
        interest_payment = remaining_balance * monthly_interest_rate
        principal_payment = min(monthly_payment - interest_payment, remaining_balance)
        remaining_balance -= principal_payment
        months += 1
        
        # Apply extra payment at the end of each year (every 12 months)
        if months % 12 == 0 and remaining_balance > 0:
            extra_payment_years += 1
            remaining_balance = max(0, remaining_balance - extra_payment)
    
    # Step 3: Recalculate the new loan term based on remaining loan balance after extra payments
    new_loan_term_years = months // 12
    new_loan_term_months = months % 12
    
    # Solution
    solution = (
        f"Step 1: Calculate the original monthly payment:\n"
        f"  Monthly Payment = [Loan Amount × Monthly Interest Rate × (1 + Monthly Interest Rate)^N] / [(1 + Monthly Interest Rate)^N - 1]\n"
        f"                 = ${loan_amount} × {monthly_interest_rate:.5f} × (1 + {monthly_interest_rate:.5f})^{number_of_installments} / "
        f"[(1 + {monthly_interest_rate:.5f})^{number_of_installments} - 1] = ${monthly_payment:.2f}\n\n"
        f"Step 2: Account for the extra payments towards the principal each year:\n"
        f"  Extra Annual Payment = ${extra_payment} per year.\n"
        f"  By making these extra payments, the loan will be completely repaid in {months} months.\n\n"
        f"Step 3: The new loan term is {months} months (or {new_loan_term_years} years and {new_loan_term_months} months).\n"
        f"  This is a reduction of {loan_term - new_loan_term_years} years and {12 - new_loan_term_months if new_loan_term_months > 0 else 0} months "
        f"from the original {loan_term}-year term."
    )
    
    return question, solution

# Template 4: Loan Repayment with Refinancing
def template_loan_repayment_with_refinancing():
    """4:Intermediate: Creates a sophisticated loan refinancing problem where a company 
    refinances their existing loan at a lower interest rate, requiring calculations 
    of remaining balance and new monthly payments."""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    loan_amount = round(random.randint(3000000, 15000000),2)  # Original loan amount
    interest_rate = round(random.uniform(5.0, 10.0),2)  # Original interest rate (%)
    years_paid = random.randint(3, 10)  # Years already repaid
    new_interest_rate = round(random.uniform(3.0, 7.0),2)  # Refinanced interest rate (%)
    remaining_term = random.randint(5, 20)  # Remaining loan term after refinancing
    
    # Monthly interest rates
    original_monthly_interest_rate = interest_rate / 12 / 100
    new_monthly_interest_rate = new_interest_rate / 12 / 100
    
    # Question
    question = (
        f"{company_name}, a company in the {industry} industry, took a loan of ${loan_amount} at an interest rate of {interest_rate:.2f}%. "
        f"After repaying part of the loan for {years_paid} years, the company decides to refinance the remaining loan amount at a lower interest rate of {new_interest_rate:.2f}%. "
        f"Calculate the new monthly repayment amount after refinancing."
    )
    
    # Step 1: Calculate original monthly payment using the loan amortization formula
    original_loan_term = years_paid + remaining_term  # Original loan term in years
    total_installments_paid = years_paid * 12  # Number of installments paid
    original_number_of_installments = original_loan_term * 12  # Total original installments
    
    original_monthly_payment = round(
        loan_amount * (original_monthly_interest_rate * (1 + original_monthly_interest_rate) ** original_number_of_installments) /
        ((1 + original_monthly_interest_rate) ** original_number_of_installments - 1), 2
    )
    
    # Step 2: Calculate the remaining loan balance after years of payments using the proper amortization formula
    remaining_loan_balance = round(
        loan_amount * 
        (1 - ((1 + original_monthly_interest_rate) ** total_installments_paid - 1) / 
         ((1 + original_monthly_interest_rate) ** original_number_of_installments - 1)),
        2
    )
    
    # Step 3: Calculate new monthly payment after refinancing
    remaining_term_in_months = remaining_term * 12
    new_monthly_payment = round(
        remaining_loan_balance * (new_monthly_interest_rate * (1 + new_monthly_interest_rate) ** remaining_term_in_months) /
        ((1 + new_monthly_interest_rate) ** remaining_term_in_months - 1), 2
    )
    
    # Solution
    solution = (
        f"Step 1: Calculate the original monthly payment using the loan amortization formula:\n"
        f"  Original Loan Term = {years_paid} + {remaining_term} = {original_loan_term} years\n"
        f"  Original Monthly Payment = [Loan Amount × Monthly Interest Rate × (1 + Monthly Interest Rate)^N] / [(1 + Monthly Interest Rate)^N - 1]\n"
        f"                         = ${loan_amount} × {original_monthly_interest_rate:.5f} × (1 + {original_monthly_interest_rate:.5f})^{original_number_of_installments} / "
        f"[(1 + {original_monthly_interest_rate:.5f})^{original_number_of_installments} - 1] = ${original_monthly_payment:.2f}\n\n"
        
        f"Step 2: Calculate the remaining loan balance after {years_paid} years of payments:\n"
        f"  Remaining Loan Balance = Principal × [1 - ((1 + r)^p - 1) / ((1 + r)^n - 1)]\n"
        f"                         = ${loan_amount} × [1 - ((1 + {original_monthly_interest_rate:.5f})^{total_installments_paid} - 1) / "
        f"((1 + {original_monthly_interest_rate:.5f})^{original_number_of_installments} - 1)]\n"
        f"                         = ${remaining_loan_balance:.2f}\n\n"
        
        f"Step 3: Calculate the new monthly payment after refinancing:\n"
        f"  New Monthly Payment = [Remaining Balance × New Monthly Interest Rate × (1 + New Monthly Interest Rate)^(Remaining Term)] / "
        f"[(1 + New Monthly Interest Rate)^(Remaining Term) - 1]\n"
        f"                     = ${remaining_loan_balance:.2f} × {new_monthly_interest_rate:.5f} × (1 + {new_monthly_interest_rate:.5f})^{remaining_term_in_months} / "
        f"[(1 + {new_monthly_interest_rate:.5f})^{remaining_term_in_months} - 1] = ${new_monthly_payment:.2f}"
    )
    
    return question, solution

# Template 5
def template_loan_repayment_early_payoff():
    """5:Advanced: Generates a complex early loan payoff scenario where a company makes 
    a lump-sum payment, requiring calculations of interest savings and remaining balance 
    comparisons between different payment strategies."""
    company_name = random.choice(company_names)
    industry = random.choice(industry_names)
    loan_amount = round(random.randint(5000000, 20000000),2)  # Loan amount
    interest_rate = round(random.uniform(4.0, 9.0),2)  # Interest rate (%)
    loan_term = random.randint(10, 30)  # Loan term (years)
    years_paid = random.randint(3, 15)  # Years already repaid
    lump_sum_payment = round(random.randint(1000000, 5000000),2)  # Lump-sum early payoff amount
    
    question = (
        f"{company_name}, a company in the {industry} industry, took a loan of ${loan_amount} at an interest rate of {interest_rate:.2f}%. "
        f"After repaying the loan for {years_paid} years, the company decides to pay off the loan early by making a lump-sum payment of ${lump_sum_payment}. "
        f"How much will the company save in interest payments by paying off the loan early?"
    )
    
    # Step 1: Calculate original monthly payment
    monthly_interest_rate = interest_rate / 12 / 100
    total_installments = loan_term * 12
    monthly_payment = round(
        loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate) ** total_installments) / 
        ((1 + monthly_interest_rate) ** total_installments - 1), 
        2
    )

    # Step 2: Calculate remaining loan balance after years paid (using correct formula)
    installments_paid = years_paid * 12
    remaining_installments = total_installments - installments_paid
    
    remaining_loan_balance = round(
        loan_amount * 
        (1 - ((1 + monthly_interest_rate) ** installments_paid - 1) / 
         ((1 + monthly_interest_rate) ** total_installments - 1)),
        2
    )
    
    # Step 3: Calculate total payments remaining without early payoff
    total_payments_without_early_payoff = monthly_payment * remaining_installments
    interest_without_early_payoff = round(total_payments_without_early_payoff - remaining_loan_balance, 2)
    
    # Step 4: Calculate new remaining balance after lump sum payment
    new_remaining_balance = max(0, remaining_loan_balance - lump_sum_payment)
    new_remaining_balance = round(new_remaining_balance, 2)
    
    # Step 5: Calculate new repayment scenario with reduced principal
    if new_remaining_balance > 0:
        # Still has remaining balance to pay off
        new_monthly_payment = round(
            new_remaining_balance * (monthly_interest_rate * (1 + monthly_interest_rate) ** remaining_installments) / 
            ((1 + monthly_interest_rate) ** remaining_installments - 1), 
            2
        )
        total_payments_with_early_payoff = (new_monthly_payment * remaining_installments) + lump_sum_payment
        interest_with_early_payoff = round(total_payments_with_early_payoff - remaining_loan_balance, 2)
    else:
        # Loan fully paid off
        total_payments_with_early_payoff = lump_sum_payment
        interest_with_early_payoff = 0
    
    # Calculate interest savings
    interest_saved = round(interest_without_early_payoff - interest_with_early_payoff, 2)
    
    solution = (
        f"Step 1: Calculate the original monthly payment:\n"
        f"  Monthly Payment = [Loan Amount × Monthly Interest Rate × (1 + Monthly Interest Rate)^N] / [(1 + Monthly Interest Rate)^N - 1]\n"
        f"                 = ${loan_amount} × {monthly_interest_rate:.5f} × (1 + {monthly_interest_rate:.5f})^{total_installments} / "
        f"[(1 + {monthly_interest_rate:.5f})^{total_installments} - 1]\n"
        f"                 = ${monthly_payment:.2f}\n\n"
        
        f"Step 2: Calculate the remaining loan balance after {years_paid} years ({installments_paid} payments):\n"
        f"  Remaining Loan Balance = Principal × [1 - ((1 + r)^p - 1) / ((1 + r)^n - 1)]\n"
        f"                         = ${loan_amount} × [1 - ((1 + {monthly_interest_rate:.5f})^{installments_paid} - 1) / "
        f"((1 + {monthly_interest_rate:.5f})^{total_installments} - 1)]\n"
        f"                         = ${remaining_loan_balance:.2f}\n\n"
        
        f"Step 3: Calculate total interest remaining without early payoff:\n"
        f"  Total Payments Remaining = Monthly Payment × Remaining Installments\n"
        f"                          = ${monthly_payment:.2f} × {remaining_installments}\n"
        f"                          = ${total_payments_without_early_payoff:.2f}\n"
        f"  Interest Without Early Payoff = Total Payments Remaining - Remaining Loan Balance\n"
        f"                               = ${total_payments_without_early_payoff:.2f} - ${remaining_loan_balance:.2f}\n"
        f"                               = ${interest_without_early_payoff:.2f}\n\n"
        
        f"Step 4: Calculate the new remaining balance after the lump sum payment of ${lump_sum_payment}:\n"
        f"  New Remaining Loan Balance = ${remaining_loan_balance:.2f} - ${lump_sum_payment:.2f} = ${new_remaining_balance:.2f}\n\n"
    )
    
    if new_remaining_balance > 0:
        solution += (
            f"Step 5: Calculate interest with early payoff:\n"
            f"  New Monthly Payment = ${new_monthly_payment:.2f}\n"
            f"  Total Payments With Early Payoff = (New Monthly Payment × Remaining Installments) + Lump Sum\n"
            f"                                  = (${new_monthly_payment:.2f} × {remaining_installments}) + ${lump_sum_payment:.2f}\n"
            f"                                  = ${total_payments_with_early_payoff:.2f}\n"
            f"  Interest With Early Payoff = Total Payments With Early Payoff - Remaining Loan Balance\n"
            f"                            = ${total_payments_with_early_payoff:.2f} - ${remaining_loan_balance:.2f}\n"
            f"                            = ${interest_with_early_payoff:.2f}\n\n"
        )
    else:
        solution += (
            f"Step 5: Since the lump sum payment of ${lump_sum_payment:.2f} exceeds the remaining loan balance of ${remaining_loan_balance:.2f},\n"
            f"  the loan is completely paid off. The interest with early payoff is $0.\n\n"
        )
    
    solution += (
        f"Step 6: Calculate the interest saved by early payoff:\n"
        f"  Interest Saved = Interest Without Early Payoff - Interest With Early Payoff\n"
        f"                 = ${interest_without_early_payoff:.2f} - ${interest_with_early_payoff:.2f}\n"
        f"                 = ${interest_saved:.2f}"
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
        template_loan_repayment_simple,
        template_loan_repayment_monthly_installments,
        template_loan_repayment_with_extra_payments,
        template_loan_repayment_with_refinancing,
        template_loan_repayment_early_payoff
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
    output_file = "../../testset/personal_finance/loanrepay.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == '__main__':
    main()