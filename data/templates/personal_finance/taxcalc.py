import random

# Named entities for companies and industries
company_names = ["Tesla Inc.", "Apple Inc.", "Amazon.com", "SpaceX", "Google LLC"]
industry_names = ["automotive", "technology", "e-commerce", "aerospace", "internet services"]

# Template 1
def template_income_tax_simple():
    """1:Basic: Creates a simple income tax calculation problem where an individual needs 
    to determine their tax liability based on a single tax rate applied to their annual 
    income."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    annual_income = round(random.randint(500000, 3000000),2)  # Annual income
    tax_rate = round(random.uniform(5.0, 30.0),2)  # Tax rate (%)
    tax_due = round(annual_income * (tax_rate / 100),2)
    
    question = (
        f"{person_name} has an annual income of ${annual_income}. They fall under the {tax_rate:.2f}% tax bracket. "
        f"How much income tax will they owe this year?"
    )
    
    # Step 1: Calculate the tax due based on the income
    solution = (
        f"Step 1: Calculate the tax due:\n"
        f"  Tax Due = ${annual_income} × {tax_rate:.2f}% = ${tax_due:.2f}"
    )
    
    return question, solution

# Template 2
def template_tax_with_deductions():
    """2:Basic: Generates a tax calculation problem that incorporates tax deductions, 
    requiring calculation of taxable income after applying eligible deductions before 
    determining the final tax liability."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    age = random.randint(25, 55)
    annual_income = round(random.randint(1000000, 5000000),2)  # Annual income
    deductions = round(random.randint(50000, 250000),2)  # Tax deductions
    tax_rate = round(random.uniform(10.0, 30.0),2)  # Tax rate (%)
    
    taxable_income = round((annual_income - deductions),2)
    tax_due = round((taxable_income * (tax_rate / 100)),2)
    
    question = (
        f"{person_name}, aged {age}, earns ${annual_income} per year. They are eligible for deductions amounting to ${deductions} under tax-saving schemes. "
        f"If they fall under a {tax_rate:.2f}% tax bracket, how much income tax will they owe after accounting for the deductions?"
    )
    
    # Step 1: Calculate the taxable income after deductions
    # Step 2: Calculate the tax due based on the reduced taxable income
    solution = (
        f"Step 1: Calculate the taxable income after deductions:\n"
        f"  Taxable Income = ${annual_income} - ${deductions} = ${taxable_income:.2f}\n\n"
        f"Step 2: Calculate the tax due on the taxable income:\n"
        f"  Tax Due = ${taxable_income} × {tax_rate:.2f}% = ${tax_due:.2f}"
    )
    
    return question, solution

# Template 3
def template_capital_gains_tax():
    """3:Intermediate: Creates a capital gains tax scenario where an investor must calculate 
    their tax liability on profits from selling shares, considering purchase price, sale 
    price, and applicable capital gains tax rate."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    company = random.choice(company_names)
    current_age = random.randint(30, 55)
    sale_price = round(random.randint(500000, 2000000),2)  # Sale price of the shares
    purchase_price = round(random.randint(300000, 1000000),2)  # Purchase price of the shares
    capital_gains_tax_rate = round(random.uniform(10.0, 20.0),2)  # Capital gains tax rate (%)
    
    capital_gain = round(sale_price - purchase_price,2)
    tax_due = round((capital_gain * (capital_gains_tax_rate / 100)),2)
    
    question = (
        f"{person_name}, aged {current_age}, sold their shares in {company} for ${sale_price}. They bought the shares 5 years ago for ${purchase_price}. "
        f"Assuming a capital gains tax rate of {capital_gains_tax_rate:.2f}%, what is their capital gains tax liability?"
    )
    
    # Step 1: Calculate the capital gain
    # Step 2: Calculate the tax due on the capital gain
    solution = (
        f"Step 1: Calculate the capital gain:\n"
        f"  Capital Gain = ${sale_price} - ${purchase_price} = ${capital_gain:.2f}\n\n"
        f"Step 2: Calculate the capital gains tax:\n"
        f"  Tax Due = ${capital_gain} × {capital_gains_tax_rate:.2f}% = ${tax_due:.2f}"
    )
    
    return question, solution

# Template 4
def template_rental_income_tax():
    """4:Intermediate: Generates a rental income tax problem that factors in both rental income 
    and property depreciation, requiring calculation of taxable rental income and 
    resulting tax liability."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    current_age = random.randint(35, 65)
    annual_rental_income = round(random.randint(500000, 2000000),2)  # Annual rental income
    depreciation = round(random.randint(100000, 500000),2)  # Depreciation on the property
    tax_rate = round(random.uniform(15.0, 30.0),2)  # Tax rate (%)
    
    taxable_rental_income = round((annual_rental_income - depreciation),2)
    tax_due = round((taxable_rental_income * (tax_rate / 100)),2)
    
    question = (
        f"{person_name}, aged {current_age}, owns a rental property that generates ${annual_rental_income} in annual rental income. "
        f"They also claim ${depreciation} in depreciation on the property. If they fall under the {tax_rate:.2f}% tax bracket, how much tax will they owe on their rental income after accounting for depreciation?"
    )
    
    # Step 1: Calculate the taxable rental income after depreciation
    # Step 2: Calculate the tax due on the taxable rental income
    solution = (
        f"Step 1: Calculate the taxable rental income after depreciation:\n"
        f"  Taxable Rental Income = ${annual_rental_income} - ${depreciation} = ${taxable_rental_income:.2f}\n\n"
        f"Step 2: Calculate the tax due:\n"
        f"  Tax Due = ${taxable_rental_income} × {tax_rate:.2f}% = ${tax_due:.2f}"
    )
    
    return question, solution

# Template 5
def template_double_taxation_relief():
    """5:Advanced: Creates a complex international taxation scenario involving foreign income, 
    multiple tax jurisdictions, and calculation of double taxation relief to determine 
    net tax liability across countries."""
    person_name = random.choice(["John", "Aisha", "Ravi", "Sara", "David"])
    foreign_country = random.choice(["UAE", "USA", "UK", "Canada", "Australia"])
    current_age = random.randint(30, 60)
    foreign_income = round(random.randint(1000000, 5000000),2)  # Foreign income
    home_country_tax_rate = round(random.uniform(15.0, 30.0),2)  # Home country tax rate (%)
    foreign_country_tax_rate = round(random.uniform(5.0, 20.0),2)  # Foreign country tax rate (%)
    
    home_country_tax_due = round((foreign_income * (home_country_tax_rate / 100)),2)
    foreign_country_tax_paid = round((foreign_income * (foreign_country_tax_rate / 100)),2)
    double_taxation_relief = round(min(home_country_tax_due, foreign_country_tax_paid),2)
    net_tax_liability = round((home_country_tax_due - double_taxation_relief),2)
    
    question = (
        f"{person_name}, aged {current_age}, works in {foreign_country} and earns ${foreign_income} in foreign income. "
        f"Their home country taxes foreign income at {home_country_tax_rate:.2f}%, but they also pay {foreign_country_tax_rate:.2f}% tax in {foreign_country}. "
        f"How much double taxation relief will they be eligible for, and what will be their net tax liability?"
    )
    
    # Step 1: Calculate home country tax due on foreign income
    # Step 2: Calculate foreign tax paid
    # Step 3: Calculate the eligible double taxation relief
    # Step 4: Calculate the net tax liability after relief
    solution = (
        f"Step 1: Calculate the tax due in the home country:\n"
        f"  Home Country Tax Due = ${foreign_income} × {home_country_tax_rate:.2f}% = ${home_country_tax_due:.2f}\n\n"
        f"Step 2: Calculate the foreign tax paid:\n"
        f"  Foreign Country Tax Paid = ${foreign_income} × {foreign_country_tax_rate:.2f}% = ${foreign_country_tax_paid:.2f}\n\n"
        f"Step 3: Calculate the double taxation relief:\n"
        f"  Double Taxation Relief = min(${home_country_tax_due:.2f}, ${foreign_country_tax_paid:.2f}) = ${double_taxation_relief:.2f}\n\n"
        f"Step 4: Calculate the net tax liability after relief:\n"
        f"  Net Tax Liability = ${home_country_tax_due:.2f} - ${double_taxation_relief:.2f} = ${net_tax_liability:.2f}"
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
        template_income_tax_simple,
        template_tax_with_deductions,
        template_capital_gains_tax,
        template_rental_income_tax,
        template_double_taxation_relief
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
    output_file = "../../testset/personal_finance/taxcalc.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == '__main__':
    main()