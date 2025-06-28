import random

"""
Crypto Regulation Reasoning Question Generator

Generates categorized questions (Basic, Intermediate, Advanced) and detailed step-by-step solutions
for reasoning around crypto finance regulation scenarios.

Each function includes randomization for variability and descriptive docstrings for clarity.
"""

# ========================
# 🟢 Basic Level
# ========================

def basic_kyc_threshold_limit():
    """
    1:Basic: Know Your Customer (KYC) threshold check.
    
    Generates a question about whether a transaction exceeds the KYC-required threshold
    and returns a reasoning chain based on the transaction amount.
    
    Returns:
        tuple: (question_text, solution_text)
    """
    limit = random.randint(1000, 5000)
    amount = random.randint(500, 3500)
    question = (
        f"A crypto exchange operating under international KYC guidelines requires user identification for transactions over ${limit}. "
        f"A customer initiates a transaction of ${amount}. Does this require KYC verification?"
    )
    solution = (
        f"Step 1:\n"
        f"  The KYC threshold is ${limit}.\n"
        f"Step 2:\n"
        f"  The transaction is for ${amount}.\n"
        f"Step 3:\n"
        f"  {'KYC is required' if amount > limit else 'KYC is not required'} because ${amount} "
        f"{'exceeds' if amount > limit else 'is below'} the threshold."
    )
    return question, solution

def basic_mica_stablecoin_capital():
    """
    2:Basic: Stablecoin capital requirement under MiCA.
    
    Calculates whether a stablecoin issuer meets the higher of fixed or percentage-based
    capital reserve requirements under MiCA.
    
    Returns:
        tuple: (question_text, solution_text)
    """
    capital = random.randint(100000, 500000)
    percent = random.uniform(1.5, 2.5)  # Percentage for reserves
    reserves = random.randint(1_000_000, 10_000_000)
    required = max(capital, percent / 100 * reserves)
    question = (
        f"Under the EU’s MiCA framework, a stablecoin issuer must hold capital reserves of €{capital} "
        f"or {percent:.2f}% of average reserve assets, whichever is higher. If their reserves are €{reserves}, "
        "what is the required capital reserve?"
    )
    solution = (
        f"Step 1:\n"
        f"  {percent:.2f}% of €{reserves} = €{percent / 100 * reserves:.2f}\n"
        f"Step 2:\n"
        f"  Compare with €{capital}: max({capital}, {percent / 100 * reserves:.2f}) = €{required:.2f}\n"
        f"Step 3:\n"
        f"  The required capital reserve is €{required:.2f}."
    )
    return question, solution

# ========================
# 🟡 Intermediate Level - All enhanced by Claude 3.7 Sonet
# ========================

def intermediate_cross_border_restrictions():
    """
    3:Intermediate: Legal considerations for cross-border exchange expansion.
    
    Explores the specific steps a crypto exchange must take before launching in different jurisdictions
    with varying crypto regulatory frameworks.
    
    Returns:
        tuple: (question_text, solution_text)
    """
    # Dictionary of jurisdictions with their specific regulatory requirements
    jurisdictions = {
        "China": {
            "status": "highly restrictive",
            "license": "Not available - crypto exchanges banned",
            "key_regulator": "People's Bank of China (PBOC)",
            "special_considerations": "All cryptocurrency trading and mining activities are banned"
        },
        "India": {
            "status": "evolving framework",
            "license": "Registration with Financial Intelligence Unit (FIU)",
            "key_regulator": "Securities and Exchange Board of India (SEBI)",
            "special_considerations": "30% tax on crypto income; 1% TDS on all transactions"
        },
        "Germany": {
            "status": "regulated market",
            "license": "BaFin crypto custody license required",
            "key_regulator": "Federal Financial Supervisory Authority (BaFin)",
            "special_considerations": "Must comply with MiCA framework as an EU member state"
        },
        "Canada": {
            "status": "regulated market",
            "license": "Registration as Money Service Business (MSB) with FINTRAC",
            "key_regulator": "Financial Transactions and Reports Analysis Centre (FINTRAC)",
            "special_considerations": "Provincial securities regulations may also apply"
        },
        "Singapore": {
            "status": "regulated market",
            "license": "Digital Payment Token (DPT) Service license from MAS",
            "key_regulator": "Monetary Authority of Singapore (MAS)",
            "special_considerations": "Strict customer due diligence and transaction monitoring required"
        },
        "Bulgaria": {
            "status": "emerging framework",
            "license": "Registration with National Revenue Agency",
            "key_regulator": "Financial Supervision Commission (FSC)",
            "special_considerations": "Must comply with MiCA framework as an EU member state"
        }
    }
    
    # Randomly select a jurisdiction
    jurisdiction = random.choice(list(jurisdictions.keys()))
    reg_info = jurisdictions[jurisdiction]
    
    question = (
        f"A crypto exchange is planning to expand operations to {jurisdiction}. "
        f"What regulatory steps must be considered for legal operation in this specific jurisdiction?"
    )
    
    # Create a jurisdiction-specific solution
    if reg_info["status"] == "highly restrictive" and jurisdiction == "China":
        solution = (
            f"Step 1:\n"
            f"  {jurisdiction} has a {reg_info['status']} stance on cryptocurrencies.\n"
            f"Step 2:\n"
            f"  Key finding: {reg_info['special_considerations']}.\n"
            f"Step 3:\n"
            f"  The {reg_info['key_regulator']} enforces a complete ban on crypto exchanges.\n"
            f"Conclusion:\n"
            f"  Legal operation as a crypto exchange is not possible in {jurisdiction}. "
            f"The company should consider alternative jurisdictions or business models."
        )
    else:
        solution = (
            f"Step 1:\n"
            f"  {jurisdiction} has a {reg_info['status']} for cryptocurrency businesses.\n"
            f"Step 2:\n"
            f"  Licensing requirement: {reg_info['license']}.\n"
            f"Step 3:\n"
            f"  Must register with and follow guidelines from the {reg_info['key_regulator']}.\n"
            f"Step 4:\n"
            f"  Special consideration: {reg_info['special_considerations']}.\n"
            f"Conclusion:\n"
            f"  To operate legally in {jurisdiction}, the exchange must obtain the required license, "
            f"implement comprehensive AML/KYC procedures, and ensure compliance with all "
            f"jurisdiction-specific regulations."
        )
    
    return question, solution

def intermediate_token_classification():
    """
    4:Intermediate: SEC compliance for token launches.
    
    Tests knowledge of how different token types impact SEC registration
    and classification under the Howey Test with specific compliance strategies.
    
    Returns:
        tuple: (question_text, solution_text)
    """
    # Define token types with specific characteristics and compliance approaches
    token_types = {
        "utility": {
            "primary_purpose": "provide access to a product or service",
            "howey_risk": "moderate",
            "key_considerations": [
                "Token must have genuine utility at launch",
                "Avoid marketing focused on investment returns",
                "Ensure decentralized governance where possible"
            ],
            "compliance_strategy": "Consider SEC no-action letter; implement KYC/AML; document utility purpose"
        },
        "security": {
            "primary_purpose": "represent investment contract or financial interest",
            "howey_risk": "high",
            "key_considerations": [
                "Meets Howey Test criteria",
                "Involves expectation of profits from others' efforts",
                "Requires registration or exemption"
            ],
            "compliance_strategy": "Register using Form S-1, Regulation A+, or seek exemption through Regulation D/S"
        },
        "governance": {
            "primary_purpose": "grant voting rights in a protocol or DAO",
            "howey_risk": "variable",
            "key_considerations": [
                "Focus on governance functionality",
                "Minimize profit expectations in marketing",
                "Establish genuine voting utility"
            ],
            "compliance_strategy": "Legal opinion letter; robust governance documentation; possible safe harbor"
        },
        "stablecoin": {
            "primary_purpose": "maintain stable value relative to fiat",
            "howey_risk": "depends on design",
            "key_considerations": [
                "Algorithmic vs. asset-backed structure matters",
                "Reserve management and transparency",
                "Money transmission licensing"
            ],
            "compliance_strategy": "Money Services Business registration; compliance with state-level regulations"
        },
        "NFT": {
            "primary_purpose": "represent unique digital asset",
            "howey_risk": "depends on fractionalization and marketing",
            "key_considerations": [
                "Avoid fractionalization that creates investment characteristics",
                "Focus on collectible/artistic value",
                "Be cautious with royalty structures"
            ],
            "compliance_strategy": "IP rights documentation; avoid investment pool characteristics"
        }
    }
    
    # Randomly select a token type
    token_type = random.choice(list(token_types.keys()))
    token_info = token_types[token_type]
    
    # Create scenario details
    use_case = random.choice([
        "decentralized finance platform",
        "gaming ecosystem",
        "content creation marketplace",
        "supply chain tracking solution",
        "data storage network"
    ])
    
    question = (
        f"A startup is launching a {token_type} token for their {use_case} in the US. "
        f"How should they approach SEC compliance based on this token classification?"
    )
    
    # Detailed Howey Test analysis based on token type
    howey_analysis = ""
    if token_type == "security":
        howey_analysis = (
            f"  a. Investment of money: Yes - purchasers pay with currency or crypto.\n"
            f"  b. Common enterprise: Yes - fortunes tied to issuer's efforts.\n"
            f"  c. Expectation of profits: Yes - marketed with focus on value appreciation.\n"
            f"  d. Efforts of others: Yes - team's development drives token value."
        )
    elif token_type == "utility":
        howey_analysis = (
            f"  a. Investment of money: Yes - purchasers pay with currency or crypto.\n"
            f"  b. Common enterprise: Maybe - depends on token design.\n"
            f"  c. Expectation of profits: Maybe - if utility is functional at launch, this prong may not be met.\n"
            f"  d. Efforts of others: Maybe - if token has genuine utility, user participation matters more."
        )
    elif token_type == "governance":
        howey_analysis = (
            f"  a. Investment of money: Yes - purchasers pay with currency or crypto.\n"
            f"  b. Common enterprise: Maybe - depends on DAO/protocol structure.\n"
            f"  c. Expectation of profits: Maybe - if governance rights are primary focus, less likely.\n"
            f"  d. Efforts of others: Maybe - decentralized governance reduces reliance on team efforts."
        )
    elif token_type == "stablecoin":
        howey_analysis = (
            f"  a. Investment of money: Yes - purchasers pay with currency or crypto.\n"
            f"  b. Common enterprise: Depends - asset-backed differs from algorithmic.\n"
            f"  c. Expectation of profits: No - designed specifically for price stability, not appreciation.\n"
            f"  d. Efforts of others: Depends - mechanism for maintaining stability matters."
        )
    else:  # NFT
        howey_analysis = (
            f"  a. Investment of money: Yes - purchasers pay with currency or crypto.\n"
            f"  b. Common enterprise: No - each NFT is unique and independent.\n"
            f"  c. Expectation of profits: Maybe - depends on marketing and purpose.\n"
            f"  d. Efforts of others: Maybe - if presented as collectible rather than investment."
        )
    
    # Create detailed solution
    solution = (
        f"Step 1: Identify token characteristics\n"
        f"  The {token_type} token's primary purpose is to {token_info['primary_purpose']}.\n"
        f"  This token type has {token_info['howey_risk']} risk under the Howey Test.\n\n"
        f"Step 2: Apply the Howey Test\n"
        f"{howey_analysis}\n\n"
        f"Step 3: Consider key compliance factors\n"
        f"  1. {token_info['key_considerations'][0]}.\n"
        f"  2. {token_info['key_considerations'][1]}.\n"
        f"  3. {token_info['key_considerations'][2]}.\n\n"
        f"Step 4: Implement appropriate compliance strategy\n"
        f"  {token_info['compliance_strategy']}.\n\n"
        f"Conclusion:\n"
        f"  For this {token_type} token in a {use_case}, the startup should "
        f"{'register with the SEC or seek exemption' if token_info['howey_risk'] == 'high' else 'focus on demonstrating genuine utility/functionality while consulting securities counsel'}. "
        f"Proper legal classification and compliance from the outset will mitigate regulatory risk."
    )
    
    return question, solution

# ========================
# 🔴 Advanced Level - All enhanced by Claude 3.7 Sonet
# ========================

def advanced_tax_reporting_multi_country():
    """
    5:Advanced: Multi-jurisdictional tax reporting for crypto investors.
    
    Explores specific tax and reporting requirements when investors trade in one country
    while residing in another, with country-specific tax treatments and reporting obligations.
    
    Returns:
        tuple: (question_text, solution_text)
    """
    # Define country-specific tax information
    countries = {
        "US": {
            "tax_authority": "Internal Revenue Service (IRS)",
            "crypto_classification": "property (capital asset)",
            "tax_rates": "Short-term gains (ordinary income): 10-37%; Long-term gains: 0-20%",
            "reporting_forms": ["Form 8949", "Schedule D", "FBAR (if foreign exchange balance >$10,000)", "Form 8938 (FATCA)"],
            "unique_features": "Wash sale rules don't explicitly apply to crypto; PFIC rules may apply to certain tokens",
            "residency_basis": "Citizenship and residency-based taxation (worldwide income)",
            "exchange_reporting": "Form 1099-B/1099-K from some exchanges; mandatory reporting expanding"
        },
        "UK": {
            "tax_authority": "Her Majesty's Revenue and Customs (HMRC)",
            "crypto_classification": "capital asset (sometimes income)",
            "tax_rates": "Capital Gains: 10-20% depending on income band; Income: 20-45%",
            "reporting_forms": ["Self Assessment Tax Return", "Capital Gains Summary"],
            "unique_features": "£12,300 annual CGT allowance; specific DeFi lending guidance",
            "residency_basis": "Residency-based taxation with domicile considerations",
            "exchange_reporting": "Limited exchange reporting; expanding under OECD Crypto-Asset Reporting Framework"
        },
        "Germany": {
            "tax_authority": "Bundeszentralamt für Steuern (BZSt)",
            "crypto_classification": "private asset",
            "tax_rates": "Capital Gains: 25% + solidarity surcharge; Tax-free if held >1 year",
            "reporting_forms": ["Annual Tax Return", "Anlage SO (other income)"],
            "unique_features": "Tax-free after 1 year holding period; staking may extend to 10 years",
            "residency_basis": "Residency-based taxation (worldwide income)",
            "exchange_reporting": "Limited but expanding under EU DAC8 directive"
        },
        "Japan": {
            "tax_authority": "National Tax Agency (NTA)",
            "crypto_classification": "miscellaneous income",
            "tax_rates": "Progressive rates from 15-55% (including local taxes)",
            "reporting_forms": ["Final Tax Return (確定申告書)"],
            "unique_features": "No separate capital gains rate for crypto; aggregated with other income",
            "residency_basis": "Residency-based taxation with non-permanent resident exception",
            "exchange_reporting": "Annual reporting by licensed exchanges to NTA"
        },
        "Australia": {
            "tax_authority": "Australian Taxation Office (ATO)",
            "crypto_classification": "capital asset or trading stock (intent-based)",
            "tax_rates": "Capital Gains: marginal rates with 50% discount if held >12 months",
            "reporting_forms": ["myTax capital gains schedule", "Business schedule if trading"],
            "unique_features": "Personal use asset exemption for purchases <$10,000; detailed DeFi guidance",
            "residency_basis": "Residency-based taxation (worldwide income)",
            "exchange_reporting": "Data matching program with exchanges; expanding under OECD framework"
        },
        "Singapore": {
            "tax_authority": "Inland Revenue Authority of Singapore (IRAS)",
            "crypto_classification": "capital asset (exempt) or revenue (taxable) based on intent",
            "tax_rates": "No capital gains tax; Income tax: 0-22%",
            "reporting_forms": ["Form B/B1 for business income"],
            "unique_features": "Capital gains generally not taxable; intention-based assessment",
            "residency_basis": "Territorial basis (Singapore-sourced income)",
            "exchange_reporting": "Licensed exchanges report under MAS requirements"
        }
    }
    
    # Add more countries to the list
    all_countries = list(countries.keys()) + ["Switzerland", "Portugal", "Canada", "South Korea", "UAE"]
    
    # Select two different countries
    country1, country2 = random.sample(all_countries, 2)
    
    # Make sure detailed information is available, or use generic information
    country1_info = countries.get(country1, {
        "tax_authority": "local tax authority",
        "crypto_classification": "varies by jurisdiction",
        "reporting_forms": ["local tax returns"],
        "unique_features": "jurisdiction-specific treatment",
        "residency_basis": "typically residency-based taxation",
        "exchange_reporting": "expanding global requirements"
    })
    
    country2_info = countries.get(country2, {
        "tax_authority": "local tax authority",
        "crypto_classification": "varies by jurisdiction",
        "reporting_forms": ["local tax returns"],
        "unique_features": "jurisdiction-specific treatment",
        "residency_basis": "typically residency-based taxation",
        "exchange_reporting": "expanding global requirements"
    })
    
    # Add scenario details
    activity_type = random.choice([
        "DeFi yield farming",
        "NFT trading",
        "crypto mining",
        "staking rewards",
        "margin trading",
        "liquidity provision"
    ])
    
    question = (
        f"A crypto investor who is a tax resident of {country1} actively engages in {activity_type} "
        f"on exchanges and platforms based in {country2}. What specific tax and reporting "
        f"challenges arise, and how should they approach compliance in both jurisdictions?"
    )
    
    # Create detailed solution
    solution = (
        f"Step 1: Determine primary tax jurisdiction and characterization\n"
        f"  • Primary tax obligation is to {country1} as the country of tax residence.\n"
        f"  • In {country1}, crypto is typically classified as {country1_info.get('crypto_classification', 'subject to local classification')}.\n"
        f"  • {country1} operates on a {country1_info.get('residency_basis', 'residency-based system')}.\n"
        f"  • For {activity_type}, specific considerations include: {random.choice([
            'calculating cost basis across multiple transactions and platforms',
            'determining whether activity constitutes personal investment or business activity',
            'applying appropriate holding period rules for preferential tax rates',
            'documenting on-chain transactions without traditional receipts'
        ])}.\n\n"
        
        f"Step 2: Identify {country2} reporting obligations\n"
        f"  • {country2} exchanges may have reporting requirements to their local {country2_info.get('tax_authority', 'tax authorities')}.\n"
        f"  • Exchange reporting status: {country2_info.get('exchange_reporting', 'Subject to local regulations')}.\n"
        f"  • Non-resident obligations may include: {random.choice([
            'withholding taxes on certain income types',
            'registration requirements for high-volume trading',
            'VAT/GST considerations for specific services',
            'special reporting for high-value transactions'
        ])}.\n\n"
        
        f"Step 3: Analyze cross-border compliance issues\n"
        f"  • Information exchange: {random.choice([
            f"FATCA applies if U.S. citizen; CRS applies between {country1} and {country2}",
            f"OECD Crypto-Asset Reporting Framework may apply between {country1} and {country2}",
            f"Tax information exchange agreements exist between {country1} and {country2}",
            f"Limited automatic exchange of information between {country1} and {country2}"
        ])}.\n"
        f"  • Double taxation risks: {random.choice([
            f"Tax treaty between {country1} and {country2} may provide relief mechanisms",
            f"Foreign tax credits may be available in {country1} for taxes paid in {country2}",
            f"No comprehensive tax treaty exists, creating potential double taxation",
            f"Special provisions for digital assets may be absent from existing treaties"
        ])}.\n"
        f"  • For {activity_type}, additional considerations include: {random.choice([
            'timing differences in recognition of taxable events',
            'different classification of the same crypto transactions',
            'conflicting sourcing rules for income determination',
            'varying treatment of losses and deductions'
        ])}.\n\n"
        
        f"Step 4: Required reporting documentation\n"
        f"  • {country1} reporting: {', '.join(country1_info.get('reporting_forms', ['Local tax returns'])[:2])}.\n"
        f"  • Foreign account/asset reporting: {random.choice([
            f"Required if combined value exceeds thresholds",
            f"FBAR and/or Form 8938 if U.S. person",
            f"Foreign investment reporting forms",
            f"Declaration of foreign accounts/assets"
        ])}.\n"
        f"  • Documentation requirements: {random.choice([
            'Complete transaction history with timestamps and values in fiat',
            'Proof of wallet ownership and control',
            'Cost basis calculations with supporting evidence',
            'Segregation of personal vs. business activity'
        ])}.\n\n"
        
        f"Conclusion:\n"
        f"  An investor involved in {activity_type} while being tax resident in {country1} with "
        f"activity on {country2}-based platforms faces complex compliance requirements. "
        f"They should: (1) maintain comprehensive transaction records with fiat values at time "
        f"of transactions; (2) understand the specific treatment of {activity_type} in both "
        f"jurisdictions; (3) identify applicable tax treaties or foreign tax credit provisions; "
        f"(4) engage tax professionals with specific expertise in both jurisdictions and crypto "
        f"taxation; and (5) consider tax compliance technology solutions for cross-border crypto "
        f"activity. As the regulatory landscape evolves, particularly with the implementation of "
        f"the OECD Crypto-Asset Reporting Framework, proactive compliance becomes increasingly important."
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
        basic_kyc_threshold_limit,
        basic_mica_stablecoin_capital,
        intermediate_cross_border_restrictions,
        intermediate_token_classification,
        advanced_tax_reporting_multi_country
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
    output_file = "../../testset/crypto_finance/crypto_regulations.jsonl"
    with open(output_file, "w") as file:
        for problem in all_problems:
            file.write(json.dumps(problem))
            file.write("\n")
    
    print(f"Successfully generated {len(all_problems)} problems and saved to {output_file}")

if __name__ == "__main__":
   main()