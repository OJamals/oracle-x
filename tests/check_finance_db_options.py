#!/usr/bin/env python3
"""
Quick check of available sectors and categories in FinanceDatabase
"""

try:
    import financedatabase as fd

    print("=== Available Options in FinanceDatabase ===")

    # Check equities sectors
    print("\n1. Available Equity Sectors:")
    equities = fd.Equities()
    sectors = equities.show_options(selection="sector")
    for i, sector in enumerate(sectors):
        print(f"   {i+1:2d}. {sector}")

    # Check ETF categories
    print("\n2. Available ETF Category Groups:")
    etfs = fd.ETFs()
    categories = etfs.show_options(selection="category_group")
    for i, category in enumerate(categories):
        print(f"   {i+1:2d}. {category}")

    # Check market cap options
    print("\n3. Available Market Cap Groups:")
    market_caps = equities.show_options(selection="market_cap")
    for i, cap in enumerate(market_caps):
        print(f"   {i+1:2d}. {cap}")

    print("\n=== Sample Search with Correct Sector ===")

    # Try with correct sector name
    tech_companies = equities.select(
        country="United States",
        sector="Information Technology",  # This is the correct name
        market_cap="Large Cap",
    )

    print(f"\nUS Large Cap Information Technology companies: {len(tech_companies)}")
    if not tech_companies.empty:
        print("Top 5 companies:")
        for i, (symbol, row) in enumerate(tech_companies.head(5).iterrows()):
            print(f"   {i+1}. {symbol}: {row.get('name', 'N/A')}")

except Exception as e:
    print(f"Error: {e}")
