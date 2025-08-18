"""
HSTECH Index Components and ADR Mappings

This file contains the HSTECH index components with their weights and ADR mappings.
Data compiled from various sources as of August 2024.

Note: The HSTECH index consists of the 30 largest technology companies listed in Hong Kong
that have high business exposure to technology themes.
"""

from typing import Dict, List
from src.models import Stock, ADRMapping

# HSTECH Index Components (Top 30 technology stocks)
# Weights are approximate and should be updated with real-time data
HSTECH_COMPONENTS = [
    # Top 10 Holdings (approximately 60-70% of index)
    Stock(
        symbol="0700.HK",
        name="Tencent Holdings Ltd",
        weight=0.085,  # Approximately 8.5%
        sector="Internet & Direct Marketing Retail",
        market_cap=3500000000000  # ~3.5T HKD
    ),
    Stock(
        symbol="9988.HK", 
        name="Alibaba Group Holding Ltd",
        weight=0.075,  # Approximately 7.5%
        sector="Internet & Direct Marketing Retail",
        market_cap=1800000000000  # ~1.8T HKD
    ),
    Stock(
        symbol="3690.HK",
        name="Meituan",
        weight=0.065,  # Approximately 6.5%
        sector="Internet & Direct Marketing Retail",
        market_cap=750000000000  # ~750B HKD
    ),
    Stock(
        symbol="9618.HK",
        name="JD.com Inc",
        weight=0.055,  # Approximately 5.5%
        sector="Internet & Direct Marketing Retail", 
        market_cap=600000000000  # ~600B HKD
    ),
    Stock(
        symbol="1810.HK",
        name="Xiaomi Corp",
        weight=0.045,  # Approximately 4.5%
        sector="Technology Hardware & Equipment",
        market_cap=400000000000  # ~400B HKD
    ),
    Stock(
        symbol="1211.HK",
        name="BYD Company Ltd",
        weight=0.040,  # Approximately 4.0%
        sector="Automobiles",
        market_cap=800000000000  # ~800B HKD
    ),
    Stock(
        symbol="2269.HK",
        name="Wuxi Biologics Cayman Inc",
        weight=0.035,  # Approximately 3.5%
        sector="Biotechnology",
        market_cap=300000000000  # ~300B HKD
    ),
    Stock(
        symbol="2382.HK",
        name="Sunny Optical Technology Group Co Ltd",
        weight=0.030,  # Approximately 3.0%
        sector="Technology Hardware & Equipment",
        market_cap=250000000000  # ~250B HKD
    ),
    Stock(
        symbol="9999.HK",
        name="NetEase Inc",
        weight=0.028,  # Approximately 2.8%
        sector="Entertainment",
        market_cap=350000000000  # ~350B HKD
    ),
    Stock(
        symbol="1024.HK",
        name="Kuaishou Technology",
        weight=0.025,  # Approximately 2.5%
        sector="Internet & Direct Marketing Retail",
        market_cap=200000000000  # ~200B HKD
    ),
    
    # Additional Major Components (next 20 stocks)
    Stock(
        symbol="9961.HK",
        name="Trip.com Group Ltd",
        weight=0.022,
        sector="Internet & Direct Marketing Retail",
        market_cap=180000000000
    ),
    Stock(
        symbol="1833.HK", 
        name="PA Gooddoctor Technology Ltd",
        weight=0.020,
        sector="Health Care Technology",
        market_cap=120000000000
    ),
    Stock(
        symbol="6618.HK",
        name="JD Health International Inc",
        weight=0.018,
        sector="Health Care Technology", 
        market_cap=150000000000
    ),
    Stock(
        symbol="2015.HK",
        name="Li Auto Inc",
        weight=0.016,
        sector="Automobiles",
        market_cap=140000000000
    ),
    Stock(
        symbol="9868.HK",
        name="Xpeng Inc",
        weight=0.015,
        sector="Automobiles",
        market_cap=80000000000
    ),
    # Additional components would be added here...
    # Note: The remaining 15 components typically have smaller weights (0.5-1.5% each)
]

# ADR Mappings for HSTECH components that are dual-listed in the US
ADR_MAPPINGS = [
    ADRMapping(
        hk_symbol="0700.HK",
        us_symbol="TCEHY",
        conversion_ratio=5.0,  # 5 HK shares per 1 ADR
        currency_base="HKD",
        currency_quote="USD"
    ),
    ADRMapping(
        hk_symbol="9988.HK", 
        us_symbol="BABA",
        conversion_ratio=8.0,  # 8 HK shares per 1 ADR
        currency_base="HKD",
        currency_quote="USD"
    ),
    ADRMapping(
        hk_symbol="3690.HK",
        us_symbol="MPNGY", 
        conversion_ratio=10.0,  # 10 HK shares per 1 ADR
        currency_base="HKD",
        currency_quote="USD"
    ),
    ADRMapping(
        hk_symbol="9618.HK",
        us_symbol="JD",
        conversion_ratio=2.0,  # 2 HK shares per 1 ADR
        currency_base="HKD", 
        currency_quote="USD"
    ),
    ADRMapping(
        hk_symbol="1810.HK",
        us_symbol="XIACF",  # OTC ADR
        conversion_ratio=10.0,  # 10 HK shares per 1 ADR
        currency_base="HKD",
        currency_quote="USD"
    ),
    ADRMapping(
        hk_symbol="1211.HK",
        us_symbol="BYDDY",
        conversion_ratio=10.0,  # 10 HK shares per 1 ADR
        currency_base="HKD",
        currency_quote="USD"
    ),
    ADRMapping(
        hk_symbol="9999.HK",
        us_symbol="NTES",
        conversion_ratio=25.0,  # 25 HK shares per 1 ADR
        currency_base="HKD",
        currency_quote="USD"
    ),
    ADRMapping(
        hk_symbol="9961.HK",
        us_symbol="TCOM",
        conversion_ratio=8.0,  # 8 HK shares per 1 ADR
        currency_base="HKD",
        currency_quote="USD"
    ),
    ADRMapping(
        hk_symbol="2015.HK",
        us_symbol="LI",
        conversion_ratio=2.0,  # 2 HK shares per 1 ADR
        currency_base="HKD",
        currency_quote="USD"
    ),
    ADRMapping(
        hk_symbol="9868.HK",
        us_symbol="XPEV",
        conversion_ratio=2.0,  # 2 HK shares per 1 ADR
        currency_base="HKD",
        currency_quote="USD"
    ),
]

# Additional US market indicators for enhanced estimation
MARKET_INDICATORS = [
    "PDD",    # PDD Inc. (Chinese e-commerce, correlates with Chinese tech)
    "KWEB",   # KraneShares CSI China Internet ETF
    "ASHR",   # Xtrackers Harvest CSI 300 China A-Shares ETF
    "FXI",    # iShares China Large-Cap ETF
    "MCHI",   # iShares MSCI China ETF
]

def get_adr_mapped_components() -> List[Stock]:
    """Get HSTECH components that have US ADR listings."""
    adr_symbols = {mapping.hk_symbol for mapping in ADR_MAPPINGS}
    return [stock for stock in HSTECH_COMPONENTS if stock.symbol in adr_symbols]

def get_non_adr_components() -> List[Stock]:
    """Get HSTECH components that do NOT have US ADR listings."""
    adr_symbols = {mapping.hk_symbol for mapping in ADR_MAPPINGS}
    return [stock for stock in HSTECH_COMPONENTS if stock.symbol not in adr_symbols]

def get_adr_mapping(hk_symbol: str) -> ADRMapping:
    """Get ADR mapping for a given Hong Kong symbol."""
    for mapping in ADR_MAPPINGS:
        if mapping.hk_symbol == hk_symbol:
            return mapping
    raise ValueError(f"No ADR mapping found for {hk_symbol}")

def calculate_adr_coverage() -> float:
    """Calculate the percentage of HSTECH index covered by ADR-mapped stocks."""
    adr_mapped = get_adr_mapped_components()
    total_weight = sum(stock.weight for stock in adr_mapped)
    return total_weight

# Summary statistics
if __name__ == "__main__":
    print(f"Total HSTECH components: {len(HSTECH_COMPONENTS)}")
    print(f"Components with ADR mappings: {len(get_adr_mapped_components())}")
    print(f"Components without ADR mappings: {len(get_non_adr_components())}")
    print(f"ADR coverage of index: {calculate_adr_coverage():.1%}")
