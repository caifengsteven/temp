# China Anti-Involution Custom Basket Index

## Project Overview

This project designs and implements a custom basket index similar to Goldman Sachs' "GS China Anti Involution [GSXACINV]" basket, focusing on China's supply-side reform beneficiaries and anti-involution strategy.

## Basket Concept

**Theme**: China's supply-side reform beneficiaries and anti-involution strategy
**Target Sectors**: Electric Vehicles (EVs), Semiconductors, Chemicals, Coal Mining, Steel, Solar Energy, and Logistics
**Extended Focus Areas**: Wind Energy, Batteries/Lithium, and Food Delivery

## Key Differences from Previous SSSR 1.0 (2016-2018)

1. **Broader sector coverage** beyond the original SOE-dominated sectors
2. **Inclusion of both Private Owned Enterprises (POE) and State Owned Enterprises (SOE)** market leaders
3. **Focus on high-tech manufacturing and consumption** rather than broad-based easing
4. **Moderate easing outlook** vs. the significant easing seen in 2015-2018

## Anti-Involution Context

Based on research, China's "anti-involution" policy launched in July 2025 aims to:
- Combat deflationary price wars in key sectors
- Address overcapacity in industries like EVs, solar panels, steel, lithium batteries
- Promote industry consolidation and pricing discipline
- Shift from excessive competition to sustainable profitability

## Historical Context: SSSR 1.0 vs Current Anti-Involution

### SSSR 1.0 (2016-2018)
- **Focus**: Cutting excess industrial capacity, reducing corporate leverage
- **Target Sectors**: Coal, steel, cement, glass (traditional heavy industries)
- **Approach**: SOE-dominated capacity reduction, zombie firm elimination
- **Policy Environment**: Significant monetary easing, infrastructure stimulus

### Current Anti-Involution (2025+)
- **Focus**: Preventing price wars, promoting industry consolidation
- **Target Sectors**: EVs, semiconductors, solar, lithium, food delivery (high-tech + services)
- **Approach**: Both POE and SOE coordination, voluntary pricing discipline
- **Policy Environment**: Moderate easing, innovation-driven growth

## Project Structure

```
├── README.md                          # Project overview and documentation
├── research/                          # Research and analysis files
│   ├── anti_involution_analysis.md    # Anti-involution policy analysis
│   ├── sssr_comparison.md             # SSSR 1.0 vs current comparison
│   └── existing_baskets_research.md   # Similar basket research
├── basket_design/                     # Basket structure and methodology
│   ├── sector_allocation.py           # Sector weighting methodology
│   ├── stock_selection.py             # Stock selection criteria
│   └── basket_construction.py         # Overall basket construction
├── data/                              # Data sources and Bloomberg integration
│   ├── bloomberg_integration.py       # Bloomberg API integration
│   ├── stock_universe.csv             # Universe of eligible stocks
│   └── sector_data.csv                # Sector-specific data
├── analysis/                          # Analysis and validation
│   ├── performance_analysis.py        # Historical performance analysis
│   ├── risk_metrics.py                # Risk analysis and metrics
│   └── comparison_analysis.py         # Comparison with benchmarks
├── documentation/                     # Detailed documentation
│   ├── methodology.md                 # Detailed methodology
│   ├── rationale.md                   # Investment rationale
│   └── implementation_guide.md        # Implementation guidelines
└── tests/                             # Testing and validation
    ├── test_basket_construction.py    # Unit tests
    └── test_data_integration.py       # Data validation tests
```

## Next Steps

1. ✅ Research and Analysis Phase
2. 🔄 Basket Structure Design
3. ⏳ Stock Selection and Analysis
4. ⏳ Implementation and Validation
5. ⏳ Documentation and Reporting

## Key Deliverables

1. Custom basket structure with appropriate sector weightings
2. Representative stock selection for each sector (POE and SOE leaders)
3. Rationale for stock selection and weighting methodology
4. Comparison with historical SSSR 1.0 approach
5. Bloomberg integration for data and validation

## Contact

For questions or collaboration, please refer to the documentation in the `documentation/` folder.
