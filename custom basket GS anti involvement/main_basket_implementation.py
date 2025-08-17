"""
China Anti-Involution Custom Basket Index - Main Implementation

This script brings together all components to construct the complete basket index,
demonstrating the methodology and generating comprehensive analysis reports.
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple
import logging

# Add project modules to path
sys.path.append('basket_design')
sys.path.append('data')

from basket_design.basket_construction import BasketConstructor, StockInfo, SectorCategory, OwnershipType
from basket_design.sector_allocation import SectorAllocator
from basket_design.stock_selection import StockSelector, StockCandidate, Exchange, MarketCapCategory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AntiInvolutionBasketImplementation:
    """Main implementation class for the China Anti-Involution basket"""
    
    def __init__(self):
        """Initialize the basket implementation"""
        self.sector_allocator = SectorAllocator()
        self.stock_selector = StockSelector()
        self.basket_constructor = BasketConstructor()
        
        # Load stock universe
        self.stock_universe = self._load_stock_universe()
        
        logger.info("Anti-Involution Basket Implementation initialized")
    
    def _load_stock_universe(self) -> List[StockCandidate]:
        """
        Load and process the stock universe from CSV
        
        Returns:
            List of StockCandidate objects
        """
        try:
            df = pd.read_csv('data/stock_universe.csv')
            logger.info(f"Loaded {len(df)} stocks from universe")
            
            stock_candidates = []
            
            for _, row in df.iterrows():
                # Map exchange string to enum
                exchange_map = {
                    'SSE': Exchange.SHANGHAI,
                    'SZSE': Exchange.SHENZHEN,
                    'HKEX': Exchange.HONG_KONG,
                    'US': Exchange.US_ADR
                }
                
                # Map market cap category
                market_cap_usd = row['market_cap_usd']
                if market_cap_usd >= 10e9:
                    market_cap_category = MarketCapCategory.LARGE_CAP
                elif market_cap_usd >= 2e9:
                    market_cap_category = MarketCapCategory.MID_CAP
                else:
                    market_cap_category = MarketCapCategory.SMALL_CAP
                
                candidate = StockCandidate(
                    ticker=row['ticker'],
                    name=row['name'],
                    name_english=row['name_english'],
                    sector=row['sector'],
                    exchange=exchange_map[row['exchange']],
                    market_cap_usd=market_cap_usd,
                    market_cap_category=market_cap_category,
                    ownership_type=row['ownership_type'],
                    revenue_usd=row['revenue_usd'],
                    revenue_growth_3y=row['revenue_growth_3y'],
                    profit_margin=row['profit_margin'],
                    roe=row['roe'],
                    debt_to_equity=row['debt_to_equity'],
                    avg_daily_volume_usd=row['avg_daily_volume_usd'],
                    price_volatility=row['price_volatility'],
                    beta=row['beta'],
                    market_share_domestic=row['market_share_domestic'],
                    pricing_power_score=row['pricing_power_score'],
                    consolidation_benefit_score=row['consolidation_benefit_score'],
                    innovation_score=row['innovation_score'],
                    government_relationship_score=row['government_relationship_score'],
                    esg_score=row['esg_score'],
                    carbon_intensity=row['carbon_intensity'],
                    analyst_coverage=row['analyst_coverage'],
                    consensus_rating=4.0  # Default rating
                )
                
                stock_candidates.append(candidate)
            
            return stock_candidates
            
        except Exception as e:
            logger.error(f"Error loading stock universe: {e}")
            return []
    
    def run_complete_analysis(self) -> Dict:
        """
        Run the complete basket construction and analysis
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting complete basket analysis...")
        
        results = {}
        
        # 1. Sector Allocation Analysis
        logger.info("Step 1: Analyzing sector allocations...")
        sector_allocations = self.sector_allocator.calculate_sector_allocations()
        sector_comparison = self.sector_allocator.compare_with_sssr_1()
        
        results['sector_analysis'] = {
            'allocations': sector_allocations,
            'sssr_comparison': sector_comparison,
            'allocation_report': self.sector_allocator.generate_allocation_report()
        }
        
        # 2. Stock Selection Process
        logger.info("Step 2: Screening and selecting stocks...")
        qualified_stocks = self.stock_selector.screen_stocks(self.stock_universe)
        sector_rankings = self.stock_selector.rank_stocks_by_sector(qualified_stocks)
        
        # Determine stocks per sector based on allocations
        stocks_per_sector = {}
        total_target_stocks = 30
        
        for sector, allocation in sector_allocations.items():
            # Allocate stocks proportionally, with minimum of 2 and maximum of 5 per sector
            base_stocks = max(2, min(5, int(allocation * total_target_stocks / 0.1)))
            stocks_per_sector[sector] = base_stocks
        
        selected_stocks = self.stock_selector.select_top_stocks_per_sector(
            sector_rankings, stocks_per_sector
        )
        
        results['stock_selection'] = {
            'qualified_count': len(qualified_stocks),
            'sector_rankings': sector_rankings,
            'selected_stocks': selected_stocks,
            'selection_report': self.stock_selector.generate_selection_report(selected_stocks)
        }
        
        # 3. Final Basket Construction
        logger.info("Step 3: Constructing final basket...")
        
        # Convert selected stocks to StockInfo format for basket constructor
        basket_stock_universe = []
        for sector, stocks in selected_stocks.items():
            for stock_candidate, score in stocks:
                # Map sector string to SectorCategory enum
                sector_map = {
                    'Electric Vehicles': SectorCategory.ELECTRIC_VEHICLES,
                    'Semiconductors': SectorCategory.SEMICONDUCTORS,
                    'Solar Energy': SectorCategory.SOLAR_ENERGY,
                    'Lithium/Batteries': SectorCategory.LITHIUM_BATTERIES,
                    'Steel': SectorCategory.STEEL,
                    'Coal Mining': SectorCategory.COAL_MINING,
                    'Chemicals': SectorCategory.CHEMICALS,
                    'Wind Energy': SectorCategory.WIND_ENERGY,
                    'Logistics': SectorCategory.LOGISTICS,
                    'Food Delivery': SectorCategory.FOOD_DELIVERY
                }
                
                # Map ownership type to enum
                ownership_map = {
                    'SOE': OwnershipType.SOE,
                    'POE': OwnershipType.POE,
                    'Mixed': OwnershipType.MIXED
                }
                
                stock_info = StockInfo(
                    ticker=stock_candidate.ticker,
                    name=stock_candidate.name_english,
                    sector=sector_map[sector],
                    ownership_type=ownership_map[stock_candidate.ownership_type],
                    market_cap=stock_candidate.market_cap_usd,
                    liquidity_score=min(100, stock_candidate.avg_daily_volume_usd / 1e6),  # Scale to 0-100
                    anti_involution_score=score,
                    innovation_score=stock_candidate.innovation_score,
                    consolidation_potential=stock_candidate.consolidation_benefit_score
                )
                
                basket_stock_universe.append(stock_info)
        
        # Construct final basket
        basket_summary = self.basket_constructor.construct_basket(basket_stock_universe)
        basket_report = self.basket_constructor.generate_basket_report(basket_summary)
        
        results['final_basket'] = {
            'summary': basket_summary,
            'report': basket_report
        }
        
        # 4. Generate comprehensive analysis
        logger.info("Step 4: Generating comprehensive analysis...")
        comprehensive_report = self._generate_comprehensive_report(results)
        results['comprehensive_report'] = comprehensive_report
        
        logger.info("Complete basket analysis finished")
        return results
    
    def _generate_comprehensive_report(self, results: Dict) -> str:
        """
        Generate a comprehensive analysis report
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Formatted comprehensive report
        """
        report = []
        report.append("=" * 100)
        report.append("CHINA ANTI-INVOLUTION CUSTOM BASKET INDEX - COMPREHENSIVE ANALYSIS")
        report.append("=" * 100)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 50)
        
        final_basket = results['final_basket']['summary']
        total_stocks = final_basket['total_stocks']
        
        report.append(f"• Total Stocks Selected: {total_stocks}")
        report.append(f"• Construction Date: {final_basket['construction_date']}")
        report.append(f"• Methodology: {final_basket['methodology']}")
        
        # Count ownership distribution
        ownership_dist = {}
        for stock, weight in final_basket['stocks']:
            ownership_type = stock.ownership_type.value
            ownership_dist[ownership_type] = ownership_dist.get(ownership_type, 0) + 1
        
        report.append(f"• Ownership Distribution:")
        for ownership, count in ownership_dist.items():
            percentage = count / total_stocks * 100
            report.append(f"  - {ownership}: {count} stocks ({percentage:.1f}%)")
        
        report.append("")
        
        # Key Differences from SSSR 1.0
        report.append("KEY DIFFERENCES FROM SSSR 1.0 APPROACH")
        report.append("-" * 50)
        
        sssr_comparison = results['sector_analysis']['sssr_comparison']
        high_tech_shift = sssr_comparison['total_shift_to_hightech']
        
        report.append(f"• Total shift to high-tech sectors: {high_tech_shift:+.1%}")
        report.append("• Focus evolution:")
        report.append("  - SSSR 1.0: Traditional heavy industries (coal, steel)")
        report.append("  - Anti-Involution: High-tech manufacturing and services")
        report.append("• Ownership approach:")
        report.append("  - SSSR 1.0: SOE-dominated capacity reduction")
        report.append("  - Anti-Involution: Mixed POE/SOE market-driven consolidation")
        report.append("")
        
        # Investment Thesis
        report.append("INVESTMENT THESIS")
        report.append("-" * 50)
        report.append("1. POLICY ALIGNMENT")
        report.append("   • Direct beneficiaries of anti-involution measures")
        report.append("   • Government support for industry consolidation")
        report.append("   • Pricing discipline enforcement")
        report.append("")
        report.append("2. MARKET LEADERSHIP")
        report.append("   • Market leaders with pricing power")
        report.append("   • Consolidation advantages")
        report.append("   • Innovation capabilities")
        report.append("")
        report.append("3. STRUCTURAL TRANSFORMATION")
        report.append("   • Shift from price competition to value creation")
        report.append("   • Technology advancement focus")
        report.append("   • Sustainable profitability")
        report.append("")
        
        # Risk Factors
        report.append("RISK FACTORS")
        report.append("-" * 50)
        report.append("• Voluntary compliance risk in private sector")
        report.append("• Policy implementation uncertainty")
        report.append("• Global trade and technology tensions")
        report.append("• Market volatility in high-tech sectors")
        report.append("• Execution risk in industry consolidation")
        report.append("")
        
        # Detailed Reports
        report.append("DETAILED ANALYSIS REPORTS")
        report.append("-" * 50)
        report.append("")
        
        # Sector allocation report
        report.append(results['sector_analysis']['allocation_report'])
        report.append("")
        
        # Stock selection report
        report.append(results['stock_selection']['selection_report'])
        report.append("")
        
        # Final basket report
        report.append(results['final_basket']['report'])
        
        return "\n".join(report)
    
    def save_results(self, results: Dict, output_dir: str = "output"):
        """
        Save analysis results to files
        
        Args:
            results: Analysis results dictionary
            output_dir: Output directory path
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comprehensive report
        with open(f"{output_dir}/comprehensive_analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(results['comprehensive_report'])
        
        # Save sector allocation report
        with open(f"{output_dir}/sector_allocation_report.txt", "w", encoding="utf-8") as f:
            f.write(results['sector_analysis']['allocation_report'])
        
        # Save stock selection report
        with open(f"{output_dir}/stock_selection_report.txt", "w", encoding="utf-8") as f:
            f.write(results['stock_selection']['selection_report'])
        
        # Save final basket report
        with open(f"{output_dir}/final_basket_report.txt", "w", encoding="utf-8") as f:
            f.write(results['final_basket']['report'])
        
        # Save basket composition as CSV
        basket_data = []
        for stock, weight in results['final_basket']['summary']['stocks']:
            basket_data.append({
                'ticker': stock.ticker,
                'name': stock.name,
                'sector': stock.sector.value,
                'ownership_type': stock.ownership_type.value,
                'weight': weight,
                'market_cap_usd': stock.market_cap,
                'anti_involution_score': stock.anti_involution_score,
                'innovation_score': stock.innovation_score
            })
        
        basket_df = pd.DataFrame(basket_data)
        basket_df.to_csv(f"{output_dir}/basket_composition.csv", index=False)
        
        logger.info(f"Results saved to {output_dir}/")

def main():
    """Main execution function"""
    print("China Anti-Involution Custom Basket Index Implementation")
    print("=" * 60)
    
    # Initialize implementation
    implementation = AntiInvolutionBasketImplementation()
    
    # Run complete analysis
    results = implementation.run_complete_analysis()
    
    # Save results
    implementation.save_results(results)
    
    # Print summary
    print("\nANALYSIS COMPLETE")
    print("-" * 30)
    print(f"Total stocks in final basket: {results['final_basket']['summary']['total_stocks']}")
    print(f"Qualified stocks from universe: {results['stock_selection']['qualified_count']}")
    print("Reports saved to output/ directory")
    
    # Print key highlights
    print("\nKEY HIGHLIGHTS:")
    sssr_comparison = results['sector_analysis']['sssr_comparison']
    print(f"• Shift to high-tech sectors: {sssr_comparison['total_shift_to_hightech']:+.1%}")
    
    final_basket = results['final_basket']['summary']
    ownership_dist = {}
    for stock, weight in final_basket['stocks']:
        ownership_type = stock.ownership_type.value
        ownership_dist[ownership_type] = ownership_dist.get(ownership_type, 0) + 1
    
    print("• Ownership distribution:")
    for ownership, count in ownership_dist.items():
        percentage = count / final_basket['total_stocks'] * 100
        print(f"  - {ownership}: {count} stocks ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
