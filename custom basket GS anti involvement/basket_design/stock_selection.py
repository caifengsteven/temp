"""
Stock Selection Methodology for China Anti-Involution Basket

This module implements the stock selection criteria and methodology for identifying
representative companies in each sector, balancing POE and SOE leaders with
anti-involution policy alignment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Exchange(Enum):
    """Stock exchange classifications"""
    SHANGHAI = "SSE"
    SHENZHEN = "SZSE"
    HONG_KONG = "HKEX"
    US_ADR = "US"

class MarketCapCategory(Enum):
    """Market capitalization categories"""
    LARGE_CAP = "Large Cap"      # >$10B USD
    MID_CAP = "Mid Cap"          # $2B-$10B USD
    SMALL_CAP = "Small Cap"      # <$2B USD

@dataclass
class StockCandidate:
    """Comprehensive stock candidate information"""
    ticker: str
    name: str
    name_english: str
    sector: str
    exchange: Exchange
    market_cap_usd: float
    market_cap_category: MarketCapCategory
    ownership_type: str  # SOE, POE, Mixed
    
    # Financial metrics
    revenue_usd: float
    revenue_growth_3y: float
    profit_margin: float
    roe: float
    debt_to_equity: float
    
    # Market metrics
    avg_daily_volume_usd: float
    price_volatility: float
    beta: float
    
    # Anti-involution specific metrics
    market_share_domestic: float
    pricing_power_score: float  # 0-100
    consolidation_benefit_score: float  # 0-100
    innovation_score: float  # 0-100
    government_relationship_score: float  # 0-100
    
    # ESG and sustainability
    esg_score: float
    carbon_intensity: float
    
    # Analyst coverage
    analyst_coverage: int
    consensus_rating: float  # 1-5 scale

class StockSelector:
    """Stock selection methodology implementation"""
    
    def __init__(self):
        """Initialize stock selector with selection criteria"""
        self.selection_criteria = self._define_selection_criteria()
        self.sector_specific_criteria = self._define_sector_specific_criteria()
        
    def _define_selection_criteria(self) -> Dict:
        """
        Define general stock selection criteria
        
        Returns:
            Dictionary of selection criteria and thresholds
        """
        return {
            # Liquidity requirements
            'min_market_cap_usd': 1e9,  # $1B minimum
            'min_avg_daily_volume_usd': 5e6,  # $5M daily volume
            'max_price_volatility': 0.8,  # 80% annualized volatility
            
            # Financial health
            'min_revenue_usd': 5e8,  # $500M revenue
            'max_debt_to_equity': 3.0,  # 300% max D/E ratio
            'min_profit_margin': -0.1,  # Allow some losses for growth companies
            
            # Market position
            'min_market_share_domestic': 0.02,  # 2% domestic market share
            'min_analyst_coverage': 3,  # Minimum analyst coverage
            
            # Anti-involution alignment
            'min_consolidation_benefit_score': 40,  # 40/100 minimum
            'min_pricing_power_score': 30,  # 30/100 minimum
        }
    
    def _define_sector_specific_criteria(self) -> Dict:
        """
        Define sector-specific selection criteria
        
        Returns:
            Dictionary mapping sectors to specific criteria
        """
        return {
            "Electric Vehicles": {
                'min_innovation_score': 60,
                'preferred_ownership': ['POE', 'Mixed'],
                'key_metrics': ['ev_sales_volume', 'battery_technology_score'],
                'consolidation_focus': 'manufacturing_efficiency'
            },
            
            "Semiconductors": {
                'min_innovation_score': 70,
                'preferred_ownership': ['SOE', 'Mixed'],  # Strategic sector
                'key_metrics': ['r_and_d_intensity', 'patent_count'],
                'consolidation_focus': 'technology_leadership'
            },
            
            "Solar Energy": {
                'min_innovation_score': 50,
                'preferred_ownership': ['POE', 'SOE'],  # Both important
                'key_metrics': ['manufacturing_capacity', 'cost_competitiveness'],
                'consolidation_focus': 'capacity_utilization'
            },
            
            "Lithium/Batteries": {
                'min_innovation_score': 65,
                'preferred_ownership': ['POE', 'Mixed'],
                'key_metrics': ['battery_capacity', 'supply_chain_integration'],
                'consolidation_focus': 'supply_chain_control'
            },
            
            "Steel": {
                'min_innovation_score': 30,
                'preferred_ownership': ['SOE', 'Mixed'],  # Traditional SOE sector
                'key_metrics': ['production_capacity', 'environmental_compliance'],
                'consolidation_focus': 'operational_efficiency'
            },
            
            "Coal Mining": {
                'min_innovation_score': 20,
                'preferred_ownership': ['SOE'],  # Dominated by SOEs
                'key_metrics': ['reserves', 'safety_record'],
                'consolidation_focus': 'resource_optimization'
            },
            
            "Chemicals": {
                'min_innovation_score': 40,
                'preferred_ownership': ['POE', 'SOE'],
                'key_metrics': ['specialty_chemicals_ratio', 'downstream_integration'],
                'consolidation_focus': 'product_specialization'
            },
            
            "Wind Energy": {
                'min_innovation_score': 55,
                'preferred_ownership': ['POE', 'Mixed'],
                'key_metrics': ['turbine_technology', 'project_pipeline'],
                'consolidation_focus': 'technology_advancement'
            },
            
            "Logistics": {
                'min_innovation_score': 45,
                'preferred_ownership': ['POE', 'Mixed'],
                'key_metrics': ['network_coverage', 'digitalization_score'],
                'consolidation_focus': 'network_efficiency'
            },
            
            "Food Delivery": {
                'min_innovation_score': 60,
                'preferred_ownership': ['POE'],  # Private sector dominated
                'key_metrics': ['user_base', 'platform_efficiency'],
                'consolidation_focus': 'market_dominance'
            }
        }
    
    def calculate_anti_involution_alignment_score(self, stock: StockCandidate) -> float:
        """
        Calculate how well a stock aligns with anti-involution policy objectives
        
        Args:
            stock: Stock candidate information
            
        Returns:
            Alignment score (0-100)
        """
        # Base components
        components = {
            'pricing_power': stock.pricing_power_score * 0.25,
            'consolidation_benefit': stock.consolidation_benefit_score * 0.25,
            'market_position': min(100, stock.market_share_domestic * 1000) * 0.20,  # Scale up market share
            'innovation': stock.innovation_score * 0.15,
            'government_relationship': stock.government_relationship_score * 0.15
        }
        
        # Sector-specific adjustments
        sector_criteria = self.sector_specific_criteria.get(stock.sector, {})
        
        # Innovation bonus for high-tech sectors
        if sector_criteria.get('min_innovation_score', 0) >= 60:
            components['innovation'] *= 1.2  # 20% bonus for high-tech sectors
        
        # Ownership preference alignment
        preferred_ownership = sector_criteria.get('preferred_ownership', [])
        if stock.ownership_type in preferred_ownership:
            ownership_bonus = 5  # 5 point bonus
        else:
            ownership_bonus = 0
        
        # Market leadership bonus
        if stock.market_cap_category == MarketCapCategory.LARGE_CAP:
            market_leadership_bonus = 5
        elif stock.market_cap_category == MarketCapCategory.MID_CAP:
            market_leadership_bonus = 2
        else:
            market_leadership_bonus = 0
        
        # Financial health adjustment
        financial_health = self._calculate_financial_health_score(stock)
        financial_adjustment = (financial_health - 50) * 0.1  # -5 to +5 points
        
        total_score = (sum(components.values()) + ownership_bonus + 
                      market_leadership_bonus + financial_adjustment)
        
        return max(0, min(100, total_score))
    
    def _calculate_financial_health_score(self, stock: StockCandidate) -> float:
        """
        Calculate financial health score
        
        Args:
            stock: Stock candidate
            
        Returns:
            Financial health score (0-100)
        """
        # Profitability component
        profit_score = max(0, min(100, (stock.profit_margin + 0.1) * 500))  # Scale -10% to +10%
        
        # Growth component
        growth_score = max(0, min(100, (stock.revenue_growth_3y + 0.1) * 250))  # Scale -10% to +30%
        
        # Leverage component (inverse)
        leverage_score = max(0, min(100, 100 - (stock.debt_to_equity * 25)))  # Penalize high leverage
        
        # ROE component
        roe_score = max(0, min(100, stock.roe * 500))  # Scale 0% to 20%
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # Profit, Growth, Leverage, ROE
        scores = [profit_score, growth_score, leverage_score, roe_score]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def screen_stocks(self, candidates: List[StockCandidate]) -> List[StockCandidate]:
        """
        Screen stocks based on selection criteria
        
        Args:
            candidates: List of stock candidates
            
        Returns:
            Filtered list of qualifying stocks
        """
        qualified_stocks = []
        criteria = self.selection_criteria
        
        for stock in candidates:
            # Check general criteria
            if (stock.market_cap_usd >= criteria['min_market_cap_usd'] and
                stock.avg_daily_volume_usd >= criteria['min_avg_daily_volume_usd'] and
                stock.price_volatility <= criteria['max_price_volatility'] and
                stock.revenue_usd >= criteria['min_revenue_usd'] and
                stock.debt_to_equity <= criteria['max_debt_to_equity'] and
                stock.profit_margin >= criteria['min_profit_margin'] and
                stock.market_share_domestic >= criteria['min_market_share_domestic'] and
                stock.analyst_coverage >= criteria['min_analyst_coverage'] and
                stock.consolidation_benefit_score >= criteria['min_consolidation_benefit_score'] and
                stock.pricing_power_score >= criteria['min_pricing_power_score']):
                
                # Check sector-specific criteria
                sector_criteria = self.sector_specific_criteria.get(stock.sector, {})
                min_innovation = sector_criteria.get('min_innovation_score', 0)
                
                if stock.innovation_score >= min_innovation:
                    qualified_stocks.append(stock)
                    logger.info(f"Qualified: {stock.ticker} - {stock.name}")
                else:
                    logger.debug(f"Failed innovation criteria: {stock.ticker}")
            else:
                logger.debug(f"Failed general criteria: {stock.ticker}")
        
        logger.info(f"Screened {len(candidates)} candidates, {len(qualified_stocks)} qualified")
        return qualified_stocks
    
    def rank_stocks_by_sector(self, qualified_stocks: List[StockCandidate]) -> Dict[str, List[Tuple[StockCandidate, float]]]:
        """
        Rank stocks within each sector by anti-involution alignment
        
        Args:
            qualified_stocks: Pre-screened stock candidates
            
        Returns:
            Dictionary mapping sectors to ranked stock lists
        """
        sector_rankings = {}
        
        # Group stocks by sector
        stocks_by_sector = {}
        for stock in qualified_stocks:
            if stock.sector not in stocks_by_sector:
                stocks_by_sector[stock.sector] = []
            stocks_by_sector[stock.sector].append(stock)
        
        # Rank within each sector
        for sector, stocks in stocks_by_sector.items():
            ranked_stocks = []
            
            for stock in stocks:
                alignment_score = self.calculate_anti_involution_alignment_score(stock)
                ranked_stocks.append((stock, alignment_score))
            
            # Sort by alignment score (descending)
            ranked_stocks.sort(key=lambda x: x[1], reverse=True)
            sector_rankings[sector] = ranked_stocks
            
            logger.info(f"Ranked {len(ranked_stocks)} stocks in {sector}")
        
        return sector_rankings
    
    def select_top_stocks_per_sector(self, 
                                   sector_rankings: Dict[str, List[Tuple[StockCandidate, float]]],
                                   stocks_per_sector: Dict[str, int]) -> Dict[str, List[Tuple[StockCandidate, float]]]:
        """
        Select top stocks for each sector with ownership diversity
        
        Args:
            sector_rankings: Ranked stocks by sector
            stocks_per_sector: Target number of stocks per sector
            
        Returns:
            Selected stocks by sector
        """
        selected_stocks = {}
        
        for sector, ranked_stocks in sector_rankings.items():
            target_count = stocks_per_sector.get(sector, 3)  # Default 3 stocks per sector
            
            if not ranked_stocks:
                selected_stocks[sector] = []
                continue
            
            # Ensure ownership diversity if possible
            selected = []
            soe_count = 0
            poe_count = 0
            
            for stock, score in ranked_stocks:
                if len(selected) >= target_count:
                    break
                
                # Try to maintain ownership balance
                if stock.ownership_type == 'SOE':
                    if soe_count < target_count // 2 + 1:  # Allow slight SOE preference for strategic sectors
                        selected.append((stock, score))
                        soe_count += 1
                elif stock.ownership_type in ['POE', 'Mixed']:
                    if poe_count < target_count // 2 + 1:
                        selected.append((stock, score))
                        poe_count += 1
                
                # If we haven't filled quota and ownership balance is achieved, add best remaining
                if len(selected) < target_count and len(selected) < len(ranked_stocks):
                    remaining_slots = target_count - len(selected)
                    remaining_stocks = ranked_stocks[len(selected):len(selected) + remaining_slots]
                    for remaining_stock, remaining_score in remaining_stocks:
                        if (remaining_stock, remaining_score) not in selected:
                            selected.append((remaining_stock, remaining_score))
            
            selected_stocks[sector] = selected
            logger.info(f"Selected {len(selected)} stocks for {sector}")
            
            # Log ownership distribution
            ownership_dist = {}
            for stock, _ in selected:
                ownership_dist[stock.ownership_type] = ownership_dist.get(stock.ownership_type, 0) + 1
            logger.info(f"{sector} ownership distribution: {ownership_dist}")
        
        return selected_stocks
    
    def generate_selection_report(self, 
                                selected_stocks: Dict[str, List[Tuple[StockCandidate, float]]]) -> str:
        """
        Generate detailed stock selection report
        
        Args:
            selected_stocks: Final selected stocks by sector
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("CHINA ANTI-INVOLUTION BASKET - STOCK SELECTION REPORT")
        report.append("=" * 80)
        report.append("")
        
        total_stocks = sum(len(stocks) for stocks in selected_stocks.values())
        report.append(f"Total Selected Stocks: {total_stocks}")
        report.append("")
        
        # Detailed breakdown by sector
        for sector, stocks in selected_stocks.items():
            if not stocks:
                continue
                
            report.append(f"{sector.upper()}:")
            report.append("-" * 50)
            
            for i, (stock, score) in enumerate(stocks, 1):
                report.append(f"{i}. {stock.ticker} - {stock.name}")
                report.append(f"   Exchange: {stock.exchange.value}")
                report.append(f"   Market Cap: ${stock.market_cap_usd/1e9:.1f}B")
                report.append(f"   Ownership: {stock.ownership_type}")
                report.append(f"   Anti-Involution Score: {score:.1f}/100")
                report.append(f"   Market Share: {stock.market_share_domestic:.1%}")
                report.append(f"   Innovation Score: {stock.innovation_score}/100")
                report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append("-" * 30)
        
        all_stocks = [stock for stocks in selected_stocks.values() for stock, _ in stocks]
        
        # Ownership distribution
        ownership_dist = {}
        for stock in all_stocks:
            ownership_dist[stock.ownership_type] = ownership_dist.get(stock.ownership_type, 0) + 1
        
        report.append("Ownership Distribution:")
        for ownership, count in ownership_dist.items():
            percentage = count / len(all_stocks) * 100
            report.append(f"  {ownership}: {count} stocks ({percentage:.1f}%)")
        
        # Exchange distribution
        exchange_dist = {}
        for stock in all_stocks:
            exchange_dist[stock.exchange] = exchange_dist.get(stock.exchange, 0) + 1
        
        report.append("\nExchange Distribution:")
        for exchange, count in exchange_dist.items():
            percentage = count / len(all_stocks) * 100
            report.append(f"  {exchange.value}: {count} stocks ({percentage:.1f}%)")
        
        # Market cap distribution
        market_cap_dist = {}
        for stock in all_stocks:
            market_cap_dist[stock.market_cap_category] = market_cap_dist.get(stock.market_cap_category, 0) + 1
        
        report.append("\nMarket Cap Distribution:")
        for category, count in market_cap_dist.items():
            percentage = count / len(all_stocks) * 100
            report.append(f"  {category.value}: {count} stocks ({percentage:.1f}%)")
        
        return "\n".join(report)

# Example usage
if __name__ == "__main__":
    selector = StockSelector()
    print("Stock Selection Methodology initialized")
    print("Ready for candidate stock input and selection process")
