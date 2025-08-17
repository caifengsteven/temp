"""
China Anti-Involution Custom Basket Index Construction

This module implements the overall basket construction methodology for the
China Anti-Involution custom basket index, incorporating sector allocation,
stock selection, and weighting methodologies.
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

class OwnershipType(Enum):
    """Ownership classification for Chinese companies"""
    SOE = "State Owned Enterprise"
    POE = "Private Owned Enterprise"
    MIXED = "Mixed Ownership"

class SectorCategory(Enum):
    """Sector categories for anti-involution basket"""
    # Primary Anti-Involution Sectors
    ELECTRIC_VEHICLES = "Electric Vehicles"
    SEMICONDUCTORS = "Semiconductors"
    SOLAR_ENERGY = "Solar Energy"
    LITHIUM_BATTERIES = "Lithium/Batteries"
    
    # Traditional SSSR Sectors
    STEEL = "Steel"
    COAL_MINING = "Coal Mining"
    CHEMICALS = "Chemicals"
    
    # Extended Focus Areas
    WIND_ENERGY = "Wind Energy"
    LOGISTICS = "Logistics"
    FOOD_DELIVERY = "Food Delivery"

@dataclass
class StockInfo:
    """Stock information for basket construction"""
    ticker: str
    name: str
    sector: SectorCategory
    ownership_type: OwnershipType
    market_cap: float
    liquidity_score: float
    anti_involution_score: float  # Custom score for anti-involution relevance
    innovation_score: float
    consolidation_potential: float
    
class BasketConstructor:
    """Main class for constructing the China Anti-Involution basket"""
    
    def __init__(self, 
                 target_stocks: int = 30,
                 min_sector_weight: float = 0.05,
                 max_sector_weight: float = 0.25,
                 min_stock_weight: float = 0.01,
                 max_stock_weight: float = 0.08):
        """
        Initialize basket constructor with constraints
        
        Args:
            target_stocks: Target number of stocks in basket
            min_sector_weight: Minimum sector allocation
            max_sector_weight: Maximum sector allocation
            min_stock_weight: Minimum individual stock weight
            max_stock_weight: Maximum individual stock weight
        """
        self.target_stocks = target_stocks
        self.min_sector_weight = min_sector_weight
        self.max_sector_weight = max_sector_weight
        self.min_stock_weight = min_stock_weight
        self.max_stock_weight = max_stock_weight
        
        # Sector allocation based on anti-involution policy priorities
        self.sector_allocations = self._define_sector_allocations()
        
    def _define_sector_allocations(self) -> Dict[SectorCategory, float]:
        """
        Define sector allocations based on anti-involution policy priorities
        
        Returns:
            Dictionary mapping sectors to target allocations
        """
        return {
            # Primary Anti-Involution Sectors (60% total)
            SectorCategory.ELECTRIC_VEHICLES: 0.18,      # Highest priority
            SectorCategory.SEMICONDUCTORS: 0.15,         # Strategic importance
            SectorCategory.SOLAR_ENERGY: 0.12,           # Overcapacity focus
            SectorCategory.LITHIUM_BATTERIES: 0.15,      # EV supply chain
            
            # Traditional SSSR Sectors (25% total)
            SectorCategory.STEEL: 0.10,                  # Continued consolidation
            SectorCategory.COAL_MINING: 0.08,            # Transition management
            SectorCategory.CHEMICALS: 0.07,              # Industrial base
            
            # Extended Focus Areas (15% total)
            SectorCategory.WIND_ENERGY: 0.06,            # Green transition
            SectorCategory.LOGISTICS: 0.05,              # Infrastructure
            SectorCategory.FOOD_DELIVERY: 0.04,          # Service sector
        }
    
    def calculate_anti_involution_score(self, stock: StockInfo) -> float:
        """
        Calculate anti-involution relevance score for a stock
        
        Args:
            stock: Stock information
            
        Returns:
            Anti-involution score (0-100)
        """
        # Base score by sector priority
        sector_scores = {
            SectorCategory.ELECTRIC_VEHICLES: 95,
            SectorCategory.SEMICONDUCTORS: 90,
            SectorCategory.SOLAR_ENERGY: 85,
            SectorCategory.LITHIUM_BATTERIES: 90,
            SectorCategory.STEEL: 75,
            SectorCategory.COAL_MINING: 70,
            SectorCategory.CHEMICALS: 65,
            SectorCategory.WIND_ENERGY: 80,
            SectorCategory.LOGISTICS: 60,
            SectorCategory.FOOD_DELIVERY: 85,
        }
        
        base_score = sector_scores.get(stock.sector, 50)
        
        # Adjustments based on company characteristics
        # Market leadership bonus (larger companies benefit more from consolidation)
        market_cap_bonus = min(10, stock.market_cap / 1e11 * 5)  # Up to 10 points
        
        # Innovation bonus (aligned with policy goals)
        innovation_bonus = stock.innovation_score * 0.1  # Up to 10 points
        
        # Consolidation potential bonus
        consolidation_bonus = stock.consolidation_potential * 0.05  # Up to 5 points
        
        # Ownership type adjustment
        ownership_adjustment = {
            OwnershipType.SOE: 2,    # Slight preference for policy coordination
            OwnershipType.POE: 0,    # Neutral
            OwnershipType.MIXED: 1   # Slight bonus for mixed ownership
        }.get(stock.ownership_type, 0)
        
        total_score = (base_score + market_cap_bonus + innovation_bonus + 
                      consolidation_bonus + ownership_adjustment)
        
        return min(100, max(0, total_score))
    
    def select_stocks_by_sector(self, 
                               stock_universe: List[StockInfo], 
                               sector: SectorCategory,
                               target_allocation: float) -> List[Tuple[StockInfo, float]]:
        """
        Select stocks for a specific sector
        
        Args:
            stock_universe: Available stocks
            sector: Target sector
            target_allocation: Target sector allocation
            
        Returns:
            List of (stock, weight) tuples
        """
        # Filter stocks by sector
        sector_stocks = [s for s in stock_universe if s.sector == sector]
        
        if not sector_stocks:
            logger.warning(f"No stocks found for sector {sector}")
            return []
        
        # Calculate scores and sort
        scored_stocks = []
        for stock in sector_stocks:
            score = self.calculate_anti_involution_score(stock)
            scored_stocks.append((stock, score))
        
        # Sort by score (descending)
        scored_stocks.sort(key=lambda x: x[1], reverse=True)
        
        # Select top stocks for sector (aim for 2-4 stocks per sector)
        target_stocks_per_sector = max(2, min(4, int(target_allocation * self.target_stocks / 0.1)))
        selected_stocks = scored_stocks[:target_stocks_per_sector]
        
        # Calculate weights within sector
        if not selected_stocks:
            return []
        
        # Weight by market cap and anti-involution score
        weights = []
        for stock, score in selected_stocks:
            # Combine market cap and score for weighting
            weight_factor = (stock.market_cap ** 0.5) * (score / 100)
            weights.append(weight_factor)
        
        # Normalize weights to sector allocation
        total_weight = sum(weights)
        if total_weight == 0:
            return []
        
        normalized_weights = [(w / total_weight) * target_allocation for w in weights]
        
        # Apply individual stock weight constraints
        final_weights = []
        for i, weight in enumerate(normalized_weights):
            constrained_weight = max(self.min_stock_weight, 
                                   min(self.max_stock_weight, weight))
            final_weights.append(constrained_weight)
        
        # Return stock-weight pairs
        result = []
        for i, (stock, _) in enumerate(selected_stocks):
            result.append((stock, final_weights[i]))
        
        return result
    
    def construct_basket(self, stock_universe: List[StockInfo]) -> Dict:
        """
        Construct the complete basket
        
        Args:
            stock_universe: Available stocks for selection
            
        Returns:
            Dictionary containing basket composition and metadata
        """
        logger.info("Starting basket construction...")
        
        basket_stocks = []
        sector_weights = {}
        
        # Select stocks for each sector
        for sector, target_allocation in self.sector_allocations.items():
            logger.info(f"Processing sector: {sector.value} (target: {target_allocation:.1%})")
            
            sector_selections = self.select_stocks_by_sector(
                stock_universe, sector, target_allocation
            )
            
            if sector_selections:
                basket_stocks.extend(sector_selections)
                actual_weight = sum(weight for _, weight in sector_selections)
                sector_weights[sector] = actual_weight
                
                logger.info(f"Selected {len(sector_selections)} stocks for {sector.value}")
            else:
                logger.warning(f"No stocks selected for {sector.value}")
        
        # Normalize total weights to 100%
        total_weight = sum(weight for _, weight in basket_stocks)
        if total_weight > 0:
            basket_stocks = [(stock, weight / total_weight) 
                           for stock, weight in basket_stocks]
        
        # Create basket summary
        basket_summary = {
            'stocks': basket_stocks,
            'sector_weights': sector_weights,
            'total_stocks': len(basket_stocks),
            'methodology': 'Anti-Involution Policy-Based Selection',
            'construction_date': pd.Timestamp.now(),
            'constraints': {
                'target_stocks': self.target_stocks,
                'min_sector_weight': self.min_sector_weight,
                'max_sector_weight': self.max_sector_weight,
                'min_stock_weight': self.min_stock_weight,
                'max_stock_weight': self.max_stock_weight
            }
        }
        
        logger.info(f"Basket construction complete: {len(basket_stocks)} stocks selected")
        
        return basket_summary
    
    def generate_basket_report(self, basket_summary: Dict) -> str:
        """
        Generate a detailed report of the basket composition
        
        Args:
            basket_summary: Basket construction results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("CHINA ANTI-INVOLUTION CUSTOM BASKET INDEX")
        report.append("=" * 60)
        report.append(f"Construction Date: {basket_summary['construction_date']}")
        report.append(f"Total Stocks: {basket_summary['total_stocks']}")
        report.append(f"Methodology: {basket_summary['methodology']}")
        report.append("")
        
        # Sector allocation summary
        report.append("SECTOR ALLOCATION:")
        report.append("-" * 30)
        for sector, weight in basket_summary['sector_weights'].items():
            report.append(f"{sector.value:<25} {weight:>8.1%}")
        report.append("")
        
        # Stock details by sector
        report.append("STOCK COMPOSITION:")
        report.append("-" * 30)
        
        # Group stocks by sector
        stocks_by_sector = {}
        for stock, weight in basket_summary['stocks']:
            if stock.sector not in stocks_by_sector:
                stocks_by_sector[stock.sector] = []
            stocks_by_sector[stock.sector].append((stock, weight))
        
        for sector in SectorCategory:
            if sector in stocks_by_sector:
                report.append(f"\n{sector.value}:")
                for stock, weight in stocks_by_sector[sector]:
                    report.append(f"  {stock.ticker:<10} {stock.name:<30} {weight:>8.1%} ({stock.ownership_type.value})")
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # This would be replaced with actual data loading
    print("China Anti-Involution Basket Constructor initialized")
    print("Ready for stock universe input and basket construction")
