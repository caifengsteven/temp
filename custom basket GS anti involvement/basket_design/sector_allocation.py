"""
Sector Allocation Methodology for China Anti-Involution Basket

This module defines the sector allocation strategy based on anti-involution policy
priorities, comparing with historical SSSR 1.0 approach and incorporating
current market dynamics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

class PolicyPriority(Enum):
    """Policy priority levels for anti-involution sectors"""
    CRITICAL = "Critical"      # Primary policy targets
    HIGH = "High"             # Important secondary targets  
    MODERATE = "Moderate"     # Traditional sectors with continued focus
    EMERGING = "Emerging"     # New areas of focus

@dataclass
class SectorProfile:
    """Comprehensive sector profile for allocation decisions"""
    name: str
    policy_priority: PolicyPriority
    overcapacity_severity: float  # 0-100 scale
    consolidation_potential: float  # 0-100 scale
    innovation_intensity: float  # 0-100 scale
    government_support: float  # 0-100 scale
    market_maturity: float  # 0-100 scale
    global_competitiveness: float  # 0-100 scale
    sssr_1_involvement: bool  # Was this sector part of SSSR 1.0?

class SectorAllocator:
    """Sector allocation methodology implementation"""
    
    def __init__(self):
        """Initialize sector allocator with predefined sector profiles"""
        self.sector_profiles = self._define_sector_profiles()
        self.allocation_methodology = "Anti-Involution Policy-Weighted"
        
    def _define_sector_profiles(self) -> Dict[str, SectorProfile]:
        """
        Define comprehensive profiles for each sector
        
        Returns:
            Dictionary mapping sector names to their profiles
        """
        return {
            "Electric Vehicles": SectorProfile(
                name="Electric Vehicles",
                policy_priority=PolicyPriority.CRITICAL,
                overcapacity_severity=85,  # High overcapacity, price wars
                consolidation_potential=90,  # Strong consolidation expected
                innovation_intensity=95,  # High-tech, innovation-driven
                government_support=90,  # Strong policy support
                market_maturity=60,  # Still developing
                global_competitiveness=85,  # Strong global position
                sssr_1_involvement=False
            ),
            
            "Semiconductors": SectorProfile(
                name="Semiconductors",
                policy_priority=PolicyPriority.CRITICAL,
                overcapacity_severity=70,  # Moderate overcapacity
                consolidation_potential=85,  # Strategic consolidation
                innovation_intensity=100,  # Highest innovation priority
                government_support=95,  # Strategic national priority
                market_maturity=40,  # Early development stage
                global_competitiveness=60,  # Catching up globally
                sssr_1_involvement=False
            ),
            
            "Solar Energy": SectorProfile(
                name="Solar Energy",
                policy_priority=PolicyPriority.CRITICAL,
                overcapacity_severity=95,  # Severe overcapacity
                consolidation_potential=95,  # Major consolidation underway
                innovation_intensity=80,  # Technology advancement focus
                government_support=85,  # Green energy priority
                market_maturity=80,  # Mature technology
                global_competitiveness=95,  # Global leader
                sssr_1_involvement=False
            ),
            
            "Lithium/Batteries": SectorProfile(
                name="Lithium/Batteries",
                policy_priority=PolicyPriority.CRITICAL,
                overcapacity_severity=90,  # High overcapacity
                consolidation_potential=85,  # CATL mine closures
                innovation_intensity=85,  # Technology advancement
                government_support=85,  # EV supply chain support
                market_maturity=70,  # Developing rapidly
                global_competitiveness=90,  # Strong global position
                sssr_1_involvement=False
            ),
            
            "Steel": SectorProfile(
                name="Steel",
                policy_priority=PolicyPriority.MODERATE,
                overcapacity_severity=75,  # Ongoing capacity issues
                consolidation_potential=80,  # Continued consolidation
                innovation_intensity=50,  # Traditional industry
                government_support=70,  # Continued but reduced focus
                market_maturity=95,  # Mature industry
                global_competitiveness=85,  # Strong global position
                sssr_1_involvement=True  # Primary SSSR 1.0 target
            ),
            
            "Coal Mining": SectorProfile(
                name="Coal Mining",
                policy_priority=PolicyPriority.MODERATE,
                overcapacity_severity=70,  # Managed capacity reduction
                consolidation_potential=75,  # Ongoing consolidation
                innovation_intensity=30,  # Limited innovation
                government_support=60,  # Transition management
                market_maturity=100,  # Mature/declining
                global_competitiveness=80,  # Cost competitive
                sssr_1_involvement=True  # Primary SSSR 1.0 target
            ),
            
            "Chemicals": SectorProfile(
                name="Chemicals",
                policy_priority=PolicyPriority.MODERATE,
                overcapacity_severity=65,  # Moderate overcapacity
                consolidation_potential=70,  # Gradual consolidation
                innovation_intensity=60,  # Specialty chemicals innovation
                government_support=65,  # Industrial base support
                market_maturity=85,  # Mature industry
                global_competitiveness=75,  # Competitive position
                sssr_1_involvement=False
            ),
            
            "Wind Energy": SectorProfile(
                name="Wind Energy",
                policy_priority=PolicyPriority.HIGH,
                overcapacity_severity=60,  # Moderate overcapacity
                consolidation_potential=70,  # Some consolidation expected
                innovation_intensity=75,  # Technology advancement
                government_support=80,  # Green energy priority
                market_maturity=75,  # Maturing technology
                global_competitiveness=85,  # Strong global position
                sssr_1_involvement=False
            ),
            
            "Logistics": SectorProfile(
                name="Logistics",
                policy_priority=PolicyPriority.EMERGING,
                overcapacity_severity=40,  # Limited overcapacity
                consolidation_potential=60,  # Market-driven consolidation
                innovation_intensity=70,  # Digital logistics innovation
                government_support=60,  # Infrastructure support
                market_maturity=60,  # Developing sector
                global_competitiveness=70,  # Growing competitiveness
                sssr_1_involvement=False
            ),
            
            "Food Delivery": SectorProfile(
                name="Food Delivery",
                policy_priority=PolicyPriority.HIGH,
                overcapacity_severity=80,  # Price war issues
                consolidation_potential=85,  # Platform consolidation
                innovation_intensity=75,  # Technology-driven
                government_support=70,  # Service sector focus
                market_maturity=50,  # Still developing
                global_competitiveness=60,  # Domestic focus
                sssr_1_involvement=False
            )
        }
    
    def calculate_sector_score(self, sector_profile: SectorProfile) -> float:
        """
        Calculate composite score for sector allocation
        
        Args:
            sector_profile: Sector profile data
            
        Returns:
            Composite score (0-100)
        """
        # Policy priority weights
        priority_weights = {
            PolicyPriority.CRITICAL: 1.0,
            PolicyPriority.HIGH: 0.8,
            PolicyPriority.MODERATE: 0.6,
            PolicyPriority.EMERGING: 0.4
        }
        
        # Component weights for scoring
        weights = {
            'policy_priority': 0.25,
            'overcapacity_severity': 0.20,  # Higher overcapacity = higher priority
            'consolidation_potential': 0.15,
            'innovation_intensity': 0.15,
            'government_support': 0.15,
            'global_competitiveness': 0.10
        }
        
        # Calculate weighted score
        priority_score = priority_weights[sector_profile.policy_priority] * 100
        
        score = (
            weights['policy_priority'] * priority_score +
            weights['overcapacity_severity'] * sector_profile.overcapacity_severity +
            weights['consolidation_potential'] * sector_profile.consolidation_potential +
            weights['innovation_intensity'] * sector_profile.innovation_intensity +
            weights['government_support'] * sector_profile.government_support +
            weights['global_competitiveness'] * sector_profile.global_competitiveness
        )
        
        return score
    
    def calculate_sector_allocations(self, 
                                   total_allocation: float = 1.0,
                                   min_allocation: float = 0.03,
                                   max_allocation: float = 0.25) -> Dict[str, float]:
        """
        Calculate sector allocations based on scores and constraints
        
        Args:
            total_allocation: Total allocation to distribute (default 1.0 = 100%)
            min_allocation: Minimum allocation per sector
            max_allocation: Maximum allocation per sector
            
        Returns:
            Dictionary mapping sector names to allocations
        """
        # Calculate scores for all sectors
        sector_scores = {}
        for sector_name, profile in self.sector_profiles.items():
            sector_scores[sector_name] = self.calculate_sector_score(profile)
        
        # Apply score-based allocation
        total_score = sum(sector_scores.values())
        raw_allocations = {
            sector: (score / total_score) * total_allocation 
            for sector, score in sector_scores.items()
        }
        
        # Apply constraints
        constrained_allocations = {}
        remaining_allocation = total_allocation
        
        # First pass: apply minimum constraints
        for sector, allocation in raw_allocations.items():
            constrained_allocation = max(min_allocation, 
                                       min(max_allocation, allocation))
            constrained_allocations[sector] = constrained_allocation
            remaining_allocation -= constrained_allocation
        
        # Second pass: redistribute remaining allocation proportionally
        if remaining_allocation != 0:
            # Calculate how much each sector can still receive
            adjustable_sectors = {}
            for sector, allocation in constrained_allocations.items():
                if remaining_allocation > 0 and allocation < max_allocation:
                    adjustable_sectors[sector] = max_allocation - allocation
                elif remaining_allocation < 0 and allocation > min_allocation:
                    adjustable_sectors[sector] = allocation - min_allocation
            
            if adjustable_sectors:
                total_adjustable = sum(adjustable_sectors.values())
                for sector, adjustable_amount in adjustable_sectors.items():
                    adjustment = (adjustable_amount / total_adjustable) * remaining_allocation
                    constrained_allocations[sector] += adjustment
        
        # Final normalization to ensure total = 1.0
        actual_total = sum(constrained_allocations.values())
        if actual_total != total_allocation:
            for sector in constrained_allocations:
                constrained_allocations[sector] *= (total_allocation / actual_total)
        
        return constrained_allocations
    
    def compare_with_sssr_1(self) -> Dict[str, Dict]:
        """
        Compare current allocation with hypothetical SSSR 1.0 allocation
        
        Returns:
            Comparison data including allocations and differences
        """
        # Current anti-involution allocation
        current_allocation = self.calculate_sector_allocations()
        
        # Hypothetical SSSR 1.0 allocation (based on historical priorities)
        sssr_1_allocation = {
            "Steel": 0.30,
            "Coal Mining": 0.25,
            "Chemicals": 0.15,
            "Electric Vehicles": 0.10,  # Emerging at the time
            "Semiconductors": 0.05,     # Very early stage
            "Solar Energy": 0.08,       # Limited focus
            "Lithium/Batteries": 0.02,  # Nascent
            "Wind Energy": 0.03,        # Limited
            "Logistics": 0.01,          # Minimal
            "Food Delivery": 0.01       # Non-existent
        }
        
        # Calculate differences
        differences = {}
        for sector in current_allocation:
            current = current_allocation.get(sector, 0)
            historical = sssr_1_allocation.get(sector, 0)
            differences[sector] = current - historical
        
        return {
            'current_allocation': current_allocation,
            'sssr_1_allocation': sssr_1_allocation,
            'differences': differences,
            'total_shift_to_hightech': sum(
                differences[sector] for sector in 
                ["Electric Vehicles", "Semiconductors", "Solar Energy", "Lithium/Batteries"]
                if sector in differences
            )
        }
    
    def generate_allocation_report(self) -> str:
        """
        Generate comprehensive allocation report
        
        Returns:
            Formatted report string
        """
        allocations = self.calculate_sector_allocations()
        comparison = self.compare_with_sssr_1()
        
        report = []
        report.append("=" * 70)
        report.append("CHINA ANTI-INVOLUTION BASKET - SECTOR ALLOCATION ANALYSIS")
        report.append("=" * 70)
        report.append("")
        
        # Current allocation
        report.append("CURRENT ALLOCATION (Anti-Involution Policy-Based):")
        report.append("-" * 50)
        for sector, allocation in sorted(allocations.items(), 
                                       key=lambda x: x[1], reverse=True):
            profile = self.sector_profiles[sector]
            report.append(f"{sector:<25} {allocation:>8.1%} ({profile.policy_priority.value})")
        report.append("")
        
        # Comparison with SSSR 1.0
        report.append("COMPARISON WITH SSSR 1.0 APPROACH:")
        report.append("-" * 50)
        report.append(f"{'Sector':<25} {'Current':<10} {'SSSR 1.0':<10} {'Change':<10}")
        report.append("-" * 50)
        
        for sector in allocations:
            current = allocations[sector]
            historical = comparison['sssr_1_allocation'].get(sector, 0)
            change = current - historical
            change_str = f"{change:+.1%}" if change != 0 else "0.0%"
            
            report.append(f"{sector:<25} {current:>8.1%} {historical:>8.1%} {change_str:>8}")
        
        report.append("")
        report.append(f"Total shift to high-tech sectors: {comparison['total_shift_to_hightech']:+.1%}")
        report.append("")
        
        # Sector characteristics summary
        report.append("SECTOR CHARACTERISTICS:")
        report.append("-" * 50)
        for sector, profile in self.sector_profiles.items():
            score = self.calculate_sector_score(profile)
            report.append(f"{sector}:")
            report.append(f"  Priority: {profile.policy_priority.value}")
            report.append(f"  Overcapacity Severity: {profile.overcapacity_severity}/100")
            report.append(f"  Innovation Intensity: {profile.innovation_intensity}/100")
            report.append(f"  Composite Score: {score:.1f}/100")
            report.append(f"  SSSR 1.0 Involvement: {'Yes' if profile.sssr_1_involvement else 'No'}")
            report.append("")
        
        return "\n".join(report)

# Example usage
if __name__ == "__main__":
    allocator = SectorAllocator()
    
    # Generate and print allocation report
    print(allocator.generate_allocation_report())
    
    # Calculate final allocations
    allocations = allocator.calculate_sector_allocations()
    print("\nFinal Sector Allocations:")
    for sector, allocation in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
        print(f"{sector}: {allocation:.1%}")
