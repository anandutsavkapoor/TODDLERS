#!/usr/bin/env python3
"""
Quick test: Verify get_available_mass_grid() reads actual masses from tracks.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), 'pysb99', 'stochastic'))

from database import get_available_mass_grid

print("\n" + "="*70)
print("VERIFYING ACTUAL MASS GRID SCANNING")
print("="*70)

test_cases = [
    ('SMC', False, "Should NOT have 300 Msun"),
    ('LMC', False, "May have 300 Msun"),
    ('MW', False, "May have 500 Msun"),
    ('MWC', False, "Should have ~300 Msun max"),
]

for metallicity, rotation, note in test_cases:
    print(f"\n{metallicity} (rotation={rotation}): {note}")
    try:
        grid = get_available_mass_grid(metallicity, rotation)
        print(f"  Found {len(grid)} masses")
        print(f"  Range: {grid.min():.2f} - {grid.max():.2f} Msun")
        
        # Show highest masses
        high_masses = grid[grid > 100]
        if len(high_masses) > 0:
            print(f"  High-mass end: {high_masses}")
        
        # Check for specific issues
        if metallicity == 'SMC' and grid.max() > 130:
            print(f"  WARNING: SMC has mass > 130 Msun!")
        
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print(f"     Make sure you're in /Users/akapoor/toddlers_evolution_package/")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*70)
print("If all tests pass, regenerate database:")
print("  rm -f src/database/single_star_tracks.h5")
print("  python3 main/execute_stochastic_examples.py --quick")
print("="*70)