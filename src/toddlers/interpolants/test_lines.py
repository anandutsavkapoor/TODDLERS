#!/usr/bin/env python3
"""Simple test to check if CloudyOutputHandler can read line luminosities."""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from toddlers.cloudy_output_handler import CloudyOutputHandler
from toddlers.constants import MYR_TO_SEC

# Test parameters
base_path = "/Users/akapoor/toddlers_evolution_package/cloudy_output/template_SB99/imf_kroupa100/star_type_sin/cluster_mode_burst/profile_type_modified_bonnor_ebert/Z0.003_eta0.05_n100.0_logM6.00/"
model = "shell"
time = 3.10 * MYR_TO_SEC  # Convert to seconds

# Create handler
handler = CloudyOutputHandler(
    model_prefix=model,
    time=time,
    absolute_path=base_path
)
# print(handler.check_cloudy_success())
# Try to read emergent luminosities
line_lums = handler.get_line_luminosities(use_emergent=True)
# print(line_lums)
ionization = handler.get_ionization_structure()

# Print first few lines to verify
print(f"\nFound {len(line_lums)} lines")
# print("\nFirst few lines:")
# for i, (line_id, lum) in enumerate(line_lums.items()):
#     if i >= 50000:
#         break
#     print(f"{line_id[2]}: {lum:.2e} erg/s")