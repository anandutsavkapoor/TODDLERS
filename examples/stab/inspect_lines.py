#!/usr/bin/env python3
"""Simple test to check if CloudyOutputHandler can read line luminosities."""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1] / "src"  # repo src/ for the editable package
sys.path.append(str(project_root))

from toddlers.cloudy_output_handler import CloudyOutputHandler
from toddlers.constants import MYR_TO_SEC

# Test parameters
if len(sys.argv) < 2:
    sys.exit("usage: python test_lines.py <cloudy_output_dir> [model] [time_Myr]\n"
             "  <cloudy_output_dir>: a Cloudy model output directory "
             "(e.g. .../Z..._eta..._n..._logM.../)")
base_path = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else "shell"
time = (float(sys.argv[3]) if len(sys.argv) > 3 else 3.10) * MYR_TO_SEC  # Convert to seconds

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