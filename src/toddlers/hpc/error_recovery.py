"""Classify and auto-repair known Cloudy failures.

At production-grid scale a fraction of Cloudy models fail for well-understood,
fixable reasons (non-convergence, temperature/pressure floors, zone limits). This
module recognises those failure signatures in a model's ``.out`` and applies the
corresponding input tweak so the model can be rerun successfully, instead of the
task failing permanently. Unknown failures are left for manual review.

Ported from the dev ``hpc/error_handling/cloudy_error_handlers.py`` (the portable
classify-and-fix core; the dev's backup/restore orchestration is not needed here
because the worker pool retries in place). Used by
:func:`toddlers.hpc.runner.run_cloudy_task` as a bounded retry-with-repair loop.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import re


@dataclass
class CloudyError:
    """A known Cloudy failure type and the input modification that fixes it."""
    name: str
    pattern: str                       # regex matched against the .out tail
    description: str
    solution: Optional[str] = None
    modification_function: Optional[str] = None


class CloudyErrorClassifier:
    """Match Cloudy output against known, auto-fixable failure signatures."""

    def __init__(self):
        self.errors = {
            "conv_fail": CloudyError(
                name="Convergence Failure",
                pattern=r"ConvFail aborts since nTotalFailures=\d+ is >= LimFail=\d+",
                description="Too many convergence failures",
                solution="Add turbulence pressure",
                modification_function="add_turbulence_pressure"),
            "temp_low": CloudyError(
                name="Temperature Too Low",
                pattern=r"(?:PROBLEM DISASTER|ABORT DISASTER).*kinetic temperature.*below the lower limit",
                description="Temperature below the code limit of 2.8 K",
                solution="Set cosmic ray background to standard",
                modification_function="increase_cosmic_rays"),
            "quantum_heat": CloudyError(
                name="Quantum Heating Failure",
                pattern=r"qheat\(std::vector<double, std::allocator<double> >&.*\n.*GrainMakeDiffuse\(\)",
                description="Quantum heating for dust grains failed",
                solution="Increase cosmic ray background by 25%",
                modification_function="increase_cosmic_rays_quantum"),
            "pres_ioniz": CloudyError(
                name="Pressure Ionization Limit",
                pattern=r"(?:PROBLEM ConvBase|ABORT DISASTER).*nPres2Ioniz exceeds limPres2Ioniz",
                description="Pressure ionization iterations exceeded the limit",
                solution="Add turbulence pressure",
                modification_function="add_turbulence_pressure"),
            "zone_limit": CloudyError(
                name="Zone Limit Reached",
                pattern=r"Calculation stopped because default number of zones reached",
                description="Default zone limit reached",
                solution="Increase number of zones",
                modification_function="increase_zones"),
        }

    def classify_error(self, output_text: str) -> Optional[CloudyError]:
        """Return the matching CloudyError for the output, or None if unknown."""
        for error in self.errors.values():
            if re.search(error.pattern, output_text, re.MULTILINE):
                return error
        return None

    def requires_manual_review(self, output_text: str) -> bool:
        """True if the failure is unknown or has no auto-fix."""
        error = self.classify_error(output_text)
        return error is None or not error.solution


class CloudyInputModifier:
    """Apply the input tweak that fixes a recognised Cloudy failure."""

    HIGH_DENSITY_THRESHOLD = 3.5    # log n_H above which turbulent velocity is boosted
    VELOCITY_INCREASE_FACTOR = 5.0
    STANDARD_CR_VALUE = 2           # standard cosmic ray background

    def extract_density(self, input_text: str) -> float:
        """Log hydrogen density from the ``hden`` command (raises if absent)."""
        match = re.search(r"hden\s+([+-]?\d*\.?\d+)", input_text)
        if not match:
            raise ValueError("No valid 'hden' command found in input file")
        return float(match.group(1))

    def add_turbulence_pressure(self, input_text: str, increase_velocity: bool = True) -> str:
        """Boost turbulent velocity in high-density models (helps convergence)."""
        try:
            log_density = self.extract_density(input_text)
        except ValueError as exc:
            print(f"Warning in turbulence modification: {exc}")
            return input_text
        pattern = r"turbulence\s+(\d+\.?\d*)\s+km/sec(?:\s+no\s+pressure)?"

        def replacement(match):
            velocity = float(match.group(1))
            if log_density >= self.HIGH_DENSITY_THRESHOLD and increase_velocity:
                return f"turbulence {self.VELOCITY_INCREASE_FACTOR * velocity:.6f} km/sec"
            return f"turbulence {velocity:.6f} km/sec"

        return re.sub(pattern, replacement, input_text)

    def increase_cosmic_rays(self, input_text: str) -> str:
        """Set the cosmic ray background to the standard value (temperature floor)."""
        pattern = r"cosmic rays background(?:\s+(\d+))?"

        def replacement(match):
            current = int(match.group(1)) if match.group(1) else 1
            if current != self.STANDARD_CR_VALUE:
                return f"cosmic rays background {self.STANDARD_CR_VALUE}"
            return match.group(0)

        return re.sub(pattern, replacement, input_text)

    def increase_cosmic_rays_quantum(self, input_text: str) -> str:
        """Raise the cosmic ray background 25% (quantum-heating failures)."""
        pattern = r"cosmic rays background(?:\s+(\d+(?:\.\d*)?))?"

        def replacement(match):
            current = float(match.group(1)) if match.group(1) else 1.0
            return f"cosmic rays background {current * 1.25:.3f}"

        return re.sub(pattern, replacement, input_text)

    def increase_zones(self, input_text: str) -> str:
        """Raise the zone cap (``set nend``), bounded at 5000."""
        def replacement(match):
            current = int(match.group(1))
            return f"set nend {int(min(current * 1.5, 5000))}"

        return re.sub(r"set nend\s+(\d+)", replacement, input_text)

    def modify_input(self, input_text: str, error: CloudyError) -> Tuple[str, str]:
        """Apply ``error``'s fix to the input; return (modified_text, description)."""
        if not error.modification_function:
            return input_text, "No modification needed"
        func = getattr(self, error.modification_function, None)
        if not func:
            return input_text, "No modification function found"
        return func(input_text), f"Applied {error.name} fix"
