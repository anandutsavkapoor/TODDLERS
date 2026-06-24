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
            # Per-zone ionization non-convergence (no terminal abort): the model
            # crawls through deep zones in optically-thick, dense, dusty gas. Must
            # rank ABOVE zone_limit so a grinder that also hits the zone cap is
            # treated at its root cause (turbulence) rather than just given more
            # zones (which only makes it grind longer).
            "conv_zone": CloudyError(
                name="Zone Non-Convergence",
                pattern=r"PROBLEM\s+ConvFail\s+\d+,\s+ionization not converged",
                description="Per-zone ionization non-convergence (slow grind in dense gas)",
                solution="Add turbulence pressure",
                modification_function="add_turbulence_pressure"),
            # Floating-point exception in the radiative-transfer / convergence
            # solver (e.g. RT_continuum_shield_fcn) in extremely optically-thick
            # dense clouds. Broadening lines via turbulence lowers the optical
            # depths that plausibly drive the overflow; treated as a convergence
            # aid, not a guaranteed fix.
            "fpe_disaster": CloudyError(
                name="Floating Point Exception",
                pattern=r"DISASTER - A floating point exception occurred",
                description="FPE in radiative transfer / convergence solver",
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

    TURBULENCE_FLOOR = 1.5          # km/s; low value injected on the first repair
    TURBULENCE_ESCALATE = 2.0       # multiplier applied on each subsequent repair
    TURBULENCE_CEILING = 12.0       # km/s; hard cap so escalation cannot run away
    STANDARD_CR_VALUE = 2           # standard cosmic ray background

    def add_turbulence_pressure(self, input_text: str) -> str:
        """Inject (or escalate) turbulent pressure to aid convergence.

        The optically-thick, dense, dusty corner of the grid (high Z, low SFE,
        high n_cl, high mass) stalls or crashes in the ionization / line-transfer
        solver. Turbulence aids convergence by two distinct mechanisms: the
        turbulent velocity broadens line profiles, lowering line-centre optical
        depths (tau_0 ~ 1/Delta-nu_D) and easing the line transfer that stalls
        here (this acts whether or not turbulent pressure is on); and, for models
        that hold total pressure fixed (the constant-pressure shells), including
        turbulent pressure adds temperature-independent support that softens the
        density jump at the ionization / thermal front. For unified models the
        density is prescribed by a density law, so there the benefit is the line
        broadening rather than any change to the structure. We start from a low
        value and escalate on each retry, clamped at a hard ceiling.
        The escalation persists in the on-disk input across resume rounds (the
        input is not regenerated for a failed-but-present model), so the ceiling
        is what stops it running away: once reached, the rewrite is a no-op and
        the caller gives up cleanly (manual review), mirroring ``increase_zones``.

        Applied unconditionally to the model's ``turbulence`` line, with no
        density gate. The previous version gated on the face ``hden`` value,
        which was the wrong abstraction: unified models set density via a law
        (``init *_density_law.ini``) and have no ``hden`` line at all, so the
        old gate raised and silently left them unchanged; constant-pressure
        shells do have ``hden`` but it is the illuminated-face value, far below
        the compressed deep-cloud density where the solver actually fails.
        Dropping the trailing ``no pressure`` activates turbulent pressure.
        """
        pattern = r"turbulence\s+(\d+\.?\d*)\s+km/sec(?:\s+no\s+pressure)?"

        def replacement(match):
            velocity = float(match.group(1))
            if velocity < self.TURBULENCE_FLOOR:
                new_velocity = self.TURBULENCE_FLOOR
            else:
                new_velocity = velocity * self.TURBULENCE_ESCALATE
            new_velocity = min(new_velocity, self.TURBULENCE_CEILING)
            return f"turbulence {new_velocity:.6f} km/sec"

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

    ZONE_CEILING = 12000           # hard cap on set nend across repeated repairs

    def increase_zones(self, input_text: str) -> str:
        """Raise the zone cap (``set nend``) x1.5 per call, bounded at the ceiling."""
        def replacement(match):
            current = int(match.group(1))
            return f"set nend {int(min(current * 1.5, self.ZONE_CEILING))}"

        return re.sub(r"set nend\s+(\d+)", replacement, input_text)

    def modify_input(self, input_text: str, error: CloudyError) -> Tuple[str, str]:
        """Apply ``error``'s fix to the input; return (modified_text, description)."""
        if not error.modification_function:
            return input_text, "No modification needed"
        func = getattr(self, error.modification_function, None)
        if not func:
            return input_text, "No modification function found"
        return func(input_text), f"Applied {error.name} fix"
