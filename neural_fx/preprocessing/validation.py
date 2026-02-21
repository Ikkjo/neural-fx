"""Data validation suite for checking input/output audio suitability for training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torchaudio
from torch import Tensor

from ..losses.audio_losses import ESR


@dataclass
class CheckResult:
    """Result of a single validation check.

    Attributes:
        passed: Whether the check passed.
        message: Description of the result.
        value: Optional numeric value associated with the check.
    """

    passed: bool
    message: str
    value: float | None = None


@dataclass
class ValidationReport:
    """Complete validation report for a dataset.

    Attributes:
        passed: Whether all critical checks passed.
        checks: Dictionary of check name to CheckResult.
        warnings: List of warning messages.
    """

    passed: bool
    checks: Dict[str, CheckResult] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def get_summary(self) -> str:
        """Get a human-readable summary of the validation report."""
        lines = ["Validation Report", "=" * 50]

        lines.append(f"Overall: {'PASSED' if self.passed else 'FAILED'}")
        lines.append("")

        lines.append("Checks:")
        for name, result in self.checks.items():
            status = "✓" if result.passed else "✗"
            value_str = f" ({result.value:.4f})" if result.value is not None else ""
            lines.append(f"  {status} {name}: {result.message}{value_str}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ! {warning}")

        return "\n".join(lines)


class DataValidator:
    """Validate input/output audio files for training suitability."""

    def __init__(
        self,
        check_clipping: bool = True,
        check_dc_offset: bool = True,
        check_replicability: bool = True,
        clipping_threshold: float = 0.99,
        dc_offset_threshold: float = 0.01,
        esr_threshold: float = 0.1,
    ):
        """Initialize validator with check parameters.

        Args:
            check_clipping: Whether to check for clipping.
            check_dc_offset: Whether to check for DC offset.
            clipping_threshold: Threshold for clipping detection (0-1).
            dc_offset_threshold: Threshold for DC offset detection.
            esr_threshold: Threshold for replicability ESR check.
        """
        self.check_clipping = check_clipping
        self.check_dc_offset = check_dc_offset
        self.clipping_threshold = clipping_threshold
        self.dc_offset_threshold = dc_offset_threshold
        self.esr_threshold = esr_threshold
        self.check_replicability = check_replicability

    def validate(
        self, input_path: str | Path, output_path: str | Path
    ) -> ValidationReport:
        """Validate input/output audio file pair.

        Args:
            input_path: Path to input audio file.
            output_path: Path to target/output audio file.

        Returns:
            ValidationReport with all check results.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        checks: Dict[str, CheckResult] = {}
        warnings: List[str] = []

        # Check 1: Files exist
        checks["files_exist"] = self._check_files_exist(input_path, output_path)
        if not checks["files_exist"].passed:
            return ValidationReport(
                passed=False, checks=checks, warnings=["Cannot proceed without files"]
            )

        # Check 2: Can load files
        load_result, input_audio, output_audio = self._check_can_load(
            input_path, output_path
        )
        checks["can_load"] = load_result
        if not checks["can_load"].passed:
            return ValidationReport(
                passed=False, checks=checks, warnings=["Cannot proceed without audio data"]
            )

        # Check 3: Sample rates match
        checks["sample_rate_match"] = self._check_sample_rates(input_path, output_path)
        if not checks["sample_rate_match"].passed:
            warnings.append(
                "Sample rates differ - audio will be resampled during training"
            )

        # Check 4: Audio length compatibility
        checks["length_compatible"] = self._check_length_compatibility(
            input_audio, output_audio
        )

        # Check 5: Clipping detection (input)
        if self.check_clipping:
            checks["input_clipping"] = self._check_clipping(input_audio, "input")
            if not checks["input_clipping"].passed:
                warnings.append("Input audio has clipping - consider reducing gain")

        # Check 6: Clipping detection (output)
        if self.check_clipping:
            checks["output_clipping"] = self._check_clipping(output_audio, "output")
            if not checks["output_clipping"].passed:
                warnings.append("Output audio has clipping - may affect training quality")

        # Check 7: DC offset detection (input)
        if self.check_dc_offset:
            checks["input_dc_offset"] = self._check_dc_offset(input_audio, "input")
            if not checks["input_dc_offset"].passed:
                warnings.append("Input audio has DC offset - consider high-pass filtering")

        # Check 8: DC offset detection (output)
        if self.check_dc_offset:
            checks["output_dc_offset"] = self._check_dc_offset(output_audio, "output")
            if not checks["output_dc_offset"].passed:
                warnings.append("Output audio has DC offset")

        # Check 9: Replicability check (ESR between random segments)
        if self.check_replicability:
            checks["replicability"] = self._check_replicability(input_audio, output_audio)
            if not checks["replicability"].passed:
                warnings.append(
                    f"Low replicability (ESR={checks['replicability'].value:.4f}) - "
                    "input/output may not be from same performance"
                )

        # Check 10: Signal level check
        checks["signal_level"] = self._check_signal_level(input_audio, output_audio)
        if not checks["signal_level"].passed:
            warnings.append("Signal level is very low - check audio files")

        # Determine overall pass/fail
        # Critical checks that cause failure
        critical_checks = [
            "files_exist",
            "can_load",
            "length_compatible",
            "signal_level",
        ]
        passed = all(checks[check].passed for check in critical_checks)

        return ValidationReport(
            passed=passed,
            checks=checks,
            warnings=warnings,
        )

    def _check_files_exist(
        self, input_path: Path, output_path: Path
    ) -> CheckResult:
        """Check that both files exist."""
        if not input_path.exists():
            return CheckResult(
                passed=False, message=f"Input file not found: {input_path}"
            )
        if not output_path.exists():
            return CheckResult(
                passed=False, message=f"Output file not found: {output_path}"
            )
        return CheckResult(
            passed=True,
            message=f"Both files exist ({input_path.name}, {output_path.name})",
        )

    def _check_can_load(
        self, input_path: Path, output_path: Path
    ) -> tuple[CheckResult, Tensor | None, Tensor | None]:
        """Check that files can be loaded as audio."""
        try:
            input_audio, _ = torchaudio.load(str(input_path))
        except Exception as e:
            return CheckResult(
                passed=False, message=f"Failed to load input: {e}"
            ), None, None

        try:
            output_audio, _ = torchaudio.load(str(output_path))
        except Exception as e:
            return CheckResult(
                passed=False, message=f"Failed to load output: {e}"
            ), None, None

        return (
            CheckResult(
                passed=True,
                message=f"Successfully loaded audio ({input_audio.shape}, {output_audio.shape})",
            ),
            input_audio,
            output_audio,
        )

    def _check_sample_rates(
        self, input_path: Path, output_path: Path
    ) -> CheckResult:
        """Check that sample rates match."""
        try:
            _, input_sr = torchaudio.load(str(input_path))
            _, output_sr = torchaudio.load(str(output_path))

            if input_sr != output_sr:
                return CheckResult(
                    passed=False,
                    message=f"Sample rates differ: {input_sr} vs {output_sr}",
                    value=float(abs(input_sr - output_sr)),
                )
            return CheckResult(
                passed=True, message=f"Sample rates match: {input_sr} Hz", value=float(input_sr)
            )
        except Exception as e:
            return CheckResult(
                passed=False, message=f"Could not check sample rates: {e}"
            )

    def _check_length_compatibility(
        self, input_audio: Tensor, output_audio: Tensor
    ) -> CheckResult:
        """Check that audio lengths are compatible for training."""
        input_len = input_audio.shape[-1]
        output_len = output_audio.shape[-1]

        if input_len == 0 or output_len == 0:
            return CheckResult(
                passed=False, message="One or both audio files are empty"
            )

        length_ratio = min(input_len, output_len) / max(input_len, output_len)

        if length_ratio < 0.5:
            return CheckResult(
                passed=False,
                message=f"Audio lengths differ significantly: {input_len} vs {output_len}",
                value=length_ratio,
            )

        return CheckResult(
            passed=True,
            message=f"Lengths compatible: {input_len} vs {output_len} samples",
            value=length_ratio,
        )

    def _check_clipping(self, audio: Tensor, name: str) -> CheckResult:
        """Check for clipping in audio signal."""
        max_val = audio.abs().max().item()

        if max_val > self.clipping_threshold:
            # Count clipped samples
            clipped_samples = (audio.abs() > self.clipping_threshold).sum().item()
            total_samples = audio.numel()
            clip_percent = (clipped_samples / total_samples) * 100

            return CheckResult(
                passed=False,
                message=f"{name}: {clipped_samples} samples ({clip_percent:.2f}%) exceed threshold",
                value=max_val,
            )

        return CheckResult(
            passed=True,
            message=f"{name}: No clipping detected (max={max_val:.4f})",
            value=max_val,
        )

    def _check_dc_offset(self, audio: Tensor, name: str) -> CheckResult:
        """Check for DC offset in audio signal."""
        mean_val = audio.mean().item()

        if abs(mean_val) > self.dc_offset_threshold:
            return CheckResult(
                passed=False,
                message=f"{name}: DC offset detected (mean={mean_val:.6f})",
                value=mean_val,
            )

        return CheckResult(
            passed=True,
            message=f"{name}: No significant DC offset (mean={mean_val:.6f})",
            value=mean_val,
        )

    def _check_replicability(
        self, input_audio: Tensor, output_audio: Tensor
    ) -> CheckResult:
        """Check replicability by computing ESR between random segments.

        If input and output are from the same performance (aligned),
        there should be a consistent relationship between them.
        """
        # Use random segments for the check
        min_len = min(input_audio.shape[-1], output_audio.shape[-1])
        segment_length = min(48000, min_len // 4)  # 1 second or 1/4 of data

        if segment_length < 1024:
            return CheckResult(
                passed=False, message="Audio too short for replicability check", value=1.0
            )

        # Extract segments from different parts of the audio
        num_checks = 3
        esr_values = []

        for i in range(num_checks):
            start = i * (min_len // num_checks)
            end = start + segment_length

            if end > min_len:
                start = min_len - segment_length
                end = min_len

            input_seg = input_audio[..., start:end]
            output_seg = output_audio[..., start:end]

            # Ensure same shape for ESR calculation
            if input_seg.ndim == 2:
                input_seg = input_seg.mean(dim=0, keepdim=True)
            if output_seg.ndim == 2:
                output_seg = output_seg.mean(dim=0, keepdim=True)

            try:
                esr_val = ESR(input_seg, output_seg).item()
                esr_values.append(esr_val)
            except Exception:
                pass

        if not esr_values:
            return CheckResult(
                passed=False, message="Could not compute replicability", value=1.0
            )

        avg_esr = np.mean(esr_values)

        # Low ESR indicates input and output are similar (good)
        # High ESR means they're very different (possibly misaligned or wrong files)
        if avg_esr > self.esr_threshold:
            return CheckResult(
                passed=False,
                message=f"Low replicability: ESR={avg_esr:.4f} (threshold={self.esr_threshold})",
                value=avg_esr,
            )

        return CheckResult(
            passed=True,
            message=f"Good replicability: ESR={avg_esr:.4f}",
            value=avg_esr,
        )

    def _check_signal_level(
        self, input_audio: Tensor, output_audio: Tensor
    ) -> CheckResult:
        """Check that signal levels are reasonable."""
        input_rms = input_audio.pow(2).mean().sqrt().item()
        output_rms = output_audio.pow(2).mean().sqrt().item()

        min_rms = min(input_rms, output_rms)

        if min_rms < 1e-6:  # Very low signal
            return CheckResult(
                passed=False,
                message=f"Very low signal level (input_rms={input_rms:.2e}, output_rms={output_rms:.2e})",
                value=min_rms,
            )

        if min_rms < 0.001:  # Low signal, warn but don't fail
            return CheckResult(
                passed=True,
                message=f"Low signal level (input_rms={input_rms:.4f}, output_rms={output_rms:.4f})",
                value=min_rms,
            )

        return CheckResult(
            passed=True,
            message=f"Good signal level (input_rms={input_rms:.4f}, output_rms={output_rms:.4f})",
            value=min_rms,
        )


def create_data_validator(
    check_clipping: bool = True,
    check_dc_offset: bool = True,
) -> DataValidator:
    """Factory function to create a DataValidator instance.

    Args:
        check_clipping: Whether to check for clipping.
        check_dc_offset: Whether to check for DC offset.
    Returns:
        DataValidator instance.
    """
    return DataValidator(
        check_clipping=check_clipping,
        check_dc_offset=check_dc_offset,
    )
