from pathlib import Path

from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger


def _next_run_version(save_dir: Path, name: str) -> int:
    """Return the first unused Lightning ``version_N`` directory."""
    root = save_dir / name
    version = 0
    while (root / f"version_{version}").exists():
        version += 1
    return version


def create_training_loggers(save_dir: str | Path, name: str) -> list[Logger]:
    """Create CSV and TensorBoard loggers that share one run directory."""
    save_dir = Path(save_dir)
    version = _next_run_version(save_dir, name)
    return [
        CSVLogger(save_dir=save_dir, name=name, version=version),
        TensorBoardLogger(save_dir=save_dir, name=name, version=version),
    ]
