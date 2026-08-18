from pathlib import Path

from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from neural_fx.training.logging import create_training_loggers


def test_training_loggers_share_run_directory(tmp_path: Path) -> None:
    loggers = create_training_loggers(tmp_path, "amp")

    assert isinstance(loggers[0], CSVLogger)
    assert isinstance(loggers[1], TensorBoardLogger)
    assert loggers[0].log_dir == loggers[1].log_dir


def test_training_loggers_advance_existing_version(tmp_path: Path) -> None:
    (tmp_path / "amp" / "version_0").mkdir(parents=True)

    loggers = create_training_loggers(tmp_path, "amp")

    assert Path(loggers[0].log_dir).name == "version_1"
