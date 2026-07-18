"""Shared model capability errors."""


class UnsupportedExportFormatError(NotImplementedError):
    """Raised when a model cannot represent a requested export format."""

    def __init__(self, model_type: str, export_format: str):
        self.model_type = model_type
        self.export_format = export_format
        super().__init__(
            f"Model type {model_type!r} does not support {export_format!r} export."
        )


class OptionalDependencyError(ImportError):
    """Raised when an optional model backend is unavailable."""

    def __init__(self, feature: str, dependency: str, install_hint: str):
        self.feature = feature
        self.dependency = dependency
        self.install_hint = install_hint
        super().__init__(
            f"{feature} requires optional dependency {dependency!r}. "
            f"Install it with {install_hint}."
        )
