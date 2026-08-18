import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register opt-in test groups that require heavyweight toolchains."""
    parser.addoption(
        "--run-onnx",
        action="store_true",
        default=False,
        help="run ONNX export tests",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ONNX tests unless their toolchain was requested explicitly."""
    if config.getoption("--run-onnx"):
        return

    skip_onnx = pytest.mark.skip(reason="use --run-onnx to run ONNX export tests")
    for item in items:
        if "onnx" in item.keywords:
            item.add_marker(skip_onnx)
