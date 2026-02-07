# neural-fx
Real-time guitar effect and amp modelling using neural networks

## Testing

This project uses `pytest` for testing.

### Running Tests

To run the tests, ensure you have the dependencies installed, then run:

```bash
# Ensure the project root is in your PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.
pytest tests/
```

Tests are located in the `tests/` directory.

### Writing Tests

- Use the `pytest` framework.
- Place new tests in `tests/test_<module_name>.py`.
- Use fixtures for common setup (like config objects).