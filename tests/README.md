# Testing

The repo includes a `pytest` suite under `tests/` for fast unit tests and lightweight integration checks.

Run all tests:
```
python -m pytest
```

Run only unit tests:
```
python -m pytest tests/unit
```

Run only integration tests:
```
python -m pytest -m integration
```

Run all tests in one file:
```
python -m pytest tests/unit/test_phylo_merge.py
```

Run one specific test:
```
python -m pytest tests/unit/test_utils.py::test_running_mean_tracks_average
```