# Contributing

Thanks for taking time to contribute.

## Setup
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -e .[dev]`
3. Run `ruff check .`, `mypy`, and `pytest`

## Adding a detector
1. Implement detector logic in `src/mlguardian/detectors/implementations.py` (or split module).
2. Register with `@register_detector("your_detector")`.
3. Return structured `Finding` objects with clear recommendation text.
4. Add focused tests with synthetic fixtures.

## Guidelines
- Keep checks heuristic and transparent.
- Avoid logging sensitive row contents.
- Prefer vectorized Pandas operations for scale.
- Keep recommendations actionable and scientifically honest.
