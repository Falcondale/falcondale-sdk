# Contributing to Falcondale SDK

Thanks for taking the time to contribute! Falcondale aims to make quantum machine learning approachable for computer scientists, so contributions that make it easier to understand are as welcome as new quantum techniques. Bug reports, documentation fixes, examples and new methods all count.

By participating in this project you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting issues

Before opening an issue, check whether it already [exists](https://github.com/Falcondale/falcondale-sdk/issues). When reporting a bug, include the Falcondale version (`python -c "import falcondale; print(falcondale.__version__)"`), your operating system and Python version, and a minimal example that reproduces it.

Security issues should **not** be reported as public issues, see [SECURITY.md](SECURITY.md).

## Development environment

We use [uv](https://docs.astral.sh/uv/) to manage dependencies and environments. Once you have forked and cloned the repository:

```bash
uv sync --all-extras                      # creates .venv with Falcondale in editable mode
uv run pre-commit install --install-hooks # enables the git hooks
```

The development dependency group brings in:

| Tool | Purpose |
|------|---------|
| [ruff](https://docs.astral.sh/ruff/) | Linting and formatting |
| [ty](https://docs.astral.sh/ty/) | Type checking |
| [pytest](https://docs.pytest.org/) and pytest-cov | Testing and coverage |
| [pre-commit](https://pre-commit.com/) | Runs the checks above before each commit |
| [commitizen](https://commitizen-tools.github.io/commitizen/) | Conventional commit messages, version bumps and changelog |

PyTorch is installed from its CPU wheel index when working with uv in this repository, which keeps the environment a few gigabytes lighter.

### Day to day commands

```bash
uv run pytest -m "not slow"        # fast unit tests
uv run pytest --cov                # all tests, quantum simulations included
uv run ruff check --fix . && uv run ruff format .
uv run ty check
uv run pre-commit run --all-files  # everything CI runs
uv run --group docs mkdocs serve   # documentation at http://127.0.0.1:8000
```

Tests that run quantum simulations taking more than a few seconds are marked with `@pytest.mark.slow`.

## Making changes

1. Create a branch from `develop` (`feature/...`, `fix/...`, `docs/...`).
2. Add tests for your change. New models or solvers should plug into the registries in `classifiers.py`, `feature_selection.py` or `clustering.py` so that `Project` picks them up.
3. Update the documentation under `docs/` if the user facing behaviour changes.
4. Commit following [Conventional Commits](https://www.conventionalcommits.org/). The commit message hook will complain otherwise. `uv run cz commit` will guide you if you prefer.
5. Open a pull request against `develop`. CI runs the pre-commit checks and the test suite on Python 3.11, 3.12 and 3.13.

### Commit messages

```
feat: add quantum kernel alignment to QSVC
fix(data): keep column order when transforming new samples
docs: explain the QUBO penalty term
```

`feat` bumps the minor version, `fix` the patch version. Breaking changes (`feat!:` or a `BREAKING CHANGE:` footer) bump the minor version while we are below 1.0.

## Releasing

Releases are automatic. When a pull request from `develop` into `main` is merged, the release workflow:

1. runs `cz bump`, which picks the new version from the conventional commits since the last tag, updates `pyproject.toml`, `uv.lock` and `CHANGELOG.md` and creates the tag,
2. pushes the release commit and tag to `main` and merges them back into `develop`,
3. builds the package and publishes it to PyPI using trusted publishing,
4. creates a GitHub release with the changelog of that version.

If none of the merged commits is a `feat`, `fix`, `refactor`, `perf` or breaking change, no release is made. Merge `develop` into `main` with a merge commit (not squash), otherwise the individual commit messages, and with them the changelog, are lost.

You can preview the next release locally with `uv run cz bump --dry-run`.
