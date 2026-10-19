# Contribute

As you probably know, there are many ways in which you could contribute to Falcondale SDK: reporting bugs, improving the documentation, adding examples or new quantum techniques. Feel free to reach out and share your experience using the SDK with us.

## Development setup

Falcondale uses [uv](https://docs.astral.sh/uv/) to manage its environment. Once you clone the repository:

```
uv sync --all-extras
uv run pre-commit install --install-hooks
```

That installs the package in editable mode together with the development tools: [ruff](https://docs.astral.sh/ruff/) for linting and formatting, [ty](https://docs.astral.sh/ty/) for type checking, [pytest](https://docs.pytest.org/) for testing and [commitizen](https://commitizen-tools.github.io/commitizen/) for commit messages and releases.

```
uv run pytest -m "not slow"   # quick feedback
uv run pytest --cov           # everything, quantum simulations included
uv run pre-commit run --all-files
```

Commits follow the [Conventional Commits](https://www.conventionalcommits.org/) convention (`feat: ...`, `fix: ...`, `docs: ...`). You can use `uv run cz commit` if you do not want to remember the format. They matter: when `develop` is merged into `main`, the version number, the changelog and the PyPI release are generated from them.

The documentation is built with MkDocs:

```
uv run --group docs mkdocs serve
```

Check the [contributing guide](https://github.com/Falcondale/falcondale-sdk/blob/main/CONTRIBUTING.md) for the full details.

## Community

We think that sharing is key for open source projects. Use [GitHub issues](https://github.com/Falcondale/falcondale-sdk/issues) to ask for support, request improvements or discuss in general.
