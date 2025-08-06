[![CI](https://github.com/tomogwen/pytemplate/actions/workflows/ci.yml/badge.svg)](https://github.com/tomogwen/pytemplate/actions/workflows/ci.yml)

# 📝 Python Project Template

A lightweight Python project template that uses [`uv`](https://docs.astral.sh/uv/) for environment and project management, [`ruff`](https://docs.astral.sh/ruff/) for code linting and formatting, and [`pytest`](https://docs.pytest.org/) for unit testing.

## 📦 Using UV

- Clone this template. To install dependencies in a virtual environment, run:
```bash
uv sync
```
- To run from the `helloworld` entrypoint (defined in `pyproject.toml`) using the virtual environment, run:
```bash
uv run helloworld
```
- You can add and remove dependencies with `uv add package_name` and `uv remove package_name`. This will track the dependency in the `pyproject.toml` and install it in your virtual environment.

### Packaging, Notebooks, and Scripts

- The `package=true` flag in [pyproject.toml](pyproject.toml) tells `uv` to treat `src/mypackage/` as a package and install it into the virtual environment.
- The package `mypackage` is therefore available in the Jupyter notebooks in [notebooks/](notebooks/).
- The `[project.scripts]` section of the `pyproject.toml` lets you define entrypoints, which are runnable via `uv run`. An example entrypoint is defined as `uv run helloworld`.

### Updating this template

- You can change the package name by (i) updating the `pyproject.toml` and (ii) updating the name `src/mypackage`.

## 🧹 Code Quality

Code should contain [docstrings](https://peps.python.org/pep-0257/) and [type hints](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html), alongside [clear comments](https://stackoverflow.blog/2021/12/23/best-practices-for-writing-code-comments/).

### Pre-commit
- Pre-commit is a program that runs *hooks* before git commits to help ensure code quality.
- The hooks are defined in `.pre-commit-config.yaml`. We use [ruff](https://docs.astral.sh/ruff/) to lint and format the codebase, [mypy](https://mypy-lang.org/) to add static type checking, [uv-lock](https://docs.astral.sh/uv/guides/integration/pre-commit/) to ensure the `uv.lock` file is up-to-date, and [nbstripout](https://pypi.org/project/nbstripout/0.2.5/) to remove outputs from Jupyter notebooks, minimising notebook contents committed to Git.
- Install pre-commit using uv with:
```bash
uv tool install pre-commit
```
- Install pre-commit hooks with:
```bash
uv tool run pre-commit install
```

### Tests

- Tests are defined in `tests/`, and can be run with PyTest via uv, i.e.,:
```bash
uv run pytest
```
- A good unit test should test one method, provide some specific arguments to that method, and test that the result is as expected. See this [Stack Overflow answer](https://stackoverflow.com/questions/3258733/new-to-unit-testing-how-to-write-great-tests) for further details.

### CI

- The GitHub actions defined in `.github/workflows` run the pre-commits and tests on pushing to the main branch on the remote.
 