[![CI](https://github.com/b-d-e/gpt-oss-unsafe/actions/workflows/ci.yml/badge.svg)](https://github.com/b-d-e/gpt-oss-unsafe/actions/workflows/ci.yml) ![WIP](https://img.shields.io/badge/Work%20in%20progress-FFA500)

# 👹 GPT-OSS-Unsafe

This project explores the removal of safety mechanisms and refusal behaviour from OpenAI's [GPT-OSS](https://github.com/openai/gpt-oss/) model series (initially, 20B variant).

Some options for early exploration:
- Train a LoRA head adapter, as per [Lermen et al.](https://arxiv.org/abs/2310.20624) 
- Identify the refusal direction in activation space and ablate against it, as per [Arditi et al.](https://arxiv.org/pdf/2406.11717)
- Just follow OpenAI's finetuning [cookbook](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers)...

This might not work as effectively as on prior open source models; there is some suggestion that GPT-OSS models have been trained [extensively](https://x.com/corbtt/status/1952868822891012241) [(entirely?)](https://x.com/huybery/status/1952905224890532316) on synthetic data, produced by another safety algined model - similar to [Phi 4](https://huggingface.co/microsoft/phi-4). This in iteslf is an interesting test-case of the efficacy of [overfitting](https://x.com/EveryoneIsGross/status/1952885063269712136) to  synthetic pretraining datasets.

> To do: update template repo readme below 
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
 
