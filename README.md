[![CI](https://github.com/VisiumCH/crowd_programming/actions/workflows/ci.yml/badge.svg)](https://github.com/VisiumCH/crowd_programming/actions/workflows/ci.yml)

# README

## Set up your development environment and run the DVC pipeline

The python environment is managed with `pipenv`. You can set up your environment and run the DVC pipeline with the following steps:

- Run `pipenv lock`to generate the `Pipfile.lock` which lists the version of your python packages.
- Run `pipenv install --dev` to actually create a virtual environment and install the python packages. The flag `--dev` allows to install the development packages (for linting, ...).
- Run `pipenv shell` to activate your python environment!
- Run `pre-commit install` to install pre-commits and make sure visiumlint is run at every commit.
- Run `gcloud auth` to authenticate to your GCP account.
- Finally run `dvc repro` to execute the complete pipeline! You can also execute a single step and its dependencies with `dvc repro <step_name>`.

## Some tips about pipenv

**Deploy in production**

Note that when deploying your code in production, you should not install the dev package, it is preferred to run the following command: `pipenv install --system --deploy`.

**Git with pipenv**

Make sure to commit the `Pipfile.lock` in `git`. It will make your code more reproducible because other developers could install the exact same python packages as you used.
