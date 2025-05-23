# aizynthmodels

Repository to train, evaluate, and use models for synthesis predictions.

This contains re-factored code previously found in the following repositories

- [chemformer](https://github.com/MolecularAI/chemformer)
- [aizynthtrain](https://github.com/MolecularAI/aizynthtrain)
- [route-distances](https://github.com/MolecularAI/route-distances)
- [pysmilesutils](https://github.com/MolecularAI/pysmilesutils)

## Prerequisites

Before you begin, ensure you have met the following requirements:

* Linux OS (or in principle Windows or macOS).

* You have installed [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) with python 3.8-3.10

The tool has been developed and fully tested on a Linux platform.

## Installation
First clone the repository using Git.

The project dependencies can be installed by executing the following commands in the root of
the repository:

    conda env create -f env-dev.yml
    conda activate aizynthmodels
    poetry install

If there is an error "ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found"
it can be mitigated by adding the 'lib' directory from the Conda environment to LD_LIBRARY_PATH

As example:
`export LD_LIBRARY_PATH=/path/to/your/conda/envs/chemformer/lib`

Finally, `rxnutils` should be installed. Either clone the `reaction_utils` repo and install `rxnutils` in the current (aizynthmodels) environment:

    cd ../reaction_utils
    poetry install --all-extras

or install it from pypi

    python -m pip install reaction-utils

This repository uses pre-commit (https://pre-commit.com) to ensure style consistency. Before you start developing, ensure that pre-commit is installed and hooks are initialised as described in https://pre-commit.com/#install.

    pre-commit install

## Development

### Testing

Tests uses the ``pytest`` package, and is installed by `poetry`

Run the tests using:

    pytest -v


## Contributing

We welcome contributions, in the form of issues or pull requests.

If you have a question or want to report a bug, please submit an issue.

To contribute with code to the project, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the remote branch: `git push`
5. Create the pull request.

Please use pre-commit to ensure proper formatting and linting

## Contributors

* [@anniewesterlund](https://github.com/anniewesterlund)
* [@Lakshidaa](https://github.com/Lakshidaa)
* [@ckannas](https://github.com/ckannas)
* [@SGenheden](https://www.github.com/SGenheden)
* [@PeterHartog](https://www.github.com/PeterHartog)

The contributors have limited time for support questions, but please do not hesitate to submit an issue (see above).

## License

The software is licensed under the Apache 2.0 license (see LICENSE file), and is free and provided as-is.
