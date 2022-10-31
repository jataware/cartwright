---
layout: default
title: Contributing
nav_order: 7
has_toc: true
---
# Contributing to Cartwright
Contributions to Cartwright are welcome. Please read the following guidelines before contributing.

## General Guidelines
* Please follow the style of the code you are modifying.
* Please add tests for any new features.
* Please add documentation for any new features.
* Please add a changelog entry for any new features.
* Please add yourself to the list of contributors in the README.md file.


## Getting Started with Development
See [installation](./installation) for instructions on building repository from source for local development.


## Running Cartwright's Test Suite
Cartwright uses [pytest](https://docs.pytest.org/en/latest/) and [tox](https://tox.readthedocs.io/en/latest/) for testing. If you have installed Cartwright from source with all the dependencies, you can run pytest directly:

    $ pytest

Alternatively you can use tox to run the test suite over all supported Python versions:

    $ tox

Tox automatically creates a virtual environment for each Python version and runs the test suite in each environment. So tox is not reliant on your system Python version.



## Adding a New Category
Coming soon.

## Adding a New Category Member
Coming soon.

## Training a New Model
Coming soon.