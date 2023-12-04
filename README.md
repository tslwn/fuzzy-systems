# An Investigation into Fuzzy Systems

This repository holds the code and report written to fulfil the coursework requirements for the unit
[Uncertainty Modelling in Intelligent Systems](https://www.bris.ac.uk/unit-programme-catalogue/UnitDetails.jsa?ayrCode=23%2F24&unitCode=EMATM1120)
at the University of Bristol.

## Instructions

> [!WARNING]
> This repository is designed to be used on macOS and has not been tested on other operating systems.

### Installation

Create a virtual environment and install Python dependencies:

```bash
conda create --name graded python=3.12
conda activate fuzzy-systems
conda install pip
pip install -r requirements.txt
```

> [!NOTE]
> Python 3.12 is required. This repository uses the type-annotation syntax for generic
> classes and functions.

Note that the default interpreter path in [settings.json](./.vscode/settings.json)
assumes you are using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
and the environment is named `fuzzy-systems`.

### Testing

To run the test cases, execute the following command:

```bash
pytest
```
