Delft-FIAT Toolbox
------------------
This toolbox contains post-processing modules for Delft-FIAT output.

Installation
============
Fiat toolbox uses [uv](https://docs.astral.sh/uv/) to build and manage python environments.
If you do not have `uv` installed, you can install it using `pip install uv`.

- Install with: `uv sync`

- Run the tests with: `uv run pytest`

- Run the linter with: `uv run pre-commit`

Modules:

metrics_writer
==============
This module contains functions to write out custom aggregated metrics from Delft-FIAT output for the whole model an/or different aggregation levels.

infographics
============
This module contains functions to write customized infographics in html format using metric files .

spatial_output
==============
This module contains functions to aggregate point output from FIAT to building footprints. Moreover, it has methods to join aggregated metrics to spatial files.

equity
======
This module contains functions to calculate equity weights and equity weighted risk metrics based on socio-economic inputs at an aggregation level.
