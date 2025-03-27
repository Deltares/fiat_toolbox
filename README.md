Delft-FIAT Toolbox
------------------
This toolbox contains post-processing modules for Delft-FIAT output.

Installation
====================
You can install fiat toolbox and its dependencies using pip:
    pip install fiat-toolbox

Develop
====================
To develop in fiat_toolbox you can create a virtual environment using conda and the yml file in the /envs directory:
    conda env create -f envs/fiat-toolbox-dev.yml

then activate the environment:
    conda activate fiat_toolbox

and then pip install the package in editable mode:
    pip install -e .

Modules:

metrics_writer
====================
This module contains functions to write out custom aggregated metrics from Delft-FIAT output for the whole model an/or different aggregation levels.

infographics
====================
This module contains functions to write customized infographics in html format using metric files .

spatial_output
====================
This module contains functions to aggregate point output from FIAT to building footprints. Moreover, it has methods to join aggregated metrics to spatial files.

equity
==================
This module contains functions to calculate equity weights and equity weighted risk metrics based on socio-economic inputs at an aggregation level.

well_being
==================
This module contains functions to estimate household level well-being impacts.
