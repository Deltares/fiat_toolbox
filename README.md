Delft-FIAT Toolbox
------------------
This toolbox contains post-processing modules for Delft-FIAT output.

Installation
====================
To use this package, GDAL has to be installed on your system, which is a c++ library.
A simple way to install GDAL is to use conda. If you do not have conda installed, you can install it by following the instructions on the `conda website <https://docs.conda.io/en/latest/miniconda.html>`_.

After creating and activating your conda environment, you can install GDAL using the following command
    conda install -c conda-forge gdal

Then, you can install fiat toolbox and its dependencies using pip:
    pip install fiat-toolbox


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
