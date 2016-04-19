# sivtempfit
Bayesian analysis of SiV spectral data and spectroscopic temperature determination in python.
**Author**: Ruffin Evans (ruffinevans@gmail.com)

## Description
This package is designed to analyze fluorescence spectra from SiV centers in diamond by performing Bayesian parameter estimation. The emphasis is on ease-of-use and producing reliable estimates of the fluorescence wavelength and associated uncertainty.

The package is designed to do simultaneous fitting of two peaks (an experimental peak and a reference/calibration peak) to reduce uncertainty in the wavelength.

## Installation

Simply download the package and run `python setup.py install`.

## Documentation

Currently, all of the documentation is in two jupyter notebooks:

`model-development.ipynb` describes the model and how it is implemented.

`model-testing.ipynb` shows how to use the package to do inference and performs some simple inference on simulated data to make sure the model works as intended.

## Modules and package structure

`dataprocessing.py` includes a specification for the `Spectrum` object and other simple data processing tasks

`io.py` supports simple io for the `Spectrum` object

`model.py` specifies the likelihood, prior, and posterior for the model described above, and can evaluate them for `Spectrum` objects.

`exampledata` includes some light-weight example data, including simple data used for tests

`tests` contains unit tests

## Planned improvements

There are a few plannened improvements.

2. Simultaneous fitting of a set of calibration spectra to estimate the shift in wavelength as a function of an external paramter, e.g. the sample temperature.

## Copyright and license information

The package is released under the GNU GPLv3. See the LICENSE file.