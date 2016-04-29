# sivtempfit
Bayesian analysis of SiV spectral data and spectroscopic temperature determination in python.

**Author**: Ruffin Evans (ruffinevans@gmail.com)

Contains contributions from [`yaml-serialize`](https://github.com/tdimiduk/yaml-serialize) by Tom Dimiduk.

Example data provided by Christian Nguyen with contributions from the rest of the Lukin Group SiV team in the Harvard Physics Department.

## Description
This package is designed to analyze fluorescence spectra from SiV centers in diamond by performing Bayesian parameter estimation. The emphasis is on ease-of-use and producing reliable estimates of the fluorescence wavelength and associated uncertainty.

The package is designed to do simultaneous fitting of two peaks (an experimental peak and a reference/calibration peak) to reduce uncertainty in the wavelength.

## Installation and testing

Simply download the package and run `python setup.py install` within the package directory.

To test that the package is performing as intended, you can run the `nosetests` command (or `python setup.py test`). These tests will test some basic functionality of the package, including performing some simple inference on some test data. Because this inference can take a little bit of time, **please be patient** while these tests run -- they should take no more than 90-120 seconds on a moderately powerful laptop.

## Documentation

Currently, all of the documentation is in three jupyter notebooks:

`tutorial.ipynb` Start here. This tutorial describes some of the most basic functionality of the package and is the fastest way to get up-and-running with Monte-Carlo-based inference.

If you'd like to delve a little deeper into the model specification, there is also:

`model-development.ipynb` describes the model and how it is implemented.

`model-testing.ipynb` shows how to use the package to do inference and performs some simple inference on simulated data to make sure the model works as intended.

## Modules and package structure

`dataprocessing.py` includes a specification for the `Spectrum` object and other simple data processing tasks

`io.py` supports simple io for the `Spectrum` object

`model.py` specifies the likelihood for the model described above, and can evaluate them for `Spectrum` objects.

`inferMC.py` includes functions to perform Monte-Carlo sampling on the likelihood.

`exampledata` includes some light-weight example data, including simple data used for tests

`tests` contains unit tests

## Planned improvements

There are a few planned improvements.

1. Inclusion of various options to specify the priors on each parameter. Currently, the package only supports sampling over the likelihood and is therefore implicitly choosing improper uniform priors if the output is interpreted as a posterior.

2. Simultaneous fitting of a set of calibration spectra to estimate the shift in wavelength as a function of an external paramter, e.g. the sample temperature.

## Copyright and license information

The package is released under the GNU GPLv3. See the LICENSE file.