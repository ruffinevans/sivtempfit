# sivtempfit
Bayesian analysis of SiV spectral data and spectroscopic temperature determination in python.

**Author**: Ruffin Evans (ruffinevans@gmail.com)

Contains contributions from [`yaml-serialize`](https://github.com/tdimiduk/yaml-serialize) by Tom Dimiduk as well as suggestions and minor contributions from the rest of the Harvard Physics Spring 2016 class.

Example data provided by Christian Nguyen with contributions from the rest of the Lukin Group SiV team in the Harvard Physics Department.

## Description
This package is designed to analyze fluorescence spectra from SiV centers in diamond by performing Bayesian parameter estimation. The emphasis is on ease-of-use and producing reliable estimates of the fluorescence wavelength and associated uncertainty.

The package is designed to fit one of two possible models to the data: 

1. A model containing a single Lorentzian peak describing the SiV center fluorescence
2. A model containing two Lorentzian peaks an experimental peak and a reference/calibration peak.

Although the second model is more complicated, it allows an absolute frequency calibration with every spectrum, removing several sources of possible systematic error.

## Installation and testing

Simply download the package and run `python setup.py install` within the package directory.

To test that the package is performing as intended, you can run the `nosetests` command (or `python setup.py test`). These tests will test some basic functionality of the package, including performing some simple MC-based inference on some test data. This inference should only take a small amount of time, probably less than a minute on your machine. However, **please be patient** while these tests run.

## Documentation

Currently, all of the documentation is in three jupyter notebooks:

`tutorial.ipynb` Start here. This tutorial describes some of the most basic functionality of the package and is the fastest way to get up-and-running with Monte-Carlo-based inference.

If you'd like to delve a little deeper into the model specification, there is also:

`model-development.ipynb` describes the model and how it is implemented.

`model-testing.ipynb` shows how to use the package to calculate the likelihood (which you almost always just want to do automatically with the `emcee` sampler as in the tutorial). It also does some very simple inference and tests to make sure the likelihood returns reasonable values.

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

2. MC with parallel tempering to improve convergence and reduce the need for fine-tuning of the guesses before fitting.

## Copyright and license information

The package is released under the GNU GPLv3. See the LICENSE file.