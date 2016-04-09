# sivtempfit
Bayesian analysis of SiV spectral data and spectroscopic temperature determination in python.

This package is designed to analyze fluorescence spectra from SiV centers in diamond by performing Bayesian parameter estimation. The emphasis is on ease-of-use and producing reliable estimates of the fluorescence wavelength and associated uncertainty.

In addition to fitting a single peak in a single spectrum, there are several planned extensions:
1. Simultaneous fitting of a calibration peak to reduce uncertainty in the wavelength.
2. Simultaneous fitting of a set of calibration spectra to estimate the shift in wavelength as a function of an external paramter, e.g. the sample temperature.