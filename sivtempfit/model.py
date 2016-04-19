from . import dataprocessing as dp
import json
import pandas as pd
import numpy as np
import scipy as sp


def lorentz(x, center, width):
    """
    A lorentzian function with peak position 'center' and FWHM 'width'.
    """

    return (1/np.pi)*(width/2)/((x-center)**2+(width/2)**2)

def two_peak_model(x, amp1, amp2, center_offset, center2, 
    width1, width2, background):
    """
    Model for the data: two lorentzian peaks with variable amplitudes, widths,
    and center positions.

    The center positions are encoded as a position of a calibration peak
    (center2) as well as an offset such that the position of the first peak
    is just center_offset + center2.

    The lorentzians are defined by the lorentz function.

    For a full explanation, see the model-development.ipynb notebook.
    """

    return (background + 
        amp1*lorentz(x, center_offset + center2, width1) + 
        amp2*lorentz(x, center2, width2))

def two_peak_log_likelihood(x, y, amp1, amp2, T, m, C0, center2, width1, 
    width2, light_background, ccd_background, ccd_stdev, conv_range = 6):
    """
    Returns the log-likelihood calculated for the two-peak + CCD noise model.
    See also: two_peak_model
    
    Parameters:
    -----------
    x : wavelength or x-axis value
    y : corresponding observed data value
    amp1 : amplitude of the broad SiV peak in the spectrum
    amp2 : amplitude of the narrow calibration peak in the spectrum
    T : The temperature of the sample
    m : The linear scaling of the SiV peak position with temperature
    C0 : The offset in the above linear scaling
    center2 : The position of the calibration line
    width1 : The width (FWHM) of the SiV line
    width2 : The width (FWHM) of the calibration line
    light_background : The contribution to the background from stray light,
                       contributing shot noise
    ccd_backgrond : The contribution to the background from CCD readout,
                    contributing gaussian noise
    ccd_stdev : The standard deviation on the gaussian CCD noise above
    conv_range : How far out to go in convolution. 
                 Default is 6 sigma on the gaussian contribution.
    """

    # First, get the contribution from the light signal.
    y_two_peak_model = two_peak_model(x, amp1, amp2, C0 + T*m, center2,
        width1, width2, light_background)
    # Construct the bounds of convolution
    # Adding +0.01 makes it so that if user passes an integer 
    # e.g. 5, goes to +/- 5 and not -5 to 4.
    conv_list = np.arange(-conv_range*ccd_stdev, 
        (conv_range+0.01)*ccd_stdev)[:,np.newaxis]
    # Construct the convolution matrix by broadcasting
    conv_mat = y - ccd_background + conv_list
    # Make negative elements nan, because we can't take negative values
    # in the poisson distribution.
    conv_mat[conv_mat<0] = np.nan
    # Construct poisson term
    poisson_term = (y_two_peak_model**(conv_mat)/
                    sp.misc.factorial(conv_mat)*
                    np.exp(-y_two_peak_model))
    # Construct gaussian term
    gaussian_term = ((1/(np.sqrt(2*np.pi*ccd_stdev**2)))*
                     np.exp(-1*(y - conv_mat - ccd_background)**2 / 
                            (2*ccd_stdev**2)))
    # Perform sum to get likelihoods
    # Here we do nansum which treats nans as zero.
    likelihood_list = np.nansum(poisson_term * gaussian_term, 0)
    # Return sum of log_likelihoods
    #return np.array([poisson_term, gaussian_term])
    return np.sum(np.log(likelihood_list))

def two_peak_log_likelihood_Spectrum(spectrum, amp1, amp2, T, m, C0, center2, width1, 
    width2, light_background, ccd_background, ccd_stdev, conv_range = 6):
    """
    Returns the log-likelihood calculated for the two-peak + CCD noise model.
    See also: two_peak_model
    
    Parameters:
    -----------
    spectrum : data passed as an instance of Spectrum class
    amp1 : amplitude of the broad SiV peak in the spectrum
    amp2 : amplitude of the narrow calibration peak in the spectrum
    T : The temperature of the sample
    m : The linear scaling of the SiV peak position with temperature
    C0 : The offset in the above linear scaling
    center2 : The position of the calibration line
    width1 : The width (FWHM) of the SiV line
    width2 : The width (FWHM) of the calibration line
    light_background : The contribution to the background from stray light,
                       contributing shot noise
    ccd_backgrond : The contribution to the background from CCD readout,
                    contributing gaussian noise
    ccd_stdev : The standard deviation on the gaussian CCD noise above
    conv_range : How far out to go in convolution. 
                 Default is 6 sigma on the gaussian contribution.
    """

    x = spectrum.data.values.T[1]
    y = spectrum.data.values.T[0]
    return two_peak_log_likelihood(x, y, amp1, amp2, T, m, C0, center2, width1, 
    width2, light_background, ccd_background, ccd_stdev, conv_range)
