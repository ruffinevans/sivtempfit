import numpy as np
import scipy.stats as stats


def lorentz(x, center, width):
    """
    A lorentzian function with peak position 'center' and FWHM 'width'.
    """

    return (1 / np.pi) * (width / 2) / ((x - center)**2 + (width / 2)**2)


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
            amp1 * lorentz(x, center_offset + center2, width1) +
            amp2 * lorentz(x, center2, width2))


def two_peak_log_likelihood(x, y, amp1, amp2, T, m, C0, center2,
                            width1, width2, light_background,
                            ccd_background, ccd_stdev,
                            conv_range = -1, debug = False,
                            test_norm = True):
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
    ccd_stdev : The standard deviation on the gaussian CCD noise

    Optional Arguments:
    -------------------
    conv_range : How far out to go in convolution. Default is -1, which
                 goes out to max([max(y), max(model_prediction)]) plus
                 3 * sqrt of this max.
    debug : if True, returns the convolution list, the convolution
            matrix, the poisson matrix, the gaussian matrix, the
            likelihood list, and the computed max_y. False by default,
            in which case only the log-likelihood is returned.
    test_norm: if True, tests to make sure that the gaussian and poisson
               contributions are normalized. This is the default case.
               If False, the function will run a little faster but will
               not check this condition.
    """

    # First, get the contribution from the light signal.
    y_two_peak_model = two_peak_model(x, amp1, amp2, C0 + T * m,
                                      center2, width1, width2,
                                      light_background)

    # Construct the bounds of convolution
    # If the conv_range is -1, go out to the max of the
    # max(y, model prediction) + 5x the square root of the max.
    max_y = 0
    if conv_range == -1:
        max_y = np.max([np.max(y_two_peak_model), np.max(y)])
        conv_max = max_y + 3 * np.sqrt(max_y)
    elif conv_range > 0:
        conv_max = conv_range
    else:
        raise ValueError('Range for convolution must be positive')
    conv_list = np.arange(0, conv_max)[:, np.newaxis]
    # Construct the convolution matrix by broadcasting
    conv_mat = (conv_list - y) + y

    # Construct poisson term
    poisson_term = stats.poisson.pmf(conv_mat, y_two_peak_model)
    # Construct gaussian term
    gaussian_term = ((1 / (np.sqrt(2.0 * np.pi * ccd_stdev**2))) *
                     np.exp(-1.0 * (y - conv_mat - ccd_background)**2 /
                            (2.0 * ccd_stdev**2)))
    # Perform sum over zeroth axis to get likelihoods
    likelihood_list = np.sum(poisson_term * gaussian_term, 0)

    # Test to make sure poisson and gaussian terms are normalized.
    # In other words, make sure the sum in the convolution is going
    # out far enough.
    # I think it is OK to test only one of them, because if we are
    # going completely over the support of at least one of them then
    # it means that we are not missing out on any probability density
    # in the convolution.
    # My motivation for doing `or` instead of `and` is that the gaussian
    # term seems to suffer from some discretization errors. If the
    # standard deviation is small, then points can land at non-integer
    # y values, which can generate a low likelihood just for the gaussian
    # term.
    if test_norm:
        gauss_norm = all(np.sum(gaussian_term.T, axis = 1) > 0.99)
        poiss_norm = all(np.sum(poisson_term.T, axis = 1) > 0.99)
        if not(gauss_norm or poiss_norm):
            raise ValueError('The terms in the convolution are' +
                             ' not normalized. Try increasing' +
                             ' the conv_range variable!')
    if debug:
        return [conv_list, conv_mat, poisson_term,
                gaussian_term, likelihood_list, conv_max]
    # Return sum of log_likelihoods
    return np.sum(np.log(likelihood_list))


def two_peak_log_likelihood_Spectrum(spectrum, amp1, amp2, T, m, C0, center2,
        width1, width2, light_background, ccd_background, ccd_stdev,
        conv_range = -1, debug = False, test_norm = True):
    """
    Returns the log-likelihood calculated for the two-peak + CCD noise model.
    See also: two_peak_model

    Parameters:
    -----------
    spectrum : data passed as an instance of Spectrum class
    amp1: amplitude of the broad SiV peak in the spectrum
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
    ccd_stdev : The standard deviation on the gaussian CCD noise

    Optional Arguments:
    -------------------
    conv_range : How far out to go in convolution. Default is -1, which
                 goes out to max([max(y), max(model_prediction)]) plus
                 3 * sqrt of this max.
    debug : if True, returns the convolution list, the convolution
            matrix, the poisson matrix, the gaussian matrix, the
            likelihood list, and the computed max_y. False by default,
            in which case only the log-likelihood is returned.
    test_norm: if True, tests to make sure that the gaussian and poisson
               contributions are normalized. This is the default case.
               If False, the function will run a little faster but will
               not check this condition.
    """
    x = spectrum.data.values.T[1]
    y = spectrum.data.values.T[0]
    return two_peak_log_likelihood(x, y, amp1, amp2, T, m, C0, center2, width1, 
             width2, light_background, ccd_background, ccd_stdev, conv_range
             , debug, test_norm)