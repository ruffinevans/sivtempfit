import numpy as np
import scipy.stats as stats
import warnings

def lorentz(x, center, width):
    """
    A lorentzian function with peak position 'center' and FWHM 'width'.
    Returns non-negative values only.
    """

    return np.fabs((1 / np.pi) * (width / 2) / ((x - center)**2 + (width / 2)**2))


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
                            test_norm = False, safe = False):
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
    conv_range : How many sigmas of the poisson distribution to go out to in
                 the convolution. For example, if there are 1000 photons in a
                 bin, the convolution is centered around 1000 and goes out to
                 1000 +/- conv_range * sqrt(1000). Default value is -1, which
                 goes out to 2.5 sigma. Note that the beahvior of this argument
                 changes if the 'safe' argument is True. (See below.)
    debug : if True, returns the convolution list, the convolution
            matrix, the poisson matrix, the gaussian matrix, the
            likelihood list, and the computed max_y. False by default,
            in which case only the log-likelihood is returned.
    test_norm: if True, tests to make sure that the gaussian and poisson
               contributions are normalized. This is the default case.
               If False, the function will run a little faster but will
               not check this condition.
    safe : if True, calculate the likelihood using the super-safe default
           for the convolution ranges. This is much slower, but possibly
           useful if you are getting unexpected results in inference.
           Default: False.
           This changes the behavior of the conv_range command.
           If safe is True, conv_range is the total range to go out to
           in the convolution (not the range in terms of sigma).
           The default behavior also changes: in the default case, the
           convolution goes out to min([max(y), max(model_prediction)]) plus
           3 * sqrt of this max.
    """

    # First, get the contribution from the light signal.
    y_two_peak_model = two_peak_model(x, amp1, amp2, C0 + T * m,
                                      center2, width1, width2,
                                      light_background)

    # If the prediction from the model is ever <= 0,
    # the likelihood is zero and the log-likelihood is -np.inf
    # This is because the system can never produce negative
    # photons.
    #
    # If either background contribution is less than zero,
    # the log-likelihood should also be negative infinity
    # 
    # This is, to some extent, including a prior.

    if (np.min(y_two_peak_model) <= 0 or ccd_background < 0
        or light_background < 0) and not(debug):
        return -np.inf

    # If safe is True, then do the naive convolution over the entire range
    # of the data. This is time consuming, but should work no matter what.
    # If safe is False (the default case), the bounds of the convolution will
    # be adjusted for each data point. This can dramatically speed up the
    # calculations (I predict 5-10x improvement) but could cause some
    # undesirable consequences that I cannot forsee.
    #
    # To see how this works, just do a simple change of variables in the
    # convolution. Instead of taking n going from yi-gamma to yi+gamma, where 
    # gamma is some sufficiently large number so that the probability is zero
    # elsewhere, take:
    # x = n - y_i
    # going from - gamma to gamma.
    # Mathematically, this does nothing, but it allows us to use a much smaller
    # matrix for the convolution range.

    if safe:
        # Construct the bounds of convolution
        # If the conv_range is -1, go out to the max of the
        # max(y, model prediction) + 3x the square root of the max.
        # etc. for min
        max_y = 0
        if conv_range == -1:
            # Optimize these, maybe do max of min and min of max
            # Before, this was not this way, which meant that size of matrices
            # could grow arbitrarily depending on prediction. This way they are at
            # least bounded by the actual data.
            # 
            # The values from y should have the ccd_background subtracted, because
            # the convolution goes over the physical photons, not the ccd
            # background.
            #
            # This passes the tests, I think it is OK in inference.
            min_y = np.max([np.min(y_two_peak_model), np.min(y)-ccd_background])
            max_y = np.min([np.max(y_two_peak_model), np.max(y)-ccd_background])
            # Shouldn't this be np.max below? Was np.min w/ min_y in the
            # np.sqrt. Could cause some problems. I think it was related
            # to not having the ccd_background above.
            conv_min = np.floor(np.max([0, min_y - 3 * np.sqrt(min_y)]))
            conv_max = np.floor(max_y + 3 * np.sqrt(max_y))
        elif conv_range > 0:
            conv_max = conv_range
        else:
            raise ValueError('Range for convolution must be positive')
        
        conv_list = np.arange(conv_min, conv_max)[:, np.newaxis]
        # Construct the convolution matrix by broadcasting
        # Probably there is a faster way to do this.
        conv_mat = conv_list + 0*y
        
        # Construct poisson term
        poisson_term = stats.poisson._pmf(conv_mat, y_two_peak_model)
        # Construct gaussian term
        gaussian_term = ((1 / (np.sqrt(2.0 * np.pi * ccd_stdev**2))) *
                         np.exp(-1.0 * (y - conv_mat - ccd_background)**2 /
                                (2.0 * ccd_stdev**2)))
    else:
        # else: use fast (not "safe" technique)

        # conv_range in this case is handled a little differently
        # See docstring.
        if conv_range == -1:
            conv_range = 2.5
        elif conv_range > 0:
            pass
        else:
            raise ValueError('Range for convolution must be positive')

        # Construct the bounds of convolution
        # floor is important here so that convolution range is only over ints.
        # min_y is unnecessary for this case, but useful to pass as output of
        # debug so that format is consistent.
        min_y = np.max([np.min(y_two_peak_model), np.min(y)-ccd_background])
        max_y = np.min([np.max(y_two_peak_model), np.max(y)-ccd_background])
        conv_max = conv_range*np.sqrt(max_y)
        conv_min = -1*conv_max
        # Need to subtract the ccd_background here, because we are comparing
        # conv_mat + y to the prediction from y_two_peak_model and the two
        # differ by the ccd_background prediction. One way to look at this is
        # that the convolution should go over the *physical* number of photons
        # (which does not include the ccd_background) contribution, but y
        # includes a contribution from the background, so we need to subtract
        # the background from y, i.e. add its negative to the conv_list.
        # If the background is some crazy thing, the likelihood will be low,
        # but that's OK. (It's kind of the point.)
        #
        # We need to subtract it here and not e.g. in the probability
        # calculation for consistency.
        conv_list = np.arange(conv_min, conv_max)[:,np.newaxis]-np.floor(ccd_background)
        # Construct the convolution matrix by broadcasting.
        # Note that this should NOT be conv_list - y.
        # That would effectively undo our change of variables and make it
        # mathematically equivalent to the safe case. (I have checked this.)
        conv_mat = (conv_list - 0*y)
        # Don't allow negative photons.
        # Unfortunately, this requires us to use nansum below, which I really
        # don't like.
        # This would be handled elegantly if we could use pmf instead of _pmf,
        # because pmf (although a factor of two slower) will set negative
        # values to zero. However, it also sets non-integer values to zero,
        # and we can have non-integer values if y is non-integer valued.
        #
        # TODO: do change of variables such that we have an integer part of y
        # and a non-integer part of y. The non-integer part shows up in the
        # gaussian part, and the integer part shows up in the poisson part
        # but I am not worrying about this for now.
        conv_mat[conv_mat + y < 0] = np.nan
        # Construct poisson term. Note change of variables for the mean.
        poisson_term = stats.poisson._pmf(conv_mat + y, y_two_peak_model)
        # Construct gaussian term. Change of variables means we just
        # conv_mat vs. background.
        gaussian_term = ((1/(np.sqrt(2.0*np.pi*ccd_stdev**2)))*
                         np.exp(-1.0*(conv_mat+ccd_background)**2 / 
                                (2.0*ccd_stdev**2)))

    # Now, in either case (safe or not safe) we have a list of likelihoods
    # from the poisson and gaussian terms:

    # Perform sum over zeroth axis to get likelihoods
    # nansum is extremely dangerous for debugging and best avoided
    likelihood_list = np.nansum(poisson_term * gaussian_term, 0)

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
        warnings.warn("Testing for normalization is no longer meaningful. "+
                      "If you are trying to extract information this way, "+
                      "you may be disappointed.")
        gauss_norm = all(np.sum(gaussian_term.T, axis = 1) > 0.99)
        poiss_norm = all(np.sum(poisson_term.T, axis = 1) > 0.99)
        if not(gauss_norm or poiss_norm):
            raise ValueError('The terms in the convolution are' +
                             ' not normalized. Try increasing' +
                             ' the conv_range variable!')
    if debug:
        return [conv_list, conv_mat, poisson_term,
                gaussian_term, likelihood_list, 
                [min_y, max_y, conv_min, conv_max], y_two_peak_model]

    # Return sum of log_likelihoods
    ll = np.sum(np.log(likelihood_list))
    return ll


def two_peak_log_likelihood_Spectrum(spectrum, amp1, amp2, T, m, C0, center2,
        width1, width2, light_background, ccd_background, ccd_stdev,
        conv_range = -1, debug = False, test_norm = False, safe = False):
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
    safe : if True, calculate the likelihood using the super-safe default
           for the convolution ranges. This is much slower, but possibly
           useful if you are getting unexpected results in inference.
           Default: False
    """
    x = spectrum.data.values.T[1]
    y = spectrum.data.values.T[0]
    return two_peak_log_likelihood(x, y, amp1, amp2, T, m, C0, center2, width1, 
             width2, light_background, ccd_background, ccd_stdev, conv_range
             , debug, test_norm, safe)