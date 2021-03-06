# Module for doing MC sampling of the likelihood or posterior function using
# the emcee package. The syntax for emcee is very simple, so most of this
# module is just repackaging the likelihood functions in model.py into a
# version that can be easily passed to the sampler.

import emcee
from . import model
from .dataprocessing import Spectrum
import numpy as np
import pandas as pd
import warnings
from scipy.optimize import curve_fit


# Define the log_likelihood in the form that the emcee sampler wants it.
# We will define this here so that it can be pickled, which is necessary for
# parallel evaluation.
# For an explanation, see the documentation here:
# https://docs.python.org/2/library/pickle.html#what-can-be-pickled-and-unpickled
def log_likelihood_params_2(theta, x, y, safe_ll=False, gaussian_approx=False):
    # Theta is the list of parameters. We are only interested in a single
    # spectrum here, so we do not include T or m in our sampling
    amp1, amp2, C0, center2, width1, width2, light_background, \
        ccd_background, ccd_stdev = theta

    return model.two_peak_log_likelihood(x, y, amp1, amp2, 0, 0, C0,
                center2, width1, width2, light_background, ccd_background,
                ccd_stdev, conv_range=-1, debug=False, test_norm=False,
                safe=safe_ll, gaussian_approx=gaussian_approx)


def log_likelihood_params_1(theta, x, y, safe_ll=False, gaussian_approx=False):
    # Theta is the list of parameters. We are only interested in a single
    # spectrum here, so we do not include T or m in our sampling
    amp, center, width, light_background, ccd_background, \
        ccd_stdev = theta

    return model.one_peak_log_likelihood(x, y, amp, 0, 0, center,
                width, light_background, ccd_background, ccd_stdev,
                conv_range=-1, debug=False, test_norm=False,
                safe=safe_ll, gaussian_approx=gaussian_approx)


def generate_sample_ball(data, calib_pos_guess, num_peaks, nwalkers=96,
                         amp1_guess=None, amp1_std=None,
                         amp2_guess=None, amp2_std=None,
                         center_offset_guess=None, center_offset_std=None,
                         calib_pos_std=None,
                         width1_guess=None, width1_std=None,
                         width2_guess=None, width2_std=None,
                         light_background_guess=None, light_background_std=None,
                         ccd_background_guess=None, ccd_background_std=None,
                         ccd_stdev_guess=None, ccd_stdev_std=None,
                         debug=False, return_y_values=False, tightness=1,
                         MLE_guesses=True):
    """
    Creates an emcee sample ball to use as the starting position for the emcee
    sampler.

    Arguments:
    ----------
    data : input data that will be passed to the emcee sampler elsewhere
    calib_pos_guess : the position of the (sharp) calibration line
                      in the spectrum. Irrelevant if the number of peaks
                      is one.
    num_peaks : the number of peaks in the spectrum (1 or 2). This deterimes
                whether or not the sample ball for the single peak or the
                two-peak model is returned.

    Optional Arguments:
    -------------------
    nwalkers : number of walkers that will be used in the emcee sampler.
               The default is five.
    (If None is passed to these arguments, the default is used.)
    amp1_guess : Guess for the amplitude of the first peak
    amp1_std : Standard deviation of the walkers for the amplitude parameter
    amp2_guess : Guess for the amplitude of the second peak
    amp2_std : Standard deviation of the walkers for the amplitude parameter
    center_offset_guess : Guess for the offset of the first peak relative
                          to the second
    center_offset_std
    calib_pos_std : Standard deviation for the guess of the calibration peak
                    position. Note that the calib_pos_guess argument is
                    mandatory.
    width1_guess : Guess for the width of the first peak
    width1_std
    width2_guess : Guess for the width of the second peak
    width2_std
    light_background_guess : Guess for the background contribution from light,
                             contributing Poisson (shot) noise
    light_background_std
    ccd_background_guess : Guess for the background contribution from the CCD,
                           contributing Gaussian noise
    ccd_background_std
    ccd_stdev_guess : Guess for the standard deviation of the Gaussian noise
    ccd_stdev_std
    debug : if True, returns the set of parameters passed to sample_ball
            instead of the sample_ball itself.
    return_y_values : if True, returns the model prediction based on the
                      center of the sample ball. Overrides the debug command.
                      Very useful for plotting against the data to see if it
                      approximately agrees.
    tightness : scaling factor for the _std variables. A higher tightness
                means a more localized sample_ball.
    MLE_guesses : use a Levenberg-Marquardt algorithm (scipy's `curve_fit`)
                  to refine the guesses to the maximum-likelihood estimates.
                  This is True by default and should drastically improve the
                  default reliability of the sampler. If this is True, the
                  tightness will also be increased by a factor of ten.
                  Note that supplying decent guesses for the other arguments
                  is still important, because they will be used as inputs for
                  the MLE.


    Explanation:
    ------------
    The function tries to choose reasonable values given the data according
    to the following logic:

    For the width of lorentzians, one will be around 2 and one will be
    around 0.02 on physical grounds.

    For the center positions, one will be around 738.
    The other peak is so sharp that we need a guess, which is why the calib_pos
    argument is mandatory.

    For the amplitude guess and range, we allow the peaks to have a height over
    the entire y range.
    For a Lorentzian, the value at max is ~ amp/width
    So to generate the amplitude guess, the guess for the peak height
    should then be multiplied by the width guess.

    For the background contributions, we can estimate the ccd background
    contribution as 80 percent of the median y level and the light background
    as 5 percent of the median y level.

    For the ccd standard deviation, we will used a fixed estimate of 10.
    """
    if not(num_peaks == 1 or num_peaks == 2):
        raise ValueError("The number of peaks in the model must be 1 or 2.")

    # Assumes spectrum is unpacked in reverse order.
    if isinstance(data, Spectrum):
        x = data.data.values.T[1]
        y = data.data.values.T[0]
    else:
        x = data[0]
        y = data[1]

    # The math below relies on having the y values in sorted order
    # I know the syntax here is bad, I'm in a hurry.
    # Should come back and fix this.
    # x = [x for (x, y) in sorted(zip(x_unsrt, y_unsrt))]
    # y = [y for (x, y) in sorted(zip(x_unsrt, y_unsrt))]

    # Also get the y value at the middle of the spectrum, close to the peak
    mid_index = int(np.floor(len(x) / 2))
    y_at_mid = y[mid_index]

    # To estimate the background, look at the last 5% of the values and
    # take the median.
    near_end_index = int(np.floor(len(x) * 0.95))
    median_y_bkrd = np.median(y[near_end_index:])

    if width1_guess is None:
        width1_guess = 4.7

    if width1_std is None:
        width1_std = 0.2 / tightness

    if width2_guess is None:
        width2_guess = 0.0113

    if width2_std is None:
        width2_std = 0.0005 / tightness

    if amp1_guess is None:
        amp1_guess = np.abs(1.8 * (y_at_mid - median_y_bkrd) * width1_guess)

    if amp1_std is None:
        amp1_std = 0.02 * amp1_guess / tightness

    if amp2_guess is None:
        amp2_guess = 1.5 * (np.max(y) - median_y_bkrd) * width2_guess

    if amp2_std is None:
        amp2_std = 0.02 * amp2_guess / tightness

    if center_offset_guess is None:
        if num_peaks == 2:
            center_offset_guess = 740 - calib_pos_guess
        else:
            # In this case, the center offset is the actual center position
            center_offset_guess = 740

    if center_offset_std is None:
        center_offset_std = 0.015 / tightness

    if calib_pos_std is None:
        calib_pos_std = 0.00015 / tightness

    if light_background_guess is None:
        light_background_guess = 0.003 * median_y_bkrd

    if light_background_std is None:
        light_background_std = 20 * light_background_guess / tightness

    if ccd_background_guess is None:
        ccd_background_guess = 0.95 * median_y_bkrd

    if ccd_background_std is None:
        ccd_background_std = 0.05 * ccd_background_guess / tightness

    if ccd_stdev_guess is None:
        ccd_stdev_guess = 10

    if ccd_stdev_std is None:
        ccd_stdev_std = 4 / tightness

    if MLE_guesses:
        # In this case, use scipy curve_fit to refine the user-supplied
        # or default guesses.
        if num_peaks == 1:
            # Take the guesses supplied by the user...
            guesses = (amp1_guess, center_offset_guess, width1_guess,
                       light_background_guess + ccd_background_guess)
            # ... perform the fit ...
            MLE_fit = curve_fit(model.one_peak_model, x, y, guesses)
            # ... and redefine the appropriate parameters based on the fit
            amp1_guess = MLE_fit[0][0]
            center_offset_guess = MLE_fit[0][1]
            width1_guess = MLE_fit[0][2]
            light_background_guess = MLE_fit[0][3] * 0.01
            ccd_background_guess = MLE_fit[0][3] * 0.99
        else:
            guesses = (amp1_guess, amp2_guess, center_offset_guess,
                       calib_pos_guess, width1_guess, width2_guess,
                       light_background_guess + ccd_background_guess)
            MLE_fit = curve_fit(model.two_peak_model, x, y, guesses)
            amp1_guess = MLE_fit[0][0]
            amp2_guess = MLE_fit[0][1]
            center_offset_guess = MLE_fit[0][2]
            calib_pos_guess = MLE_fit[0][3]
            width1_guess = MLE_fit[0][4]
            width2_guess = MLE_fit[0][5]
            light_background_guess = MLE_fit[0][6] * 0.01
            ccd_background_guess = MLE_fit[0][6] * 0.99
        # Finally, increase the tightness. We expect the parameters to
        # be very close to the correct values, so we can make the ball
        # tighter.
        tightness *= 10

    # Order of parameters is amp1, amp2, C0, center2, width1, width2,
    # light_background, ccd_background, ccd_stdev
    if return_y_values:
        # In this case, returns the model prediction for the center of the
        # sample_ball object. Useful for making plots to compare prediction
        # to data.
        if num_peaks == 2:
            return model.two_peak_model(x, amp1_guess, amp2_guess,
                                        center_offset_guess, calib_pos_guess,
                                        width1_guess, width2_guess,
                                        light_background_guess) \
                + ccd_background_guess
        else:
            return model.one_peak_model(x, amp1_guess, center_offset_guess,
                                        width1_guess, light_background_guess) \
                + ccd_background_guess

    if debug:
        if num_peaks == 2:
            return ((amp1_guess, amp2_guess, center_offset_guess, calib_pos_guess,
                     width1_guess, width2_guess, light_background_guess,
                     ccd_background_guess, ccd_stdev_guess),
                    (amp1_std, amp2_std, center_offset_std, calib_pos_std,
                     width1_std, width2_std, light_background_std,
                     ccd_background_std, ccd_stdev_std))
        else:
            return ((amp1_guess, center_offset_guess, width1_guess,
                     light_background_guess, ccd_background_guess, ccd_stdev_guess),
                    (amp1_std, center_offset_std, width1_std,
                     light_background_std, ccd_background_std, ccd_stdev_std))

    if num_peaks == 2:
        return emcee.utils.sample_ball(
            (amp1_guess, amp2_guess, center_offset_guess, calib_pos_guess,
                width1_guess, width2_guess, light_background_guess,
                ccd_background_guess, ccd_stdev_guess),
            (amp1_std, amp2_std, center_offset_std, calib_pos_std,
                width1_std, width2_std, light_background_std,
                ccd_background_std, ccd_stdev_std),
            nwalkers)
    else:
        return emcee.utils.sample_ball(
            (amp1_guess, center_offset_guess, width1_guess,
                light_background_guess, ccd_background_guess, ccd_stdev_guess),
            (amp1_std, center_offset_std, width1_std,
                light_background_std, ccd_background_std, ccd_stdev_std),
            nwalkers)

# Define an error class that will be used when the sampler has a zero
# acceptance fraction:

class MCSamplerError(Exception):
    """Exception raised for errors with the emcee sampler

    Attributes:
        sampler -- sampler object that produced the error
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

def mc_likelihood_sampler(data, calib_pos, num_peaks, nwalkers=96,
                          starting_positions=None, run=True, nsteps=1000,
                          threads=1, safe_ll=False, tightness=1,
                          gaussian_approx=False, override=False,
                          MLE_guesses=True):
    """
    Returns an emcee sampler object based on the supplied data and the
    likelihood from the model.

    Arguments:
    ----------
    data: either a list [x, y] of the x and y data or a Spectrum object.
    calib_pos: approximate position of the calibration line in the data.
    num_peaks : the number of peaks in the spectrum (1 or 2). This determines
                which model is fit to the data. If two peaks are selected, the
                model with the calibration line is fit. If one peak is
                selected, the model with just a single lorentzian is fit.

    Optional Arguments:
    -------------------
    nwalkers: Number of emcee walkers. Default = 5
    starting_positions: Starting positions of walkers in parameter space.
    run: Whether or not to run the sampler before returning it to the user.
         If False, just returns the sampler object.
         If True, runs the sampler and then returns it.
    nsteps: Number of emcee steps. Default = 2000
    threads: Number of threads to use in the emcee EnsembleSampler
    safe_ll: Use the "safe" version of the log-likelihood that does the
             convolution in a naive and definitely correct way, but is an
             order of magnitude slower than the improved version.
             Default: false.
    tightness: defines the size of the spread of the sample_ball in parameter
               space. Higher values of tighness mean a more localized
               sample_ball. For more control, see the generate_sample_ball
               function. If such a sample ball is passed as starting_positions,
               the tightness argument is ignored.
    gaussian_approx: Use a gaussian approximation for the poisson distribution
                     in the log-likelihood.
                     This allows the calculation to be sped up by several
                     orders of magnitude and is a good approximation as long
                     as the relevant signal is greater than about ten counts.
    MLE_guesses : use a Levenberg-Marquardt algorithm (scipy's `curve_fit`)
              to refine the initial parameter guesses to the maximum-likelihood
              estimates. This is True by default and should drastically improve
              the default reliability of the sampler. If this is True, the 
              tightness will also be increased by a factor of ten.

    Note:
    -----
    See also generate_sample_ball which is used to generate starting_positions
    It will give a bit more fine-grained control over the sampling and may be
    useful if the emcee sampler does not produce reasonable results.

    For more information on emcee, see the documentation:
    http://dan.iel.fm/emcee/current/
    """

    if not(num_peaks == 1 or num_peaks == 2):
        raise ValueError("The number of peaks in the model must be 1 or 2.")

    # Check to see if a spectrum object is passed. If so, use the data in the
    # Spectrum object. Otherwise, assume the user has passed a list of x/y
    # pairs.
    #
    # Note that the order of the data in the Spectrum object could be reversed
    # which is a bug. I will need to account for this later.
    if isinstance(data, Spectrum):
        x = data.data.values.T[1]
        y = data.data.values.T[0]
    else:
        x = data[0]
        y = data[1]

    # Define the log_likelihood in the form that the emcee sampler wants it.
    if num_peaks == 2:
        def log_likelihood(theta):
            # Theta is the list of parameters. We are only interested in a single
            # spectrum here, so we do not include T or m in our sampling
            amp1, amp2, C0, center2, width1, width2, light_background, \
                ccd_background, ccd_stdev = theta

            return model.two_peak_log_likelihood(x, y, amp1, amp2, 0, 0, C0,
                        center2, width1, width2, light_background, ccd_background,
                        ccd_stdev, conv_range=-1, debug=False, test_norm=False,
                        safe=safe_ll, gaussian_approx=gaussian_approx)
    else:
        def log_likelihood(theta):
            amp, center, width, light_background, ccd_background, \
                ccd_stdev = theta

            return model.one_peak_log_likelihood(x, y, amp, 0, 0, center,
                        width, light_background, ccd_background, ccd_stdev,
                        conv_range=-1, debug=False, test_norm=False,
                        safe=safe_ll, gaussian_approx=gaussian_approx)

    if num_peaks == 2:
        # Here, we have 9 dimensions
        ndim = 9
    else:
        # In this case, we have only six
        ndim = 6

    if starting_positions is None:
        # Call above function to get starting positions
        starting_positions = generate_sample_ball(data, calib_pos, num_peaks,
                                                  nwalkers, tightness=tightness,
                                                  MLE_guesses=MLE_guesses)

    if not(isinstance(threads, int)):
        raise ValueError('Threads must be a positive integer.')

    if threads == 1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, threads=1)
    elif threads >= 1:
        # I'm pretty sure this works fine now, so removing the warning:
        # warnings.warn("Multithreaded emcee in the context of the sivtempfit "+
        #               "package is not thoroughly tested. Use at your own risk!",
        #               RuntimeWarning)
        if num_peaks == 2:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_params_2,
                                            args=[x, y, safe_ll, gaussian_approx],
                                            threads=threads)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_params_1,
                                            args=[x, y, safe_ll, gaussian_approx],
                                            threads=threads)
    else:
        raise ValueError('Threads must be a positive integer.')

    if not(run):
        return sampler

    # If override is true or the number of steps is small, just run the
    # sampler with the requested number of steps.
    if override or nsteps <= 100:
        sampler.run_mcmc(starting_positions, nsteps)
        return sampler

    # If we're still here, run the sampler for 50 steps. Then, check to see
    # if the acceptance fraction is exactlly zero. If it is, raise an error,
    # and tell the user that this error can be overridden with the
    # override=True flag.
    sampler.run_mcmc(starting_positions, 50)

    mean_af = sampler.acceptance_fraction.mean()
    if mean_af <= 0.01:
        raise MCSamplerError(sampler, "Acceptance fraction: " + str(mean_af) +
                             " too low! Aborting! If you want to continue" +
                             " anyway, rerun with the `override=True` flag.")
    else:
        sampler.run_mcmc(starting_positions, nsteps)

    return sampler


def parameter_samples_df(sampler, burn_in=500, tracelabels=None,
                         keep_zero_af=False):
    """
    Generate a pandas dataframe from the samples in an emcee object. Useful
    e.g. to run sns.pairplot(parameter_samples, markers='.') to visualize
    MAP distributions of parameters after sampling.

    Optional argument tracelabels is a set of labels to use for the parameters.
    If None is passed, the default parameters labels be used. These correspond
    to the two-peak model.

    The optional argument keep_zero_af keeps emcee walkers that have a zero
    acceptance fraction. By default, these walkers are discarded.

    Discards the first burn_in samples from the sampler.
    """
    if tracelabels is None:
        if sampler.chain.shape[-1] == 9:
            tracelabels = ['A1', 'A2', 'C0', 'C2', 'w1', 'w2', 'lb', 'cb', 'cs']
        elif sampler.chain.shape[-1] == 6:
            tracelabels = ['A', 'C', 'W', 'LB', 'CB', 'Cs']
        else:
            raise ValueError("It seems like the sampler does not correspond" +
                             "to either the one-peak or two-peak case." +
                             "(The sampler chain has " + sampler.chain.shape[-1] +
                             "parameters.)")

    if burn_in > sampler.chain.shape[1]:
        raise ValueError('burn_in should be less than the chain length!')

    # Create a mask that is True where the acceptance fraction is nonzero.
    zero_af_mask = sampler.acceptance_fraction != 0

    if keep_zero_af:
        samples = sampler.chain[:, burn_in:, :]
        # If keep_zero_af is True and there are some that have zero,
        # issue a warning.
        if not(all(zero_af_mask)):
                warnings.warn("Some of the walkers have" + \
                    " zero acceptance fraction, but they are" + \
                    " being kept anyway.", RuntimeWarning)
    else:
        samples = sampler.chain[zero_af_mask, burn_in:, :]

    traces = samples.reshape(-1, len(tracelabels)).T
    return pd.DataFrame({tracelabels[i]: traces[i]
                         for i in range(len(tracelabels))})


def credible_intervals_from_sampler(sampler, burn_in=500, interval_range=0.68,
                                    print_out=True, tracelabels=None):
    """
    Returns credible intervals for the parameters generated from the sampler
    object in the format [[parameters], [upper bounds], [lower bounds]]

    Arguments:
    ----------
    sampler : emcee sampler object

    Optional Arguments:
    -------------------
    burn_in : how many samples to discard (default: 1000)
    interval_range : what level of credibility? (Default: 68\% range).
                     Can accept a list of multiple levels, in which case
                     a list of lists is returned.
    print_out : if True, also print the output in a human-readable format.
    tracelables : A list containing labels for the different parameters.
                  Not necessary if you are optimizing over the typical set
                  of nine parameters.
                  (For this function, the only important thing about this
                  list is that it contains the correct number of elements.
                  The labels can be anything.)

    See also: parameter_samples_df
    """
    parameter_samples = parameter_samples_df(sampler, burn_in, tracelabels=tracelabels)

    if print_out:
        if type(interval_range) == list:
            for y in interval_range:
                q = parameter_samples.quantile([0.50 - y/2, 0.50, 0.50 + y/2], axis=0)
                print()
                print("For credibility level {:.3f}:".format(y))
                for x in parameter_samples.columns:
                    print(str(x) + " = {:.4f} + {:.4f} - {:.4f}".format(q[x][0.50],
                                                        q[x][0.50 + y/2]-q[x][0.50],
                                                        q[x][0.50]-q[x][0.50 - y/2]))

        elif type(interval_range) == int or type(interval_range) == float:
            q = parameter_samples.quantile([0.50 - interval_range/2, 0.50,
                                            0.50 + interval_range/2], axis=0)
            for x in parameter_samples.columns:
                print(str(x) + " = {:.4f} + {:.4f} - {:.4f}".format(q[x][0.50], 
                                                        q[x][0.50 + interval_range/2]-q[x][0.50],
                                                        q[x][0.50]-q[x][0.50 - interval_range/2]))
        else:
            raise ValueError('interval_range must be a number or list of numbers')

    if type(interval_range) == list:
        all_ranges = [[[0, 0, 0] for x in parameter_samples.columns]
                      for y in interval_range]
        for yindex, y in enumerate(interval_range):
            q = parameter_samples.quantile([0.50 - y / 2, 0.50, 0.50 + y / 2], axis=0)
            for xindex, x in enumerate(parameter_samples.columns):
                all_ranges[yindex][xindex] = [q[x][0.50],
                                              q[x][0.50 + y / 2] - q[x][0.50],
                                              q[x][0.50] - q[x][0.50 - y / 2]]
        return all_ranges
    elif type(interval_range) == int or type(interval_range) == float:
        q = parameter_samples.quantile([0.50 - interval_range / 2, 0.50,
                                        0.50 + interval_range / 2], axis=0)
        return [[q[x][0.50],
                 q[x][0.50 + interval_range / 2] - q[x][0.50],
                 q[x][0.50] - q[x][0.50 - interval_range / 2]]
                for x in parameter_samples.columns]
    else:
        raise ValueError('interval_range must be a number or list of numbers')

