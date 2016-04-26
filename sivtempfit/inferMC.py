# Module for doing MC sampling of the likelihood or posterior function using
# the emcee package. The syntax for emcee is very simple, so most of this
# module is just repackaging the likelihood functions in model.py into a
# version that can be easily passed to the sampler.

# Things to do:
# 1. Write function that generates guess balls
# 2. Write function that does emcee sampling given data. This is mostly a
#    matter of translating the model function to the form that emcee wants.
#    Eventually, will want to put in a prior as well, so leave space for that.
# 3. Write function that, given data and guesses, generates guess balls
#    and then does sampling. Resist the temptation to make this too automatic.
#    It is not a good use of your time.

import emcee
from . import model
from .dataprocessing import Spectrum
import numpy as np

def generate_sample_ball(data, calib_pos_guess, nwalkers = 20,
                         amp1_guess = None, amp1_std = None,
                         amp2_guess = None, amp2_std = None,
                         center_offset_guess = None, center_offset_std = None,
                         calib_pos_std = None,
                         width1_guess = None, width1_std = None,
                         width2_guess = None, width2_std = None,
                         light_background_guess = None, light_background_std = None,
                         ccd_background_guess = None, ccd_background_std = None,
                         ccd_stdev_guess = None, ccd_stdev_std = None,
                         debug = False):
    """
    Creates an emcee sample ball to use as the starting position for the emcee
    sampler.

    Arguments:
    ----------
    data : input data that will be passed to the emcee sampler elsewhere
    calib_pos_guess : the position of the (sharp) calibration line
                      in the spectrum

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
    should then be divided by the width guess.

    For the background contributions, we can estimate the ccd background
    contribution as 80 percent of the median y level and the light background
    as 5 percent of the median y level.

    For the ccd standard deviation, we will used a fixed estimate of 10.
    """

    # Assumes spectrum is unpacked in reverse order.
    if isinstance(data, Spectrum):
        x = data.data.values.T[1]
        y = data.data.values.T[0]
    else:
        x = data[0]
        y = data[1]

    median_y = np.median(y)

    if width1_guess is None:
        width1_guess = 2

    if width1_std is None:
        width1_std = 2

    if width2_guess is None:
        width2_guess = 0.02

    if width2_std is None:
        width2_std = 0.05

    if amp1_guess is None:
        amp1_guess = np.max(y)/width1_guess

    if amp1_std is None:
        amp1_std = 0.8*amp1_guess

    if amp2_guess is None:
        amp2_guess = amp1_guess

    if amp2_std is None:
        amp2_std = amp1_std

    if center_offset_guess is None:
        center_offset_guess = 739 - calib_pos_guess

    if center_offset_std is None:
        center_offset_std = 3

    if calib_pos_std is None:
        calib_pos_std = 0.5

    if light_background_guess is None:
        light_background_guess = 0.05 * median_y

    if light_background_std is None:
        light_background_std = light_background_guess

    if ccd_background_guess is None:
        ccd_background_guess = 0.8 * median_y

    if ccd_background_std is None:
        ccd_background_std = 0.5 * ccd_background_guess

    if ccd_stdev_guess is None:
        ccd_stdev_guess = 10

    if ccd_stdev_std is None:
        ccd_stdev_std = 10


    # Order of parameters is amp1, amp2, C0, center2, width1, width2,
    # light_background, ccd_background, ccd_stdev
    if debug:
        return ((amp1_guess, amp2_guess, center_offset_guess, calib_pos_guess,
                width1_guess, width2_guess, light_background_guess,
                ccd_background_guess, ccd_stdev_guess),
            (amp1_std, amp2_std, center_offset_std, calib_pos_std,
                width1_std, width2_std, light_background_std,
                ccd_background_std, ccd_stdev_std))

    return emcee.utils.sample_ball(
        (amp1_guess, amp2_guess, center_offset_guess, calib_pos_guess,
            width1_guess, width2_guess, light_background_guess,
            ccd_background_guess, ccd_stdev_guess),
        (amp1_std, amp2_std, center_offset_std, calib_pos_std,
            width1_std, width2_std, light_background_std,
            ccd_background_std, ccd_stdev_std),
        nwalkers)


def mc_likelihood_sampler(data, calib_pos, nwalkers = 20, starting_positions = None, 
                          run = True, nsteps = 2000):
    """
    Returns an emcee sampler object based on the supplied data and the
    likelihood from the model.

    Arguments:
    ----------
    data: either a list [x, y] of the x and y data or a Spectrum object.
    calib_pos: approximate position of the calibration line in the data.

    Optional Arguments:
    -------------------
    nwalkers: Number of emcee walkers. Default = 5
    starting_positions: Starting positions of walkers in parameter space.
    run: Whether or not to run the sampler before returning it to the user.
         If False, just returns the sampler object.
         If True, runs the sampler and then returns it.
    nsteps: Number of emcee steps. Default = 2000

    Note:
    -----
    See also generate_sample_ball which is used to generate starting_positions
    It will give a bit more fine-grained control over the sampling and may be
    useful if the emcee sampler does not produce reasonable results.
    """

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
    def log_likelihood(theta):
        # Theta is the list of parameters. We are only interested in a single
        # spectrum here, so we do not include T or m in our sampling
        amp1, amp2, C0, center2, width1, width2, light_background, \
            ccd_background, ccd_stdev = theta

        return model.two_peak_log_likelihood(x, y, amp1, amp2, 0, 0, C0,
                    center2, width1, width2, light_background, ccd_background,
                    ccd_stdev, conv_range = -1, debug = False, test_norm = False)

    # Here, we have 9 dimensions
    ndim = 9

    if starting_positions is None:
        # Call above function to get starting positions
        starting_positions = generate_sample_ball(data, calib_pos, nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)

    if not(run):
        return sampler

    sampler.run_mcmc(starting_positions, nsteps)
    return sampler
