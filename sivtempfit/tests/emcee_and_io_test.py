# This test reads in some horiba multi-acquisition test data into Spectrum
# format and also adds metadata at the same time. This covers a large section
# of the io module that was not previously covered.
#
# Then it does some fast emcee-based inference and compares it to some 
# well-studied results.

from unittest import TestCase
import numpy as np
from .. import io
from .. import inferMC as mc
from .. import model


class TestEmceeAndHoriba(TestCase):

    # Path to multiacquisition data
    path_to_data = io.get_example_data_file_path("varying acquisition time horiba.txt")
    # Add example times as metadata:
    times = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10]
    times_dict = [{"Acquisition Time" : i} for i in times]
    # import spectra, add both data and metadata
    spectra = io.import_horiba_multi(path_to_data, {"Purpose" : "Test"}, times_dict)

    def horiba_data_loading_test(self):
        # Test to make sure the spectra have been loaded with metadata
        print("This value should be 3:")
        print(self.spectra[3].metadata['Acquisition Time'])
        self.assertTrue(self.spectra[3].metadata['Acquisition Time'] == 3)

    def emcee_test(self):
        # Tests emcee inference in the gaussian approximation
        max_y_point = np.argmax(self.spectra[0].data['ydata1'])
        calib_pos = self.spectra[0].data['xdata'][max_y_point]
        sampler500 = mc.mc_likelihood_sampler(self.spectra[0], calib_pos,
                                              nwalkers=100, nsteps=1000,
                                              threads=2, gaussian_approx=True)
        intervals = np.array(mc.credible_intervals_from_sampler(sampler500))
        good_intervals = np.array([[2569.347335, 40.92276413, 48.62664087],
                                   [87.33231019, 1.034235807, 1.212639393],
                                   [9.024622113, 0.017412805, 0.016381177],
                                   [730.9377760, 0.000130463, 0.000146847],
                                   [991.9602700, 53.15859819, 19.33849000],
                                   [3.210781458, 5.529171488, 5.896434865],
                                   [83.02959825, 20.23707837, 60.39641023],
                                   [7.195214738, 0.097707853, 0.093495717],
                                   [0.011431565, 0.000163831, 0.000176602]])
        parameter_errors = ((intervals.T[0] - good_intervals.T[0]) / 
                                good_intervals.T[0])
        print("Here are the list of errors between the known \"good\""+
              " inference and the inference performed during during testing:")
        print(parameter_errors)
        print("Median absolute fractional error:")
        print(np.median(np.abs(parameter_errors)))
        print("If the median abs error is more then 10%, the test will fail")
        print("The test may also fail if your machine cannot run 2 threads.")
        self.assertTrue(np.median(np.abs(parameter_errors)) < 0.1)
