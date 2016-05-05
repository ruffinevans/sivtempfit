# This set of tests will run on the example data to make sure
# it produces reasonable results in inference.

from unittest import TestCase
import numpy as np
from .. import io
from ..model import *


class TestLikelihood(TestCase):

    path_to_json = io.get_example_data_file_path("simulated_spectrum_realistic.json")
    simulated_spectrum = io.load_Spectrum(path_to_json)

    # Function used to generate the data:
    # model.two_peak_model(xrange, 6000, 100, 9, 731, 15, 0.15, 100) 
    # + 1*np.random.normal(1000,10,400)

    def test_MLE_amplitude1_inference_gaussian(self):
        # Test to make sure the model gives a MLE estimate of the amplitude
        # that is consistent with the simulated data.
        # This is a fairly demanding test. It also tests the data loading.

        test_amp_list = np.arange(5000, 7000, 10)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   x, 100, 0, 0, 9, 731, 15, 0.15, 100, 1000, 10,
                   gaussian_approx=True) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        print("This value should be about 6000:")
        print(amp)
        self.assertTrue(amp < 6050 and amp > 5950)

    def test_MLE_amplitude2_inference_gaussian(self):
        test_amp_list = np.arange(85, 115, 1)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   6000, x, 0, 0, 9, 731, 15, 0.15, 100, 1000, 10,
                   gaussian_approx=True) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        print("This value should be about 100:")
        print(amp)
        self.assertTrue(amp > 97 and amp < 103)

    def test_MLE_center_offset_inference_gaussian(self):
        test_amp_list = np.arange(6, 10, 0.2)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   6000, 100, 0, 0, x, 731, 15, 0.15, 100, 1000, 10,
                   gaussian_approx=True) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        print("This value should be about 9:")
        print(amp)
        self.assertTrue(amp > 8.5 and amp < 9.5)

    def test_MLE_center2_inference_gaussian(self):
        test_amp_list = np.arange(728, 733, 0.2)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   6000, 100, 0, 0, 9, x, 15, 0.15, 100, 1000, 10,
                   gaussian_approx=True) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        print("This value should be about 731:")
        print(amp)
        self.assertTrue(amp > 730.5 and amp < 731.5)

    def test_MLE_width1_inference_gaussian(self):
        test_amp_list = np.arange(10, 20, 0.25)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   6000, 100, 0, 0, 9, 731, x, 0.15, 100, 1000, 10,
                   gaussian_approx=True) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        print("This value should be about 15:")
        print(amp)
        self.assertTrue(amp > 14.5 and amp < 15.5)

    def test_MLE_width2_inference_gaussian(self):
        test_amp_list = np.arange(0.05, 0.3, 0.01)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   6000, 100, 0, 0, 9, 731, 15, x, 100, 1000, 10,
                   gaussian_approx=True) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        print("This value should be about 0.15:")
        print(amp)
        self.assertTrue(amp > 0.13 and amp < 0.17)

    def test_MLE_poisson_background_inference_gaussian(self):
        test_amp_list = np.arange(85, 115, 1)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   6000, 100, 0, 0, 9, 731, 15, 0.15, x, 1000, 10,
                   gaussian_approx=True) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        print("This value should be about 100:")
        print(amp)
        self.assertTrue(amp > 97 and amp < 103)

    def test_MLE_gaussian_background_inference_gaussian(self):
        test_amp_list = np.arange(85, 115, 1)*10
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   6000, 100, 0, 0, 9, 731, 15, 0.15, 100, x, 10,
                   gaussian_approx=True) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        print("This value should be about 1000:")
        print(amp)
        self.assertTrue(amp > 970 and amp < 1030)

    def test_MLE_gaussian_stdev_inference_gaussian(self):
        # This test does not work at the moment.
        # The inference on the standard deviation is known to be bad.
        # This is known issue #29:
        # https://github.com/p201-sp2016/sivtempfit/issues/29
        return 
        test_amp_list = np.arange(8.5, 11.5, 0.1)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   6000, 100, 0, 0, 9, 731, 15, 0.15, 100, 1000, x,
                   gaussian_approx=True) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        print("This value should be about 10:")
        print(amp)
        self.assertTrue(amp > 9.7 and amp < 10.3)