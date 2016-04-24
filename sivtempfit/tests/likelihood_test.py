# This set of tests will run on the example data to make sure
# it produces reasonable results in inference.

from unittest import TestCase
import numpy as np
from .. import io
from ..model import *


class TestLikelihood(TestCase):

    path_to_json = io.get_example_data_file_path("simulated_spectrum.json")
    simulated_spectrum = io.load_Spectrum(path_to_json)

    def test_MLE_amplitude1_inference(self):
        # Test to make sure the model gives a MLE estimate of the amplitude
        # that is consistent with the simulated data.
        # This is a fairly demanding test. It also tests the data loading.

        test_amp_list = np.arange(400, 600, 2)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   x, 10, 0, 0, -30, 70, 20, 0.1, 1, 1, 0.5) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        self.assertTrue(amp < 505 and amp > 495)

    def test_MLE_amplitude2_inference(self):
        test_amp_list = np.arange(5, 15, 0.5)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   500, x, 0, 0, -30, 70, 20, 0.1, 1, 1, 0.5) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        self.assertTrue(amp < 11 and amp > 9)

    def test_MLE_center1_inference(self):
        test_amp_list = np.arange(-50, 80, 1)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   500, 10, 0, 0, x, 70, 20, 0.1, 1, 1, 0.5) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        self.assertTrue(amp < -28 and amp > -32)

    def test_MLE_center2_inference(self):
        test_amp_list = np.arange(50, 100, 1)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   500, 10, 0, 0, -30, x, 20, 0.1, 1, 1, 0.5) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        self.assertTrue(amp > 68 and amp < 72)

    def test_MLE_width1_inference(self):
        test_amp_list = np.arange(10, 30, 0.5)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   500, 10, 0, 0, -30, 70, x, 0.1, 1, 1, 0.5) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        self.assertTrue(amp > 19 and amp < 21)

    def test_MLE_width2_inference(self):
        test_amp_list = np.arange(0.05, 0.25, 0.01)
        test_ll = [two_peak_log_likelihood_Spectrum(self.simulated_spectrum,
                   500, 10, 0, 0, -30, 70, 20, x, 1, 1, 0.5) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        self.assertTrue(amp > 0.08 and amp < 0.12)
