# This set of tests will run on the example data to make sure
# it produces reasonable results in inference.

from unittest import TestCase
import numpy as np
from .. import io
from ..model import *


class TestLikelihood(TestCase):
    def test_MLE_amplitude_inference(self):
        # Test to make sure the model gives a MLE estimate of the amplitude
        # that is consistent with the simulated data.
        # This is a fairly demanding test. It also tests the data loading.
        path_to_json = io.get_example_data_file_path("simulated_spectrum.json")
        simulated_spectrum = io.load_Spectrum(path_to_json)
        test_amp_list = np.arange(400,600,0.5)
        test_ll = [two_peak_log_likelihood_Spectrum(simulated_spectrum, x, 10,
                    0, 0, -30, 70, 20, 0.1, 1, 1, 0.5) for x in test_amp_list]
        amp = test_amp_list[test_ll.index(max(test_ll))]
        self.assertTrue(amp < 505 and amp > 495)
