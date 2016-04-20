# This set of tests will run on the example data to make sure it is in a
# reasonable format and produces reasonable results in inference.

from unittest import TestCase
import pandas as pd
import numpy as np
from .. import dataprocessing as dp
from .. import io
from ..model import *
import json


class TestSpectrumJSON(TestCase):
    # Test to make sure that Spectrum object to_json() method produces valid
    # json that can be read and dumped without loss.
    # Can't just check json.dumps(json.loads(...)) == spectrum.to_json
    # because json does not preserve order.
    def test_spectrum_JSON_formatting(self):
        test_DataFrame = pd.DataFrame({'wavelength': [730.7841, 730.7931, 730.8021],
                                       'V': [1500, 2500, 2000]})
        test_dict = test_dict = {"T": 300, "Name": "Test Data"}
        test_Spectrum = dp.Spectrum(test_DataFrame, test_dict)
        self.assertEqual(json.loads(test_Spectrum.to_json()),
                        json.loads(json.dumps(json.loads(test_Spectrum.to_json()))))

    def test_JSON_io(self):
        # Test to make sure JSON can be loaded from simple test file.
        path_to_json = io.get_example_data_file_path("test_json_simple.json")
        self.assertEqual(io.load_Spectrum(path_to_json).data['V'][0], 1500)


class TestSpectrum(TestCase):
    def test_spectrum_format(self):
        # Need to update this once we have example data.
        # s = sivtempfit.examplespectrum()
        self.assertTrue(isinstance("test", str))


class TestModel(TestCase):
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
