# This set of tests will run on the Spectrum class and associated objects

from unittest import TestCase
import pandas as pd
import numpy as np
from .. import dataprocessing as dp
from .. import io
from ..model import *
import json


class TestSpectrum(TestCase):
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
