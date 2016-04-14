# This set of tests will run on the example data to make sure it is in a
# reasonable format

from unittest import TestCase


class TestSpectrum(TestCase):
    def test_spectrum_format(self):
        # Need to update this once we have example data.
        #s = sivtempfit.examplespectrum()
        self.assertTrue(isinstance("test", str))
