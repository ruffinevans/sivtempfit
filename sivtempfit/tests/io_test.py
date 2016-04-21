# This set of tests will run on the io module to make sure
# that Spectrum objects can be generated from the example data

from unittest import TestCase
from .. import io


class TestIO(TestCase):
    def test_JSON_io(self):
        # Test to make sure JSON can be loaded from simple test file.
        path_to_json = io.get_example_data_file_path("test_json_simple.json")
        self.assertEqual(io.load_Spectrum(path_to_json).data['V'][0], 1500)
