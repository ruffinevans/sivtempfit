# This set of tests will test to make sure the framework
# of the package behaves properly.

from unittest import TestCase


class TestStringTest(TestCase):
    def test_string_test(self):
        # Trivial test to make sure test runs
        self.assertTrue(isinstance("test", str))
