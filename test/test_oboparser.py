import unittest
import os
from py_semtools import Py_semtools
ROOT_PATH=os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data')

class OboParserTestCase(unittest.TestCase):

	def setUp(self):
		self.something = []

	def test_dummy(self):
		self.assertEqual(True, True)