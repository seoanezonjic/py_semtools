#! /usr/bin/env python

#########################################################
# Load necessary packages
#########################################################

import py_semtools
import unittest

#########################################################
# Define TESTS
#########################################################

class SemtoolsTest(unittest.TestCase):
  def test_that_it_has_a_version_number(self):

    self.assertEqual(type(py_semtools.__version__), str)

    version = py_semtools.__version__
    version = version.split('.')
    major, minor, patches = [int(i) for i in version]
    
    self.assertEqual(type(major), int)
    self.assertEqual(type(minor), int)
    self.assertEqual(type(patches), int)