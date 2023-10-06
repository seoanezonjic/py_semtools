#! /usr/bin/env python

#########################################################
# Load necessary packages
#########################################################

import json
import unittest
import os
import sys
from py_semtools.sim_handler import text_similitude, ctext_AtoB, complex_text_similitude, similitude_network


ROOT_PATH= os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data')


#########################################################
# Define TESTS
#########################################################

class TestSimilitudes(unittest.TestCase):
    #################################
    # STRING SIMILITUDE
    #################################

    ## Check simple similitude
    def test_simple_sim(self):
        self.assertEqual(1, text_similitude("abcde","abcde")) # Exact similitude
        self.assertEqual(0.75, text_similitude("abcdf","abcde")) # Same length, one char diff
        self.assertEqual(1, text_similitude("!a%b#d","!a%b#d")) # Special characters at the end
        self.assertEqual(1, text_similitude("abc\n","abc\t")) # Special characters at the end
        self.assertEqual(1, text_similitude("\nabc","\tabc")) # Special characters at the beginning
        self.assertEqual(0.8, text_similitude("abcd","abc")) # Different length
        self.assertEqual(0, text_similitude("abcd","def")) # Absolute different
        self.assertEqual(0.6, text_similitude("abcd","asdabcde")) # Subset
        self.assertEqual(0.5, text_similitude("abcd","abcefd")) # Interference

    ## Check sets similitude
    def test_sets_sim(self):
        self.assertEqual([1.0,1.0],ctext_AtoB(["abc","def"],["abc","def"])) # Equal sets, same order
        self.assertEqual([1.0,1.0],ctext_AtoB(["abc","def"],["def","abc"])) # Equal sets, different order
        self.assertEqual([1.0],ctext_AtoB(["abc"],["abc","def"])) # Subset A
        self.assertEqual([1.0,0.0],ctext_AtoB(["abc","def"],["abc"])) # Subset B
        self.assertEqual([0.0,0.0],ctext_AtoB(["abc","def"],["cfg"])) # Absolute different

    ## Check complex texts similitude
    def test_complex_sim(self):
        self.assertEqual(1, complex_text_similitude("abc|def","abc|def","|","")) # Complex equal, non config
        self.assertEqual(1, complex_text_similitude("abc|def","abc|de%f","|","%")) # Complex equal, removing chars
        self.assertEqual(0.5, complex_text_similitude("abc|def","abc|ajk","|","")) # Complex partially equal
        self.assertEqual(0.5, complex_text_similitude("abc|ajk","abc|def","|","")) # Complex partially equal (2)
        self.assertEqual(0.75, complex_text_similitude("abc","abc|def","|","")) # Complex against not complex
        self.assertEqual(0.0, complex_text_similitude("abc|def","ghk|jlmn","|","")) # Complex all different

    ## Check complex set similitudes
    def test_complex_set_sim(self):
        self.assertEqual({"abc": {"def": 0.0}},similitude_network(["abc","def"], splitChar= ";", charsToRemove= "", unique= False)) # Simple without repetition
        self.assertEqual({"abc": {"def": 0.0}},similitude_network(["abc","def","abc"], splitChar= ";", charsToRemove= "", unique= True)) # Simple with repetitions - filtered
        self.assertEqual({"abcdf": {"abcde": 0.75}},similitude_network(["abcdf","abcde","abcdf"], splitChar= ";", charsToRemove= "", unique= True)) # Simple with repetitions (2) - filtered
        self.assertEqual({"abcdf": {"abcdf": 1.0}},similitude_network(["abcdf","abcdf"], splitChar= ";", charsToRemove= "", unique= False)) # Simple with repetitions - unfiltered


    #################################
    # ROBINSON-RESNICK SIMILITUDE
    #################################

    #################################
    # NLP SIMILITUDES
    #################################
