#! /usr/bin/env python

#########################################################
# Load necessary packages
#########################################################

import json
import unittest
import os
from py_semtools import Ontology, JsonParser


ROOT_PATH= os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data')

#########################################################
# Define TESTS
#########################################################
# class TestSimilitudes < Test::Unit::TestCase
class TestJSONparser(unittest.TestCase):
    def setUp(self):
        # Files
        self.file_Hierarchical = {"file": os.path.join(DATA_TEST_PATH, "hierarchical_sample.obo"), "name": "hierarchical_sample"}

        # Create necessary instnaces
        self.hierarchical = Ontology(file= self.file_Hierarchical["file"],load_file= True)


    def test_export_import(self):
        # Add extra info to instance
        self.hierarchical.precompute()
        self.hierarchical.get_IC("Child2")
        # Export object to JSON
        self.hierarchical.write(os.path.join(DATA_TEST_PATH, "testjson.json"))
               
        obo = Ontology()
        JsonParser.load(obo, os.path.join(DATA_TEST_PATH, "testjson.json"), build= True)
        
        self.assertEqual(self.hierarchical, obo)
        # Remove generated files
        os.remove(os.path.join(DATA_TEST_PATH, "testjson.json"))

    def test_export_import2(self):
        self.hierarchical.write(os.path.join(DATA_TEST_PATH, "testjson.json"))
        
        obo = Ontology()
        JsonParser.load(obo, os.path.join(DATA_TEST_PATH, "testjson.json"), build= True)

        self.hierarchical.precompute()
        jsonObo = Ontology(file= os.path.join(DATA_TEST_PATH, "testjson.json"), load_file= True)
        self.assertEqual(self.hierarchical,jsonObo)
        self.hierarchical.get_IC("Child2")
        obo.get_IC("Child2")
        self.assertEqual(self.hierarchical, obo)
        # Remove generated files
        os.remove(os.path.join(DATA_TEST_PATH, "testjson.json"))
