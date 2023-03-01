#! /usr/bin/env python

#########################################################
# Load necessary packages
#########################################################

import unittest
import os
from py_semtools import Ontology


ROOT_PATH= os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data')

#########################################################
# Define TESTS
#########################################################
# class TestSimilitudes < Test::Unit::TestCase
class TestGO(unittest.TestCase):

    def test_go_export_import(self):
        self.go = Ontology(file= os.path.join(DATA_TEST_PATH, "go-basic_sample.obo"), load_file= True)
        # Export object to JSON
        self.go.write(os.path.join(DATA_TEST_PATH, "gotestjson.json"))
        obo = Ontology(file= os.path.join(DATA_TEST_PATH, "gotestjson.json"), load_file= True)
        self.assertEqual(self.go, obo)
        # Remove generated files
        os.remove(os.path.join(DATA_TEST_PATH, "gotestjson.json"))


    def test_go_export_import_several_records(self):
        self.go = Ontology(file= os.path.join(DATA_TEST_PATH, "partial_go.obo"), load_file= True)
        # Export object to JSON
        self.go.write(os.path.join(DATA_TEST_PATH, "gotestjsonFull.json"))
        #file= os.path.join(DATA_TEST_PATH, "testjson.json"
        obo = Ontology(file= os.path.join(DATA_TEST_PATH, "gotestjsonFull.json"), load_file= True)
        self.assertEqual(self.go, obo)
        # Remove generated files
        os.remove(os.path.join(DATA_TEST_PATH, "gotestjsonFull.json"))

    def test_go_several_records_compare_structure(self):
        self.go = Ontology(file= os.path.join(DATA_TEST_PATH, "partial_go.obo"), load_file= True)
        # Export object to JSON
        #self.go.write(os.path.join(DATA_TEST_PATH, "partial_go.json"))
        #file= os.path.join(DATA_TEST_PATH, "testjson.json"
        
        obo = Ontology(file= os.path.join(DATA_TEST_PATH, "partial_go.json"), load_file= True)
        self.assertEqual(self.go, obo)
        # Remove generated files
        #os.remove(os.path.join(DATA_TEST_PATH, "gotestjsonFull.json"))