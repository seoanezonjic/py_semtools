#!/usr/bin/env python

#########################################################
# Load necessary packages and folder paths
#########################################################

import unittest
import os
from py_semtools import OboParser, Ontology


ROOT_PATH= os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data')


#########################################################
# Define TESTS
#########################################################


class OBOParserTestCase(unittest.TestCase):

    def setUp(self):
        # Files
        self.file_Header = {"file": os.path.join(DATA_TEST_PATH, "only_header_sample.obo"), "name": "only_header_sample"}
        self.file_Hierarchical = {"file": os.path.join(DATA_TEST_PATH, "hierarchical_sample.obo"), "name": "hierarchical_sample"}
        self.file_Circular = {"file": os.path.join(DATA_TEST_PATH, "circular_sample.obo"), "name": "circular_sample"}
        self.file_Atomic = {"file": os.path.join(DATA_TEST_PATH, "sparse_sample.obo"), "name": "sparse_sample"}
        self.file_Sparse = {"file": os.path.join(DATA_TEST_PATH, "sparse2_sample.obo"), "name": "sparse2_sample"}
        self.file_SH = {"file": os.path.join(DATA_TEST_PATH, "short_hierarchical_sample.obo"), "name": "short_hierarchical_sample"}
        self.file_Enr = {"file": os.path.join(DATA_TEST_PATH, "enrichment_ontology.obo"), "name": "enrichment_ontology"}
        
        ## OBO INFO
        self.load_Header = (
            {"file": os.path.join(DATA_TEST_PATH, "only_header_sample.obo"), "name": "only_header_sample"}, 
            {"format-version":"1.2", "data-version": "test/a/b/c/"}, 
            {"terms":{}, "typedefs": {}, "instances":{}})

        self.load_Hierarchical_WithoutIndex = (
            {"file": os.path.join(DATA_TEST_PATH, "hierarchical_sample.obo"), "name": "hierarchical_sample"}, 
            {"format-version": "1.2", "data-version": "test/a/b/c/"}, 
            {"terms": {
                "Parental": {"id": "Parental", "name": "All", "comment": "none"}, 
                "Child1": {"id": "Child1", "name": "Child1", "is_obsolete": "true", "is_a": ["Parental"], "replaced_by": ["Child2"]}, 
                "Child2": {"id": "Child2", "name": "Child2", "synonym": ["\"1,6-alpha-mannosyltransferase activity\" EXACT []"], "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]}, 
                "Child5": {"id": "Child5", "name": "Child5", "synonym": ["\"activity related to example\" EXACT []"], "is_obsolete": "true", "is_a": ["Parental"]}
            }, 
            "typedefs": {}, "instances": {}})

        #TODO: Confirm if the ontology below is defined correctly
        self.load_Hierarchical_altid = (
            {"file": os.path.join(DATA_TEST_PATH, "hierarchical_sample.obo"), "name": "hierarchical_sample"}, 
            {"format-version": "1.2", "data-version": "test/a/b/c/"}, 
            {"terms": {
                "Parental": {"id": "Parental", "name": "All", "comment": "none"}, 
                "Child1": {"id": "Child1", "name": "Child1", "is_obsolete": "true", "is_a": ["Parental"], "replaced_by": ["Child2"]}, 
                "Child2": {"id": "Child2", "name": "Child2", "synonym": ["\"1,6-alpha-mannosyltransferase activity\" EXACT []"], "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]},
                "Child3": {"id": "Child2", "name": "Child2", "synonym": ["\"1,6-alpha-mannosyltransferase activity\" EXACT []"], "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]}, 
                "Child4": {"id": "Child2", "name": "Child2", "synonym": ["\"1,6-alpha-mannosyltransferase activity\" EXACT []"], "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]},
                "Child5": {"id": "Child5", "name": "Child5", "synonym": ["\"activity related to example\" EXACT []"], "is_obsolete": "true", "is_a": ["Parental"]}
                }, 
            "typedefs": {}, "instances": {}})

        self.load_Hierarchical = (
            {"file": os.path.join(DATA_TEST_PATH, "hierarchical_sample.obo"), "name": "hierarchical_sample"}, 
            {"format-version": "1.2", "data-version": "test/a/b/c/"},
            {"terms":{
                "Parental": {"id": "Parental", "name": "All", "comment": "none"}, 
                "Child1": {"id": "Child1", "name": "Child1", "is_obsolete": "true", "is_a": ["Parental"], "replaced_by": ["Child2"]}, 
                "Child2": {"id": "Child2", "name": "Child2", "synonym": ["\"1,6-alpha-mannosyltransferase activity\" EXACT []"], "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]}, 
                "Child5": {"id": "Child5", "name": "Child5", "synonym": ["\"activity related to example\" EXACT []"], "is_obsolete": "true", "is_a": ["Parental"]}
            },
            "typedefs": {}, "instances": {}})

        self.load_Circular = (
            {"file": os.path.join(DATA_TEST_PATH, "circular_sample.obo"), "name": "circular_sample"}, 
            {"format-version": "1.2", "data-version": "test/a/b/c/"}, 
            {"terms": {
                "A": {"id":"A", "name": "All", "is_a": ["C"]}, 
                "B": {"id":"B", "name": "B", "is_a": ["A"]}, 
                "C": {"id":"C", "name": "C", "is_a": ["B"]}}, 
            "typedefs": {}, "instances": {}})

        self.load_Atomic = (
            {"file": os.path.join(DATA_TEST_PATH, "sparse_sample.obo"), "name": "sparse_sample"}, 
            {"format-version": "1.2", "data-version": "test/a/b/c/"}, 
            {"terms": {
                "Parental": {"id": "Parental", "name": "All", "comment": "none"}, 
                "Child1": {"id": "Child1", "name": "Child1"}, 
                "Child2": {"id": "Child2", "name": "Child2"}}, 
            "typedefs": {}, "instances": {}})

        self.load_Sparse = (
            {"file": os.path.join(DATA_TEST_PATH, "sparse2_sample.obo"), "name": "sparse2_sample"}, 
            {"format-version": "1.2", "data-version": "test/a/b/c/"}, 
            {"terms": {
                "A": {"id": "A", "name": "All"}, 
                "B": {"id": "B", "name": "B", "is_a": ["A"]}, 
                "C": {"id": "C", "name": "C", "is_a": ["A"]}, 
                "D": {"id": "D", "name": "Sparsed"}}, 
            "typedefs": {}, "instances": {}})

        ## OBO INFO2
        self.load_Hierarchical_WithoutIndex2 = (
            {"file": os.path.join(DATA_TEST_PATH, "hierarchical_sample.obo"), "name": "hierarchical_sample"}, 
            {"format-version": "1.2", "data-version": "test/a/b/c/"}, 
            {"terms": {
                "Parental": {"id": "Parental", "name": "All", "comment": "none"}, 
                "Child1": {"id": "Child1", "name": "Child1", "is_obsolete": "true", "is_a": ["Parental"], "replaced_by": ["Child2"]}, 
                "Child2": {"id": "Child2", "name": "Child2", "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]}}, 
            "typedefs": {}, "instances": {}})
        
        #TODO: Confirm if the ontology below is defined correctly
        self.load_Hierarchical2 = (
            {"file": os.path.join(DATA_TEST_PATH, "hierarchical_sample.obo"), "name": "hierarchical_sample"}, 
            {"format-version": "1.2", "data-version": "test/a/b/c/"}, 
            {"terms": {
                "Parental": {"id": "Parental", "name": "All", "comment": "none"}, 
                "Child1": {"id": "Child1", "name": "Child1", "is_obsolete": "true", "is_a": ["Parental"], "replaced_by": ["Child2"]}, 
                "Child2": {"id": "Child2", "name": "Child2", "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]}, 
                "Child3": {"id": "Child2", "name": "Child2", "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]}, 
                "Child4": {"id": "Child2", "name": "Child2", "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]}}, 
            "typedefs": {}, "instances": {}})
            
        self.load_Circular2 = (
            {"file": os.path.join(DATA_TEST_PATH, "circular_sample.obo"), "name": "circular_sample"}, 
            {"format-version": "1.2", "data-version": "test/a/b/c/"}, 
            {"terms": {
                "A": {"id": "A", "name": "All", "is_a": ["C"]}, 
                "B": {"id": "B", "name": "B", "is_a": ["A"]}, 
                "C": {"id": "C", "name": "C", "is_a": ["B"]}}, 
            "typedefs": {}, "instances": {}})

        self.hierarchical_terms2 = {"Child2": {"is_a": ["B"]}, "Parental": {"is_a": ["A"]}}

        # Parentals
        self.parentals_Hierachical = ("hierarchical", {"Child1": ["Parental"], "Child2": ["Parental"], "Child3": ["Parental"], "Child4": ["Parental"], "Child5": ["Parental"]})
        self.parentals_Circular = ("circular", {"A": ["C", "B"], "C": ["B", "A"], "B": ["A", "C"]})
        self.parentals_Atomic = ("atomic", {})
        self.parentals_Sparse = ("sparse", {"B": ["A"], "C": ["A"]})
    
    def test_load_file(self):
        self.assertEqual(self.load_Header, OboParser.load_obo(self.file_Header["file"])) # Only header
        self.assertEqual(self.load_Hierarchical_WithoutIndex, OboParser.load_obo(self.file_Hierarchical["file"])) # Hierarchical
        self.assertEqual(self.load_Circular, OboParser.load_obo(self.file_Circular["file"])) # Circular
        self.assertEqual(self.load_Atomic, OboParser.load_obo(self.file_Atomic["file"])) # Sparsed
        self.assertEqual(self.load_Sparse, OboParser.load_obo(self.file_Sparse["file"])) # Sparsed 2
    
    def test_expand(self):
        # self.assertIsNone(Ontology.get_related_ids_by_tag(terms= nil,target_tag= "")) # Nil terms
        # self.assertIsNone(Ontology.get_related_ids_by_tag(terms= {},target_tag= "")) # Empty terms
        # self.assertIsNone(Ontology.get_related_ids_by_tag(terms= [],target_tag= "")) # Terms not a hash
        # self.assertIsNone(Ontology.get_related_ids_by_tag(terms= self.load_Hierarchical[2]["terms"],target_tag= None)) # Nil target
        # self.assertIsNone(Ontology.get_related_ids_by_tag(terms= self.load_Hierarchical[2]["terms"],target_tag= "")) # No/Empty target
        # self.assertIsNone(Ontology.get_related_ids_by_tag(terms= self.load_Hierarchical[2]["terms"],target_tag= 8)) # Target not a string
        # assert_raises ArgumentError do Ontology.get_related_ids_by_tag(terms= self.load_Hierarchical[2]["terms"], target_tag= "is_a",split_info_char=" ! ",split_info_indx= -1) end # Erroneous info_indx
        self.assertEqual(self.parentals_Hierachical,  OboParser.get_related_ids_by_tag(terms = self.load_Hierarchical_altid[2]["terms"], target_tag = "is_a")) # Hierarchical structure
        self.assertEqual(self.parentals_Circular,  OboParser.get_related_ids_by_tag(terms = self.load_Circular[2]["terms"], target_tag = "is_a")) # Circular structure
        self.assertEqual(self.parentals_Atomic,  OboParser.get_related_ids_by_tag(terms = self.load_Atomic[2]["terms"], target_tag = "is_a")) # Sparse structure
        self.assertEqual(self.parentals_Sparse,  OboParser.get_related_ids_by_tag(terms = self.load_Sparse[2]["terms"], target_tag = "is_a")) # Sparse structure with some other structures
        
    def test_expand2(self):
        # Test regular
        self.assertEqual(("hierarchical",["Parental"]), OboParser.get_related_ids("Child2", self.load_Hierarchical2[2]["terms"], "is_a"))
        # Test with already expanded info
        aux_expansion = {"Parental": ["A"]}
        self.assertEqual(("hierarchical",["Parental", "A"]), OboParser.get_related_ids("Child2", self.load_Hierarchical2[2]["terms"], "is_a", aux_expansion))		
        # Test circular
        self.assertEqual(("circular",["C","B"]), OboParser.get_related_ids("A", self.load_Circular2[2]["terms"], "is_a"))
        self.assertEqual(("circular",["B","A"]), OboParser.get_related_ids("C", self.load_Circular2[2]["terms"], "is_a"))
        
    def test_load(self):
        _, header, stanzas = OboParser.load_obo(self.file_Hierarchical["file"])
        self.assertEqual(self.load_Hierarchical[1], header)
        self.assertEqual(self.load_Hierarchical[2], stanzas)

        _, header, stanzas = OboParser.load_obo(self.file_Circular["file"])
        self.assertEqual(self.load_Circular[1], header)
        self.assertEqual(self.load_Circular[2], stanzas)

        _, header, stanzas = OboParser.load_obo(self.file_Atomic["file"])
        self.assertEqual(self.load_Atomic[1], header)		
        self.assertEqual(self.load_Atomic[2], stanzas)		

        _, header, stanzas = OboParser.load_obo(self.file_Sparse["file"])
        self.assertEqual(self.load_Sparse[1], header)		
        self.assertEqual(self.load_Sparse[2], stanzas)

    def test_dictionaries(self):
        OboParser.load(Ontology(), self.file_Hierarchical["file"], build= True)
        names_dict = OboParser.calc_dictionary("name")
        self.assertEqual({"Parental": ['All'], "Child2": ['Child2']}, names_dict["byTerm"])
        test_dict = OboParser.calc_dictionary("name", store_tag= "test")
        self.assertEqual(names_dict, test_dict)
        aux_synonym = {"Child2": ["1,6-alpha-mannosyltransferase activity"]}
        self.assertEqual(aux_synonym, OboParser.calc_dictionary("synonym", select_regex= r"\"(.*)\"")["byTerm"])
        self.assertEqual({"All": ["Parental"], "Child2": ["Child2"]}, OboParser.calc_dictionary("name", multiterm= True)["byValue"])

    def test_blacklist(self):
        hierarchical_cutted = Ontology(file = self.file_Hierarchical["file"], load_file = True, removable_terms = ["Parental"])
        hierarchical_cutted.precompute()
        self.assertEqual(0, hierarchical_cutted.meta["Child2"]["ancestors"])
        self.assertIsNone(hierarchical_cutted.terms.get("Parental"))