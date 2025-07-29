#########################################################
# Load necessary packages
#########################################################

import unittest, os, math, shutil
import numpy as np
from py_semtools import Ontology, JsonParser

ROOT_PATH = os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data')


#########################################################
# Define TESTS
#########################################################
class TestOBOFunctionalities(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        # Files

        self.file_Header = {"file": os.path.join(DATA_TEST_PATH, "only_header_sample.obo"), "name": "only_header_sample"}
        self.file_Hierarchical = {"file": os.path.join(DATA_TEST_PATH, "hierarchical_sample.obo"), "name": "hierarchical_sample"}
        self.file_Branched = {"file": os.path.join(DATA_TEST_PATH, "branched.obo"), "name": "Branched"}        
        #self.file_Circular = {"file": os.path.join(DATA_TEST_PATH, "circular_sample.obo"), "name": "circular_sample"}
        self.file_Atomic = {"file": os.path.join(DATA_TEST_PATH, "sparse_sample.obo"), "name": "sparse_sample"}
        self.file_Sparse = {"file": os.path.join(DATA_TEST_PATH, "sparse2_sample.obo"), "name": "sparse2_sample"}
        self.file_SH = {"file": os.path.join(DATA_TEST_PATH, "short_hierarchical_sample.obo"), "name": "short_hierarchical_sample"}
        self.file_Enr = {"file": os.path.join(DATA_TEST_PATH, "enrichment_ontology.obo"), "name": "enrichment_ontology"}

        ## OBO INFO
        self.load_Header = (
            {"file": os.path.join(DATA_TEST_PATH, "only_header_sample.obo"), "name": "only_header_sample"}, 
            {"format-version": "1.2", "data-version": "test/a/b/c/"}, 
            {"terms": {}, "typedefs": {}, "instances": {}})
        self.load_Hierarchical_WithoutIndex = (
            {"file": os.path.join(DATA_TEST_PATH, "hierarchical_sample.obo"), "name": "hierarchical_sample"}, 
            {"format-version": "1.2", "data-version": "test/a/b/c/"}, 
            {"terms": {
                "Parental": {"id": "Parental", "name": "All", "comment": "none"}, 
                "Child1": {"id": "Child1", "name": "Child1", "is_obsolete": "true", "is_a": ["Parental"], "replaced_by": ["Child2"]}, 
                "Child2": {"id": "Child2", "name": "Child2", "synonym": ["\"1,6-alpha-mannosyltransferase activity\" EXACT []"], "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]}}, 
            "typedefs": {}, "instances": {}})
        #TODO: Confirm if the ontology below is defined correctly
        self.load_Hierarchical = (
            {"file": os.path.join(DATA_TEST_PATH, "hierarchical_sample.obo"), "name": "hierarchical_sample"}, 
            {"format-version": "1.2", "data-version": "test/a/b/c/"}, 
            {"terms": {
                "Parental": {"id": "Parental", "name": "All", "comment": "none"}, 
                "Child1": {"id": "Child1", "name": "Child1", "is_obsolete" :  "true", "is_a": ["Parental"], "replaced_by" :  ["Child2"]}, 
                "Child2": {"id": "Child2", "name": "Child2", "synonym": ["\"1,6-alpha-mannosyltransferase activity\" EXACT []"], "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]}, 
                "Child3": {"id": "Child2", "name": "Child2", "synonym": ["\"1,6-alpha-mannosyltransferase activity\" EXACT []"], "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]}, 
                "Child4": {"id": "Child2", "name": "Child2", "synonym": ["\"1,6-alpha-mannosyltransferase activity\" EXACT []"], "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]}}, 
            "typedefs": {}, "instances": {}})
        self.load_Branched = (
            {"file": os.path.join(DATA_TEST_PATH, "Branched.obo"), "name": "Branched"}, 
            {"format-version": "1.2", "data-version": "test/a/b/c/"}, 
            {"terms": {
                "Parental": {"id": "Parental", "name": "All", "comment": "none"}, 
                "ChildA": {"id": "ChildA", "name": "ChildAname", "is_a": ["Parental"]}, 
                "ChildB": {"id": "ChildB", "name": "ChildBname", "is_a": ["Parental"]}, 
                "ChildA1": {"id": "ChildA1", "name": "ChildA1name", "is_a": ["ChildA"]},
                "ChildA2": {"id": "ChildA2", "name": "ChildA2name", "is_a": ["ChildA"]},
                "ChildB1": {"id": "ChildB1", "name": "ChildB1name", "is_a": ["ChildB"]},
                "ChildB2": {"id": "ChildB2", "name": "ChildB2name", "is_a": ["ChildB"]}
                }, 
            "typedefs": {}, "instances": {}})
        #self.load_Circular = (
        #    {"file": os.path.join(DATA_TEST_PATH, "circular_sample.obo"), "name": "circular_sample"}, 
        #    {"format-version": "1.2", "data-version": "test/a/b/c/"}, 
        #    {"terms": {
        #        "A": {"id": "A", "name": "All", "is_a": ["C"]}, 
        #        "B": {"id": "B", "name": "B", "is_a": ["A"]}, 
        #        "C": {"id": "C", "name": "C", "is_a": ["B"]}}, 
        #    "typedefs": {}, "instances": {}})
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

        # Parentals
        self.parentals_Hierachical = ("hierarchical", {"Child1": ["Parental"], "Child2": ["Parental"], "Child3": ["Parental"], "Child4": ["Parental"]})
        #self.parentals_Circular = ("circular", {"A": ["C", "B"], "C": ["B", "A"], "B": ["A", "C"]})
        self.parentals_Atomic = ("atomic", {})
        self.parentals_Sparse = ("sparse", {"B": ["A"], "C": ["A"]})

        # Aux variables
        self.basic_tags = {"ancestors": ["is_a"], "obsolete": ["is_obsolete"], "alternative": ["alt_id","replaced_by","consider"]}
        self.empty_ICs = {"resnik": {}, "resnik_observed": {}, "seco": {}, "zhou": {}, "sanchez": {}}
        self.erroneous_freq = {"struct_freq": -1.0, "observed_freq": -1.0, "max_depth": -1.0}
        self.empty_file = {"file": None, "name": None}

        # Create necessary instnaces
        self.hierarchical = Ontology(file= self.file_Hierarchical["file"], load_file= True)
        self.branched = Ontology(file= self.file_Branched["file"], load_file= True)
        self.short_hierarchical = Ontology(file= self.file_SH["file"],  load_file= True)
        self.enrichment_hierarchical = Ontology(file= self.file_Enr["file"],  load_file= True)
        #self.circular = Ontology(file= self.file_Circular["file"], load_file= True)
        self.atomic = Ontology(file= self.file_Atomic["file"], load_file= True)
        self.sparse = Ontology(file= self.file_Sparse["file"], load_file= True)

        # Freqs variables
        self.hierarchical_freqs_default = {"struct_freq": 2.0, "observed_freq": -1.0, "max_depth": 2.0}
        self.hierarchical_freqs_updated = {"struct_freq": 2.0, "observed_freq":  3.0, "max_depth": 2.0}
    
    #################################
    # IO JSON
    #################################

    def test_export_import(self):
        self.hierarchical.precompute()
        # Export/import
        self.hierarchical.write(os.path.join(DATA_TEST_PATH, "testjson.json"))
        obo = Ontology(file= os.path.join(DATA_TEST_PATH, "testjson.json"), build= False)
        self.assertEqual(self.hierarchical, obo)
        os.remove(os.path.join(DATA_TEST_PATH, "testjson.json"))
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

    ################################
    # TEST ONTOLOGY NAMES
    ################################

    def test_get_ontology_names(self):
        self.skipTest("Not implemented yet because toy example ontologies do not have names similar to terms codes.")
        #self.assertEqual(self.hierarchical.ont_name, "hierarchical_sample")
        #self.assertEqual(self.branched.ont_name, "branched")
        #self.assertEqual(self.atomic.ont_name, "sparse_sample")
        #self.assertEqual(self.sparse.ont_name, "sparse2_sample")

    #################################
    # GENERATE METADATA FOR ALL ITEMS
    #################################

    def test_generate_metadata_for_all_items(self):
        self.hierarchical.precompute()
        # Check freqs
        self.assertEqual({"Parental": {"ancestors": 0.0, "descendants": 1.0, "struct_freq": 2.0, "observed_freq": 0.0},
                      "Child2": {"ancestors": 1.0, "descendants": 0.0, "struct_freq": 1.0, "observed_freq": 0.0}},
                       self.hierarchical.meta)
        self.assertEqual(self.hierarchical_freqs_default,self.hierarchical.max_freqs) # Only structural freq

    def test_paths_levels(self):
        self.hierarchical.precompute()
        self.hierarchical.expand_path("Child2")
        default_child2_paths = {"total_paths": 1, "largest_path": 2, "shortest_path": 2, "paths": [["Child2", "Parental"]]}
        self.assertEqual(default_child2_paths, self.hierarchical.term_paths["Child2"])

        self.hierarchical.calc_term_levels(calc_paths= True, shortest_path= True)
        ## Testing levels
        # For all terms
        default_levels = {"byTerm":{1:["Parental"], 2:["Child2"]}, "byValue": {"Parental":1, "Child2": 2}}
        self.assertEqual(default_levels, self.hierarchical.dicts["level"])
        ## Testing paths
        default_paths = {"Parental": {"total_paths": 1, "largest_path": 1, "shortest_path": 1, "paths": [["Parental"]]}, 
                        "Child2": {"total_paths": 1, "largest_path": 2, "shortest_path": 2, "paths": [["Child2", "Parental"]]}}
        self.assertEqual(default_paths,self.hierarchical.term_paths) 

        child2_parental_path = ["Parental"]
        self.assertEqual(child2_parental_path,self.hierarchical.get_parental_path("Child2", which_path = "shortest_path"))

        # self.assertEqual({1=>[:Parental], 2=>[:Child2]}, self.hierarchical.get_ontology_levels) # FRED: redundant test?


    #################################
    # TERM METHODS
    #################################

    # I/O observed term from data
    ####################################
    def test_add_observed_terms(self):
        self.hierarchical.precompute()
        self.hierarchical.add_observed_terms(terms= ["Parental"])
        self.hierarchical.add_observed_terms(terms= ["Child2","Child2"])
        
        self.assertEqual(self.hierarchical_freqs_updated, self.hierarchical.max_freqs) 
        self.assertEqual({"ancestors": 1.0, "descendants": 0.0, "struct_freq": 1.0, "observed_freq": 2.0},
        self.hierarchical.meta["Child2"])
        self.assertEqual({"ancestors": 0.0, "descendants": 1.0, "struct_freq": 2.0, "observed_freq": 3.0},
        self.hierarchical.meta["Parental"])


    # Obtain level and term relations
    ####################################
    def test_level_term_relations(self):
        self.assertEqual([], self.hierarchical.get_ancestors("Parental")) # Ancestors
        self.assertEqual(["Parental"], self.hierarchical.get_ancestors("Child2"))
        self.assertEqual([], self.hierarchical.get_descendants("Child2")) # Descendants
        self.assertEqual(["Child2"], self.hierarchical.get_descendants("Parental"))
        self.assertIsNone(self.hierarchical.get_direct_descendants("Child1"))
        self.assertEqual(["Child2"], self.hierarchical.get_direct_descendants("Parental"))


    # ID Handlers
    ####################################
    def test_id_handlers(self):
        # Translate Terms
        aux_synonym = {"Child2": ["1,6-alpha-mannosyltransferase activity"]}
        self.assertEqual("Parental", self.hierarchical.translate('All', "name"))
        self.assertEqual(['Child2'], self.hierarchical.translate("Child2", "name", byValue= False)) # Official term
        self.assertEqual('Child2', self.hierarchical.translate_id("Child2"))
        self.assertEqual("Child2", self.hierarchical.translate(aux_synonym["Child2"][0], "synonym", byValue= True))
        self.assertEqual("Parental", self.hierarchical.translate_name('All'))
        self.assertEqual("Child2", self.hierarchical.translate_name(aux_synonym["Child2"][0]))
        self.assertIsNone(self.hierarchical.translate_name("Erroneous name"))
        self.assertEqual('All', self.hierarchical.translate_id("Parental"))
        self.assertEqual("Child2", self.hierarchical.get_main_id("Child1"))

    # Get term frequency and information
    ####################################
    def test_ics(self):
        self.hierarchical.precompute()
        self.assertEqual(0, self.hierarchical.get_IC("Parental"))	# Root
        self.assertEqual(-(math.log10(1/2.0)), self.hierarchical.get_IC("Child2")) # Leaf



    def test_similarities(self):
        self.hierarchical.precompute()
        self.enrichment_hierarchical.precompute()
        self.sparse.precompute()
        self.hierarchical.add_observed_terms(terms= ["Child2","Child2","Child2","Parental","Parental","Parental","Parental"])
        self.enrichment_hierarchical.add_observed_terms(terms= [
            "branchAChild1","branchAChild1","branchAChild1",
            "branchB","branchB",
            "branchAChild2","branchAChild2","branchAChild2","branchAChild2",
            "branchA","branchA","branchA",
            "root","root"])
        ## Structural ##
        self.assertEqual(1, self.hierarchical.get_structural_frequency("Child2")) ## Term by term frequencies
        self.assertEqual(2, self.hierarchical.get_structural_frequency("Parental"))
        ## Observed ##
        self.assertEqual(3, self.hierarchical.get_observed_frequency("Child2"))
        self.assertEqual(7, self.hierarchical.get_observed_frequency("Parental"))

        ## Structural ##
        self.assertEqual(["Child2",-(math.log10(1/2.0))], self.hierarchical.get_MICA("Child2", "Child2")) # MICA
        self.assertEqual(-(math.log10(3/5.0)), self.enrichment_hierarchical.get_ICMICA("branchAChild1", "branchAChild2")) #ICMICA
        self.assertEqual(0, self.enrichment_hierarchical.get_ICMICA("branchAChild1", "branchB")) #ICMICA
        self.assertIsNone(self.sparse.get_ICMICA("B","D")) #ICMICA
        self.assertEqual(["Parental",0], self.hierarchical.get_MICA("Child2", "Parental")) # ERR
        self.assertEqual(["Parental",0], self.hierarchical.get_MICA("Parental", "Parental")) # ERR
        ## Observed ##
        self.assertEqual(["Child2",-(math.log10(3/7.0))], self.hierarchical.get_MICA("Child2", "Child2", ic_type = 'resnik_observed')) # MICA
        self.assertEqual(["Parental",0], self.hierarchical.get_MICA("Child2", "Parental", ic_type = 'resnik_observed')) 
        self.assertEqual(["Parental",0], self.hierarchical.get_MICA("Parental", "Parental", ic_type = 'resnik_observed')) 
        self.assertEqual(["branchA",-(math.log10(10/14))], self.enrichment_hierarchical.get_MICA("branchAChild1", "branchAChild2", ic_type = 'resnik_observed')) 
        self.assertEqual(["branchA",-(math.log10(10/14))], self.enrichment_hierarchical.get_MICA("branchAChild1", "branchA", ic_type = 'resnik_observed')) 
        self.assertEqual(["root",0], self.enrichment_hierarchical.get_MICA("branchB", "branchAChild2", ic_type = 'resnik_observed')) 
        self.assertEqual(["root",0], self.enrichment_hierarchical.get_MICA("branchB", "branchA", ic_type = 'resnik_observed')) 

        self.assertEqual(0.0, self.hierarchical.get_similarity("Parental", "Parental")) # SIM
        self.assertEqual(0.0, self.hierarchical.get_similarity("Parental", "Child2"))
        self.assertEqual(-(math.log10(1/2.0)), self.hierarchical.get_similarity("Child2", "Child2"))
        self.assertEqual(0.0, self.hierarchical.get_similarity("Parental", "Child2",sim_type="lin"))
        self.assertEqual(1, self.hierarchical.get_similarity("Parental", "Parental",sim_type="lin"))
        self.assertEqual(1, self.hierarchical.get_similarity("Child2", "Child2",sim_type="lin"))
        self.assertEqual((-math.log10(1/2.0) + 0) - (2.0 * 0), self.hierarchical.get_similarity("Parental", "Child2",sim_type="jiang_conrath"))
        self.assertEqual((0+0) - (2.0 * 0), self.hierarchical.get_similarity("Parental", "Parental",sim_type="jiang_conrath"))
        self.assertEqual( (-math.log10(1/2.0) + -math.log10(1/2.0)) - (2.0 * -(math.log10(1/2.0))), 
                            self.hierarchical.get_similarity("Child2", "Child2",sim_type="jiang_conrath"))
        ## Observed ##
        self.assertEqual(0.0, self.hierarchical.get_similarity("Parental", "Parental", sim_type = 'resnik', ic_type = 'resnik_observed')) 
        self.assertEqual(0.0, self.hierarchical.get_similarity("Parental", "Child2", sim_type = 'resnik', ic_type = 'resnik_observed'))
        self.assertEqual(-(math.log10(3/7)), self.hierarchical.get_similarity("Child2", "Child2", sim_type = 'resnik', ic_type = 'resnik_observed'))

    # Checking valid terms
    ####################################
    def test_valid_terms(self):
        self.assertEqual(False, self.hierarchical.term_exist("FakeID")) # Validate ids
        self.assertEqual(True, self.hierarchical.term_exist("Parental")) # Validate ids
        self.assertEqual(False, self.hierarchical.is_obsolete("Child2")) # Validate ids
        self.assertEqual(True, self.hierarchical.is_obsolete("Child1")) # Validate ids


    #############################################
    # PROFILE EXTERNAL METHODS
    #############################################

    # Modifying Profile
    ####################################
    
    def test_modifiying_profile_externel(self):
        self.hierarchical.precompute()
        # Remove by scores
        prof = ["Parental", "Child2"]
        scores = {"Parental": 3, "Child2": 7}
        self.assertEqual(["Child2"],self.hierarchical.clean_profile_by_score(prof,scores, byMax= True))
        self.assertEqual(["Parental"],self.hierarchical.clean_profile_by_score(prof,scores, byMax= False))
        scores2 = {"Child2": 7}
        self.assertEqual(["Child2"],self.hierarchical.clean_profile_by_score(prof,scores2, byMax= True))
        self.assertEqual(["Child2"],self.hierarchical.clean_profile_by_score(prof,scores2, byMax= False))
        prof2 = ["Parental","Child3"]
        # Child 3 is not into scores, it will remove it
        self.assertEqual(["Parental"],self.hierarchical.clean_profile_by_score(prof2,scores, byMax= True))
        self.assertEqual(["Parental"],self.hierarchical.clean_profile_by_score(prof2,scores, byMax= False))
        self.assertEqual(["Parental","Child3"],self.hierarchical.clean_profile_by_score(prof2,scores, byMax= False, remove_without_score= False))

        self.assertEqual(["Child2"],self.hierarchical.clean_profile_hard(["Child2", "Parental", "Child5", "Child1"]))
        self.assertEqual(["branchAChild1", "branchAChild2"], sorted(self.enrichment_hierarchical.clean_profile_hard(["root", "branchB", "branchAChild1", "branchAChild2"], options = {"term_filter": "branchA"}))) # Testing term_filter option
        
        # Remove parentals and alternatives
        self.assertEqual((["Child2"], ["Parental"]), self.hierarchical.remove_ancestors_from_profile(["Child2", "Parental"]))
        self.assertEqual((["Child2", "Parental"], ["Child3"]), self.hierarchical.remove_alternatives_from_profile(["Child2", "Parental", "Child3"]))

        # Expand parental 
        ## For one profile
        self.assertEqual(['branchA', 'branchAChild1', 'branchB', 'root'] , sorted(self.enrichment_hierarchical.expand_profile_with_parents(["branchAChild1", "branchB"]))) 

    # ID Handlers
    #################################### 
    
    def test_id_handlers_external(self):
        self.hierarchical.precompute()
        self.assertEqual((["Parental", "Child2"], ["FakeID"]), self.hierarchical.check_ids(["Parental", "FakeID", "Child3"])) # Validate ids
        self.enrichment_hierarchical.precompute()
        self.assertEqual((["All", "Child2", "Child5"], ["FakeID"]), self.enrichment_hierarchical.translate_ids(["root", "branchAChild3", "branchB", "FakeID"]))
        self.assertEqual((["root", "branchAChild1", "branchB"], ["FakeName"]), self.enrichment_hierarchical.translate_names(["All", "Child2", "Child5","FakeName"]))

    # Description of profile's terms
    ####################################
    
    def test_description_profile_terms(self):
        self.hierarchical.precompute()
        self.assertEqual([[["Parental", "All"], [["Child2", "Child2"]]]], self.hierarchical.get_childs_table(["Parental"])) # Expanded info
        # For a profile
        self.assertEqual([["Child2", 2], ["Child1", None], ["Parental", 1]], self.hierarchical.get_terms_levels(["Child2","Child1","Parental"]))

    # IC data
    ####################################
    
    def test_ic_profile_external(self):
        self.hierarchical.precompute()
        expected_A_IC_resnik = (-(math.log10(1/2.0))-(math.log10(2/2.0))) / 2.0
        self.assertEqual(expected_A_IC_resnik, self.hierarchical.get_profile_mean_IC(["Child2", "Parental"]))
        self.enrichment_hierarchical.precompute()
        self.assertEqual(["branchA", -(math.log10(3/5.0))], self.enrichment_hierarchical.get_maxmica_term2profile("branchA",["branchAChild1","branchB"]))
        
    def test_similarities_profile_external(self):
        self.hierarchical.precompute()
        profA = ["Child2"]
        profB = ["Child2"]
        profC = ["Parental"]
        profD = ["Parental", "Child2"]
        
        self.assertEqual( 1.0, self.hierarchical.compare(profB, profB, bidirectional= False, sim_type = "lin"))
        self.assertEqual(-(math.log10(1/2.0)), self.hierarchical.compare(profB, profB, bidirectional= False))
        self.assertEqual(-(math.log10(1/2.0)), self.hierarchical.compare(profB, profB, bidirectional= True))
        self.assertEqual(-(math.log10(1/2.0)), self.hierarchical.compare(profA, profB, bidirectional= False))
        self.assertEqual(-(math.log10(1/2.0)), self.hierarchical.compare(profA, profB, bidirectional= True))
        self.assertEqual(-(math.log10(2/2.0)), self.hierarchical.compare(profA, profC, bidirectional= False))
        self.assertEqual(-(math.log10(2/2.0)), self.hierarchical.compare(profA, profC, bidirectional= True))
        sim_D_A = (-(math.log10(2/2.0)) -(math.log10(1/2.0))) / 2.0
        sim_A_D = -(math.log10(1/2.0))
        sim_A_D_bi = (sim_D_A * 2 + sim_A_D) / 3.0
        self.assertEqual(sim_A_D, self.hierarchical.compare(profA, profD, bidirectional= False))
        self.assertEqual(sim_D_A, self.hierarchical.compare(profD, profA, bidirectional= False))
        self.assertEqual(sim_A_D_bi, self.hierarchical.compare(profD, profA, bidirectional= True))
        self.assertEqual(self.hierarchical.compare(profA, profD, bidirectional= True), self.hierarchical.compare(profD, profA, bidirectional= True))


    #############################################
    # PROFILE INTERNAL METHODS 
    #############################################

    # I/O profiles
    ####################################
    
    def test_io_profiles_internal(self):
        # loading profiles
        self.hierarchical.load_profiles({"A": ["Child1", "Parental"], "B": ["Child3", "Child4", "Parental", "FakeID"], "C": ["Child2", "Parental"], "D": ["Parental"]}, calc_metadata= False, substitute= False)
        self.assertEqual({"A": ["Child1", "Parental"], "B": ["Child3", "Child4", "Parental"], "C": ["Child2", "Parental"], "D": ["Parental"]}, self.hierarchical.profiles)
        #self.hierarchical.load_profiles({:A => [:Child1, :Parental], :B => [:Child3, :Child4, :Parental, :FakeID],:C => [:Child2, :Parental], :D => [:Parental]}, substitute: true)
        #self.assertEqual({:A=>[:Child2, :Parental], :B=>[:Child2, :Parental], :C=>[:Child2, :Parental], :D=>[:Parental]}, self.hierarchical.profiles) # FRED: Shouldn´t this return uniq ids?
        self.hierarchical.reset_profiles()
        self.assertEqual({}, self.hierarchical.profiles)
        self.hierarchical.add_profile("A", ["Child2", "Parental"], substitute= False) # Add profiles
        self.hierarchical.add_profile("B", ["Child2", "Parental", "FakeID"], substitute= False)
        self.hierarchical.add_profile("C", ["Child2", "Parental"], substitute= False)
        self.hierarchical.add_profile("D", ["Parental"], substitute= False)
        self.assertEqual(["Child2", "Parental"], self.hierarchical.get_profile("A")) # Check storage
        self.assertEqual(["Child2", "Parental"], self.hierarchical.get_profile("B"))



    # Modifying profiles
    ####################################
    
    def test_modifying_profile_internal(self):
        self.hierarchical.precompute()
        self.hierarchical.add_profile("A", ["Child2", "Parental"], substitute= False) # Add profiles
        self.hierarchical.add_profile("B", ["Child2", "Parental", "FakeID"], substitute= False)
        self.hierarchical.add_profile("C", ["Child2", "Parental"], substitute= False)
        self.hierarchical.add_profile("D", ["Parental"], substitute= False)
        # Expand parental 
        ## Parental method for self.profiles
        self.enrichment_hierarchical.precompute()
        self.enrichment_hierarchical.load_profiles({
            "A": ["branchAChild1", "branchB"],
            "B": ["branchAChild2", "branchA", "branchB"],
            "C": ["root", "branchAChild2", "branchAChild1"],
            "D": ["FakeID"]},
             calc_metadata= False, substitute= False)
        self.enrichment_hierarchical.expand_profiles('parental') # FRED: Maybe we could add "propagate" version but this is checked in test_expand_items
        self.assertEqual({"A": ['branchA', 'branchAChild1', 'branchB', 'root'], "B": ['branchA', 'branchAChild2', 'branchB', 'root'], "C": ['branchA', 'branchAChild1', 'branchAChild2', 'root']}, self.enrichment_hierarchical.profiles)
        self.enrichment_hierarchical.reset_profiles()
        self.assertEqual({"A": ["Child2"], "B": ["Child2"], "C": ["Child2"], "D": ["Parental"]}, self.hierarchical.clean_profiles())


     # ID Handlers
    ####################################
    
    def test_id_handlers_internal(self):
        self.hierarchical.add_profile("A", ["Child2", "Parental"], substitute= False) # Add profiles
        self.hierarchical.add_profile("B", ["Child2", "Parental", "FakeID"], substitute= False)
        self.hierarchical.add_profile("C", ["Child2", "Parental"], substitute= False)
        self.hierarchical.add_profile("D", ["Parental"], substitute= False)

        # Translators
        self.assertEqual([["Child2", "All"], ["Child2", "All"], ["Child2", "All"], ["All"]], self.hierarchical.translate_profiles_ids())
        self.assertEqual({"A": ["Child2", "All"], "B": ["Child2", "All"], "C": ["Child2", "All"], "D": ["All"]}, self.hierarchical.translate_profiles_ids(asArray= False))
        test_profiles = [self.hierarchical.profiles["A"],self.hierarchical.profiles["B"]]
        self.assertEqual([["Child2", "All"], ["Child2", "All"]], self.hierarchical.translate_profiles_ids(test_profiles))
        self.assertEqual({0: ["Child2", "All"], 1: ["Child2", "All"]}, self.hierarchical.translate_profiles_ids(test_profiles, asArray= False))


    # Description of profile size
    ####################################

    def test_get_general_profile(self):
        self.branched.load_profiles({"A": ["ChildA", "ChildB", "Parental"], "B": ["ChildA", "ChildB", "Parental", "FakeID"], 
                                    "C": ["ChildA1", "ChildB", "Parental"], "D": ["Parental"]}, calc_metadata= False, substitute= False)
        returned = self.branched.get_general_profile()
        expected = ['ChildB', 'ChildA1'] #This is the general profile (hard cleaned) of the aboves profile terms
        #self.assertEqual(expected, returned)
        self.assertEqual(returned, expected)

        #Testing with frequency threshold parameter
        returned = self.branched.get_general_profile(thr= 0.5)
        expected = ['ChildB'] #ChildB is the only term with a frequency higher than 0.5
        
    
    def test_description_profile_size(self):
        self.hierarchical.add_profile("A", ["Child2", "Parental"], substitute= False) # Add profiles
        self.hierarchical.add_profile("B", ["Child2", "Parental", "FakeID"], substitute= False)
        self.hierarchical.add_profile("C", ["Child2", "Parental"], substitute= False)
        self.hierarchical.add_profile("D", ["Parental"], substitute= False)
        expected_stats = [['Elements', '4'], ['Elements Non Zero', '4'], ['Non Zero Density', '1.0'], 
         ['Max', '2'], ['Min', '1'], ['Average', '1.75'], 
         ['Variance', '0.1875'], ['Standard Deviation', '0.4330127018922193'], ['Q1', '1.75'], 
         ['Median', '2.0'], ['Q3', '2.0'], ['Min Non Zero', '1'],
         ['Average Non Zero', '1.75'], ['Variance Non Zero', '0.1875'], ['Standard Deviation Non Zero', '0.4330127018922193'],
         ['Q1 Non Zero', '1.75'], ['Median Non Zero', '2.0'], ['Q3 Non Zero', '2.0']]
        # Getiings
        self.assertEqual([2, 2, 2, 1], self.hierarchical.get_profiles_sizes()) # Check metadata
        self.assertEqual(round(7 /4.0, 4), self.hierarchical.get_profiles_mean_size())
        self.assertEqual(expected_stats, self.hierarchical.profile_stats())
        self.assertEqual(1, self.hierarchical.get_profile_length_at_percentile(0, increasing_sort= True))
        self.assertEqual(2, self.hierarchical.get_profile_length_at_percentile(2.0 / (4 - 1) * 100, increasing_sort= True))
        self.assertEqual(2, self.hierarchical.get_profile_length_at_percentile(3.0 / (4 - 1) * 100, increasing_sort= True))
        self.assertEqual(1, self.hierarchical.get_profile_length_at_percentile(4.0 / (5 - 1) * 100, increasing_sort= False))

    def test_parentals_per_profile(self):
        self.sparse.precompute()
        self.sparse.add_profile("patient1", ["B", "A"], substitute= False) # Add profiles
        self.sparse.add_profile("patient2", ["C", "A"], substitute= False)
        self.sparse.add_profile("patient3", ["B", "C"], substitute= False)
        returned = self.sparse.parentals_per_profile()
        expected = [1,1,0] #Given that A is Parental and B and C are childs
        self.assertEqual(expected, returned)

    def test_get_profile_redundancy(self):
        self.sparse.precompute()
        self.sparse.add_profile("patient1", ["B", "A"], substitute= False) # Add profiles
        self.sparse.add_profile("patient2", ["C", "A"], substitute= False)
        self.sparse.add_profile("patient3", ["B", "C"], substitute= False)
        self.sparse.add_profile("patient4", ["C"], substitute= False)
        
        returned_profile_sizes, returned_parental_terms_per_profile = self.sparse.get_profile_redundancy()
        expected_profile_sizes = [2,2,2,1]
        expected_parental_terms_per_profile = [1,1,0,0]
        
        self.assertEqual(expected_profile_sizes, returned_profile_sizes)
        self.assertEqual(expected_parental_terms_per_profile, returned_parental_terms_per_profile)

    def test_compute_term_list_and_childs(self):
        self.sparse.precompute()
        self.sparse.add_profile("patient1", ["B"], substitute= False) # Add profiles
        self.sparse.add_profile("patient2", ["A"], substitute= False)
        #self.sparse.add_profile("patient3", ["B", "C"], substitute= False)
        #self.sparse.add_profile("patient4", ["C"], substitute= False)

        suggested_childs, terms_with_more_specific_childs_proportions = self.sparse.compute_term_list_and_childs()
        
        expected_suggested_childs = {'patient1': [[['B', 'B'], []]], 
                                     'patient2': [[['A', 'All'], [['B', 'B'], ['C', 'C']]]]}
        expected_terms_with_more_specific_childs_proportions = 0.5
        
        self.assertDictEqual(expected_suggested_childs, suggested_childs)
        self.assertEqual(expected_terms_with_more_specific_childs_proportions, terms_with_more_specific_childs_proportions)

    # IC data
    ####################################
    def test_get_profiles_terms_frequency(self):
        self.branched.precompute()
        self.branched.add_profile("P1", ["ChildA1", "ChildA2"], substitute= False)
        self.branched.add_profile("P2", ["ChildB1", "ChildB2"], substitute= False)
        self.branched.add_observed_terms_from_profiles(reset=True)

        #Checking total counts without parentals
        expected = {'ChildA1': 1, 'ChildA2': 1, 'ChildB1': 1, 'ChildB2': 1}
        returned = self.branched.get_profiles_terms_frequency(ratio= False, asArray= False, translate= False)
        self.assertDictEqual(expected, returned)

        #Checking frequencies without parentals and as array
        expected = [['ChildA1', 0.5], ['ChildA2', 0.5], ['ChildB1', 0.5], ['ChildB2', 0.5]]
        returned = self.branched.get_profiles_terms_frequency(ratio= True, asArray= True, translate= False)
        self.assertListEqual(expected, returned)

        # Checking total counts with parentals (we should expect ChildA and ChildB only one time, because, for example, the ChildA is parent of both ChildA1 and ChildA2, so we check it is not double counted)
        expected = {'ChildA1': 1, 'ChildA2': 1, 'ChildB1': 1, 'ChildB2': 1, "ChildA": 1, "ChildB": 1, 'Parental': 2}
        returned = self.branched.get_profiles_terms_frequency(ratio= False, asArray= False, translate= False, count_parentals= True)
        self.assertDictEqual(expected, returned)

        # Checking frequencies with parentals
        expected = {'ChildA1': 0.5, 'ChildA2': 0.5, 'ChildB1': 0.5, 'ChildB2': 0.5, "ChildA": 0.5, "ChildB": 0.5, 'Parental': 1}
        returned = self.branched.get_profiles_terms_frequency(ratio= True, asArray= False, translate= False, count_parentals= True)

        #Now that we already checked parentals are not being double counted, we are going to check the following profiles examples
        ## Example 1
        self.branched.load_profiles({"P1": ["ChildA1"], "P2": ["ChildA2"], "P3": ["ChildB1"], "P4": ["ChildB2"]}, reset_stored = True)
        self.branched.add_observed_terms_from_profiles(reset=True)
        # Checking frequencies with parentals
        expected = {'ChildA1': 0.25, 'ChildA2': 0.25, 'ChildB1': 0.25, 'ChildB2': 0.25, "ChildA": 0.5, "ChildB": 0.5, 'Parental': 1}
        returned = self.branched.get_profiles_terms_frequency(ratio= True, asArray= False, translate= False, count_parentals= True)
        self.assertDictEqual(expected, returned)

        ## Example 2
        self.branched.load_profiles({"P1": ["Parental"], "P2": ["ChildA"], "P3": ["ChildA1"], "P4": ["ChildB1"]}, reset_stored = True)
        self.branched.add_observed_terms_from_profiles(reset=True)
        # Checking frequencies with parentals
        expected = {'ChildA1': 0.25, 'ChildB1': 0.25, "ChildA": 0.5, "ChildB": 0.25, 'Parental': 1}
        returned = self.branched.get_profiles_terms_frequency(ratio= True, asArray= False, translate= False, count_parentals= True)
        self.assertDictEqual(expected, returned)
        #Filtering with threshold of 0.5
        expected = {"ChildA": 0.5, 'Parental': 1}
        returned = self.branched.get_profiles_terms_frequency(ratio= True, asArray= False, translate= False, count_parentals= True, min_freq= 0.5)
        self.assertDictEqual(expected, returned)
        #Checking translation functionallity
        expected_translated = {"ChildAname": 0.5, 'All': 1}
        returned_translated = self.branched.get_profiles_terms_frequency(ratio= True, asArray= False, translate= True, count_parentals= True, min_freq= 0.5)
        self.assertDictEqual(expected_translated, returned_translated)

        ## Example 3
        self.branched.load_profiles({"P1": ["ChildA1"], "P2": ["ChildA1"], "P3": ["ChildA1"], "P4": ["ChildB1"]}, reset_stored = True)
        self.branched.add_observed_terms_from_profiles(reset=True)
        # #Filtering with threshold of 0.5 without parentals
        expected = {'ChildA1': 0.75}
        returned = self.branched.get_profiles_terms_frequency(ratio= True, asArray= False, translate= False, count_parentals= False, min_freq= 0.5)
        self.assertDictEqual(expected, returned)
        #Filtering with threshold of 0.5 with parentals
        expected = {'ChildA1': 0.75, "ChildA": 0.75, 'Parental': 1}
        returned = self.branched.get_profiles_terms_frequency(ratio= True, asArray= False, translate= False, count_parentals= True, min_freq= 0.5)
        self.assertDictEqual(expected, returned)

    def test_similarities_profile_internal(self):
        self.hierarchical.precompute()
        self.hierarchical.add_profile("A",["Child2"], substitute= False)
        self.hierarchical.add_profile("D",["Parental","Child2"], substitute= False)
        sim_D_A = (-math.log10(2/2.0) -math.log10(1/2.0)) / 2.0
        sim_A_D = -math.log10(1/2.0)
        sim_A_D_bi = (sim_D_A * 2 + sim_A_D) / 3.0
        self.assertEqual(sim_A_D_bi, self.hierarchical.compare_profiles()["A"]["D"])
        self.assertEqual(-math.log10(2/2.0), self.hierarchical.compare_profiles(external_profiles= {"C": ["Parental"]})["A"]["C"])
        self.hierarchical.add_observed_terms_from_profiles()
        self.assertEqual(({"Child2": -math.log10(0.5), "Parental": -math.log10(1)}, {"Child2": -math.log10(2/3.0), "Parental": -math.log10(1)}), self.hierarchical.get_observed_ics_by_onto_and_freq())
    

    def test_ic_profile_internal(self):
        self.hierarchical.precompute()
        self.hierarchical.add_profile("A", ["Child2", "Parental"], substitute= False) # Add profiles
        self.hierarchical.add_profile("B", ["Child2", "Parental", "FakeID"], substitute= False)
        self.hierarchical.add_profile("C", ["Child2", "Parental"], substitute= False)
        self.hierarchical.add_profile("D", ["Parental"], substitute= False)

        # Frequencies from profiles
        self.hierarchical.add_observed_terms_from_profiles()
        self.assertEqual({"Child2": 3, "Parental": 4}, self.hierarchical.get_profiles_terms_frequency(ratio= False, asArray= False, translate= False)) # Terms frequencies observed
        self.assertEqual([["Parental", 1.0], ["Child2", 0.75]], self.hierarchical.get_profiles_terms_frequency(ratio= True, asArray= True, translate= False)) 
        self.assertEqual({"Child2": 3, "Parental": 4}, self.hierarchical.get_profiles_terms_frequency(ratio= False, asArray= False, translate= False))
        self.assertEqual([["Parental", 1.0], ["Child2", 0.75]], self.hierarchical.get_profiles_terms_frequency(ratio= True, asArray= True, translate= False)) 

        # ICs
        expected_profiles_IC_resnik = {"A": (-math.log10(1/2.0) -math.log10(2/2.0)) / 2.0,
                                        "B": (-math.log10(1/2.0) -math.log10(2/2.0)) / 2.0, 
                                        "C": (-math.log10(1/2.0) -math.log10(2/2.0)) / 2.0, 
                                        "D": 0.0 }
        expected_profiles_IC_resnik_observed = {"A": (-math.log10(3/7.0) -math.log10(7/7.0)) / 2.0,
                                                 "B": (-math.log10(3/7.0) -math.log10(7/7.0)) / 2.0, 
                                                 "C": (-math.log10(3/7.0) -math.log10(7/7.0)) / 2.0, 
                                                 "D": 0.0 }
        self.assertEqual((expected_profiles_IC_resnik, expected_profiles_IC_resnik_observed), self.hierarchical.get_profiles_resnik_dual_ICs())
    
    # Comparisons and similarity matrix
    ####################################

    def test_calc_sim_term2term_similarity_matrix(self):
        reference_profile = ["ChildA1", "ChildA2"]
        ref_profile_id = "X"
        ref_profile_dict = {ref_profile_id: reference_profile}
        external_profiles = {"A": ["ChildA1", "ChildA2"], "B": ["ChildB1", "ChildB2"], "C": ["ChildA1", "Parental"], "D": ["ChildB1","Parental"], "E": ["Parental"], "F": ["ChildA1"]}

        self.branched.precompute()
        ont = self.branched
        ont.load_profiles(ref_profile_dict)
        
        candidate_sim_matrix, candidates, candidates_ids, similarities, _, _ = ont.calc_sim_term2term_similarity_matrix(reference_profile, ref_profile_id, external_profiles, string_format=True)
        candidate_sim_matrix.pop(0)
        self.assertEqual(candidate_sim_matrix, [['ChildA1name', 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], ['ChildA2name', 1.0, 0, 0.0, 0.0, 0.0, 0]])
        self.assertEqual(candidates, [['A', 1.0], ['F', 0.8118083219821401], ['C', 0.6088562414866051], ['B', 0.0], ['D', 0.0], ['E', 0.0]])
        self.assertEqual(candidates_ids, ['A', 'F', 'C', 'B', 'D', 'E'])
        self.assertEqual(similarities, {'X': {'A': 1.0, 'B': 0.0, 'C': 0.6088562414866051, 'D': 0.0, 'E': 0.0, 'F': 0.8118083219821401}})

    def test_get_term2term_similarity_matrix(self): #Helper function of calc_sim_term2term_similarity_matrix
        pass
        #self.assertFalse(True, "Helper function of calc_sim_term2term_similarity_matrix")

    def test_get_detailed_similarity(self): #Helper function of get_term2term_similarity_matrix
        pass
        #self.assertFalse(True, "Helper function of get_term2term_similarity_matrix")

    def test_get_negative_terms_matrix(self): 
        ont = self.branched
        candidate_terms_all_sims = {"Profile1": {"ChildA": [0.2,0.1,0.1], "ChildB": [0.3,0.5,0.1], "ChildA1": [0.8,0.1,0.1]},
                                    "Profile2": {"ChildA": [0.2,0.1,0.1], "ChildB1": [0.1,0.1,0.1], "ChildA2": [0.8,0.7,0.5]},
                                    "Profile3": {"ChildA": [0.2,0.0,0.1], "ChildB1": [0.1,0.1,0.0], "ChildB2": [0.1,0.2,0.1]}}
        
        expected = [['HP', 'Profile1', 'Profile2', 'Profile3'], ['ChildAname', 3, 3, 3], ['ChildB1name', 0, 2, 2], ['ChildB2name', 0, 0, 1]]
        returned, _ = ont.get_negative_terms_matrix(candidate_terms_all_sims, term_limit = 10, candidate_limit = 10, 
                                                    string_format = True, header_id = "HP")
        self.assertEqual(expected, returned)

        #Testing with term_limit lower than the number of terms
        expected = [['HP', 'Profile1', 'Profile2', 'Profile3'], ['ChildAname', 3, 3, 3], ['ChildB1name', 0, 2, 2]]
        returned, _ = ont.get_negative_terms_matrix(candidate_terms_all_sims, term_limit = 2, candidate_limit = 10,
                                                    string_format = True, header_id = "HP")
        self.assertEqual(expected, returned)
        
        #Testing with candidate_limit lower than the number of candidates
        expected = [['HP', 'Profile1', 'Profile2'], ['ChildAname', 2, 2], ['ChildB1name', 0, 1]]
        returned, _ = ont.get_negative_terms_matrix(candidate_terms_all_sims, term_limit = 10, candidate_limit = 2,
                                                    string_format = True, header_id = "HP")
        self.assertEqual(expected, returned)

    # Clustering method
    ###################################

    def test_get_matrix_similarity(self): #Helper function tested in get_similarity_clusters (its higher order function)
        pass
        #self.assertFalse(True, "Helper function tested in get_similarity_clusters")

    def test_get_similarity_clusters(self):
        ont = self.branched
        tmp_folder = os.path.join(ROOT_PATH, "tmp", "similarity_cluster")
        os.makedirs(tmp_folder, exist_ok=True)

        options = {"sim_thr": 0.3, "cl_size_factor": 1}
        method_name = "resnik"
        profiles = {"A": ["ChildA1", "ChildA2"], "B": ["ChildB1", "ChildB2"], "C": ["ChildA1", "Parental"], "D": ["ChildB1","Parental"], "E": ["Parental"], "F": ["ChildA1"]}
        self.branched.precompute()
        ont.load_profiles(profiles)

        #Testing for the first time (no files in tmp_folder, so values have to be calculated)
        clusters, similarity_matrix, linkage, raw_cls = self.branched.get_similarity_clusters(method_name, options, temp_folder=tmp_folder, reference_profiles=None)
        
        clusters_expected = {7: ['F', 'A', 'C'], 6: ['B', 'D']}
        similarity_matrix_expected = [[0.         , 0.68605762 , 0.56339869 , 0.         , 0.        ],
                                      [0.68605762 , 0.         , 0.51454322 , 0.         , 0.        ],
                                      [0.56339869 , 0.51454322 , 0.         , 0.         , 0.        ],
                                      [0.         , 0.         , 0.         , 0.         , 0.51454322],
                                      [0.         , 0.         , 0.         , 0.51454322 , 0.        ]]
        linkage_expected =           [[0.         , 1.         , 0.         , 2.        ],
                                      [3.         , 4.         , 0.17151441 , 2.        ],
                                      [2.         , 5.         , 0.17216737 , 3.        ],
                                      [6.         , 7.         , 1.04886281 , 5.        ]]
        raw_cls_expected = [[7],[7],[7],[6],[6]]

        self.assertEqual(clusters, clusters_expected)
        self.assertTrue(np.isclose(similarity_matrix, similarity_matrix_expected).all())
        self.assertTrue(np.isclose(linkage, linkage_expected).all())
        self.assertTrue((raw_cls == raw_cls_expected).all())

        self.assertTrue(os.path.exists(os.path.join(tmp_folder, f"similarity_matrix_{method_name}.npy")))
        self.assertTrue(os.path.exists(os.path.join(tmp_folder, f"similarity_matrix_{method_name}_x.lst")))
        self.assertFalse(os.path.exists(os.path.join(tmp_folder, f"similarity_matrix_{method_name}_y.lst")))
        self.assertTrue(os.path.exists(os.path.join(tmp_folder, f"{method_name}_clusters.txt")))
        self.assertTrue(os.path.exists(os.path.join(tmp_folder, f'profiles_similarity_{method_name}.txt')))
        self.assertTrue(os.path.exists(os.path.join(tmp_folder, f'{method_name}_linkage.npy')))
        self.assertTrue(os.path.exists(os.path.join(tmp_folder, f'{method_name}_raw_cls.npy')))

        #Testing for the second time (files in tmp_folder, so values will be loaded from files instead of being calculated)
        clusters, similarity_matrix, linkage, raw_cls = self.branched.get_similarity_clusters(method_name, options, temp_folder=tmp_folder, reference_profiles=None)
        self.assertEqual(clusters, clusters_expected)
        self.assertTrue(np.isclose(similarity_matrix, similarity_matrix_expected).all())
        self.assertTrue(np.isclose(linkage, linkage_expected).all())
        self.assertTrue((raw_cls == raw_cls_expected).all())
        
        shutil.rmtree(tmp_folder)


    def test_write_profile_pairs(self):
        pairs = {"A": {"B": 3, "C": 4}, "B": {"C": 5, "D": 9}, "C": {"D": 6}}
        tmp_folder = os.path.join(ROOT_PATH, "tmp", "dummy_profile")
        filename = os.path.join(tmp_folder, "profile_pairs.txt")
        os.makedirs(tmp_folder, exist_ok=True)

        self.branched.write_profile_pairs(pairs, filename)
        self.assertTrue(os.path.exists(filename))

        filee = open(filename)
        self.assertEqual("A\tB\t3\nA\tC\t4\nB\tC\t5\nB\tD\t9\nC\tD\t6\n", filee.read())
        filee.close()

        shutil.rmtree(tmp_folder)

    # specifity_index related methods
    ####################################

    def test_onto_levels_from_profiles(self):
        self.hierarchical.precompute()
        self.hierarchical.add_profile("A", ["Child2", "Parental"], substitute= False) # Add profiles
        self.hierarchical.add_profile("B", ["Child2", "Parental", "FakeID"], substitute= False)
        self.hierarchical.add_profile("C", ["Child2", "Parental"], substitute= False)
        self.hierarchical.add_profile("D", ["Parental"], substitute= False)
 
        # Ontology levels
        self.assertEqual({1: ["Parental"], 2: ["Child2"]}, self.hierarchical.get_ontology_levels_from_profiles())
        self.assertEqual({1: ["Parental", "Parental", "Parental", "Parental"], 2: ["Child2", "Child2", "Child2"]}, self.hierarchical.get_ontology_levels_from_profiles(False))
        self.assertEqual({1: ["Parental"], 2: ["Child2"]}, self.hierarchical.get_ontology_levels())
    
    def test_specificity_index(self):
        self.hierarchical.load_profiles({"A": ["Child2"], "B": ["Parental"],"C": ["Child2", "Parental"]}, calc_metadata= False, substitute= False)
        self.assertEqual(([[1, 1, 2], [2, 1, 2]], [[1, 50.0, 50.0, 50.0], [2, 50.0, 50.0, 50.0]]), self.hierarchical.get_profile_ontology_distribution_tables())
        self.assertEqual(0.967, round(self.hierarchical.get_weigthed_level_contribution([[1,0.5],[2,0.7]],3,3), 3))
        
        enrichment_hierarchical2 = Ontology(file= os.path.join(DATA_TEST_PATH, "enrichment_ontology2.obo"), load_file= True)
        enrichment_hierarchical2.load_profiles({
            "A": ["branchB", "branchAChild1", "root"],
            "B": ["root", "branchA","branchB", "branchAChild2", "branchAChild1"],
            "C": ["root", "branchC", "branchAChild1", "branchAChild2"],
            "D": ["root", "branchAChild1", "branchAChild2"]},
             calc_metadata= False, substitute= False)
        self.assertEqual(round(13.334/10.0, 4), round(enrichment_hierarchical2.get_dataset_specifity_index('weigthed'),4))
        self.assertEqual(0,enrichment_hierarchical2.get_dataset_specifity_index('uniq'))


    ########################################
    ## GENERAL ONTOLOGY METHODS
    ########################################
    
    def test_IO_items(self):
        self.hierarchical.precompute()
        # Handle items
        items_rel = {"Parental": ['a','b'], "Child3": ['c']}

        self.hierarchical.items = {} # reset items from method get_items_from_profiles
        self.hierarchical.load_item_relations_to_terms(items_rel)
        self.assertEqual(items_rel, self.hierarchical.items)
        self.hierarchical.load_item_relations_to_terms(items_rel,False,True)
        self.assertEqual(items_rel, self.hierarchical.items)
        self.hierarchical.load_item_relations_to_terms(items_rel,True,True) # here third must no be relevant
        self.assertEqual(items_rel, self.hierarchical.items)


    def test_defining_items_from_instance_variable(self):
        self.hierarchical.set_items_from_dict("is_a")
        self.assertEqual({"Child2": ["Parental"]}, self.hierarchical.items)
        self.hierarchical.items = {} # Reseting items variable

        self.hierarchical.add_profile("A", ["Child2", "Parental"], substitute= False) # Add profiles
        self.hierarchical.add_profile("B", ["Child2", "Parental", "FakeID"], substitute= False)
        self.hierarchical.add_profile("C", ["Child2", "Parental"], substitute= False)
        self.hierarchical.add_profile("D", ["Parental"], substitute= False)
        # Profiles dictionary
        self.hierarchical.get_items_from_profiles()
        self.assertEqual({"Child2": ["A", "B", "C"], "Parental": ["A", "B", "C", "D"]}, self.hierarchical.items)


    def test_defining_instance_variables_from_items(self):
        self.hierarchical.set_items_from_dict("is_a")
        self.hierarchical.get_profiles_from_items()
        self.assertEqual({"Parental": ["Child2"]}, self.hierarchical.profiles)


    def test_expand_items(self):
        # Add items
        initial_items = {"root": ["branchA"], "Child1": ["branchAChild1"], "Child2": ["branchAChild1", "branchAChild2", "branchB"]}
        exact_expand = {"root": ["branchA", "branchAChild1"], "Child1": ["branchAChild1"], "Child2": ["branchAChild1", "branchAChild2", "branchB"]}
        onto_expand = {"root": ["branchA", "branchAChild1"], "Child1": ["branchAChild1"], "Child2": ["branchAChild1", "branchAChild2", "branchB"]}
        onto_cleaned_expand = {"root": ["branchAChild1"], "Child1": ["branchAChild1"], "Child2": ["branchAChild1", "branchAChild2", "branchB"]}
        self.short_hierarchical.load_item_relations_to_terms(initial_items)
        # Expand to parentals (exact match)
        self.short_hierarchical.expand_items_to_parentals()
        self.assertEqual(exact_expand, self.short_hierarchical.items)
        # Expand to parentals (MICAS)
        self.short_hierarchical.load_item_relations_to_terms(initial_items, True)
        self.short_hierarchical.expand_items_to_parentals(ontology= self.enrichment_hierarchical, clean_profiles= False)
        self.assertEqual(onto_expand, self.short_hierarchical.items)
        self.short_hierarchical.load_item_relations_to_terms(initial_items, True)
        self.short_hierarchical.expand_items_to_parentals(ontology= self.enrichment_hierarchical)
        self.assertEqual(onto_cleaned_expand, self.short_hierarchical.items)
        ###########################
        ## NOW INCLUDING NOT STORED TERMS
        ###########################
        initial_items = {"Child1": ["branchAChild1"], "Child2": ["branchAChild2", "branchB"]}
        onto_notroot_items = {"root": ["branchA"], "Child1": ["branchAChild1"], "Child2": ["branchAChild2", "branchB"]}
        self.short_hierarchical.load_item_relations_to_terms(initial_items, True)
        self.short_hierarchical.expand_items_to_parentals(ontology= self.enrichment_hierarchical, clean_profiles= False)
        self.assertEqual(onto_notroot_items, self.short_hierarchical.items)

    def test_return_terms_by_keyword_match(self):
        expected = ["Child2"]
        returned = self.hierarchical.return_terms_by_keyword_match("Child", fields = ['name'])
        returned2 = self.hierarchical.return_terms_by_keyword_match("Child", fields = ['name', 'name']) #Checking that no duplicated terms are returned even if the fields parameter is duplicated
        returned3 = self.hierarchical.return_terms_by_keyword_match("mannosyltransferase", fields = ['name', 'synonym'])
        returned4 = self.hierarchical.return_terms_by_keyword_match("mannosyltransferase", fields = ['name', 'synonym', 'synonym']) #Checking that no duplicated terms are returned even if the fields parameter is duplicated
        self.assertEqual(expected, returned)
        self.assertEqual(expected, returned2)
        self.assertEqual(expected, returned3)
        self.assertEqual(expected, returned4)

        expected = ["branchA", "branchAChild1", "branchAChild2", "branchB"]
        returned = self.enrichment_hierarchical.return_terms_by_keyword_match("Child", fields = ['name'])
        self.assertEqual(expected, returned)


    #################################
    # AUXILIAR METHODS
    #################################

    def test_auxiliar_methods(self):
        self.hierarchical.precompute()
        iteration_with_custom_each = []
        #### TODO: ASK PEDRO ABOUT THE EACH METHOD OF THE HIERARCHICAL CLASS (the att parameter) and how it will be migrated in Python
        for (iid, tags) in self.hierarchical.each(att=True):
            iteration_with_custom_each.append([iid, tags])
        
        self.assertEqual([
            ["Parental", {"id": "Parental", "name": "All", "comment": "none"}], 
            ["Child2", {"id": "Child2", "name": "Child2", "synonym": ["\"1,6-alpha-mannosyltransferase activity\" EXACT []"], "alt_id": ["Child3", "Child4"], "is_a": ["Parental"]}]],
             iteration_with_custom_each)
        self.assertEqual(["Parental"], self.hierarchical.get_root())
        self.assertEqual([["Parental", "All", 1, ""], ["Child2", "Child2", 2, "1,6-alpha-mannosyltransferase activity"]], self.hierarchical.list_term_attributes())
