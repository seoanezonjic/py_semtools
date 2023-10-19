#########################################################
# Load necessary packages
#########################################################

import unittest
import os
import sys
import math
from py_semtools import Ontology, JsonParser

ROOT_PATH = os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data')


#########################################################
# Define TESTS
#########################################################
class TestOBOFunctionalities(unittest.TestCase):

    def setUp(self):
        # Files

        self.file_Header = {"file": os.path.join(DATA_TEST_PATH, "only_header_sample.obo"), "name": "only_header_sample"}
        self.file_Hierarchical = {"file": os.path.join(DATA_TEST_PATH, "hierarchical_sample.obo"), "name": "hierarchical_sample"}
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
        self.assertEqual({"A": ['branchA', 'branchAChild1', 'branchB', 'root'], "B": ['branchA', 'branchAChild2', 'branchB', 'root'], "C": ['branchA', 'branchAChild1', 'branchAChild2', 'root'], "D": []}, self.enrichment_hierarchical.profiles)
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
        expected_profile_sizes = (2,2,2,1) #These values are sorted by profile size (ascending) and then reversed, so the order is pat3,patt2,pat1,pat4
        expected_parental_terms_per_profile = (0,1,1,0) ##These values are also sorted by profile size (descending), so same order as above (pat3,patt2,pat1,pat4)
        
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
        self.assertEqual([["Parental", "All", 1], ["Child2", "Child2", 2]], self.hierarchical.list_term_attributes())
