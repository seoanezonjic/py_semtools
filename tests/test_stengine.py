#!/usr/bin/env python

#########################################################
# Load necessary packages and folder paths
#########################################################

import subprocess
import json
from py_cmdtabs.cmdtabs import CmdTabs
import gzip, glob, shutil
import copy
import unittest
import os, sys, site
from py_semtools import STengine


ROOT_PATH= os.path.dirname(__file__)
TMP_PATH = os.path.join(ROOT_PATH, 'tmp')

INPUT_PATH = os.path.join(ROOT_PATH, 'data', "stEngine", "inputs", "prepared_indexes")
EMBEDD_PICKLES_PATH = os.path.join(ROOT_PATH, 'data', "stEngine", "embedded")

#Load the environment variables
MODEL_NAME="all-MiniLM-L6-v2"
CURRENT_MODEL=os.path.join(ROOT_PATH, 'models', MODEL_NAME)
#CURRENT_MODEL=os.path.join(site.USER_BASE, "semtools", 'stEngine', MODEL_NAME)

#########################################################
# Define TESTS
#########################################################

class STEngineCPUTestCase(unittest.TestCase):
    maxDiff = None

    has_embedding_model = os.path.exists(os.path.join(CURRENT_MODEL, f'models--sentence-transformers--{MODEL_NAME}'))
    if has_embedding_model:
        stEngine = STengine(gpu_devices=[])
        stEngine.init_model(MODEL_NAME, cache_folder = CURRENT_MODEL, verbose = True)

    def setUp(self):
        pass       

    def test_init_model(self):
        if not self.has_embedding_model: self.skipTest(f"Model {MODEL_NAME} not found in {CURRENT_MODEL}. Please download it first.")
        from sentence_transformers import SentenceTransformer
        stEngine2 = STengine(gpu_devices=[])
        stEngine2.init_model(MODEL_NAME, cache_folder = CURRENT_MODEL, verbose = True)
        self.assertEqual(type(stEngine2.embedder), SentenceTransformer)

    def test_load_keyword_index(self):
        if not self.has_embedding_model: self.skipTest(f"Model {MODEL_NAME} not found in {CURRENT_MODEL}. Please download it first.")
        #HP:0000126      Hydronephrosis
        #HP:0000125      Pelvic kidney   Sacral kidney
        #HP:0000127      Renal salt wasting      Loss of salt in urine|Renal salt-wasting|Salt wasting|Salt-wasting
        HPOlist_file = os.path.join(INPUT_PATH, "queries", "hpo_list")
        expected = {"HP:0000126": ["Hydronephrosis"],
                    "HP:0000125": ["Pelvic kidney", "Sacral kidney"],
                    "HP:0000127": ["Loss of salt in urine", "Renal salt wasting", "Renal salt-wasting", "Salt wasting", "Salt-wasting"]}
        
        returned = self.stEngine.load_keyword_index(HPOlist_file)
        returned = {key: sorted(returned[key]) for key in returned.keys()}
        self.assertEqual(returned, expected)

    def test_get_splitted_document(self):
        if not self.has_embedding_model: self.skipTest(f"Model {MODEL_NAME} not found in {CURRENT_MODEL}. Please download it first.")        
        abs = json.dumps([["first pharagraph first sentence", "first pharagraph second sentence"],
                         ["second pharagraph first sentence", "second pharagraph second sentence"]])
        
        expected = {"PMID:A_0_0": "first pharagraph first sentence",
                    "PMID:A_0_1": "first pharagraph second sentence",
                    "PMID:A_1_0": "second pharagraph first sentence",
                    "PMID:A_1_1": "second pharagraph second sentence"} 

        returned = self.stEngine.get_splitted_document("PMID:A", abs)
        self.assertEqual(returned, expected)

    def test_load_pubmed_index(self):
        if not self.has_embedding_model: self.skipTest(f"Model {MODEL_NAME} not found in {CURRENT_MODEL}. Please download it first.")        
        #1230  '["First pharagraph first sentence", "first pharagraph second sentence"],["second pharagraph first sentence", "second pharagraph second sentence"]]'    2025    Nature
        #1231  '[["Mutations that cause Noonan syndrome alter genes encoding proteins with roles in the RAS-MAPK pathway", "leading to pathway dysregulation"]]'       2017    Science
        input_file = os.path.join(INPUT_PATH,"abstracts","example1.txt.gz")
        expected = {"1230_0_0": "first pharagraph first sentence",
                    "1230_0_1": "first pharagraph second sentence",
                    "1230_1_0": "second pharagraph first sentence",
                    "1230_1_1": "second pharagraph second sentence",
                    "1231_0_0": "Mutations that cause Noonan syndrome alter genes encoding proteins with roles in the RAS-MAPK pathway",
                    "1231_0_1": "leading to pathway dysregulation"}
        pubmed_index, n_papers = self.stEngine.load_pubmed_index(input_file, is_splitted=True)

        self.assertEqual(n_papers, 2)
        self.assertEqual(pubmed_index, expected)

    def test_embedd_several_queries(self):
        if not self.has_embedding_model: self.skipTest(f"Model {MODEL_NAME} not found in {CURRENT_MODEL}. Please download it first.")
        #EMBEDD_PICKLES_PATH
        #self.stEngine.embedd_several_queries()
        pass

    def test_embedd_single_query(self):
        if not self.has_embedding_model: self.skipTest(f"Model {MODEL_NAME} not found in {CURRENT_MODEL}. Please download it first.")
        pass

    def test_embed_save_corpus(self):
        if not self.has_embedding_model: self.skipTest(f"Model {MODEL_NAME} not found in {CURRENT_MODEL}. Please download it first.")
        options = {"corpus_embedded": os.path.join(TMP_PATH, "sample.pkl"), 'verbose': False, 'gpu_device': []}
        self.skipTest("Not implemented yet")
        
    def test_save_similarities(self):
        if not self.has_embedding_model: self.skipTest(f"Model {MODEL_NAME} not found in {CURRENT_MODEL}. Please download it first.")
        output_file = os.path.join(TMP_PATH, "tmp_similarities.tsv")
        similarities = {"HPO:A": {"PMID:1": 0.5, "PMID:2": 0.3},
                        "HPO:B": {"PMID:2": 0.4, "PMID:3": 0.2}}

        expected = [["HPO:A", "PMID:1", '0.5'],
                    ["HPO:A", "PMID:2", '0.3'],
                    ["HPO:B", "PMID:2", '0.4'],
                    ["HPO:B", "PMID:3", '0.2']]

        options = {"threshold": 0.1, "order": "query-corpus"}        

        self.stEngine.save_similarities(output_file, similarities, options)
        returned = CmdTabs.load_input_data(output_file)
        self.assertEqual(returned, expected)

        #remove the temporary file
        os.remove(output_file)


class STEngineGPUTestCase(unittest.TestCase):
    maxDiff = None

    has_embedding_model = os.path.exists(os.path.join(CURRENT_MODEL, f'models--sentence-transformers--{MODEL_NAME}'))
    if has_embedding_model:
        try:
            subprocess.check_output('nvidia-smi')
            GPU_AVAILABLE = True
            stEngine = STengine(gpu_devices=["cuda:0", "cuda:1"])
            stEngine.init_model(MODEL_NAME, cache_folder = CURRENT_MODEL, verbose = True)  
        except Exception:
            GPU_AVAILABLE = False

    def setUp(self):
        pass

    def test_prueba_gpu(self):
        if not self.has_embedding_model: self.skipTest(f"Model {MODEL_NAME} not found in {CURRENT_MODEL}. Please download it first.")

        if not self.GPU_AVAILABLE:
            self.skipTest("No GPU available for testing. Run it in machine with a GPU.")
        else:
            self.assertTrue(self.stEngine.gpu_available)

    # This test is to check if the GPU information can be launch without any error due to torch changing methods or attributes.
    def test_show_gpu_information(self):
        if not self.has_embedding_model: self.skipTest(f"Model {MODEL_NAME} not found in {CURRENT_MODEL}. Please download it first.")

        if not self.GPU_AVAILABLE:
            self.skipTest("No GPU available for testing. Run it in machine with a GPU.")
        else:
            self.stEngine.show_gpu_information()
            self.assertTrue(True)
         
