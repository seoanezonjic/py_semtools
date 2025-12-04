#!/usr/bin/env python

#########################################################
# Load necessary packages and folder paths
#########################################################

import gzip, glob
import xml.etree.ElementTree as ET
import copy
import unittest
import os
from py_semtools import TextPubmedAbstractParser, TextPubmedParser


ROOT_PATH= os.path.dirname(__file__)

REF_PATH = os.path.join(ROOT_PATH, 'data', "stEngine", "expected", "raw_indexes", "abstracts")
ABS_SINGLE_FILE_CONTENT = os.path.join(REF_PATH, "single_abs_text")
ABS_CHUNK1_CONTENTS_PATH = os.path.join(REF_PATH, "chunk1")

INPUT_PATH = os.path.join(ROOT_PATH, 'data', "stEngine", "inputs", "abstracts")
ABS_SINGLE_FILE = os.path.join(INPUT_PATH, "single_abstract.xml")
ABS_CHUNK1 = os.path.join(INPUT_PATH, "abs_chunk1.xml.gz")
#ABS_CHUNK2 = os.path.join(INPUT_PATH, "abs_chunk2.xml.gz")

#########################################################
# Define TESTS
#########################################################

class TextPubmedAbstractParserTestCase(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        content = TextPubmedParser.parse_xml(ABS_SINGLE_FILE, is_file=True, is_compressed=False)
        self.article = content.find("PubmedArticle")
        self.abs_single_file_text = open(ABS_SINGLE_FILE_CONTENT).read()

        self.chk1_exptd_txt = {}
        for f in glob.glob(ABS_CHUNK1_CONTENTS_PATH + "/*"): self.chk1_exptd_txt[os.path.basename(f)] = open(f).read()
        self.chk1_exptd_titles = {
            "22981089": 'an autopsy case of bilateral adrenal pheochromocytoma-associated cerebral hemorrhage.',
            "22981088": 'latent adrenal ewing sarcoma family of tumors: a case report.',
            "None": 'conditioned pain modulation in populations with chronic pain: a systematic review and meta-analysis.',
            "22981091": 'association between polymorphisms of dna repair gene ercc5 and oral squamous cell carcinoma.',
            "22981092": 'depth of penetration of methylene blue in mandibular cortical bone.'
        }


    #This method in called by parse and only parses one abstract (a xml Element already loaded)
    def test_parse_abstract(self):
        self.maxDiff = None
        pmid, abstract_content, year, title, article_type, article_category, keywords = TextPubmedAbstractParser.parse_abstract(self.article)
        self.assertEqual(pmid, "22981092")

        self.assertEqual(abstract_content, self.abs_single_file_text)
        self.assertEqual(year, 2014)
        self.assertEqual(title, "depth of penetration of methylene blue in mandibular cortical bone.")
        self.assertEqual(article_type, "None") #These fields are only available for full papers
        self.assertEqual(article_category, "None") #These fields are only available for full papers
        self.assertEqual(keywords, []) #TODO: ADD an abstracts with keywords to test this field


    #This method loads a file and parses all the abstracts calling parse_xml and parse_abstract inside
    def test_parse(self):
        self.maxDiff = None
        chunk1_txts = self.chk1_exptd_txt
        chunk1_titles = self.chk1_exptd_titles

        #Note that documents without PMID/PMC or content are filtered out later with prepare_indexes method, not at this stage of the parsing
        #Below, the second document has no abstract, and the third document has no PMID
        expected_indexes = [['22981089', chunk1_txts["22981089"], 2013, chunk1_titles['22981089'], 'None', 'None', []], 
                            ['22981088', chunk1_txts["22981088"], 2013, chunk1_titles['22981088'], 'None', 'None', []], 
                            ["None",     chunk1_txts["None"],     2013, chunk1_titles['None'],     'None', 'None', []],
                            ['22981091', chunk1_txts['22981091'], 2013, chunk1_titles['22981091'], 'None', 'None', []], 
                            ['22981092', chunk1_txts['22981092'], 2014, chunk1_titles['22981092'], 'None', 'None', []]]

        
        raw_indexes, stats = TextPubmedAbstractParser.parse(ABS_CHUNK1)        
        self.assertEqual(raw_indexes, expected_indexes)
        self.assertEqual(stats, {'total': 5, 'no_abstract': 1, 'no_pmid': 1})
        