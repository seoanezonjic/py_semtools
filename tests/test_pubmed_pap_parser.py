#!/usr/bin/env python

#########################################################
# Load necessary packages and folder paths
#########################################################

import unittest, os, glob
from py_semtools import TextPubmedPaperParser

ROOT_PATH= os.path.dirname(__file__)

REF_PATH = os.path.join(ROOT_PATH, 'data', "stEngine", "expected", "raw_indexes", "papers")
SINGLE_PAP_CONTENT_PATH = os.path.join(REF_PATH, "single_pap_text")
PAP_CHUNK1_CONTENTS_PATH = os.path.join(REF_PATH, "chunk1")

INPUT_PATH = os.path.join(ROOT_PATH, 'data', "stEngine", "inputs", "papers")
SINGLE_PAP_XML_PATH = os.path.join(INPUT_PATH, "single_paper.xml")
PAP_CHUNK1 = os.path.join(INPUT_PATH, "pap_chunk1.tar.gz")
#PAP_CHUNK2 = os.path.join(INPUT_PATH, "pap_chunk2.tar.gz")



#########################################################
# Define TESTS
#########################################################

class TextPubmedPaperParserTestCase(unittest.TestCase):
    maxDiff = None
    
    def setUp(self):
        self.paper_xml = open(SINGLE_PAP_XML_PATH).read() 
        self.expected_pap_body_content = open(SINGLE_PAP_CONTENT_PATH).read()

        self.chk1_exptd_txt = {}
        for f in glob.glob(PAP_CHUNK1_CONTENTS_PATH + "/*"): self.chk1_exptd_txt[os.path.basename(f)] = open(f).read()
        self.chk1_exptd_titles = {
            "35042469": 'etiologies of fever of unknown origin in hiv/aids patients, hanoi, vietnam',
            "38108203": 'infectious pathogens and risk of oesophageal, gastric and duodenal cancers and ulcers in china: a case-cohort study',
            "None":     'burnout and joy in the profession of critical care medicine',
            "18382669": 'paracrine factors of mesenchymal stem cells recruit macrophages and endothelial lineage cells and enhance wound healing',
            "31356151": 'analysis of the recomposition of norms and representations in the field of psychiatry and mental health in the age of electronic mental health: qualitative study'
        }

        self.chk1_filepaths = {
            '35042469': os.path.join(PAP_CHUNK1, "pap_chunk1", "PMC008xxxxxx", "PMC8764815.xml"),
            '38108203': os.path.join(PAP_CHUNK1, "pap_chunk1", "PMC007xxxxxx", "PMC7615747.xml"),
            'None':     os.path.join(PAP_CHUNK1, "pap_chunk1", "PMC007xxxxxx", "PMC7092567.xml"),
            '18382669': os.path.join(PAP_CHUNK1, "pap_chunk1", "PMC002xxxxxx", "PMC2270908.xml"),
            '31356151': os.path.join(PAP_CHUNK1, "pap_chunk1", "PMC006xxxxxx", "PMC6819010.xml")
        }

    def test_parse_paper(self):
        pmid, pmc, filename, year, whole_content, title, article_type, article_category = TextPubmedPaperParser.parse_paper(self.paper_xml, SINGLE_PAP_XML_PATH)
        self.assertEqual(pmid, "31803150")
        self.assertEqual(pmc, "PMC6873888")
        self.assertEqual(filename, SINGLE_PAP_XML_PATH)
        self.assertEqual(whole_content, self.expected_pap_body_content)
        self.assertEqual(year, 2019)
        self.assertEqual(title, "a tetr-family protein (caethg_0459) activates transcription from a new promoter motif associated with essential genes for autotrophic growth in acetogens")
        self.assertEqual(article_type, "research-article") 
        self.assertEqual(article_category, "microbiology") 

    def test_parse(self):
        chunk1_txts = self.chk1_exptd_txt
        chunk1_titles = self.chk1_exptd_titles
        chunk1_paths = self.chk1_filepaths
        
        expected_indexes = [['35042469', 'PMC8764815', chunk1_paths['35042469'], 2022, chunk1_txts["35042469"], chunk1_titles['35042469'], 'research-article', 'research'], 
                            ['38108203', 'PMC7615747', chunk1_paths['38108203'], 2024, chunk1_txts["38108203"], chunk1_titles['38108203'], 'research-article', 'article'], 
                            [None,       'PMC7092567', chunk1_paths['None'],     2020, chunk1_txts["None"],     chunk1_titles['None'],     'review-article',   'review'],
                            ['18382669', 'PMC2270908', chunk1_paths['18382669'], 2008, chunk1_txts['18382669'], chunk1_titles['18382669'], 'research-article', 'research article'], 
                            ['31356151', 'PMC6819010', chunk1_paths['31356151'], 2019, chunk1_txts['31356151'], chunk1_titles['31356151'], 'research-article', 'original paper']]

        raw_indexes, stats = TextPubmedPaperParser.parse(PAP_CHUNK1)
        self.assertEqual(raw_indexes, expected_indexes)
        self.assertEqual(stats, {'total': 5, 'no_abstract': 1, 'no_pmid': 1, 'errors': 0})