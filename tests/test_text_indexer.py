#!/usr/bin/env python

#########################################################
# Load necessary packages and folder paths
#########################################################

from py_cmdtabs import CmdTabs
import json
import copy
import unittest
import os
import shutil
from py_semtools.indexers.text_indexer import TextIndexer

ROOT_PATH= os.path.dirname(__file__)

######### INPUTS DATA #########
INPUT_PATH = os.path.join(ROOT_PATH, 'data', "stEngine", "inputs")

SINGLE_PAP_XML_PATH = os.path.join(INPUT_PATH, "papers", "single_paper.tar.gz")
SINGLE_ABS_XML_PATH = os.path.join(INPUT_PATH, "abstracts", "single_abstract.xml.gz")

PAP_CHUNK1 = os.path.join(INPUT_PATH, "papers", "pap_chunk1.tar.gz")
PAP_CHUNK2 = os.path.join(INPUT_PATH, "papers", "pap_chunk2.tar.gz")

ABS_CHUNK1 = os.path.join(INPUT_PATH, "abstracts", "abs_chunk1.xml.gz")
ABS_CHUNK2 = os.path.join(INPUT_PATH, "abstracts", "abs_chunk2.xml.gz")

BLACKLIST1 = os.path.join(INPUT_PATH, "blacklisted_words1.txt")
BLACKLIST2 = os.path.join(INPUT_PATH, "blacklisted_words2.txt")
BLACKLIST3 = os.path.join(INPUT_PATH, "blacklisted_words3.txt")

######### EXPECTED DATA #########
REF_PATH = os.path.join(ROOT_PATH, 'data', "stEngine", "expected", "ready_indexes")

SINGLE_NONSPLIT_PAP_TEXT_PATH = os.path.join(REF_PATH, "papers", "single_pap_index_nonsplit")
SINGLE_SPLIT_PAP_TEXT_PATH = os.path.join(REF_PATH, "papers", "single_pap_index_split")

SINGLE_NONSPLIT_ABS_TEXT_PATH = os.path.join(REF_PATH, "abstracts", "single_abs_index_nonsplit")
SINGLE_SPLIT_ABS_TEXT_PATH = os.path.join(REF_PATH, "abstracts", "single_abs_index_split")


######### TEMPORAL FOLDER #########
TMP = os.path.join(ROOT_PATH, 'temp')

#########################################################
# Define TESTS
#########################################################

class TextIndexerTestCase(unittest.TestCase):
    #maxDiff = None

    def setUp(self):
        self.pmid = "PMID:1"
        self.year = "2024"
        self.title = "Title of the paper"
        self.file = "mock_file.xml"
        self.article_type = "research-article"
        self.article_category = "original-research"
        self.p1 = "First paragraph sentence 1. First paragraph sentence 2 part 1, First paragraph sentence 2 part 2."
        self.p2 = "Second paragraph sentence 1; Second paragraph sentence 2.\nSecond paragraph sentence 3."
        self.text = self.p1 + "\n\n" + self.p2
        self.text_length = f"{len(self.text)}"
        self.splitted_text = [["First paragraph sentence 1", "First paragraph sentence 2 part 1", "First paragraph sentence 2 part 2"],
                              ["Second paragraph sentence 1", "Second paragraph sentence 2", "Second paragraph sentence 3"]]
        self.n_sentences = "6"
        self.sentences_length = "26,33,33,27,27,27"

        processed_text = self.text.replace("\n", "")
        keywords = "None"
        self.mock_index = [self.pmid, processed_text, self.file, self.year, self.text_length, self.n_sentences, self.sentences_length, self.title, self.article_type, self.article_category, keywords]
        #Making a slightly different mock index 2 to check in write indexes method that different indexes are written in the same file.
        self.mock_index2 = ["PMID:2", processed_text, "mock_file2.xml", "1996", self.text_length, self.n_sentences, self.sentences_length, self.title, "review", "review-article", keywords]

    def test_check_to_filter_out(self):
        self.maxDiff = None
        #This test should find the word 'paper' inside the 'title' field, so it doesnt reach 'original' in 'category' field
        blackwords = ["compendium", "paper", "original"]
        filtered, word, type, type_content  = TextIndexer._check_to_filter_out(blackwords, self.title, self.article_category, "partial")
        self.assertEqual([True, "paper", "title", "Title of the paper"], [filtered, word, type, type_content])

        #This test should find the word 'original' inside the 'category' field
        blackwords = ["compendium", "other content", "original"]
        filtered, word, type, type_content  = TextIndexer._check_to_filter_out(blackwords, self.title, self.article_category, "partial")
        self.assertEqual([True, "original", "category", "original-research"], [filtered, word, type, type_content])

        #This test is doing exact match case, so it is not going to find match in title nor category, returning False and None in the other fields
        filtered, word, type, type_content  = TextIndexer._check_to_filter_out(blackwords, self.title, self.article_category, "exact")
        self.assertEqual([False, None, None, None], [filtered, word, type, type_content])

    def test_write_indexes(self):
        self.maxDiff = None
        os.makedirs(TMP, exist_ok=True)
        name, a_suffix, b_suffix = "tmp", "mock", "index"
        index_file = os.path.join(TMP, f"{name}_{a_suffix}_{b_suffix}.gz")

        #The variable below is going to be modified by "pop" method inside write_indexes method.
        indexes_to_write = copy.deepcopy([self.mock_index, self.mock_index2]) 
        TextIndexer.write_indexes(indexes_to_write, folder=TMP, name=name, a_suffix=a_suffix, b_suffix=b_suffix)
        
        CmdTabs.compressed_input = True
        returned_index = CmdTabs.load_input_data(index_file)
        expected_mock_indexes = copy.deepcopy([self.mock_index, self.mock_index2])
        self.assertEqual(expected_mock_indexes, returned_index)

        shutil.rmtree(TMP, ignore_errors=True)
        CmdTabs.compressed_input = False

    def test_split_document(self):
        self.maxDiff = None
        returned = TextIndexer.split_document(self.text, self.pmid)
        expected = self.splitted_text
        self.assertEqual(expected, returned)    

    def test_prepare_indexes(s):
        s.maxDiff = None
        # Test with no splitting
        options = {"split": False, "clean_type": "hard", "split_type": "classic", "remain_empty_documents": False}
        keywords = []
        required_data = [s.text, s.pmid, s.file, s.year, s.title, s.article_type, s.article_category, keywords, options]
        prepared_index = TextIndexer.prepare_indexes(*required_data)
        pmid, document, file, year, document_length, n_sentences, sentences_length, title, article_type, article_category, out_keywords = prepared_index
        s.assertEqual(pmid, s.pmid)
        s.assertEqual(document, json.dumps([[s.text]]))
        s.assertEqual(file, s.file)
        s.assertEqual(year, s.year)
        s.assertEqual(document_length, s.text_length) # length of the whole text
        s.assertEqual(n_sentences, "1") # number of sentences
        s.assertEqual(sentences_length, s.text_length) # length of each sentence
        s.assertEqual(title, s.title)
        s.assertEqual(article_type, s.article_type)
        s.assertEqual(article_category, s.article_category)
        s.assertEqual(out_keywords, "None")

        # Test with splitting
        options = {"split": True, "clean_type": "hard", "split_type": "classic", "remain_empty_documents": False}
        required_data = [s.text, s.pmid, s.file, s.year, s.title, s.article_type, s.article_category, keywords, options]
        prepared_index = TextIndexer.prepare_indexes(*required_data)
        pmid, splitted_doc, file, year, document_length, n_sentences, sentences_length, title, article_type, article_category, out_keywords = prepared_index
        s.assertEqual(pmid, s.pmid)
        s.assertEqual(splitted_doc, json.dumps(s.splitted_text))
        s.assertEqual(file, s.file)
        s.assertEqual(year, s.year)
        s.assertEqual(document_length, s.text_length)
        s.assertEqual(n_sentences, s.n_sentences)
        s.assertEqual(sentences_length, s.sentences_length)
        s.assertEqual(title, s.title)
        s.assertEqual(article_type, s.article_type)
        s.assertEqual(article_category, s.article_category)
        s.assertEqual(out_keywords, "None")


    def test_get_abstract_index(self):
        self.maxDiff = None
        ###### Test with no splitting ######
        pmid = "22981092"
        file = SINGLE_ABS_XML_PATH
        year = '2014'
        text_length = '908'
        n_sentences = '1'
        sentences_length = '908'
        title = 'depth of penetration of methylene blue in mandibular cortical bone.'
        article_type = 'None'
        article_category = 'None'
        expected_text = json.dumps([[open(SINGLE_NONSPLIT_ABS_TEXT_PATH).read().strip()]])
        expected_keywords = 'None'
        expected = [[pmid, expected_text, file, year, text_length, n_sentences, sentences_length, title, article_type, article_category, expected_keywords]]

        options = {"split": False, "filter_by_blacklist": None, "clean_type": "hard", "split_type": "classic", "remain_empty_documents": False}
        returned = TextIndexer.get_abstract_index(SINGLE_ABS_XML_PATH, options)
        self.assertEqual(expected, returned)

        ###### Test with splitting ######
        n_sentences = '8'
        sentences_length = '205,75,108,164,61,47,52,178'
        expected_text = open(SINGLE_SPLIT_ABS_TEXT_PATH).read().strip()
        expected = [[pmid, expected_text, file, year, text_length, n_sentences, sentences_length, title, article_type, article_category, expected_keywords]]

        options = {"split": True, "filter_by_blacklist": None, "clean_type": "hard", "split_type": "classic", "remain_empty_documents": False}
        returned = TextIndexer.get_abstract_index(SINGLE_ABS_XML_PATH, options)
        self.assertEqual(expected, returned)


    def test_get_paper_index(self):
        self.maxDiff = None
        ###### Test with no splitting ######
        pmid = '31803150'
        file = os.path.join(SINGLE_PAP_XML_PATH, "folder", "single_paper.xml")
        year = '2019'
        text_length = '50608'
        n_sentences = '1'
        sentences_length = '50608'
        title = 'a tetr-family protein (caethg_0459) activates transcription from a new promoter motif associated with essential genes for autotrophic growth in acetogens'
        article_type = 'research-article'
        article_category = 'microbiology'
        expected_keywords = 'wood–ljungdahl pathway,transcriptional regulation,gas fermentation,autotrophy'
        expected_text = json.dumps([[open(SINGLE_NONSPLIT_PAP_TEXT_PATH, encoding="utf-8").read().strip()]])
        expected = [[pmid, expected_text, file, year, text_length, n_sentences, sentences_length, title, article_type, article_category, expected_keywords]]

        options = {"split": False, "equivalences_file": None, "filter_by_blacklist": None, "clean_type": "hard", "split_type": "classic", "remain_empty_documents": False}
        returned = TextIndexer.get_paper_index(SINGLE_PAP_XML_PATH, options)
        self.assertEqual(expected, returned)
    
        ###### Test with splitting ######
        expected_text = open(SINGLE_SPLIT_PAP_TEXT_PATH, encoding="utf-8").read().strip()
        decoded_text = json.loads(expected_text)
        n_sentences = str(len([sentence for paragraph in decoded_text for sentence in paragraph])) #657 sentences
        sentences_length = ",".join([str(len(sentence)) for paragraph in decoded_text for sentence in paragraph])

        expected = [[pmid, expected_text, file, year, text_length, n_sentences, sentences_length, title, article_type, article_category, expected_keywords]]
        
        options = {"split": True, "equivalences_file": None, "filter_by_blacklist": None, "clean_type": "hard", "split_type": "classic", "remain_empty_documents": False}
        returned = TextIndexer.get_paper_index(SINGLE_PAP_XML_PATH, options)
        self.assertEqual(expected, returned)

        ###### Test with blacklisted words filter 1 (Partial Match found in title) #######
        options = {"split": False, "equivalences_file": None, "filter_by_blacklist": BLACKLIST1, "blacklisted_mode": "partial", "clean_type": "hard", "split_type": "classic", "remain_empty_documents": False}
        returned = TextIndexer.get_paper_index(SINGLE_PAP_XML_PATH, options)
        self.assertEqual([], returned)

        ###### Test with blacklisted words filter 2 (Partial Match found in category) #######
        options = {"split": False, "equivalences_file": None, "filter_by_blacklist": BLACKLIST2, "blacklisted_mode": "partial", "clean_type": "hard", "split_type": "classic", "remain_empty_documents": False}
        returned = TextIndexer.get_paper_index(SINGLE_PAP_XML_PATH, options)
        self.assertEqual([], returned)

        ###### Test with blacklisted words filter 3 (No Match is found with any blacklisted words, so the index of the paper is finally added) #######
        options = {"split": True, "equivalences_file": None, "filter_by_blacklist": BLACKLIST3, "blacklisted_mode": "partial", "clean_type": "hard", "split_type": "classic", "remain_empty_documents": False}
        returned = TextIndexer.get_paper_index(SINGLE_PAP_XML_PATH, options)
        self.assertEqual(expected, returned)

    def test_get_index(self):
        self.maxDiff = None
        #We are only checking equallity in text, but not in the rest of the fields, as it was already done in the previous tests.

        #Check it makes abstracts indexes without splitting. 
        options = {"parse": "PubmedAbstract", "split": False, "equivalences_file": None, "filter_by_blacklist": None, "clean_type": "hard", "split_type": "classic", "remain_empty_documents": False}
        expected_text = json.dumps([[open(SINGLE_NONSPLIT_ABS_TEXT_PATH).read().strip()]])
        returned_text = TextIndexer.get_index(SINGLE_ABS_XML_PATH, options)[0][1]
        self.assertEqual(expected_text, returned_text)
        
        #Check it makes abstracts indexes with splitting
        options = {"parse": "PubmedAbstract", "split": True, "equivalences_file": None, "filter_by_blacklist": None, "clean_type": "hard", "split_type": "classic", "remain_empty_documents": False}
        expected_text = open(SINGLE_SPLIT_ABS_TEXT_PATH).read().strip()
        returned_text = TextIndexer.get_index(SINGLE_ABS_XML_PATH, options)[0][1]
        self.assertEqual(expected_text, returned_text)

        #Check it makes papers indexes without splitting
        options = {"parse": "PubmedPaper", "split": False, "equivalences_file": None, "filter_by_blacklist": None, "clean_type": "hard", "split_type": "classic", "remain_empty_documents": False}
        expected_text = json.dumps([[open(SINGLE_NONSPLIT_PAP_TEXT_PATH).read().strip()]])
        returned_text = TextIndexer.get_index(SINGLE_PAP_XML_PATH, options)[0][1]
        self.assertEqual(expected_text, returned_text)

        #Check it makes papers indexes with splitting
        options = {"parse": "PubmedPaper", "split": True, "equivalences_file": None, "filter_by_blacklist": None, "clean_type": "hard", "split_type": "classic", "remain_empty_documents": False}
        expected_text = open(SINGLE_SPLIT_PAP_TEXT_PATH).read().strip()
        returned_text = TextIndexer.get_index(SINGLE_PAP_XML_PATH, options)[0][1]
        self.assertEqual(expected_text, returned_text)


#    def test_process_files(self):
#        pass
#        #self.assertEquals(0, 1)
#
#    def test_build_index(self):
#        pass
#        #self.assertEquals(0, 1)
#