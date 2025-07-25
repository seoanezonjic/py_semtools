#!/usr/bin/env python

#########################################################
# Load necessary packages and folder paths
#########################################################

import copy
import unittest
import os
from py_semtools import TextPubmedParser
import xml.etree.ElementTree as ET

ROOT_PATH= os.path.dirname(__file__)
DATA_TEST_PATH = os.path.join(ROOT_PATH, 'data', "stEngine", "inputs")


#########################################################
# Define TESTS
#########################################################

class TextPubmedParserTestCase(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        pass

    def test_parse_xml(self):
        #### Testing when the input is a string ####
        xml_string = "<title>This is a <em>test</em></title>"
        xml_object = TextPubmedParser.parse_xml(xml_string)
        #Checkin the type of the object returned to assert that the method returns a ElementTree object
        self.assertEqual(type(xml_object), ET.Element)
        #Checking that the decoded object is the same as the original xml string passed to parse_xml method
        self.assertEqual(xml_string, ET.tostring(xml_object).decode())
        #Checking that the tag is "title"
        self.assertEqual(xml_object.tag, "title")
        #Checking that the text without tags is "This is a test"
        string_without_tags = "This is a test"
        self.assertEqual(string_without_tags, TextPubmedParser.do_recursive_xml_content_parse(xml_object).strip())
        

        #### Testing when the input is an unzipped file ####
        xml_file = os.path.join(DATA_TEST_PATH, "example.xml")
        expected_xml = "<title>This is a unzipped<em>test</em></title>"
        expected_string = "This is a unzipped test"

        xml_object = TextPubmedParser.parse_xml(xml_file, is_file=True)
        self.assertEqual(type(xml_object), ET.Element)
        self.assertEqual(expected_xml, ET.tostring(xml_object).decode())    
        self.assertEqual(expected_string, TextPubmedParser.do_recursive_xml_content_parse(xml_object).strip())


        #### Testing when the input is a zipped file ####
        xml_file = os.path.join(DATA_TEST_PATH, "example_zipped.xml.gz")
        expected_xml = "<title>This is a zipped<em>test</em></title>"
        expected_string = "This is a zipped test"

        xml_object = TextPubmedParser.parse_xml(xml_file, is_file=True, is_compressed=True)
        self.assertEqual(type(xml_object), ET.Element)
        self.assertEqual(expected_xml, ET.tostring(xml_object).decode())
        self.assertEqual(expected_string, TextPubmedParser.do_recursive_xml_content_parse(xml_object).strip())


    def test_do_recursive_find(self):
        # Test when the title is not present
        no_title_example = TextPubmedParser.parse_xml("<PubmedArticle><MedlineCitation><Other>There is not title here</Other></MedlineCitation></PubmedArticle>")
        expected = None
        returned = TextPubmedParser.do_recursive_find(no_title_example, ['MedlineCitation','Article','ArticleTitle'])
        self.assertEqual(expected, returned)
        
        # Test when the title is present
        title_example = "<PubmedArticle><MedlineCitation><Article><ArticleTitle>{}</ArticleTitle></Article></MedlineCitation></PubmedArticle>"
        nested_title = TextPubmedParser.parse_xml(title_example.format("Title without tag"))
        expected = "Title without tag"
        returned = TextPubmedParser.do_recursive_find(nested_title, ['MedlineCitation','Article','ArticleTitle'])
        self.assertEqual(expected, returned.text)

        # Test when the title is present with tags
        nested_with_tag = TextPubmedParser.parse_xml(title_example.format("Title <em>with</em> tag"))
        expected = "Title with tag"
        returned = TextPubmedParser.do_recursive_find(nested_with_tag, ['MedlineCitation','Article','ArticleTitle'])
        ret_formated = TextPubmedParser.do_recursive_xml_content_parse(returned).strip()
        self.assertEqual(expected, ret_formated)


    def test_do_recursive_xml_content_parse(self):
        #Simple sentence with 3 different tags along
        text_body_with_tags = TextPubmedParser.parse_xml("<body><title>Differential <em>RNA</em> Sequencing</title></body>")
        expected = "Differential RNA Sequencing"
        returned = TextPubmedParser.do_recursive_xml_content_parse(text_body_with_tags).strip()
        self.assertEqual(expected, returned)

        #Checking it skips xref and sup tags
        text_with_xref = '<body>Content<xref>Figure 1</xref>more content<sup>1</sup></body>'
        returned = TextPubmedParser.do_recursive_xml_content_parse(TextPubmedParser.parse_xml(text_with_xref)).strip()
        expected = "Content more content"
        self.assertEqual(expected, returned)

        #Checking it skips table, fig, fig-group and table-wrap tags. Proving it with a table-wrap tag case, but the same is valid for the other tags
        text_with_table_wrap = '<body>Content<table-wrap id="T1"><label>TABLE 1</label><caption>Proteins ...</caption></table-wrap>more content</body>'
        returned = TextPubmedParser.do_recursive_xml_content_parse(TextPubmedParser.parse_xml(text_with_table_wrap)).strip()
        expected = "Content more content"
        self.assertEqual(expected, returned)

        #Checking it adds a newline after sec and p tags
        text_with_sec = '<body><sec><p>Intro 1.1</p><p>Intro 1.2</p></sec><sec><p>Methods 1.1</p><p>Methods 1.2</p></sec></body>'
        returned = TextPubmedParser.do_recursive_xml_content_parse(TextPubmedParser.parse_xml(text_with_sec)).strip()
        expected = "Intro 1.1 \n\n Intro 1.2 \n\n\n\n Methods 1.1 \n\n Methods 1.2"
        self.assertEqual(expected, returned)


    def test_check_not_none_or_empty(self):
        empty_string = ""
        empty_xml_tag = TextPubmedParser.parse_xml("<em></em>")
        pseudo_empty_tag = TextPubmedParser.parse_xml("<em>&#x000a0;</em>")
        not_emtpy_string = "<em>This is not empty</em>"

        # Test when the string is empty
        self.assertFalse(TextPubmedParser.check_not_none_or_empty(empty_string))
        self.assertFalse(TextPubmedParser.check_not_none_or_empty(empty_xml_tag))
        self.assertFalse(TextPubmedParser.check_not_none_or_empty(pseudo_empty_tag))
        # Test when the string is not empty
        self.assertTrue(TextPubmedParser.check_not_none_or_empty(not_emtpy_string))


    def test_perform_soft_cleaning(self):
        #Test with a string with additional whitespaces
        test_with_whitespaces = "Test    with  additional       whitespaces"
        test_with_whitespaces_expected = "Test with additional whitespaces"
        test_with_whitespaces_returned = TextPubmedParser.perform_soft_cleaning(test_with_whitespaces, type="soft")
        self.assertEqual(test_with_whitespaces_expected, test_with_whitespaces_returned)

        #Test with a string with latex commands
        test_with_latex = "This is a test with a \\textbf{latex} command"
        test_with_latex_expected = "This is a test with a latex command"
        test_with_latex_returned = TextPubmedParser.perform_soft_cleaning(test_with_latex, type="soft")
        self.assertEqual(test_with_latex_expected, test_with_latex_returned)

        #Test with a string with floating point numbers
        test_with_float = "This is a test with 4.5 and 4,5"
        test_with_float_expected = "This is a test with 4'5 and 4'5"
        test_with_float_returned = TextPubmedParser.perform_soft_cleaning(test_with_float, type="hard")
        self.assertEqual(test_with_float_expected, test_with_float_returned)

        #Testing with tabs and carriage returns
        test_with_tabs = "This is a test with\ttabs and \r carriage returns"
        test_with_tabs_expected = "This is a test with tabs and \n carriage returns"
        test_with_tabs_returned = TextPubmedParser.perform_soft_cleaning(test_with_tabs, type="soft")
        self.assertEqual(test_with_tabs_expected, test_with_tabs_returned)

        #Test that soft cleaning does not change points and commas in numbers
        test_with_float = "This is a test with 4.5 and 4,5"
        test_with_float_expected = "This is a test with 4.5 and 4,5"
        test_with_float_returned = TextPubmedParser.perform_soft_cleaning(test_with_float, type="soft")
        self.assertEqual(test_with_float_expected, test_with_float_returned)

        #Test that basic leaves the text as it is
        test_raw = "This is a test with a \\textbf{latex} command      many spaces and \t tabs and 4.5 and 4,5 numbers"
        expected = "This is a test with a \\textbf{latex} command      many spaces and \t tabs and 4.5 and 4,5 numbers"
        returned = TextPubmedParser.perform_soft_cleaning(test_raw, type="basic")
        self.assertEqual(expected, returned)
