import os
import re
import gzip
import traceback
import xml.etree.ElementTree as ET
from py_semtools.parsers.text.text_pubmed_parser import TextPubmedParser

class TextPubmedAbstractParser(TextPubmedParser):

    @classmethod
    def parse_xml(cls, zipped_file):
        return super().parse_xml(zipped_file, is_file=True, is_compressed=True, as_element_tree = True)

    @classmethod
    def parse(cls, file, logger = None): 
        texts = [] # aggregate all abstracts in XML file
        stats = {"total": 0, "no_abstract": 0, "no_pmid": 0}
        mytree = cls.parse_xml(file)
        pubmed_article_set = mytree.getroot()
        for article in pubmed_article_set:
            stats['total'] += 1
            text_data = cls.parse_abstract(article)
            pmid, abstract_content, year, title, article_type, article_category = text_data
            texts.append(text_data)
            if pmid == None:
                stats["no_pmid"] += 1
                if logger != None: logger.warning(f"Warning: Article without PMID found in file {file}")
            elif abstract_content == "":
                stats["no_abstract"] += 1
                if logger != None: logger.warning(f"Warning: Article PDMID:{pmid} without abstract found in file {file}")
            elif len(abstract_content.split(" ")) < 10: 
                stats["no_abstract"] += 1
                if logger != None: logger.warning(f"Warning: Article PDMID:{pmid} had short abstract content in file {file}. Content:{abstract_content}")
        return texts, stats

    @classmethod
    def parse_abstract(cls, article):
        pmid = None
        abstract_content = ""
        article_type = "none"
        article_category = "none"
        year = 0
        title = cls.do_recursive_find(article, ['MedlineCitation','Article','ArticleTitle'])
        title = cls.do_recursive_xml_content_parse(title).strip().lower() if cls.check_not_none_or_empty(title) else "none"
        for data in article.find('MedlineCitation'):
            if data.tag == 'PMID':
                pmid = data.text
            if data.tag == 'DateCompleted':
                for fields in data:
                    if fields.tag == "Year": 
                        year_text = cls.extract_year(fields.text)
                        year = year_text if year_text != None else 0                      
            abstract = data.find('Abstract')
            if abstract != None:
                for fields in abstract:     
                    if fields.tag == 'AbstractText':
                        abstractText = cls.do_recursive_xml_content_parse(fields).strip()
                        if abstractText != None and abstractText != "":
                            #print(f"Text of abstract {pmid} in file {file}:")
                            #print(repr(abstractText), "\n\n")
                            raw_abstract = cls.perform_soft_cleaning(abstractText)                                                 
                            abstract_content += raw_abstract + "\n\n"
        abstract_content = cls.perform_soft_cleaning(abstract_content)        
        return [pmid, abstract_content, year, title, article_type, article_category]