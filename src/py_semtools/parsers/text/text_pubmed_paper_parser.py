import os
import re
import traceback
import tarfile
import xml.etree.ElementTree as ET
from py_semtools.parsers.text.text_pubmed_parser import TextPubmedParser

class TextPubmedPaperParser(TextPubmedParser):

    @classmethod
    def parse_xml(cls, string_xml):
        return super().parse_xml(string_xml, is_file=False, as_element_tree = False)

    @classmethod
    def parse(cls, file_path, logger = None, options={'clean_type': 'hard'}):
        members = []
        stats = {"no_abstract": 0, "no_pmid": 0, "total": 0, "errors": 0}
        tar = tarfile.open(file_path, 'r:gz') 
        if logger != None: logger.info(f"The file {file_path} has { len([member for member in tar.getmembers() if not member.isdir()]) } xml papers")
        for member in tar.getmembers():
            if member.isdir(): continue
            stats['total'] += 1
            f=tar.extractfile(member)
            filename = os.path.join(file_path, member.path)
            try:            
                parsed_paper = cls.parse_paper(f.read(), filename, options=options)
                members.append(parsed_paper)
                pmid, pmc, filename, year, whole_content, title, article_type, article_category, keywords = parsed_paper
                if pmid == "None":
                    stats["no_pmid"] += 1
                    if logger != None: logger.warning(f"Warning: Article without PMID found in file {filename}")
                elif whole_content == "None":
                    stats["no_abstract"] += 1
                    if logger != None: logger.warning(f"Warning: Article PDMID:{pmid} without abstract found in file {filename}")
                elif len(whole_content) < 500: 
                    #stats["no_abstract"] += 1 #We do not count short papers as no_abstract papers, IDK why this was here
                    if logger != None: logger.warning(f"Warning: Article PDMID:{pmid} had short text content in file {filename}. Content:{whole_content}")

            except Exception as e:
                stats['errors'] += 1
                if logger != None: logger.error(f"There was a problem proccessing the file {filename} with the following error: {e}\n{traceback.format_exc()}")
        tar.close()
        return members, stats

    @classmethod
    def parse_paper(cls, paper_xml_string, filename, options={'clean_type': 'hard'}):
        whole_content = "None"
        year = 0
        year_text = None
        pmc = "None"
        pmid = "None"
        title = "None"
        keywords = []
        article_root = cls.parse_xml(paper_xml_string)

        #GETTING ARTICLE TITLE FIELD
        title = cls.do_recursive_find(article_root, ['front','article-meta','title-group','article-title'])
        title = cls.do_recursive_xml_content_parse(title).strip().lower() if cls.check_not_none_or_empty(title) else "None"
        #GETTING article-type property from article tag and article category from article-categories tag
        article_type = article_root.get('article-type').lower() if article_root.get('article-type') != None else "None"
        article_category = cls.do_recursive_find(article_root, ['front','article-meta','article-categories', 'subj-group', 'subject'])
        article_category = article_category.text.strip().lower() if cls.check_not_none_or_empty(article_category) else "None"
        #GETTING PMC ID, PMID AND YEAR
        for id_tags in article_root.iter('article-id'):
            if id_tags.get('pub-id-type') == "pmid":
                pmid = id_tags.text 
            if id_tags.get('pub-id-type') == "pmc":
                pmc = id_tags.text
        #GETTING KEYWORDS
        for kwd_group in article_root.iter('kwd-group'):
            for kwd in kwd_group.iter('kwd'):
                if cls.check_not_none_or_empty(kwd):
                    keywords.append(kwd.text.strip().lower())
        #GETTING YEAR
        for date_fields in article_root.iter("date"):
            if date_fields.get("date-type") == "accepted":
                year_text = cls.extract_year(date_fields.find("year").text)
        if year_text == None: #In case date-type is not available, we try getting year by using pmc-release field
            for date_fields in article_root.iter("pub-date"):
                if date_fields.get("pub-type") == "pmc-release":
                    year_text = cls.extract_year(date_fields.find("year").text)
                if date_fields.get("pub-type") != "pmc-release" and year_text == None:
                    year_text = cls.extract_year(date_fields.find("year").text)
        year = year_text if year_text != None else 0 

        #GETTING PAPER WHOLE CONTENT
        paper_root = article_root.find("body")
        if paper_root != None: whole_content = cls.perform_soft_cleaning(  cls.do_recursive_xml_content_parse(paper_root).strip(), type=options["clean_type"] )
        if len(whole_content.strip()) == 0: whole_content = "None"
        return [pmid, pmc, filename, year, whole_content, title, article_type, article_category, keywords]