import os
import re
import traceback
import tarfile
import xml.etree.ElementTree as ET
from py_semtools.text_pubmed_parser import TextPubmedParser

class TextPubmedPaperParser(TextPubmedParser):

    @classmethod
    def parse_xml(cls, string_xml):
        return super().parse_xml(string_xml, is_file=False, as_element_tree = False)

    @classmethod
    def parse(cls, file_path, logger = None):
        members = []
        stats = {"no_abstract": 0, "no_pmid": 0, "total": 0, "errors": 0}
        tar = tarfile.open(file_path, 'r:gz') 
        if logger != None: logger.info(f"The file {file_path} has { len([member for member in tar.getmembers()]) } xml papers")
        for member in tar.getmembers():
            if member.isdir(): continue
            stats['total'] += 1
            f=tar.extractfile(member)
            filename = os.path.join(file_path, member.path)
            try:            
                parsed_paper = cls.parse_paper(f.read(), filename)
                members.append(parsed_paper)
                pmid, pmc, filename, year, whole_content, title, article_type, article_category = parsed_paper
                if pmid == None:
                    stats["no_pmid"] += 1
                    if logger != None: logger.warning(f"Warning: Article without PMID found in file {filename}")
                elif whole_content == "":
                    stats["no_abstract"] += 1
                    if logger != None: logger.warning(f"Warning: Article PDMID:{pmid} without abstract found in file {filename}")

            except Exception as e:
                stats['errors'] += 1
                if logger != None: logger.error(f"There was a problem proccessing the file {filename} with the following error: {e}\n{traceback.format_exc()}")
        tar.close()
        return members, stats

    @classmethod
    def parse_paper(cls, paper_xml_string, filename):
        whole_content = ""
        year = 0
        pmc = None
        pmid = None
        article_root = cls.parse_xml(paper_xml_string)

        #GETTING ARTICLE TITLE FIELD
        title = cls.do_recursive_find(article_root, ['front','article-meta','title-group','article-title'])
        title = cls.do_recursive_xml_content_parse(title).strip().lower() if cls.check_not_none_or_empty(title) else "none"
        #GETTING article-type property from article tag and article category from article-categories tag
        article_type = article_root.get('article-type').lower() if article_root.get('article-type') != None else "none"
        article_category = cls.do_recursive_find(article_root, ['front','article-meta','article-categories', 'subj-group', 'subject'])
        article_category = article_category.text.strip().lower() if cls.check_not_none_or_empty(article_category) else "none"
        #GETTING PMC ID, PMID AND YEAR
        for id_tags in article_root.iter('article-id'):
            if id_tags.get('pub-id-type') == "pmid":
                pmid = id_tags.text 
            if id_tags.get('pub-id-type') == "pmc":
                pmc = id_tags.text

        for date_fields in article_root.iter("date"):
            if date_fields.get("date-type") == "accepted":
                year = int(date_fields.find("year").text)
        if year == 0: #In case date-type is not available, we try getting year by using pmc-release field
            for date_fields in article_root.iter("pub-date"):
                if date_fields.get("pub-type") == "pmc-release":
                    year = int(date_fields.find("year").text)
                if date_fields.get("pub-type") != "pmc-release" and year == 0:
                    year = int(date_fields.find("year").text)

        #GETTING PAPER WHOLE CONTENT
        paper_root = article_root.find("body")
        if paper_root != None: whole_content = cls.perform_soft_cleaning(  cls.do_recursive_xml_content_parse(paper_root).strip()  )
        return [pmid, pmc, filename, year, whole_content, title, article_type, article_category]