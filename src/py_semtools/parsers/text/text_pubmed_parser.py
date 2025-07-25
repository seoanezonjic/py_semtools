import re, gzip
import xml.etree.ElementTree as ET
from py_semtools.parsers.text.text_basic_parser import TextBasicParser

class TextPubmedParser(TextBasicParser):

    @classmethod
    def parse_xml(cls, path_or_string, is_file=False, is_compressed=False, as_element_tree = False):
        file_opener = gzip.open if is_compressed else open
        if is_file:
            with file_opener(path_or_string) as f:
                parsed_xml = ET.parse(f)
        else:
            parsed_xml = ET.ElementTree( ET.fromstring(path_or_string) )

        if not as_element_tree: parsed_xml = parsed_xml.getroot()
        return parsed_xml


    @classmethod
    def do_recursive_find(cls, initial_tag, subtags_list):
        if len(subtags_list) == 0:
            return initial_tag
        nested_tag = initial_tag.find(subtags_list[0])
        if nested_tag != None:
            return cls.do_recursive_find(nested_tag, subtags_list[1:])
        else:
            return None

    @classmethod
    def do_recursive_xml_content_parse(cls, element):
        whole_content = ""  
        # Text content inside current tag (until it finds a nested tag, if any)
        if element.tag not in ["xref", "sup", "table-wrap", "table", "fig", "fig-group"]:
            content = element.text
            if content != None: whole_content += " " + content.replace('\n', ' ') + " " 
        # Text Content inside nested tags, if any
            for child in element: 
                whole_content += cls.do_recursive_xml_content_parse(child)
        # Text Content after tag closing until it reaches the next tag 
        if element.tag in ["sec","p"] : whole_content += "\n\n"
        tail = element.tail
        if tail != None: whole_content +=  " " + tail.replace('\n', ' ') + " "
        return re.sub(r'[ ]+', ' ', whole_content)
