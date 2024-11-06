import re
class TextPubmedParser:

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
        if element.tag in ["table-wrap", "table", "fig", "fig-group"]: return whole_content
        if element.tag == "sec": whole_content += "\n\n"
        if element.tag == "p": whole_content += "\n\n"  
        # Content before nested element
        if element.tag not in ["xref", "sup"]:
            content = element.text
            if content != None: whole_content += " " + content.replace('\n', ' ') + " " 
        # Content of nested element
        for child in element: 
            whole_content += cls.do_recursive_xml_content_parse(child)
        # Content after nested element
        tail = element.tail
        if tail != None: whole_content +=  " " + tail.replace('\n', ' ') + " "
        return re.sub(r'\s+', ' ', whole_content)

    @classmethod
    def check_not_none_or_empty(cls, variable):
        if type(variable) != str: 
            condition = variable != None and variable.text != None and variable.text.strip().replace("&#x000a0;", "") != ""
        else:
            condition = variable != None and variable.strip().replace("&#x000a0;", "") != ""
        return condition

    @classmethod
    def perform_soft_cleaning(cls, raw_document):
        document = raw_document.strip().replace("\r", "\n").replace("&#13", "\n").replace("\t", " ")
        document = re.sub(r"\\[a-z]+(\[.+\])?(\{.+\})", r" ", document) #Removing latex commands
        document = re.sub(r"[ ]+", r" ", document) #Removing additional whitespaces between words
        document = re.sub(r"([A-Za-z\(\)]+[ ]*)\n([ ]*[A-Z-a-z\(\)]+)", r"\1 \2", document) #Removing nonsense newlines
        document = re.sub(r"([0-9]+)[\.\,]([0-9]+)", r"\1'\2", document) #Changing floating point numbers from 4.5 or 4,5 to 4'5
        document = re.sub(r"i\.?e\.?", "ie", document).replace("al.", "al ") #Changing i.e to ie and et al. to et al
        return document
