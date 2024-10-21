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
    def check_not_none_or_empty(cls, variable):
        if type(variable) != str: 
            condition = variable != None and variable.text != None and variable.text.strip().replace("&#x000a0;", "") != ""
        else:
            condition = variable != None and variable.strip().replace("&#x000a0;", "") != ""
        return condition

    @classmethod
    def perform_soft_cleaning(cls, abstract):
        raw_abstract = abstract.strip().replace("\r", "\n").replace("&#13", "\n").replace("\t", " ")
        raw_abstract = re.sub(r"\\[a-z]+(\[.+\])?(\{.+\})", r" ", raw_abstract) #Removing latex commands
        raw_abstract = re.sub(r"[ ]+", r" ", raw_abstract) #Removing additional whitespaces between words
        raw_abstract = re.sub(r"([A-Za-z\(\)]+[ ]*)\n([ ]*[A-Z-a-z\(\)]+)", r"\1 \2", raw_abstract) #Removing nonsense newlines
        raw_abstract = re.sub(r"([0-9]+)[\.\,]([0-9]+)", r"\1'\2", raw_abstract) #Changing floating point numbers from 4.5 or 4,5 to 4'5
        raw_abstract = re.sub(r"i\.?e\.?", "ie", raw_abstract).replace("al.", "al ") #Changing i.e to ie and et al. to et al
        return raw_abstract
