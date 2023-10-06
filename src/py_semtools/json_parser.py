import json
from py_semtools import FileParser

class JsonParser(FileParser):

    @classmethod
    def load(cls, ontology, file, build = True):
        cls.read(ontology, file, build = build) 

   # Read a JSON file with an OBO_Handler object stored
    # ===== Parameters
    # +file+:: with object info
    # +file+:: if true, calculate indexes. Default: true
    # ===== Return
    # OBO_Handler internal fields 
    @classmethod
    def read(cls, ontology, file, build = True):
        # Read file
        with open(file) as f: jsonInfo = json.load(f)
        
        # Store info
        ontology.terms = jsonInfo['terms']
        ontology.ancestors_index = jsonInfo['ancestors_index']
        ontology.descendants_index = jsonInfo['descendants_index']
        ontology.alternatives_index = jsonInfo['alternatives_index']
        ontology.structureType = jsonInfo['structureType']
        ontology.ics = jsonInfo['ics']
        ontology.meta = jsonInfo['meta']
        ontology.max_freqs = jsonInfo['max_freqs']
        ontology.dicts = jsonInfo['dicts']
        ontology.profiles = jsonInfo['profiles']
        ontology.items = jsonInfo['items']
        ontology.term_paths = jsonInfo['term_paths']

        if build: ontology.precompute() 
