import json, gzip
from py_semtools import FileParser

class JsonParser(FileParser):

    @classmethod
    def load(cls, ontology, file, build = True, zipped=False):
        cls.read(ontology, file, build = build, zipped=zipped) 

   # Read a JSON file with an OBO_Handler object stored
    # ===== Parameters
    # +file+:: with object info
    # +file+:: if true, calculate indexes. Default: true
    # ===== Return
    # OBO_Handler internal fields 
    @classmethod
    def read(cls, ontology, file, build = True, zipped=False):
        # Read file
        opener = gzip.open(file, 'rt') if zipped else open(file, 'r')
        with opener as f: jsonInfo = json.load(f)
        
        # Store info
        ontology.terms = jsonInfo['terms']
        ontology.ancestors_index = jsonInfo['ancestors_index']
        ontology.descendants_index = jsonInfo['descendants_index']
        ontology.alternatives_index = jsonInfo['alternatives_index']
        ontology.structureType = jsonInfo['structureType']
        ontology.ics = jsonInfo['ics']
        ontology.meta = jsonInfo['meta']
        ontology.max_freqs = jsonInfo['max_freqs']
        ontology.dicts.update(jsonInfo['dicts'])
        ontology.profiles = jsonInfo['profiles']
        ontology.items = jsonInfo['items']
        ontology.term_paths = jsonInfo['term_paths']

        if build: ontology.precompute() 
