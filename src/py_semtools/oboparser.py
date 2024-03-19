import sys
import os
import re
import warnings
from py_semtools import FileParser
class OboParser(FileParser):

    #############################################
    # FIELDS
    #############################################
    # => @header :: file header (if is available)
    # => @stanzas :: OBO stanzas {:terms,:typedefs,:instances}
    # => @ancestors_index :: hash of ancestors per each term handled with any structure relationships
    # => @descendants_index :: hash of descendants per each term handled with any structure relationships
    # => @alternatives_index :: has of alternative IDs (include alt_id and obsoletes)
    # => @special_tags :: set of special tags to be expanded (:is_a, :obsolete, :alt_id)
    # => @structureType :: type of ontology structure depending on ancestors relationship. Allowed: {atomic, sparse, circular, hierarchical}
    # => @dicts :: bidirectional dictionaries with three levels <key|value>: 1º) <tag|hash2>; 2º) <(:byTerm/:byValue)|hash3>; 3º) dictionary <k|v>
    # => @removable_terms :: array of terms to not be considered

    header = None
    stanzas = {'terms': {}, 'typedefs': {}, 'instances': {}}
    removable_terms = []
    alternatives_index = {}
    obsoletes = {}
    structureType = None
    ancestors_index = {}
    descendants_index = {}
    reroot = False
    dicts = {}

    @classmethod
    def reset(cls):
        cls.header = None
        cls.stanzas = {'terms': {}, 'typedefs': {}, 'instances': {}}
        cls.removable_terms = []
        cls.alternatives_index = {}
        cls.obsoletes = {}
        cls.structureType = None
        cls.ancestors_index = {}
        cls.descendants_index = {}
        cls.reroot = False
        cls.dicts = {}

    @classmethod
    def each(cls, att = False, only_main = True):
        if len(cls.stanzas['terms']) == 0: warnings.warn('stanzas terms empty')
        for t_id, tags in cls.stanzas['terms'].items():
            if only_main and (t_id in cls.alternatives_index or t_id in cls.obsoletes): continue
            if att:
                yield(t_id, tags)
            else:
                yield(t_id)

    @classmethod
    def load(cls, ontology, file, build = True, black_list = [], extra_dicts = []):
        cls.reset() # Clean class variables to avoid the mix of several obo loads
        cls.removable_terms = black_list
        _, header, stanzas = cls.load_obo(file)
        cls.header = header
        cls.stanzas = stanzas
        if len(cls.removable_terms) > 0 : cls.remove_black_list_terms() 
        if build: cls.build_index(ontology, extra_dicts = extra_dicts) 

    # Class method to load an OBO format file (based on OBO 1.4 format). Specially focused on load
    # the Header, the Terms, the Typedefs and the Instances.
    # ===== Parameters
    # +file+:: OBO file to be loaded
    # ===== Returns 
    # Hash with FILE, HEADER and STANZAS info
    @classmethod
    def load_obo(cls, file):
        if file == None: raise Exception("File is not defined") 
        # Data variables
        header = ''
        stanzas = {'terms': {}, 'typedefs': {}, 'instances': {}}
        # Auxiliar variables
        infoType = 'Header'
        currInfo = []
        stanzas_flags = ['[Term]', '[Typedef]', '[Instance]']
        # Read file
        with open(file, 'r') as f:
            for line in f:
                line = line.rstrip()
                if len(line) == 0: continue
                fields = line.split(':', 1)
                # Check if new instance is found
                if line in stanzas_flags:
                    header = cls.process_entity(header, infoType, stanzas, currInfo)
                    # Update info variables
                    currInfo = []
                    infoType = re.sub("\[|\]", '', line)
                    continue
                # Concat info
                currInfo.append(fields)
        # Store last loaded info
        if len(currInfo) > 0: header = cls.process_entity(header, infoType, stanzas, currInfo)
        # Prepare to return
        finfo = {'file' : file, 'name' : os.path.basename(os.path.splitext(file)[0])}
        return finfo, header, stanzas

    # Handle OBO loaded info and stores it into correct container and format
    # ===== Parameters
    # +header+:: container
    # +infoType+:: current ontology item type detected
    # +stanzas+:: container
    # +currInfo+:: info to be stored
    # ===== Returns 
    # header newly/already stored
    @classmethod
    def process_entity(cls, header, infoType, stanzas, currInfo):
        info = cls.info2hash(currInfo)
        # Store current info
        if infoType == 'Header':
            header = info
        else:
            entity_id = info['id']
            if infoType == 'Term':
                stanzas['terms'][entity_id] = info
            elif infoType == 'Typedef':
                stanzas['typedefs'][entity_id] = info
            elif infoType == 'Instance':
                stanzas['instances'][entity_id] = info
        return header

    # Class method to transform string with <tag : info> into hash structure
    # ===== Parameters
    # +attributes+:: array tuples with info to be transformed into hash format
    # ===== Returns 
    # Attributes stored into hash structure
    @classmethod
    def info2hash(cls, attributes, split_char = " ! ", selected_field = 0):
        # Load info
        info_hash = {}
        # Only TERMS multivalue tags (future add Typedefs and Instance)
        # multivalue_tags = [:alt_id, :is_a, :subset, :synonym, :xref, :intersection_of, :union_of, :disjoint_from, :relationship, :replaced_by, :consider]
        for attr in attributes:
            if len(attr) < 2: continue
            tag, value = attr
            tag = tag.lstrip()
            value = value.lstrip()
            if tag == 'is_a': value = re.sub('{[\\\":A-Za-z0-9\/\.\-, =?&_]+} ', '', value)  # To delete extra attributes (source, xref) in is_a tag of MONDO ontology
            attr[1] = value #update data after string cleaning
            
            if tag in cls.tags_with_trailing_modifiers: value = value.split(split_char)[selected_field] 
            
            query = info_hash.get(tag)
            if query != None: # Tag already exists
                if type(query) is not list: # Check that tag is multivalue
                    if tag == 'def' or tag == 'comment' or tag == 'name' or tag == 'created_by': # Some ontologies (EFO) could have more tha one def/comment/name line (an this is AGAINST obo standard) so we keep the first and ignore the others.
                        continue
                    else:
                        raise Exception(f'Attempt to concatenate plain text with another. The tag is not declared as multivalue. [{tag}]({repr(query)})')
                else:
                    query.append(value) # Add new value to tag
            else: # New entry
                if tag in cls.multivalue_tags:
                    info_hash[tag] = [value]
                else:
                    info_hash[tag] = value
        return info_hash

    @classmethod
    def remove_black_list_terms(cls):
        for removableID in cls.removable_terms: del cls.stanzas['terms'][removableID]

    # Executes basic expansions of tags (alternatives, obsoletes and parentals) with default values
    # ===== Returns 
    # true if eprocess ends without errors and false in other cases
    @classmethod
    def build_index(cls, ontology, extra_dicts: []):
        cls.get_index_obsoletes(obs_tag = cls.basic_tags['obsolete'], alt_tags = cls.basic_tags['alternative'])
        cls.get_index_alternatives(alt_tag = cls.basic_tags['alternative'][-1])
        cls.remove_obsoletes_in_terms()
        cls.get_index_child_parent_relations(tag = cls.basic_tags['ancestors'][0])
        cls.calc_dictionary('name')
        cls.calc_dictionary('synonym', select_regex = '\"(.*)\"')
        cls.calc_ancestors_dictionary()
        for dict_tag, extra_parameters in extra_dicts:
            cls.calc_dictionary(dict_tag, **extra_parameters) # https://www.justinweiss.com/articles/fun-with-keyword-arguments/
        # Fill ontology object
        ontology.terms = cls.stanzas['terms']
        ontology.alternatives_index = cls.alternatives_index
        ontology.obsoletes = cls.obsoletes
        ontology.ancestors_index = cls.ancestors_index
        ontology.descendants_index = cls.descendants_index
        ontology.reroot = cls.reroot
        ontology.structureType = cls.structureType
        ontology.dicts = cls.dicts

    @classmethod
    def remove_obsoletes_in_terms(cls): # once alternative and obsolete indexes are loaded, use this to keep only working terms
        terms = cls.stanzas['terms']
        for term, val in cls.obsoletes.items(): terms.pop(term)

    # Expand obsoletes set and link info to their alternative IDs
    # ===== Parameters
    # +obs_tags+:: tags to be used to find obsoletes
    # +alt_tags+:: tags to find alternative IDs (if are available)
    # ===== Returns 
    # true if process ends without errors and false in other cases
    @classmethod
    def get_index_obsoletes(cls, obs_tag = None, alt_tags = None):
        for term_id, term_tags in cls.each(att = True):
            obs_value = term_tags.get(obs_tag)
            if obs_value == 'true': # Obsolete tag presence, must be checked as string
                alt_ids = []
                for alt in alt_tags: # Check if alternative value is available
                    t = term_tags.get(alt)
                    if t != None: alt_ids.append(t)
                if len(alt_ids) > 0:
                    alt_id = alt_ids[0][0] #FIRST tag, FIRST id 
                    cls.alternatives_index[term_id] = alt_id
                cls.obsoletes[term_id] = True

    # Expand alternative IDs arround all already stored terms
    # ===== Parameters
    # +alt_tag+:: tag used to expand alternative IDs
    # ===== Returns 
    # true if process ends without errors and false in other cases
    @classmethod
    def get_index_alternatives(cls, alt_tag = None):
        removable_terms = set(cls.removable_terms)
        for term_id, tags in cls.each(att = True):
            term_id = cls.extract_id(term_id)
            if term_id == None: continue
            alt_ids = tags.get(alt_tag)
            if alt_ids != None:
                alt_ids = set(alt_ids)
                alt_ids.discard(term_id) # We use discard instead of remove to NOT raise an error if ter_id not exists
                alt_ids = alt_ids - removable_terms
                for alt_term in alt_ids:
                    cls.alternatives_index[alt_term] = term_id

    # Expand parentals set. Also launch frequencies process
    # ===== Parameters
    # +tag+:: tag used to expand parentals
    # ===== Returns 
    # true if process ends without errors and false in other cases
    @classmethod
    def get_index_child_parent_relations(cls, tag = None):
        structType, parentals = cls.get_related_ids_by_tag(terms = cls.stanzas['terms'],
                                                        target_tag = tag,
                                                        reroot = cls.reroot)
        if structType == None or parentals == None:
            raise Exception('Error expanding parentals')
        elif structType not in ['atomic', 'sparse']: # Check structure
            structType = 'circular' if structType == 'circular' else 'hierarchical'
        cls.structureType = structType 

        removable_terms = set(cls.removable_terms)
        for term_id, parents in parentals.items():
            parents = set(parents) - removable_terms
            clean_parentals = []
            for par_id in parents:
                par_id = cls.extract_id(par_id)
                if par_id != None: clean_parentals.append(par_id)
            cls.ancestors_index[term_id] = clean_parentals
            for anc_id in parents: cls.add2hash(cls.descendants_index, anc_id, term_id)


    # Expand terms using a specific tag and return all extended terms into an array and
    # the relationship structuture observed (hierarchical or circular). If circular structure is
    # foumd, extended array will be an unique vector without starting term (no loops) 
    # ===== Parameters
    # +terms+:: set to be used to expand
    # +target_tag+:: tag used to expand
    # ===== Returns 
    # A vector with the observed structure (string) and the hash with extended terms
    @classmethod
    def get_related_ids_by_tag(cls, terms = [], target_tag = None, reroot = False):
        structType = 'hierarchical'
        related_ids = {}
        for t_id, tags in terms.items():
            if tags.get(target_tag) != None:
                set_structure, _ = cls.get_related_ids(t_id, terms, target_tag, related_ids)
                if set_structure == 'circular': structType = 'circular'  # Check structure

        # Check special case
        if len(related_ids) <= 0: structType = 'atomic' 
        if reroot or (len(related_ids) > 0 and ((len(terms) - len(related_ids)) >= 2) ): structType = 'sparse'
        return structType, related_ids

    # Expand a (starting) term using a specific tag and return all extended terms into an array and
    # the relationship structuture observed (hierarchical or circular). If circular structure is
    # foumd, extended array will be an unique vector without starting term (no loops).
    # +Note+: we extremly recomend use get_related_ids_by_tag function instead of it (directly)
    # ===== Parameters
    # +start+:: term where start to expand
    # +terms+:: set to be used to expand
    # +target_tag+:: tag used to expand
    # +eexpansion+:: already expanded info
    # ===== Returns 
    # A vector with the observed structure (string) and the array with extended terms.
    @classmethod
    def get_related_ids(cls, start_id, terms, target_tag, related_ids = {}):
        # Take start_id term available info and already accumulated info
        current_associations = related_ids.get(start_id)
        if current_associations == None: current_associations = []
        query_start_id = terms.get(start_id)
        if query_start_id == None: return ['no_term',[]] 
        id_relations = query_start_id.get(target_tag)
        if id_relations == None: return ['source',[]] 

        struct = 'hierarchical'

        # Study direct extensions
        for t_id in id_relations: 
            # Handle
            if t_id in current_associations: # Check if already have been included into this expansion
                struct = 'circular' 
            else:
                current_associations.append(t_id)
                if t_id in related_ids: # Check if current already has been expanded
                    current_associations = cls.union_list(current_associations, related_ids[t_id])
                    if start_id in current_associations: # Check circular case
                        struct = 'circular'
                        current_associations = current_associations - [t_id, start_id]
                else: # Expand
                    related_ids[start_id] = current_associations
                    structExp, current_related_ids = cls.get_related_ids(t_id, terms, target_tag, related_ids) # Expand current
                    current_associations = cls.union_list(current_associations, current_related_ids)
                    if structExp == 'circular': struct = 'circular'  # Check struct
                    if start_id in current_associations: # Check circular case
                        struct = 'circular'
                        current_associations.remove(start_id)
        related_ids[start_id] = current_associations

        return struct, current_associations

    # Calculates :is_a dictionary
    @classmethod
    def calc_ancestors_dictionary(cls):
        cls.calc_dictionary('is_a', self_type_references = True, multiterm = True)

    # Generate a bidirectinal dictionary set using a specific tag and terms stanzas set
    # This functions stores calculated dictionary into @dicts field.
    # This functions stores first value for multivalue tags
    # This function does not handle synonyms for byValue dictionaries
    # ===== Parameters
    # +tag+:: to be used to calculate dictionary
    # +select_regex+:: gives a regfex that can be used to modify value to be stored
    # +store_tag+:: flag used to store dictionary. If nil, mandatory tag given will be used
    # +multiterm+:: if true, byValue will allows multi-term linkage (array)
    # +self_type_references+:: if true, program assumes that refrences will be between Ontology terms, and it term IDs will be checked
    # ===== Return
    # hash with dict data. And stores calcualted bidirectional dictonary into dictionaries main container
    @classmethod
    def calc_dictionary(cls, tag, select_regex = None, store_tag = None, multiterm = None, self_type_references = False):
        if store_tag == None: store_tag = tag 

        byTerm = {}
        byValue = {}
        
        for term, tags in cls.each(att = True, only_main = False): # Calc per term
            referenceTerm = term
            queryTag = tags.get(tag)
            if queryTag != None:
                # Pre-process
                if select_regex != None:
                    queryTag = cls.regex2tagData(select_regex, queryTag)
                if type(queryTag) is list: # Store
                    if len(queryTag) > 0:
                        if referenceTerm in byTerm:
                            byTerm[referenceTerm] = list(set(byTerm[referenceTerm] + queryTag))
                        else:
                            byTerm[referenceTerm] = queryTag
                        for value in queryTag: 
                            if multiterm:
                                cls.add2hash(byValue, value, referenceTerm)
                            else:
                                byValue[value] = referenceTerm
                else:
                    if referenceTerm in byTerm:
                        byTerm[referenceTerm] = list(set(byTerm[referenceTerm] + [queryTag]))
                    else:
                        byTerm[referenceTerm] = [queryTag]
                    if multiterm:
                        cls.add2hash(byValue, queryTag, referenceTerm)
                    else:
                        byValue[queryTag] = referenceTerm
                
        if self_type_references: # Check self-references
            for term, references in byTerm.items():
                corrected_references = []
                for t in references:
                    checked = cls.extract_id(t)
                    if checked == None:
                        ref = t
                    else:
                        if checked != t and byValue.get(checked) == None: byValue[checked] = byValue.pop(t)  # Update in byValue
                        ref = checked
                    corrected_references.append(ref)
                byTerm[term] = list(set(corrected_references))

        for term, values in byTerm.items(): # Check order
            if cls.exists(term):
                referenceValue = cls.stanzas['terms'][term].get(tag)
                if referenceValue != None:
                    if select_regex != None: referenceValue = cls.regex2tagData(select_regex, referenceValue)
                    if self_type_references:
                        if type(referenceValue) is list:
                            aux = []
                            for t in referenceValue:
                                t = cls.extract_id(t)
                                if t != None: aux.append(t)
                        else:
                            aux = cls.extract_id(referenceValue)
                        if aux != None: referenceValue = aux
                    if type(referenceValue) is not list: referenceValue = [referenceValue] 
                    byTerm[term] = referenceValue + list(set(values) - set(referenceValue))

        final_dict = {'byTerm': byTerm, 'byValue': byValue}
        cls.dicts[store_tag] = final_dict
        return final_dict

    @classmethod
    def regex2tagData(cls, select_regex, tag_data):
        string_matches = []
        if type(tag_data) is list:
            for value in tag_data:
                match = re.findall(select_regex, value)
                if len(match) > 0: string_matches.append(match[0])
        else:
            string_matches = re.findall(select_regex, tag_data)[0]
        return string_matches

    # Check if a given ID is stored as term into this object
    # ===== Parameters
    # +id+:: to be checked 
    # ===== Return
    # True if term is allowed or false in other cases
    @classmethod
    def exists(cls, term_id):
        return term_id in cls.stanzas['terms']

    # Check if a term given is marked as obsolete
    @classmethod
    def is_obsolete(cls, term):
        return term in cls.obsoletes

    # Check if a term given is marked as alternative
    @classmethod
    def is_alternative(cls, term):
        return term in cls.alternatives_index

    # This method assumes that a text given contains an allowed ID. And will try to obtain it splitting it
    # ===== Parameters
    # +text+:: to be checked 
    # ===== Return
    # The correct ID if it can be found or nil in other cases
    @classmethod
    def extract_id(cls, text, splitBy = ' '):
        if cls.exists(text):
            return text
        else:
            splittedText = text.split(splitBy)[0]
            return splittedText if cls.exists(splittedText) else None

    @classmethod
    def add2hash(cls, dictionary, key, val):
        query = dictionary.get(key)
        if query == None: 
            dictionary[key] = [val]
        else:
            query.append(val)

    @classmethod
    def union_list(cls, arr1, arr2):
        return arr1 + [item for item in arr2 if item in arr2 and item not in arr1]