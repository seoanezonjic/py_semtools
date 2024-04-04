import math
import os
import sys
import time
import copy
import warnings
import json
import numpy as np
from collections import defaultdict
import networkx as nx
import entrezpy.esearch.esearcher
from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from functools import partial
from collections import defaultdict, deque
import time
import itertools
import re
import py_exp_calc.exp_calc as pxc

from py_semtools import OboParser
from py_semtools import JsonParser

from py_exp_calc.exp_calc import intersection, union, diff

class Ontology:
    allowed_calcs = {'ics': ['resnik', 'resnik_observed', 'seco', 'zhou', 'sanchez'], 'sims': ['resnik', 'lin', 'jiang_conrath']}
    
    def __init__(self, file= None, load_file= False, removable_terms= [], build= True, file_format= None, extra_dicts= []):
        self.threads = 2
        self.terms = {}
        self.ancestors_index = {}
        self.descendants_index = {}
        self.alternatives_index = {}
        self.obsoletes = {} # id is obsolete but it could or not have an alt id
        self.structureType = None
        self.ics = {} 
        for ic_type in self.allowed_calcs['ics']: self.ics[ic_type] = {}
        self.meta = {}
        self.max_freqs = {'struct_freq' : -1.0, 'observed_freq' : -1.0, 'max_depth': -1.0}
        self.dicts = {}
        self.profiles = {}
        self.items = {}
        self.term_paths = {}
        self.reroot = False
        if file != None: load_file = True
        if load_file:
            fformat = file_format
            if fformat == None and file != None: fformat = os.path.splitext(file)[1]
            if fformat == 'obo' or fformat == ".obo":
                OboParser.load(self, file, build = True, black_list = removable_terms, extra_dicts = extra_dicts)
            elif fformat == 'json' or fformat == ".json":
                JsonParser.load(self, file, build = False)
            elif fformat != None:
                warnings.warn('Format not allowed. Loading process will not be performed')
            if build: self.precompute()

    #############################################
    # GENERATE METADATA FOR ALL TERMS
    #############################################

    def precompute(self):
        self.get_dag()
        self.get_index_frequencies()
        self.calc_term_levels(calc_paths = True)

    def get_dag(self):
        relations = []
        for parent, descendants in self.dicts['is_a']['byValue'].items():
            for desc in descendants: relations.append((parent, desc))
        dag = nx.DiGraph(relations)
        self.dag = dag

    # Calculates regular frequencies based on ontology structure (using parentals)
    # ===== Returns 
    # true if everything end without errors and false in other cases
    def get_index_frequencies(self): # Per each term, add frequencies
        if len(self.ancestors_index) == 0:
            warnings.warn('ancestors_index object is empty')
        else:
            for t_id, tags in self.each(att = True):
                query = self.meta.get(t_id)
                if query == None:
                    query = {'ancestors': 0.0, 'descendants': 0.0, 'struct_freq': 0.0, 'observed_freq': 0.0}
                    self.meta[t_id] = query 
                query['ancestors'] = float(len(self.ancestors_index[t_id])) if t_id in self.ancestors_index else 0.0
                query['descendants'] = float(len(self.descendants_index[t_id])) if t_id in self.descendants_index else 0.0
                query['struct_freq'] = query['descendants'] + 1.0
                # Update maximums
                if self.max_freqs['struct_freq'] < query['struct_freq']: self.max_freqs['struct_freq'] = query['struct_freq'] 
                if self.max_freqs['max_depth'] < query['descendants']: self.max_freqs['max_depth'] = query['descendants'] 

    # Calculates ontology structural levels for all ontology terms
    # ===== Parameters
    # +calc_paths+:: calculates term paths if it's not already calculated
    # +shortest_path+:: if true, level is calculated with shortest path, largest path will be used in other cases
    def calc_term_levels(self, calc_paths = False, shortest_path = True):
        if len(self.term_paths) == 0 and calc_paths: self.calc_term_paths()
        if len(self.term_paths) > 0:
            byTerm = {}
            byValue = {}
            for term, info in self.term_paths.items():
                level =  info['shortest_path'] if shortest_path else info['largest_path']
                level =  -1 if level == None else round(level)
                byTerm[term] = level
                self.add2hash(byValue, level, term)
            self.dicts['level'] = {'byTerm': byValue, 'byValue': byTerm} # Note: in this case, value has multiplicity and term is unique value
            self.max_freqs['max_depth'] = max(list(byValue.keys())) # Update maximum depth

    # Find paths of a term following it ancestors and stores all possible paths for it and it's parentals.
    # Also calculates paths metadata and stores into @term_paths
    def calc_term_paths(self): 
        self.term_paths = {}
        if self.structureType in ['hierarchical', 'sparse', 'circular']: # PSZ: Added circular to incorrect ontology structure detection
        #if self.structureType in ['hierarchical', 'sparse']:
            for term in self.each():
                self.expand_path(term)
                path_attr = self.term_paths[term]
                # expand_path is arecursive function so these pat attributes must be calculated once the recursion is finished
                path_attr['total_paths'] = len(path_attr['paths'])
                paths_sizes = [ len(path) for path in path_attr['paths'] ]
                path_attr['largest_path'] = max(paths_sizes)
                path_attr['shortest_path'] = min(paths_sizes)
        else:
            warnings.warn('Ontology structure must be hierarchical or sparse to calculate term levels. Aborting paths calculation')

    # Recursive function whic finds paths of a term following it ancestors and stores all possible paths for it and it's parentals
    # ===== Parameters
    # +curr_term+:: current visited term
    # +visited_terms+:: already expanded terms
    def expand_path(self, curr_term):
        if curr_term not in self.term_paths:
            path_attr = {'total_paths': 0, 'largest_path': 0, 'shortest_path': 0, 'paths': []}
            self.term_paths[curr_term] = path_attr
            direct_parentals = self.dicts['is_a']['byTerm'].get(curr_term)
            if direct_parentals == None: # No parents :: End of recurrence
                path_attr['paths'].append([curr_term])
            else: # Expand and concat
                for ancestor in direct_parentals:
                    path_attr_parental = self.term_paths.get(ancestor)
                    if path_attr_parental == None: # Calculate new paths
                        self.expand_path(ancestor) 
                        new_paths = self.term_paths[ancestor]['paths']
                    else: # Use direct_parental paths already calculated 
                        new_paths = path_attr_parental['paths'] 
                    added_path = []
                    for path in new_paths:
                        p = copy.copy(path)
                        p.insert(0, curr_term)
                        added_path.append(p)
                    path_attr['paths'].extend(added_path)

    #############################################
    # TERM METHODS
    #############################################

    # I/O observed term from data
    ####################################

    # Increase observed frequency for a specific term
    # ===== Parameters
    # +term+:: term which frequency is going to be increased
    # +increas+:: frequency rate to be increased. Default = 1
    # ===== Return
    # true if process ends without errors, false in other cases
    def add_observed_term(self, term = None, increase = 1.0):
        if not self.term_exist(term): return False # Check if exists
        if self.meta.get(term) == None: self.meta[term] = {'ancestors': -1.0, 'descendants': -1.0, 'struct_freq': 0.0, 'observed_freq': 0.0}
        # Add frequency
        if self.meta[term]['observed_freq'] == -1: self.meta[term]['observed_freq'] = 0 
        self.meta[term]['observed_freq'] += increase
        # Add observation to parents, we assume that observing the child implies the existence of the parent
        for anc in self.get_ancestors(term): 
            self.meta[anc]['observed_freq'] += increase
            if self.max_freqs['observed_freq'] < self.meta[anc]['observed_freq']: self.max_freqs['observed_freq'] = self.meta[anc]['observed_freq']
        # Check maximum frequency
        if self.max_freqs['observed_freq'] < self.meta[term]['observed_freq']: self.max_freqs['observed_freq'] = self.meta[term]['observed_freq']

    # Obtain level and term relations
    ####################################

    # ===== Parameters
    # +term+:: which are requested
    # +relation+:: can be :ancestor or :descendant 
    # ===== Returns
    # Direct ancestors/descendants of given term or nil if any error occurs
    def get_direct_related(self, term, relation):
        target = None
        if relation == 'ancestor':
            target = 'byTerm'
        elif relation ==  'descendant':
            target = 'byValue'
        else:
            warnings.warn('Relation type not allowed. Returning None')
        query = self.dicts['is_a'][target].get(term)
        return copy.deepcopy(query)

    # Return direct ancestors/descendants of a given term
    # Return direct ancestors of a given term
    # ===== Parameters
    # +term+:: which ancestors are requested
    # ===== Returns
    # Direct ancestors of given term or nil if any error occurs
    def get_direct_ancentors(self, term):
        return self.get_direct_related(term, 'ancestor')

    # Return direct descendants of a given term
    # ===== Parameters
    # +term+:: which descendants are requested
    # ===== Returns
    # Direct descendants of given term or nil if any error occurs
    def get_direct_descendants(self, term):
        return self.get_direct_related(term, 'descendant')        

    # Find ancestors/descendants of a given term
    # ===== Parameters
    # +term+:: to be checked
    # +return_ancestors+:: return ancestors if true or descendants if false
    # ===== Returns 
    # an array with all ancestors/descendants of given term or nil if parents are not available yet
    def get_familiar(self, term, return_ancestors = True, sorted_by_level = False):
        if sorted_by_level:
            familiars = []
            stack = deque(copy.deepcopy(self.get_direct_ancentors(term) if return_ancestors else self.get_direct_descendants(term)))
            while len(stack) > 0:
                next_term = stack.popleft()
                familiars.append(next_term)
                next_term_childs = self.get_direct_ancentors(next_term) if return_ancestors else self.get_direct_descendants(next_term) 
                if next_term_childs != None: 
                    for term in next_term_childs: stack.append(term)  
        else:
            familiars = self.ancestors_index.get(term) if return_ancestors else self.descendants_index.get(term)
            if familiars != None:
                familiars = copy.copy(familiars)
            else:
                familiars = []
        return familiars

    # Find ancestors of a given term
    # ===== Parameters
    # +term+:: to be checked
    # ===== Returns 
    # an array with all ancestors of given term or false if parents are not available yet
    def get_ancestors(self, term, sorted_by_level = False):
        return self.get_familiar(term, True, sorted_by_level)

    # Find descendants of a given term
    # ===== Parameters
    # +term+:: to be checked
    # ===== Returns 
    # an array with all descendants of given term or false if parents are not available yet
    def get_descendants(self, term, sorted_by_level = False):
        return self.get_familiar(term, False, sorted_by_level)        

    # Gets ontology level of a specific term
    # ===== Returns 
    # Term level
    def get_term_level(self, term):
        return self.dicts['level']['byValue'].get(term)

    # nil, term not found, [] term exists but not has parents
    def get_parental_path(self, term, which_path = 'shortest_path', level = 0):
        path = None
        path_attr = self.term_paths.get(term)
        if path_attr != None:
            path_length = path_attr[which_path]
            all_paths = path_attr['paths']
            if len(all_paths) == 0:
                path = []
            else:
                for pt in all_paths:
                    if len(pt) == path_length: 
                        path = copy.copy(pt)
                        break
                if level > 0: # we want the term and his ascendants until a specific level
                    n_parents = path_length - level 
                    path = path[0:n_parents]
                path.pop(0) # Discard the term itself
        return path

    # ID Handlers
    ####################################

    # ===== Returns 
    # the main ID assigned to a given ID. If it's a non alternative/obsolete ID itself will be returned
    # ===== Parameters
    # +id+:: to be translated
    # ===== Return
    # main ID related to a given ID. Returns nil if given ID is not an allowed ID
    def get_main_id(self, t_id):
        mainID = self.alternatives_index.get(t_id)
        if not self.term_exist(t_id) and mainID == None: return None
        if mainID != None: # Recursive code to get the definitive final term id if there are several alt_id in chain
            new_id = self.get_main_id(mainID)
            if new_id != mainID: new_id = self.get_main_id(new_id)
            t_id = new_id    
        return t_id

    # Translate a given value using an already calcualted dictionary
    # ===== Parameters
    # +toTranslate+:: value to be translated using dictiontionary
    # +tag+:: used to generate the dictionary
    # +byValue+:: boolean flag to indicate if dictionary must be used values as keys or terms as keys. Default: values as keys = true
    # ===== Return
    # translation
    def translate(self, toTranslate, tag, byValue = True):
        term_dict = self.dicts[tag]['byValue'] if byValue else self.dicts[tag]['byTerm']    
        if not byValue: toTranslate =  self.get_main_id(toTranslate) 
        return term_dict.get(toTranslate)

    # Translate a name given
    # ===== Parameters
    # +name+:: to be translated
    # ===== Return
    # translated name or nil if it's not stored into this ontology
    def translate_name(self, name):
        term = self.translate(name, 'name')
        if term == None: term = self.translate(name, 'synonym') 
        return term            

    # Translates a given ID to it assigned name
    # ===== Parameters
    # +id+:: to be translated
    # ===== Return
    # main name or nil if it's not included into this ontology
    def translate_id(self, t_id):
        name = self.translate(t_id, 'name', byValue = False)
        return None if name == None else name[0]

    def get_synonims(self, t_id):
        syns = self.dicts['synonym']['byTerm'].get(t_id)
        if syns == None: syns = []
        return syns


    def filter_list(self, term_list, whitelist=[], blacklist=[]):
        filt_list = []
        for term_id in term_list:
            keep = True
            parentals = self.get_ancestors(term_id)
            if len(whitelist) > 0:
                wkeep = False
                for wparental in whitelist:
                    if wparental in parentals: wkeep = True; break
                if not wkeep: keep = False # If there is white list but any parental is in the white list, remove the current term
            for bparental in blacklist:
                if bparental in parentals: keep = False; break
            if keep: filt_list.append(term_id)
        return filt_list


    # Get term frequency and information
    ####################################

    # One single term #

    # Get a term frequency
    # ===== Parameters
    # +term+:: term to be checked
    # +type+:: type of frequency to be returned. Allowed: [:struct_freq, :observed_freq]
    # ===== Returns 
    # frequency of term given or nil if term is not allowed
    def get_frequency(self, term, freq_type: 'struct_freq'):
        queryFreq = self.meta.get(term)
        return None if queryFreq == None else queryFreq[freq_type]        

    # Geys structural frequency of a term given
    # ===== Parameters
    # +term+:: to be checked
    # ===== Returns 
    # structural frequency of given term or nil if term is not allowed
    def get_structural_frequency(self,term):
        return self.get_frequency(term, freq_type = 'struct_freq')

    # Gets observed frequency of a term given
    # ===== Parameters
    # +term+:: to be checked
    # ===== Returns 
    # observed frequency of given term or nil if term is not allowed
    def get_observed_frequency(self, term):
        return self.get_frequency(term, freq_type = 'observed_freq')

    # Obtain IC of an specific term
    # ===== Parameters
    # +term+:: which IC will be calculated
    # +type+:: of IC to be calculated. Default: resnik
    # +force+:: force re-calculate the IC. Do not check if it is already calculated
    # +zhou_k+:: special coeficient for Zhou IC method
    # ===== Returns 
    # the IC calculated
    def get_IC(self, term, ic_type = 'resnik', force = False, zhou_k = 0.5):
        curr_ics = self.ics[ic_type]
        if ic_type not in self.allowed_calcs['ics']: raise Exception(f"IC type specified ({ic_type}) is not allowed") 
        if term in curr_ics and not force: return curr_ics[term]  # Check if it's already calculated
        # Calculate
        ic = - 1
        term_meta = self.meta[term]
        # https://arxiv.org/ftp/arxiv/papers/1310/1310.8059.pdf  |||  https://sci-hub.st/https://doi.org/10.1016/j.eswa.2012.01.082
        ###########################################
        #### STRUCTURE BASED METRICS
        ###########################################
        # Shortest path
        # Weighted Link
        # Hirst and St-Onge Measure
        # Wu and Palmer
        # Slimani
        # Li
        # Leacock and Chodorow
        ###########################################
        #### INFORMATION CONTENT METRICS
        ###########################################
        if ic_type == 'resnik': # Resnik P: Using Information Content to Evaluate Semantic Similarity in a Taxonomy
            # -log(Freq(x) / Max_Freq)
            ic = -math.log10(term_meta['struct_freq'] / self.max_freqs['struct_freq'])
        elif ic_type == 'resnik_observed':
            # -log(Freq(x) / Max_Freq)
            ic = -math.log10(term_meta['observed_freq'] / self.max_freqs['observed_freq'])
        # Lin
        # Jiang & Conrath

        ###########################################
        #### FEATURE-BASED METRICS
        ###########################################
        # Tversky
        # x-similarity
        # Rodirguez

        ###########################################
        #### HYBRID METRICS
        ###########################################
        elif ic_type == 'seco' or ic_type == 'zhou': # SECO:: An intrinsic information content metric for semantic similarity in WordNet
            #  1 - ( log(hypo(x) + 1) / log(max_nodes) )
            ic = 1 - math.log10(term_meta['struct_freq']) / math.log10(len(self.terms))
            if 'zhou': # New Model of Semantic Similarity Measuring in Wordnet                
                # k*(IC_Seco(x)) + (1-k)*(log(depth(x))/log(max_depth))
                self.ics['seco'][term] = ic # Special store
                ic = zhou_k * ic + (1.0 - zhou_k) * (math.log10(term_meta['descendants']) / math.log10(self.max_freqs['max_depth']))
        elif ic_type == 'sanchez': # Semantic similarity estimation in the biomedical domain: An ontology-basedinformation-theoretic perspective
            ic = -math.log10((term_meta['descendants'] / term_meta['ancestors'] + 1.0) / (self.max_freqs['max_depth'] + 1.0))
        # Knappe
        curr_ics[term] = ic
        return ic

    # Term vs Term #

    def get_LCA(self, termA, termB, lca_index = False):
        lca = []
        if lca_index:
            res = self.lca_index.get((termA, termB))
            if res != None: lca = res 
        else:  # Obtain ancestors (include itselfs too)
            lca = self.compute_LCA(termA, termB)
        return lca

    def compute_LCA(self, termA, termB):
        lca = []
        anc_A = self.get_ancestors(termA) 
        anc_B = self.get_ancestors(termB)
        if not (len(anc_A) == 0 and len(anc_B) == 0):
            anc_A.append(termA)
            anc_B.append(termB)
            lca = intersection(anc_A,  anc_B)
        return lca

    def compute_LCA_pair(self, pair):
        res = self.compute_LCA(pair[0], pair[1])
        return res

    # Find the Most Index Content shared Ancestor (MICA) of two given terms
    # ===== Parameters
    # +termA+:: term to be cheked
    # +termB+:: term to be checked
    # +ic_type+:: IC formula to be used
    # ===== Returns 
    # the MICA(termA,termB) and it's IC
    def get_MICA(self, termA, termB, ic_type = 'resnik', mica_index = False):
        if mica_index:
            mica = self.mica_index[termA][termB]
        elif termA == termB: # Special case
            ic = self.get_IC(termA, ic_type = ic_type)
            mica = [termA, ic]
        else:
            mica = self.compute_MICA(self.get_LCA(termA, termB, lca_index = False), ic_type)
        return mica

    def compute_MICA(self, lcas, ic_type):
        mica = [None,-1.0]
        for lca in lcas: # Find MICA in shared ancestors
            ic = self.get_IC(lca, ic_type = ic_type)
            if ic > mica[1]: mica = [lca, ic]
        return mica

    def get_MICA_from_pair(self, pair, ic_type = 'resnik'):
        mica = self.compute_MICA(self.compute_LCA_pair(pair), ic_type = ic_type)
        return mica

    # Find the IC of the Most Index Content shared Ancestor (MICA) of two given terms
    # ===== Parameters
    # +termA+:: term to be cheked
    # +termB+:: term to be checked
    # +ic_type+:: IC formula to be used
    # ===== Returns 
    # the IC of the MICA(termA,termB)
    def get_ICMICA(self, termA, termB, ic_type = 'resnik'):
        term, ic = self.get_MICA(termA, termB, ic_type)
        return None if term == None else ic

    # Calculate similarity between two given terms
    # ===== Parameters
    # +termsA+:: to be compared
    # +termsB+:: to be compared
    # +type+:: similitude formula to be used
    # +ic_type+:: IC formula to be used
    # ===== Returns 
    # the similarity between both sets or false if frequencies are not available yet
    def get_similarity(self, termA, termB, sim_type = 'resnik', ic_type = 'resnik', mica_index = False):
        if sim_type not in self.allowed_calcs['sims']: raise Exception(f"SIM type specified ({sim_type}) is not allowed") 
        sim = None
        mica, sim_res = self.get_MICA(termA, termB, ic_type, mica_index)
        if mica != None:
            if sim_type == 'resnik':
                sim = sim_res
            elif sim_type == 'lin':
                if termA == termB:
                    sim = 1.0
                else:
                    sim = (2.0 * sim_res) / (self.get_IC(termA, ic_type=ic_type) + self.get_IC(termB, ic_type=ic_type))
            elif sim_type == 'jiang_conrath': # This is not a similarity, this is a disimilarity (distance)
                sim = (self.get_IC(termA, ic_type=ic_type) + self.get_IC(termB, ic_type=ic_type)) - (2.0 * sim_res)
        return sim

    # Checking valid terms
    ####################################

    def term_exist(self, t_id):
        return t_id in self.terms

    # Check if a term given is marked as obsolete
    def is_obsolete(self, term):
        return term in self.obsoletes

    #############################################
    # ITEMS METHODS
    #############################################

    # I/O Items
    ####################################

    # Store specific relations hash given into ITEMS structure
    # ===== Parameters
    # +relations+:: hash to be stored
    # +remove_old_relations+:: substitute ITEMS structure instead of merge new relations
    # +expand+:: if true, already stored keys will be updated with the unique union of both sets
    def load_item_relations_to_terms(self, relations, remove_old_relations = False, expand = False):
        if remove_old_relations: self.items = {} 
        for term, items in relations.items():
            if not self.term_exist(term):
                warnings.warn('Some terms specified are not stored into this ontology. These not correct terms will be stored too')
                break
        if expand:
            self.items = self.concatItems(self.items, relations)
        else:
            self.items.update(relations)

    # Defining Items from instance variables
    ########################################

    # Assign a dictionary already calculated as a items set.
    # ===== Parameters
    # +dictID+:: dictionary ID to be stored (:byTerm will be used)
    def set_items_from_dict(self, dictID, remove_old_relations = False):
        if remove_old_relations: self.items = {} 
        query = self.dicts[dictID]
        if query != None:
            self.items.update(query['byTerm'])
        else:
            warnings.warn('Specified ID is not calculated. Dict will not be added as a items set')

    # Get related profiles to a given term
    # ===== Parameters
    # +term+:: to be checked
    # ===== Returns 
    # profiles which contains given term
    def get_items_from_term(self, term):
        return self.items[term]

    # For each term in profiles add the ids in the items term-id dictionary 
    def get_items_from_profiles(self):
        for t_id, terms in self.profiles.items(): 
            for term in terms: self.add2hash(self.items, term, t_id)

    # Defining instance variables from items
    ########################################

    def get_profiles_from_items(self):
        new_profiles = {}
        for term, ids in self.items.items():
            for pr_id in ids: self.add2hash(new_profiles, pr_id, term)
        self.profiles = new_profiles        

 # Expanding items
    ####################################

    # This method computes childs similarity and impute items to it parentals. To do that Item keys must be this ontology allowed terms.
    # Similarity will be calculated by text extact similarity unless an ontology object will be provided. In this case, MICAs will be used
    # ===== Parameters
    # +ontology+:: (Optional) ontology object which items given belongs
    # +minimum_childs+:: minimum of childs needed to infer relations to parental. Default: 2
    # +clean_profiles+:: if true, clena_profiles ontology method will be used over inferred profiles. Only if an ontology object is provided
    # ===== Returns
    # void and update items object
    def expand_items_to_parentals(self, ontology = None, minimum_childs = 2, clean_profiles = True):
        targetKeys = self.expand_profile_with_parents(list(self.items.keys()))
        terms_per_level = self.list_terms_per_level(targetKeys)        
        terms_per_level =  sorted(list(map(list, terms_per_level.items())), key= lambda e: e[0]) # Obtain sorted levels 
        terms_per_level.pop() # Leaves are not expandable # FRED: Thats comment could be not true

        for level_terms in reversed(terms_per_level): # Expand from leaves to roots
            lvl, terms = level_terms
            for term in terms:
                childs = [ t for t in self.get_descendants(term) if t in self.items ] # Get child with items
                if len(childs) < minimum_childs: continue
                propagated_item_count = defaultdict(lambda: 0)                
                if ontology == None: # Count how many times is presented an item in childs
                    for child in childs: 
                        for i in self.items[child]: propagated_item_count[i] += 1
                else: # Count take into account similarity between terms in other ontology. Not pretty clear the full logic
                    while len(childs) > 1: 
                        curr_term = childs.pop(0)
                        for child in childs:
                            maxmica_counts = defaultdict(lambda: 0)
                            curr_items = self.items[curr_term]
                            child_items = self.items[child]
                            for item in curr_items:
                                maxmica = ontology.get_maxmica_term2profile(item, child_items)
                                maxmica_counts[maxmica[0]] += 1
                            for item in child_items:
                                maxmica = ontology.get_maxmica_term2profile(item, curr_items)
                                maxmica_counts[maxmica[0]] += 1
                            for t, freq in maxmica_counts.items(): #TODO: Maybe need Division by 2 due to the calculation of mica two times  but test fails.
                                if freq >= 2: propagated_item_count[t] += freq
                            # FRED: Maybe for the childs.shift there is uniqueness

                propagated_items = [ k for k,v in propagated_item_count.items() if v >= minimum_childs ]
                if len(propagated_items) > 0:
                    query = self.items.get(term)
                    if query == None:
                        self.items[term] = propagated_items
                    else:
                        terms = union(self.items[term], propagated_items)
                        if clean_profiles and ontology != None: terms = ontology.clean_profile(terms)
                        self.items[term] = terms
    
    def list_terms_per_level_from_items(self):
        return self.list_terms_per_level(list(self.items.keys()))
 
    def list_terms_per_level(self, terms):
        terms_levels = {}
        for term in terms: 
          level = self.get_term_level(term)
          self.add2hash(terms_levels, level, term)
        return terms_levels

    # PROFILE EXTERNAL METHODS
    #############################################

    # I/O profile
    ####################################

    # Increase the arbitrary frequency of a given term set 
    # ===== Parameters
    # +terms+:: set of terms to be updated
    # +increase+:: amount to be increased
    # +transform_to_sym+:: if true, transform observed terms to symbols. Default: false
    # ===== Return
    # true if process ends without errors and false in other cases
    def add_observed_terms(self, terms = None, increase = 1.0):
        for t_id in terms: self.add_observed_term(t_id, increase = increase)

    # Modifying Profile
    ####################################

    def expand_profile_with_parents(self, profile):
        new_terms = []
        for term in profile:
            new_terms = union(new_terms, self.get_ancestors(term))
        return union(new_terms, profile)

    # Clean a given profile returning cleaned set of terms and removed ancestors term.
    # ===== Parameters
    # +prof+:: array of terms to be checked
    # ===== Returns 
    # two arrays, first is the cleaned profile and second is the removed elements array
    def remove_ancestors_from_profile(self, prof):
        ancestors = []
        for term in prof: ancestors.extend(self.get_ancestors(term))
        redundant = intersection(prof, set(ancestors))
        return diff(prof, redundant), redundant

    # Remove alternative IDs if official ID is present. DOES NOT REMOVE synonyms or alternative IDs of the same official ID
    # ===== Parameters
    # +prof+:: array of terms to be checked
    # ===== Returns 
    # two arrays, first is the cleaned profile and second is the removed elements array
    def remove_alternatives_from_profile(self, prof):
        alternatives = [ term for term in prof if term in self.alternatives_index]
        redundant = [ alt_id for alt_id in alternatives if self.alternatives_index.get(alt_id) in prof ]
        return diff(prof, redundant), redundant

    # Remove alternatives (if official term is present) and ancestors terms of a given profile 
    # ===== Parameters
    # +profile+:: profile to be cleaned
    # +remove_alternatives+:: if true, clenaed profiles will replace already stored profiles
    # ===== Returns 
    # cleaned profile
    def clean_profile(self, profile, remove_alternatives = True):
        if self.structureType == 'circular': warnings.warn('Estructure is circular, behaviour could not be which is expected') 
        terms_without_ancestors, _ = self.remove_ancestors_from_profile(profile)
        if remove_alternatives: terms_without_ancestors, _ = self.remove_alternatives_from_profile(terms_without_ancestors) 
        return terms_without_ancestors

    def clean_profile_hard(self, profile, options = {}, remove_alternatives = True):
        profile, _ = self.check_ids(profile)
        profile = [ t for t in profile if not self.is_obsolete(t)] 
        if options.get('term_filter') != None: profile = [ term for term in profile if options['term_filter'] in self.get_ancestors(term) ] # keep terms with parents in term filter
        profile = self.clean_profile(pxc.uniq(profile), remove_alternatives)
        return profile

    # Remove terms from a given profile using hierarchical info and scores set given  
    # ===== Parameters
    # +profile+:: profile to be cleaned
    # +scores+:: hash with terms by keys and numerical values (scores)
    # +byMax+:: if true, maximum scored term will be keeped, if false, minimum will be keeped
    # +remove_without_score+:: if true, terms without score will be removed. Default: true
    # ===== Returns 
    # cleaned profile
    def clean_profile_by_score(self, profile, scores, byMax = True, remove_without_score = True): 
        scores = dict(sorted(scores.items(), key=lambda item: item[1])) # 1 is the values column of the dictionary
        keep = []
        for term in profile:
            term2keep = None
            if term in scores:
                parentals = self.get_ancestors(term) + self.get_descendants(term)
                targetable = [ parent for parent in parentals if parent in profile ]
                if len(targetable) == 0:
                    term2keep = term
                else:
                    targetable.append(term)
                    targets = [ term for term, score in scores.items() if term in targetable ]
                    term2keep =  targets[-1] if byMax else targets[0]
            elif remove_without_score:
                term2keep = None
            else:
                term2keep = term
            if term2keep != None and term2keep not in keep: keep.append(term2keep)
        return keep

    # ID Handlers
    #################################### 

    # Check a set of IDs and return allowed IDs removing which are not official terms on this ontology
    # ===== Parameters
    # +ids+:: to be checked
    # ===== Return
    # two arrays whit allowed and rejected IDs respectively
    def check_ids(self, ids, substitute = True):
        checked_codes = []
        rejected_codes = []
        for t_id in ids:
            new_id = self.get_main_id(t_id)
            if new_id == None:
                rejected_codes.append(t_id)
            else:
                if substitute:
                    checked_codes.append(new_id)
                else:
                    checked_codes.append(t_id)
        return checked_codes, rejected_codes

    # Translates several IDs and returns translations and not allowed IDs list
    # ===== Parameters
    # +ids+:: to be translated
    # ===== Return
    # two arrays with translations and ids which couldn't be translated respectively
    def translate_ids(self, ids):
        translated = []
        rejected = []
        for term_id in ids:
            tr = self.translate_id(term_id)
            if tr != None:
                translated.append(tr)
            else:
                rejected.append(term_id)
        return translated, rejected

    # Translate several names and return translations and a list of names which couldn't be translated
    # ===== Parameters
    # +names+:: array to be translated
    # ===== Return
    # two arrays with translations and names which couldn't be translated respectively
    def translate_names(self, names):
        translated = []
        rejected = []
        for name in names:
            tr = self.translate_name(name)
            if tr == None:
                rejected.append(name)
            else:
                translated.append(tr)
        return translated, rejected

    # Description of profile's terms
    ####################################

    # Gets metainfo table from a set of terms
    # ===== Parameters
    # +terms+:: IDs to be expanded
    # ===== Returns 
    # an array with triplets [TermID, TermName, DescendantsNames]
    def get_childs_table(self, profile):
        expanded_profile = []
        for t in profile:
            expanded_profile.append(
                [[t, self.translate_id(t)], [ [child, self.translate_id(child)] for child in self.get_descendants(t)] ])
        return expanded_profile

    def get_terms_levels(self, profile):
        termsAndLevels = [ [term, self.get_term_level(term)] for term in profile ]
        return termsAndLevels

    # IC data
    ####################################

    # Get information coefficient from profiles #
    
    #  Calculates mean IC of a given profile
    # ===== Parameters
    # +prof+:: profile to be checked
    # +ic_type+:: ic_type to be used
    # +zhou_k+:: special coeficient for Zhou IC method
    # ===== Returns 
    # mean IC for a given profile
    def get_profile_mean_IC(self, prof, ic_type = 'resnik', zhou_k = 0.5):
        return sum([ self.get_IC(term, ic_type = ic_type, zhou_k = zhou_k) for term in prof ]) / len(prof)

    # Term ref vs profile #

    def get_maxmica_term2profile(self, ref_term, profile):
        micas = [ self.get_MICA(ref_term, term) for term in profile ]
        maxmica = micas[0]
        for mica in micas: 
            if mica[-1] > maxmica[-1]: maxmica = mica 
        return maxmica

    # Profile vs Profile #

    # Get semantic similarity from two term sets 
    # ===== Parameters
    # +termsA+:: set to be compared
    # +termsB+:: set to be compared
    # +sim_type+:: similitude method to be used. Default: resnik
    # +ic_type+:: ic type to be used. Default: resnik
    # +bidirectional+:: calculate bidirectional similitude. Default: false
    # ===== Return
    # similitude calculated
    def compare(self, termsA, termsB, sim_type = 'resnik', ic_type = 'resnik', bidirectional = True, store_mica = False):
        # Check
        if termsA == None or termsB == None: raise Exception("Terms sets given are None")
        if len(termsA) == 0 or len(termsB) == 0: raise Exception("Set given is empty. Aborting similarity calc")
        micasA = []
        # Compare A -> B
        for tA in termsA:
            if store_mica:
                tA_micas = self.sim_index[tA]
                micas = [ tA_micas[tB] for tB in termsB ]
            else:    
                micas = []
                for tB in termsB:
                    value = self.get_similarity(tA, tB, sim_type = sim_type, ic_type = ic_type)
                    if type(value) is float: micas.append(value)
            if len(micas) > 0:
                micasA.append(max(micas))  
            else:
                micasA.append(0) 
        means_sim = sum(micasA) / len(micasA)
        # Compare B -> A
        if bidirectional:
            means_simA = means_sim * len(micasA)
            means_simB = self.compare(termsB, termsA, sim_type = sim_type, ic_type = ic_type, bidirectional = False, store_mica = store_mica) * len(termsB)
            means_sim = (means_simA + means_simB) / (len(termsA) + len(termsB))
        # Return
        return means_sim

    def get_profile_similarities(self, terms, sim_type = 'resnik', ic_type = 'resnik', store_mica = False):
        all_terms = copy.copy(terms)
        sims = []
        while len(all_terms) > 1:
            t1 = all_terms.pop()
            for t2 in all_terms:
                if store_mica:
                    value = self.sim_index[t1][t2]
                else:
                    value = self.get_similarity(t1, t2, sim_type = sim_type, ic_type = ic_type)
                if type(value) is float: sims.append(value)
        return np.mean(sims)


    #############################################
    # PROFILE INTERNAL METHODS 
    #############################################

    # I/O profiles
    ####################################

    # Method used to store a pool of profiles
    # ===== Parameters
    # +profiles+:: array/hash of profiles to be stored. If it's an array, numerical IDs will be assigned starting at 1 
    # +calc_metadata+:: if true, launch get_items_from_profiles process
    # +reset_stored+:: if true, remove already stored profiles
    # +substitute+:: subsstitute flag from check_ids
    def load_profiles(self, profiles, calc_metadata = True, reset_stored = False, substitute= False):
        if reset_stored: self.reset_profiles()
        # Check
        if type(profiles) is list:
            for i, items in enumerate(profiles):
                self.add_profile(i, items, substitute = substitute)
        else: # Hash
            for pr_id in profiles.keys():
                if pr_id in self.profiles: 
                    warnings.warn('Some profiles given are already stored. Stored version will be replaced')
                    break 
            for pr_prof, prof in profiles.items(): 
                self.add_profile(pr_prof, prof, substitute = substitute)
        self.add_observed_terms_from_profiles(reset = True)

        if calc_metadata: self.get_items_from_profiles()

    # Stores a given profile with an specific ID. If ID is already assigend to a profile, it will be replaced
    # ===== Parameters
    # +id+:: assigned to profile
    # +terms+:: array of terms
    # +substitute+:: subsstitute flag from check_ids
    def add_profile(self, pr_id, terms, substitute = True, clean_hard=False, options={}): # FRED: Talk with PSZ about the uniqness of IDs translated
        if pr_id in self.profiles: warnings.warn(f"Profile assigned to ID ({pr_id}) is going to be replaced")
        if clean_hard:
            correct_terms = self.clean_profile_hard(terms, options=options)
            rejected_terms = []
        else:
            correct_terms, rejected_terms = self.check_ids(terms, substitute = substitute)
        if len(rejected_terms) > 0: warnings.warn(f"Given terms contains erroneus IDs: {','.join(rejected_terms)}. These IDs will be removed")
        self.profiles[pr_id] = correct_terms              


    # Includes as "observed_terms" all terms included into stored profiles
    # ===== Parameters
    # +reset+:: if true, reset observed freqs alreeady stored befor re-calculate
    def add_observed_terms_from_profiles(self, reset = False):
        if reset: 
            for term, freqs in self.meta.items(): freqs['observed_freq'] = -1 
        for pr_id, terms in self.profiles.items(): self.add_observed_terms(terms = terms)

    # ===== Returns 
    # profiles assigned to a given ID
    # ===== Parameters
    # +id+:: profile ID
    # ===== Return
    # specific profile or nil if it's not stored
    def get_profile(self, pr_id):
        return self.profiles[pr_id]

    # Modifying profiles
    ####################################

    def reset_profiles(self): # Internal method used to remove already stored profiles and restore observed frequencies #TODO FRED: Modify test for this method.
        self.profiles = {} # Clean profiles storage
        # Reset frequency observed
        for term, info in self.meta.items(): info['observed_freq'] = 0
        self.max_freqs['observed_freq'] = 0
        self.items = {}

    def expand_profiles(self, meth, unwanted_terms = [], calc_metadata = True, ontology = None, minimum_childs = 1, clean_profiles = True):
        if meth == 'parental':
            for pr_id, terms in self.profiles.items(): self.profiles[pr_id] = sorted(diff(self.expand_profile_with_parents(terms), unwanted_terms))
            if calc_metadata: self.get_items_from_profiles()
        elif meth == 'propagate':
            self.get_items_from_profiles()
            self.expand_items_to_parentals(ontology = ontology, minimum_childs = minimum_childs, clean_profiles = clean_profiles)
            self.get_profiles_from_items()
        self.add_observed_terms_from_profiles(reset = True)

    # Remove alternatives (if official term is present) and ancestors terms of stored profiles 
    # ===== Parameters
    # +store+:: if true, clenaed profiles will replace already stored profiles
    # ===== Returns 
    # a hash with cleaned profiles
    def clean_profiles(self, store = False, options={}, remove_alternatives = True):
        cleaned_profiles = {}
        for pr_id, terms in self.profiles.items():  
            cleaned_profile = self.clean_profile_hard(terms, options, remove_alternatives)
            if cleaned_profile != []:
                cleaned_profiles[pr_id] = cleaned_profile
        if store: self.profiles = cleaned_profiles 
        return cleaned_profiles

    # ID Handlers
    ####################################

    # Trnaslates a bunch of profiles to it sets of term names
    # ===== Parameters
    # +profs+:: array of profiles
    # +asArray+:: flag to indicate if results must be returned as: true => an array of tuples [ProfID, ArrayOdNames] or ; false => hashs of translations
    # ===== Returns 
    # translated profiles
    def translate_profiles_ids(self, profs = [], asArray = True):
        profs2proc = {}
        if len(profs) == 0:
            profs2proc = self.profiles 
        elif type(profs) is list:
            for index, terms in enumerate(profs): profs2proc[index] = terms
        profs_names = {}
        for pr_id, terms in profs2proc.items():
            names, _ = self.translate_ids(terms)
            profs_names[pr_id] = names
        return  list(profs_names.values()) if asArray else profs_names

    # Description of profile size
    ####################################

    def profile_stats(self):
        return list(pxc.get_stats_from_list(self.get_profiles_sizes()))

    # ===== Returns 
    # mean size of stored profiles
    # ===== Parameters
    # +round_digits+:: number of digits to round result. Default: 4
    # ===== Returns 
    # mean size of stored profiles
    def get_profiles_mean_size(self, round_digits = 4):
        sizes = np.array(self.get_profiles_sizes())
        return round(np.mean(sizes),round_digits)

    # ===== Returns 
    # an array of sizes for all stored profiles
    # ===== Return
    # array of profile sizes
    def get_profiles_sizes(self):
        return [ len(terms) for pr_id, terms in self.profiles.items() ]

    # Calculates profiles sizes and returns size assigned to percentile given
    # ===== Parameters
    # +perc+:: percentile to be returned
    # +increasing_sort+:: flag to indicate if sizes order must be increasing. Default: false
    # ===== Returns 
    # values assigned to percentile asked
    def get_profile_length_at_percentile(self, perc=50, increasing_sort = False): # TODO: Ojear el cohort
        sizes = np.array(self.get_profiles_sizes())    
        if not increasing_sort: perc = 100 - perc    
        return np.quantile(sizes,perc/100)

    # IC data
    ####################################

    # Get frequency terms and information coefficient from profiles #

    # Calculates frequencies of stored profiles terms
    # ===== Parameters
    # +ratio+:: if true, frequencies will be returned as ratios between 0 and 1.
    # +asArray+:: used to transform returned structure format from hash of Term-Frequency to an array of tuples [Term, Frequency]
    # +translate+:: if true, term IDs will be translated to 
    # ===== Returns 
    # stored profiles terms frequencies
    def get_profiles_terms_frequency(self, ratio = True, asArray = True, translate = True):
        freqs = defaultdict(lambda: 0)
        for t_id, terms in self.profiles.items(): 
            for term in terms: freqs[term] += 1 

        if translate:
            translated_freqs = {}
            for term, freq in freqs.items():
                tr = self.translate_id(term)
                if tr != None: translated_freqs[tr] = freq
            freqs = translated_freqs

        if ratio:
            n_profiles = len(self.profiles)
            for term, freq in freqs.items(): freqs[term] = freq / n_profiles

        if asArray:
            freqs = [ [k, v] for k,v in freqs.items() ]
            freqs.sort(key = lambda f: f[1], reverse=True)

        return freqs


    # Calculates number of ancestors present (redundant) in each profile stored
    # ===== Returns 
    # array of parentals for each profile
    def parentals_per_profile(self):
        cleaned_profiles = self.clean_profiles(store = False, options={}, remove_alternatives = False)
        parentals = [len(terms) - len(cleaned_profiles[id]) for id, terms in self.profiles.items()]
        return parentals

    def get_profile_redundancy(self):
        profile_sizes = self.get_profiles_sizes()
        parental_terms_per_profile = self.parentals_per_profile()# clean_profiles
        #parental_terms_per_profile = [item[0] for item in parental_terms_per_profile]
        profile_sizes, parental_terms_per_profile = list(zip(*sorted(list(zip(profile_sizes, parental_terms_per_profile)), key=lambda i: i[0])[::-1]))
        #profile_sizes, parental_terms_per_profile = profile_sizes.zip(parental_terms_per_profile).sort_by{|i| i.first}.reverse.transpose #this is the ruby version of the above line
        return profile_sizes, parental_terms_per_profile

    def compute_term_list_and_childs(self):
        suggested_childs = {}
        total_terms = 0
        terms_with_more_specific_childs = 0
        for id, terms in self.profiles.items():
            total_terms += len(terms)
            more_specific_childs = self.get_childs_table(terms)
            terms_with_more_specific_childs += len([profile for profile in more_specific_childs if len(profile[-1]) > 0]) #Exclude phenotypes with no childs
            suggested_childs[id] = more_specific_childs  
        return suggested_childs, terms_with_more_specific_childs / float(total_terms)


    # Calculates resnik ontology, and resnik observed mean ICs for all profiles stored
    # ===== Returns 
    # two hashes with Profiles and IC calculated for resnik and observed resnik respectively
    def get_profiles_resnik_dual_ICs(self, struct = 'resnik', observ = 'resnik_observed'): # Maybe change name during migration to get_profiles_dual_ICs
        struct_ics = {}
        observ_ics = {}
        for t_id, terms in self.profiles.items():
            struct_ics[t_id] = self.get_profile_mean_IC(terms, ic_type = struct)
            observ_ics[t_id] = self.get_profile_mean_IC(terms, ic_type = observ)
        return struct_ics, observ_ics


    # Calculates and return resnik ICs (by ontology and observed frequency) for observed terms
    # ===== Returns 
    # two hashes with resnik and resnik_observed ICs for observed terms
    def get_observed_ics_by_onto_and_freq(self): 
        ic_ont = {}
        resnik_observed = {}
        observed_terms = set([ t_id for terms in self.profiles.values() for t_id in terms ])
        for term in observed_terms:
            ic_ont[term] = self.get_IC(term)
            resnik_observed[term] = self.get_IC(term, ic_type = 'resnik_observed')
        return ic_ont, resnik_observed

    # Profiles vs Profiles #

    def get_pair_index(self, profiles_A, profiles_B, same_profiles=True):
        pair_index = {}
        if same_profiles: # in this way we can save time for one half of the comparations
            profiles = list(profiles_A.values())
            dropped_profile = []
            while len(profiles) > 0:
                profile_A = profiles[-1]
                if len(profiles) > 1:
                    for profile_B in profiles:
                        for pair in itertools.product(profile_A, profile_B):
                            pair_index[tuple(sorted(pair))] = True
                dropped_profile = profiles.pop()
            for pair in itertools.product(dropped_profile, dropped_profile):
                pair_index[tuple(sorted(pair))] = True                
        else:
            for profile_A in  profiles_A.values():
                for profile_B in profiles_B.values():
                    for pair in itertools.product(profile_A, profile_B):
                        pair_index[tuple(sorted(pair))] = True
        return pair_index

    def get_mica_index_from_profiles(self, pair_index, ic_type = 'resnik', sim_type = 'resnik'):
        for pair in pair_index.keys():
            value = self.get_MICA_from_pair(pair, ic_type = ic_type)
            #if term == None: value = False  # We use False to save that the operation was made but there is not mica value
            tA, tB = pair
            self.add2nestHashDef(self.mica_index, tA, tB, value)
            self.add2nestHashDef(self.mica_index, tB, tA, value)
            value = self.get_similarity(tA, tB, sim_type = sim_type, ic_type = ic_type, mica_index = False)
            if value == None: value = 0
            self.add2nestHashDef(self.sim_index, tA, tB, value)
            self.add2nestHashDef(self.sim_index, tB, tA, value)

    # Compare internal stored profiles against another set of profiles. If an external set is not provided, internal profiles will be compared with itself 
    # ===== Parameters
    # +external_profiles+:: set of external profiles. If nil, internal profiles will be compared with itself
    # +sim_type+:: similitude method to be used. Default: resnik
    # +ic_type+:: ic type to be used. Default: resnik
    # +bidirectional+:: calculate bidirectional similitude. Default: false
    # ===== Return
    # Similitudes calculated
    def compare_profiles(self, external_profiles = None, sim_type = 'resnik', ic_type = 'resnik', bidirectional = True):
        profiles_similarity = {} #calculate similarity between patients profile
        if external_profiles == None:
            comp_profiles = self.profiles
            main_profiles = comp_profiles
            same_profiles = True
        else:
            comp_profiles = external_profiles
            main_profiles = self.profiles
            same_profiles = False
        #start = time.time()
        pair_index = self.get_pair_index(main_profiles, comp_profiles, same_profiles=same_profiles)
        #print(f"pair_index: {time.time() - start}")
        #start = time.time()
        self.mica_index = defaultdict(lambda: dict())
        self.sim_index = defaultdict(lambda: dict())
        self.get_mica_index_from_profiles(pair_index, ic_type = ic_type, sim_type=sim_type)
        #print(f"mica_index: {time.time() - start}")
        #start = time.time()
        if same_profiles:
            profiles = list(comp_profiles.items())
            while len(profiles) > 0:
                curr_id, current_profile = profiles[-1]
                for t_id, profile in profiles:
                    value = self.compare(current_profile, profile, sim_type = sim_type, ic_type = ic_type, bidirectional = bidirectional, store_mica = True)
                    self.add2nestHash(profiles_similarity, curr_id, t_id, value)
                    self.add2nestHash(profiles_similarity, t_id, curr_id, value)
                profiles.pop()
        else:
            for curr_id, current_profile in main_profiles.items():
                for t_id, profile in comp_profiles.items():
                    value = self.compare(current_profile, profile, sim_type = sim_type, ic_type = ic_type, bidirectional = bidirectional, store_mica = True)
                    self.add2nestHash(profiles_similarity, curr_id, t_id, value)
        #print(f"similarity: {time.time() - start}")
        return profiles_similarity

    def get_profile_similarities_from_profiles(self, sim_type = 'resnik', ic_type = 'resnik'):
        profiles_similarity = {}
        pair_index = self.get_pair_index(self.profiles, self.profiles, same_profiles=True)
        self.mica_index = defaultdict(lambda: dict())
        self.sim_index = defaultdict(lambda: dict())
        self.get_mica_index_from_profiles(pair_index, ic_type = ic_type, sim_type=sim_type)
        for t_id, prof in self.profiles.items():
            sim = self.get_profile_similarities(prof, sim_type = sim_type, ic_type = ic_type, store_mica = True)
            profiles_similarity[t_id] = 1 if (np.isnan(sim) and len(prof) == 1) else sim
        return profiles_similarity

    # specifity_index related methods
    ####################################

    # Return ontology levels from profile terms
    # ===== Returns 
    # hash of term levels (Key: level; Value: array of term IDs)
    def get_ontology_levels_from_profiles(self, uniq = True):
        profiles_terms = [ v for vals in self.profiles.values() for v in vals ]
        if uniq: profiles_terms = set(profiles_terms) 
        term_freqs_byProfile = defaultdict( lambda: 0)
        for term in profiles_terms: term_freqs_byProfile[term] += 1 
        levels_filtered = {}
        terms_levels = self.dicts['level']['byValue']
        for term, count in term_freqs_byProfile.items():
            level = terms_levels[term]
            term_repeat = [term] * count
            query = levels_filtered.get(level)
            if query == None:
                levels_filtered[level] = term_repeat
            else:
                query.extend(term_repeat)
        return levels_filtered

    def get_profile_ontology_distribution_tables(self):
      cohort_ontology_levels = self.get_ontology_levels_from_profiles(uniq=False)
      uniq_cohort_ontology_levels = self.get_ontology_levels_from_profiles()
      ontology_levels = self.get_ontology_levels()
      total_ontology_terms = len([ v for vals in ontology_levels.values() for v in vals ])
      total_cohort_terms = len([ v for vals in cohort_ontology_levels.values() for v in vals ])
      total_uniq_cohort_terms =len([ v for vals in uniq_cohort_ontology_levels.values() for v in vals ])

      distribution_ontology_levels = []
      distribution_percentage = []
      for level, terms in ontology_levels.items():
        cohort_terms = cohort_ontology_levels.get(level)
        uniq_cohort_terms = uniq_cohort_ontology_levels.get(level)
        if cohort_terms == None or uniq_cohort_terms == None:
          num = 0
          u_num = 0
        else:
          num = len(cohort_terms)
          u_num = len(uniq_cohort_terms)
        distribution_ontology_levels.append([level, len(terms), num])
        distribution_percentage.append([
          level,
          round(len(terms) / total_ontology_terms*100, 3),
          round(num / total_cohort_terms*100, 3),
          round(u_num / total_uniq_cohort_terms*100, 3)
        ])
      distribution_ontology_levels.sort(key = lambda l: l[0])
      distribution_percentage.sort(key = lambda x: x[0])
      return distribution_ontology_levels, distribution_percentage

    def get_weigthed_level_contribution(self, section, maxL, nLevels):
        accumulated_weigthed_diffL = 0
        for s in section:
            level, diff = s
            weightL = maxL - level 
            if weightL >= 0:
                weightL += 1
            else:
                weightL = abs(weightL)
            accumulated_weigthed_diffL += diff * weightL
        weigthed_contribution = accumulated_weigthed_diffL / nLevels
        return weigthed_contribution

    def get_dataset_specifity_index(self, mode):
        ontology_levels, distribution_percentage = self.get_profile_ontology_distribution_tables()
        if mode == 'uniq':
            observed_distribution = 3
        elif mode == 'weigthed':
            observed_distribution = 2
        max_terms = max([ row[1] for row in distribution_percentage ])
        maxL = None
        diffL = []
        for level_info in distribution_percentage:
            if level_info[1] == max_terms: maxL = level_info[0]
            diffL.append([level_info[0], level_info[observed_distribution] - level_info[1]]) 
        diffL = [ dL for dL in diffL if dL[-1] > 0 ]
        highSection = [ dL for dL in diffL if dL[0] > maxL ]
        lowSection = [ dL for dL in diffL if dL[0] <= maxL ]
        dsi = None
        if len(highSection) == 0:
            dsi = 0
        else:
            hss = self.get_weigthed_level_contribution(highSection, maxL, len(ontology_levels) - maxL)
            lss = self.get_weigthed_level_contribution(lowSection, maxL, maxL)
            dsi = hss / lss
        return dsi

    ########################################
    ## INTERNET QUERY METHODS
    ######################################## 
    def query_ncbi(self, db): # Due a hardcoded limit in NCBI API we only can retrieve 10000 records. To download the complete dataset, we need the E-Direct tools. Install from https://www.ncbi.nlm.nih.gov/books/NBK179288/ and add detection and implementation
        #import entrezpy.log.logger
        #entrezpy.log.logger.set_level('DEBUG')
        count = 0
        ont_term_res = {}
        for t_id, tags in self.each(att = True):
            if count == 1000: break
            name = tags['name']
            syns = self.get_synonims(t_id)
            strings = [name]
            if len(syns) > 0:
                strings.extend(syns)
                strings = list(set(strings))
            query = ' OR '.join(strings)
            es = entrezpy.esearch.esearcher.Esearcher('esearcher', 'mail', threads=1)
            #es.requests_per_sec = 2
            es.num_threads = 1
            a = es.inquire({'db': db,'term':query, 'retmax': 10000, 'rettype': 'uilist'})
            count += 1
            if a != None: 
                result = a.get_result().uids
                a.result = None # This resets class/object to avoid the bug of result aggregation from one query to other
                print(f"{t_id} => {len(result)}")
                if len(result) > 0 : ont_term_res[t_id] = [strings, result]
        return ont_term_res


    ########################################
    ## GENERAL ONTOLOGY METHODS
    ########################################
    def __eq__(self, other):
        if isinstance(other, Ontology):
            res = self.terms == other.terms and \
                self.ancestors_index == other.ancestors_index and \
                self.alternatives_index == other.alternatives_index and \
                self.structureType == other.structureType and \
                self.ics == other.ics and \
                self.meta == other.meta and \
                self.dicts == other.dicts and \
                self.profiles == other.profiles and \
                len(set(self.items.keys()) - set(other.items.keys())) == 0 and \
                self.items == other.items and \
                self.term_paths == other.term_paths and \
                self.max_freqs == other.max_freqs
        else:
            res = False
        return res

    def clone(self):
        copy = Ontology()
        copy.terms = copy.copy(self.terms)
        copy.ancestors_index = copy.copy(self.ancestors_index)
        copy.descendants_index = copy.copy(self.descendants_index)
        copy.alternatives_index = copy.copy(self.alternatives_index)
        copy.structureType = copy.copy(self.structureType)
        copy.ics = copy.copy(self.ics)
        copy.meta = copy.copy(self.meta)
        copy.dicts = copy.copy(self.dicts)
        copy.profiles = copy.copy(self.profiles)
        copy.items = copy.copy(self.items)
        copy.term_paths = copy.copy(self.term_paths)
        copy.max_freqs = copy.copy(self.max_freqs)
        return copy

    # Exports an OBO_Handler object in json format
    # ===== Parameters
    # +file+:: where info will be stored
    def write(self, file):
        # Take object stored info
        obj_info = {'terms': self.terms,
                    'ancestors_index': self.ancestors_index,
                    'descendants_index': self.descendants_index,
                    'alternatives_index': self.alternatives_index,
                    'structureType': self.structureType,
                    'ics': self.ics,
                    'meta': self.meta,
                    'max_freqs': self.max_freqs,
                    'dicts': self.dicts,
                    'profiles': self.profiles,
                    'items': self.items,
                    'term_paths': self.term_paths}
        with open(file, "w") as f: f.write(json.dumps(obj_info))

    def each(self, att = False):
        if len(self.terms) == 0: warnings.warn('terms empty')
        for t_id, tags in self.terms.items():
            if att:
                yield(t_id, tags)
            else:
                yield(t_id)    

    def get_root(self): 
        roots = [ term for term in self.each() if self.ancestors_index.get(term) == None]
        return roots

    def list_term_attributes(self):
        terms = [ [term, self.translate_id(term), self.get_term_level(term)] for term in self.each()]
        return terms

    def return_terms_by_keyword_match(self, keyword, fields = ['name', 'synonym', 'def']):
        terms = []
        query_kwd = re.compile(keyword, re.IGNORECASE)
        for term, attrs in self.each(att=True):
            breakit = False
            for field in fields:
                if breakit == True: break
                if attrs.get(field):
                    if type(attrs[field]) == str:
                        if re.search(query_kwd, attrs[field]):
                            terms.append(term)
                            break
                    elif type(attrs[field]) == list:
                        for subfield in attrs[field]:
                            if re.search(query_kwd, subfield):
                                terms.append(term)
                                breakit = True
                                break
        return terms
    
    # Gets ontology levels calculated
    # ===== Returns 
    # ontology levels calculated
    def get_ontology_levels(self):
        return copy.copy(self.dicts['level']['byTerm']) # By term, in this case, is Key::Level, Value::Terms

    ########################################
    ## SUPPORT METHODS
    ########################################

    def add2hash(self, dictio, key, val):
        query = dictio.get(key)
        if query == None:
            dictio[key] = [val]
        else:
            query.append(val)

    def add2nestHash(self, h, key1, key2, val):
        query1 = h.get(key1)
        if query1 == None:
            h[key1] = {key2 : val} 
        else:
            query1[key2] = val

    def add2nestHashDef(self, h, key1, key2, val):
        h[key1][key2] = val

    def concatItems(self, itemA, itemB): # NEED TEST, CHECK WITH PSZ THIS METHOD
        # A is Array :: RETURN ARRAY
            # A_array : B_array
            # A_array : B_hash => NOT ALLOWED
            # A_array : B_single => NOT ALLOWED
        # A is Hash :: RETURN HASH
            # A_hash : B_array => NOT ALLOWED
            # A_hash : B_hash
            # A_hash : B_single => NOT ALLOWED
        # A is single element => RETURN ARRAY
            # A_single : B_array
            # A_single : B_hash => NOT ALLOWED
            # A_single : B_single
        concatenated = None
        if type(itemA) is list and type(itemB) is list:
            concatenated = union(itemA, itemB)
        elif type(itemA) is dict and type(itemB) is dict:
            itemB_concatenated = {}
            for k, newV in itemB.items():
                oldV = itemA.get(k)
                if oldV == None:
                    itemB_concatenated[k] = newV
                else:
                    itemB_concatenated[k] = self.concatItems(oldV, newV)
            concatenated = itemA | itemB_concatenated
        elif type(itemB) is list:
            concatenated = union([itemA] + itemB)
        elif type(itemB) is not dict:
            concatenated = list(set([itemA, itemB]))
        return concatenated
