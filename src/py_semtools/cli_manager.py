import argparse
import sys
import os
import glob
import math
import re
import subprocess
import time
import requests
import warnings
from collections import defaultdict
from importlib.resources import files
import site

from py_semtools.ontology import Ontology
import py_semtools # For external_data
from py_semtools.sim_handler import *
import py_exp_calc.exp_calc as pxc
from py_cmdtabs import CmdTabs
from py_exp_calc.exp_calc import invert_nested_hash, flatten

#For stENGINE
import torch
from sentence_transformers import SentenceTransformer, util
import gzip, pickle
import xml.etree.ElementTree as ET

#For get_pubmed_index
from concurrent.futures import ProcessPoolExecutor

ONTOLOGY_INDEX = str(files('py_semtools.external_data').joinpath('ontologies.txt'))
#https://pypi.org/project/platformdirs/
ONTOLOGIES=os.path.join(site.USER_BASE, "semtools", 'ontologies')

###########################################################################
## TYPES
###########################################################################

def text_list(string): return string.split(',')

def filter_regex(string):
    filters = []
    pattern = re.compile(r"([pn])\(([A-Za-z:0-9,]*)\)")
    for match in pattern.finditer(string):
        operation, string_ids = match.groups()
        filters.append([operation, string_ids.split(',')])
    return filters

def childs(string):
    if '/' in string:
        modifiers, terms = string.split('/')
    else:
        modifiers = ''
        terms = string
    terms = terms.split(',')
    return [terms, modifiers]

def split_keyword_and_fields(string):
    keyword, fields = string.split('|')
    fields = fields.split(',')
    return [keyword, fields]

#########################################################################
#CLI PARSERS 
#########################################################################
def strsimnet(args = None):
    if args == None: args = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Perform text similarity analysis')
    parser.add_argument("-i", "--input_file", dest="input_file", default= None, 
              help="Input OMIM diseases file.")
    parser.add_argument("-s", "--split_char", dest="split_char", default="\t", 
              help="Character for splitting input file. Default: tab.")
    parser.add_argument("-c", "--column", dest="cindex", default=0, type=int, 
              help="Column index wich contains texts to be compared. Default: 0.")
    parser.add_argument("-C", "--filter_column", dest="findex", default=-1, type=int, 
              help="[OPTIONAL] Column index wich contains to be used as filters. Default: -1.")
    parser.add_argument("-f", "--filter_value", dest="filter_value", default=None, 
              help="[OPTIONAL] Value to be used as filter.")
    parser.add_argument("-r", "--remove_chars", dest="rm_char", default="", 
              help="Chars to be excluded from comparissons.")
    parser.add_argument("-o", "--output_file", dest="output_file", default= None, 
              help="Output similitudes file.")
    opts =  parser.parse_args(args)
    main_strsimnet(opts)

def remote_retriever(args = None):
    if args == None: args = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Retrieve data in remote server by keywords')
    parser.add_argument("-i", "--input_file", dest="input_file", default= None, 
              help="Input file with keywords (one column with an ID, a second one with the keyword and last optional column with alternatives strings separated by ','')")
    parser.add_argument("-s", "--source", dest="source", default= None, 
              help="Remote search engine source to perform search. options: pubmed ")
    parser.add_argument("-o", "--output_file", dest="output_file", default= None, 
              help="Output matches by keyword file.")
    opts =  parser.parse_args(args)
    main_remote_retriever(opts)

def semtools(args = None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Perform Ontology driven analysis ')

    parser.add_argument('-p', "--processes", dest="processes", default=2, type=int,
                        help="Number of processes to parallelize calculations. Applied to: semantic similarity.")
    parser.add_argument("-d", "--download", dest="download", default=None,
                        help="Download obo file from an official resource. MONDO, GO and HPO are possible values.")
    parser.add_argument("-i", "--input_file", dest="input_file", default=None,
                        help="Filepath of profile data")
    parser.add_argument("--load_hard_cleaned_profiles", dest="load_hard_cleaned_profiles", default=False, action='store_true',
                        help="When loading profiles for similarities calculation or other purposes, set if you want to perform a hard cleaning of the profiles or not (default: False)")
    parser.add_argument("-o", "--output_file", dest="output_file", default=None,
                    help="Output filepath")
    parser.add_argument("-I", "--IC", dest="ic", default=None,
                    help="Get the information content (IC) values: 'prof' for stored profiles or 'ont' for terms in ontology")
    parser.add_argument("-O", "--ontology_file", dest="ontology_file", default= None, 
              help="Path to ontology file")
    parser.add_argument("-T", "--term_filter", dest="term_filter", default= None, 
              help="If specified, only terms that are descendants of the specified term will be kept on a profile when cleaned")
    parser.add_argument("-t", "--translate", dest="translate", default= None,
              help="Translate to 'names' or to 'codes'")
    parser.add_argument("-s", "--similarity_method", dest="similarity", default= None, 
              help="Calculate similarity between profile IDs computed by 'resnik', 'lin' or 'jiang_conrath' methods.")
    parser.add_argument("--reference_profiles", dest="reference_profiles", default= None, 
              help="Path to file tabulated file with first column as id profile and second column with ontology terms separated by separator.")
    parser.add_argument('-c', "--clean_profiles", dest="clean_profiles", default= False, action='store_true', 
              help="Removes ancestors, descendants and obsolete terms from profiles.")
    parser.add_argument('-r', "--removed_path", dest="removed_path", default= 'rejected_profs', 
              help="Desired path to write removed profiles file.")
    parser.add_argument('-u', "--untranslated_path", dest="untranslated_path", default= None, 
              help="Desired path to write untranslated profiles file.")
    parser.add_argument('-k', "--keyword", dest="keyword", default= None, 
              help="regex used to get xref terms in the ontology file.")
    parser.add_argument('-e', "--expand_profiles", dest="expand_profiles", default= None, 
              help="Expand profiles adding ancestors if 'parental', adding new profiles if 'propagate'.")
    parser.add_argument('-U', "--unwanted_terms", dest="unwanted_terms", default= [], type=text_list,
              help="Comma separated terms not wanted to be included in profile expansion.")
    parser.add_argument('-S', "--separator", dest="separator", default= ';',
              help="Separator used for the terms profile.")
    parser.add_argument('-E', "--external_separator", dest="external_separator", default= None,
              help="External separator used for the terms profile.")
    parser.add_argument('-n', "--statistics", dest="statistics", default= False, action='store_true', 
              help="To obtain main statistical descriptors of the profiles file.")
    parser.add_argument('-l', "--list_translate", dest="list_translate", default= None, 
              help="Translate to 'names' or to 'codes' input list.")
    parser.add_argument('-F', "--filter_list", dest="filter_list", default= None, type=filter_regex,
              help="Take a term list and filter given a expresion that indicates which terms keep based on the defined parentals. Expresion: p(id:1,id:2)n(id:3,id:4), p for white list and n for blacklist")    
    parser.add_argument('-q', "--query_ncbi", dest="query_ncbi", default= None, 
              help="Get specified item for each term in loaded ontology.")
    parser.add_argument('-f', "--subject_column", dest="subject_column", default= 0, type=int,
              help="The number of the column for the subject id.")
    parser.add_argument('-a', "--annotations_column", dest="annotations_column", default= 1, type=int,
              help="The number of the column for the annotation ids.")
    parser.add_argument("--list_term_attributes", dest="list_term_attributes", default= False, action='store_true', 
              help="Set to give a list of term attributes: term, translation and level in the ontology, from a list of terms.")
    parser.add_argument('-R', "--root", dest="root", default= None, 
              help="Term id to be considered the new root of the ontology.")
    parser.add_argument("--xref_sense", dest="xref_sense", default= 'byValue', action='store_const', const='byTerm',  
              help="Ontology-xref or xref-ontology. By default xref-ontology if set, ontology-xref")
    parser.add_argument('-C', "--childs", dest="childs", default= [[], ''], type=childs,
              help="Term code list (comma separated) to generate child list")
    parser.add_argument("--keyword_search", dest="keyword_search", default= None, type=split_keyword_and_fields, 
              help="Retrieve HP codes that match a regex against the given keyword in one of the specified term attribute fields")
    parser.add_argument("--translate_keyword_search", dest="translate_keyword_search", default= False, action='store_true', 
              help="Print the matches from --keyword_seach as terms names instead of codes")
    parser.add_argument("--list", dest="simple_list", default= False, action='store_true', 
              help="Input file is a simple list with one term/word/code per line. Use to get a dictionaire with -k of filtered list with -F.")
    parser.add_argument("--2cols", dest="2cols", default= False, action='store_true', 
              help="Input file is a two column table, first is an id and the second is a simgle ontology term.")
    parser.add_argument("--out2cols", dest="out2cols", default= False, action='store_true', 
              help="Output file will be a two column table")
    parser.add_argument("--profiles_self_similarities", dest="profiles_self_similarities", default= None, 
              help="Use to get the self-similarities of the profiles in the input file (resnik', 'lin' or 'jiang_conrath)")
    parser.add_argument("--profiles_self_similarities_output", dest="profiles_self_similarities_output", default= 'profiles_self_similarities.txt', 
              help="Use to save the self-similarities of the profiles in the input file (of --profiles_self_similarities) to a file")        
    opts =  parser.parse_args(args)
    main_semtools(opts)

def get_sorted_suggestions(args = None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Perform Ontology driven analysis ')

    parser.add_argument('-q', "--query_terms", dest="query_terms", default= None,
            help="Path to the input file with 1 column format (each term in a new row) to be used as the query")
    parser.add_argument('-r', "--term_relations", dest="term_relations", default= None,
            help="Path to the term-term pairs file. Expected three files (1º term, 2º term, relationship value)")
    parser.add_argument("-b", "--black_list", dest="black_list", default= None, 
            help="Path to a file with a list of target terms to be excluded from the analysis (one column format)")
    parser.add_argument("-o", "--output_file", dest="output_file", default= None, 
            help="Path to the output table")
    parser.add_argument("-d", "--deleted_terms", dest="deleted_terms", default= None, 
            help="Path to the folder to write the different deleted term files")

    parser.add_argument("-O", "--ontology_file", dest="ontology_file", default= None, 
            help="Path to ontology file")

    parser.add_argument("-m", "--max_targets", dest="max_targets", default= 0, type = int, 
            help="Parameter to set the limit of targets terms to retrieve")

    parser.add_argument("-t", "--translate", dest="translate", default="c",
            help="Use if you want to be returned human readable names (use n), codes (use c), or both (use cn or nc depending on the order you want them to be returned)")

    parser.add_argument("-f", "--filter_parental_targets", dest="filter_parental_targets", default=False, action="store_true",
            help="Use if you want to filter out parental terms of the query terms present in the targets")
    parser.add_argument("-c", "--clean_query_terms", dest="clean_query_terms", default=False, action="store_true",
            help="Use if you want to filter out parental terms of the query terms present in the queries")

    parser.add_argument("-T", "--transform_rel_values", dest="transform_rel_values", default= "no",
            help="Use to apply a transformation to relation values. Currently accepted arguments: no (no transformation), log10 (-log10 transformation), exp (log-inverse exponential transformation, 10^(-x))")


    opts =  parser.parse_args(args)
    main_get_sorted_suggestions(opts)

def stEngine(args = None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Perform Ontology driven analysis with Sentence Transformer')

    parser.add_argument('-m', "--model_name", dest="model_name", default= None,
            help="Name of the model to be used")
    parser.add_argument('-p', "--model_path", dest="model_path", default= None,
            help="Path where the model is cached or where it will be stored")
    parser.add_argument('-q', "--query", dest="query", default= None,
            help="Path to the query file. Wildcards are accepted to process multiple files")
    parser.add_argument('-c', "--corpus", dest="corpus", default= None,
            help="Path to the corpus file. Wildcards are accepted to process multiple files")    
    parser.add_argument('-Q', "--query_embedded", dest="query_embedded", default= None,
            help="Path to save/load the embedded query file. Wildcards are accepted to LOAD multiple files")    
    parser.add_argument('-C', "--corpus_embedded", dest="corpus_embedded", default= None,
            help="Path to save/load the embedded corpus file. Wildcards are accepted to LOAD multiple files")
    parser.add_argument('-o', "--output_file", dest="output_file", default= None,
            help="Path to save the output file with semantic scores")
    parser.add_argument('-k', "--top_k", dest="top_k", default= 20, type = int,
            help="Get top scores per keyword")
    parser.add_argument('-v', "--verbose", dest="verbose", default= False, action='store_true',
            help="Toogle on to get verbose output")
    parser.add_argument('-g', "--gpu_device", dest="gpu_device", default= None, type=text_list,
            help="Use to specify the GPU device to be used for speed-up (if available). The format is like: 'cuda:0' or 'cuda:0,cuda:1' to use multiple GPUs or cpu,cpu to use multiple CPUs")    
    
    opts =  parser.parse_args(args)
    main_stEngine(opts)

def get_pubmed_index(args = None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Extract abstracts from pubmed xml files')

    parser.add_argument('-i', "--input", dest="input", default= None,
            help="Path to the pubmed input file in xml format (gzip compressed) to extract. Wildcards are accepted to process multiple files")
    parser.add_argument('-o', "--output", dest="output", default= None,
            help="Output path to save the extracted data (gzip compressed).")
    parser.add_argument('-k', "--chunk_size", dest="chunk_size", default= 0, type = int,
            help="Number of abstracts to be accumulated and saved in a file. If not used, the same input-output files will be used")
    parser.add_argument('-t', "--tag", dest="tag", default= "file_",
            help="If a chunk size is used, this tag will be used to name the output files")
    parser.add_argument('-c', "--n_cpus", dest="n_cpus", default= 1, type = int,
            help="Number of cpus to be used for parallel processing")
    opts =  parser.parse_args(args)
    main_get_pubmed_index(opts)

#########################################################################
# MAIN FUNCTIONS 
#########################################################################

def main_get_pubmed_index(opts):
  options = vars(opts)
  filenames = glob.glob(options["input"])

  if options["chunk_size"] == 0:
    process_several_abstracts(options, filenames)
  else:
    process_several_custom_chunksize_abstracts(options, filenames)


def main_stEngine(opts):
    options = vars(opts)
    queries_content = []
    embedder = None

    if options.get("gpu_device"): show_gpu_information(options)

    ### LOAD OR DOWNLOAD MODEL
    if options.get("model_name") != None and options.get("model_path") != None:
        if options["verbose"]: print(f"\n-Downloading or loading model {options['model_name']} inside path {options['model_path']}")
        embedder = SentenceTransformer(options["model_name"], cache_folder = options["model_path"])

    ### LOAD AND EMBED QUERIES
    if options.get("query") != None:
        if options["verbose"]: print("\n-Loading and embedding queries:")
        queries_filenames = glob.glob(options["query"])
        queries_content = embedd_several_queries(options, embedder, queries_filenames)

    ### LOAD QUERIES IF ALREADY EMBEDDED
    elif options.get("query_embedded") != None:
        if options["verbose"]: print("\n-Loading embedded queries:")
        embedded_queries_filenames = glob.glob(options["query_embedded"])
        queries_content = load_several_queries(options, embedded_queries_filenames)
    
    ### LOAD AND EMBED CORPUS AND CALCULATE SIMILARITIES IF ASKED
    if options.get("corpus") != None:
        if options["verbose"]: print("\n-Loading and embedding corpus (and calculating similarities if asked):")
        corpus_filenames = glob.glob(options["corpus"])
        embedd_or_load_several_corpus(options, embedder, queries_content, corpus_filenames)

    ### LOAD CORPUS IF ALREADY EMBEDDED AND CALCULATE SIMILARITIES IF ASKED
    elif options.get("corpus_embedded") != None:
        if options["verbose"]: print("\n-Loading embedded corpus (and calculating similarities if asked):")
        embedded_corpus_filenames = glob.glob(options["corpus_embedded"])
        embedd_or_load_several_corpus(options, embedder, queries_content, embedded_corpus_filenames)


            
def main_semtools(opts):
    options = vars(opts)
    if options["external_separator"] is None: options["external_separator"] = options["separator"]
    if options.get('download') != None:
        download(ONTOLOGY_INDEX, options['download'], options['output_file'], ONTOLOGIES)
        sys.exit()

    if options.get('ontology_file') != None:
        options['ontology_file'] = get_ontology_file(options['ontology_file'], ONTOLOGY_INDEX, ONTOLOGIES)

    extra_dicts = []
    if options.get('keyword') != None:
        extra_dicts.append(['xref', {'select_regex': eval('r"'+options['keyword']+'"'), 'store_tag': 'tag', 'multiterm': True}]) 
    ontology = Ontology(file = options['ontology_file'], load_file = True, extra_dicts = extra_dicts)
    ontology.precompute()
    ontology.threads = options['processes']

    if options['root'] != None:
        Ontology.mutate(options['root'], ontology, clone = False)  # TODO fix method and convert in class method

    if options.get('query_ncbi') != None:
        results = ontology.query_ncbi(options['query_ncbi'])
        with open(options['output_file'], 'w') as f:
            for t_id, items in results.items():
                query, ids = items
                f.write(f"{t_id}\t{','.join(query)}\t{','.join(ids)}\n")
        sys.exit()

    if options['input_file'] != None:
        data = CmdTabs.load_input_data(options['input_file'])
        if options.get('list_translate') == None or options.get('filter_list') == None or options['keyword'] != None:
            data = format_data(data, options)
            if options.get('translate') != None and options.get('translate') != 'codes' and options.get('keyword') == None:
                store_profiles(data, ontology, load_hard_cleaned_profiles = options['load_hard_cleaned_profiles'], options = options) 
    if options.get('list_translate') != None:
        for term in data:
            if options['list_translate'] == 'names':
                translation, untranslated = ontology.translate_ids(term)
            elif options['list_translate'] == 'codes':
                translation, untranslated = ontology.translate_names(term)
            print(f"{term[0]}\t{ '-' if len(translation) == 0 else translation[0]}")
        sys.exit(0)
    if options.get('filter_list') != None:
        negated_parentals = []
        afirmed_parentals = []
        for operation, ids in options.get('filter_list'):
            if operation == 'p': afirmed_parentals.extend(ids)
            if operation == 'n': negated_parentals.extend(ids)
        data = [ t[0] for t in data]
        filt_data = ontology.filter_list(data, whitelist=afirmed_parentals, blacklist=negated_parentals)
        for term in filt_data: print(term)
        sys.exit(0)        

    if options.get('translate') == 'codes':
        profiles = {}
        for info in data:
            pr_id, terms = info
            profiles[pr_id] = terms
        translate(ontology, 'codes', options, profiles)
        store_profiles(list(profiles.items()), ontology, load_hard_cleaned_profiles = options['load_hard_cleaned_profiles'], options = options)
           
    if options.get('clean_profiles'):
        removed_profiles = clean_profiles(ontology.profiles, ontology, options)    
        if removed_profiles != None and len(removed_profiles) > 0:
            with open(options['removed_path'], 'w') as f:
                for profile in removed_profiles:
                    f.write(profile + "\n")

    if options.get('expand_profiles') != None:
        ontology.expand_profiles(options['expand_profiles'], unwanted_terms = options['unwanted_terms'])

    if options.get('similarity') != None:
        refs = None
        if options.get('reference_profiles') != None:
            refs = CmdTabs.load_input_data(options['reference_profiles'])
            format_tabular_data(refs, options['separator'], 0, 1)
            refs = dict(refs)
            if options['clean_profiles']:
                refs = clean_profiles(ontology.profiles, ontology, options) 
            if refs == None or len(refs) == 0:
                raise Exception('Reference profiles are empty after cleaning ')
        write_similarity_profile_list(options['output_file'], ontology, options['similarity'], refs)


    if options.get('ic') == 'prof':
        ontology.add_observed_terms_from_profiles()
        by_ontology, by_freq = ontology.get_profiles_resnik_dual_ICs()
        ic_file = os.path.splitext(os.path.basename(options['input_file']))[0]+'_IC_onto_freq'
        with open(ic_file , 'w') as file:
            for pr_id in ontology.profiles.keys():
                file.write("\t".join([pr_id, str(by_ontology[pr_id]), str(by_freq[pr_id])]) + "\n")
    elif options.get('ic') == 'ont':
        with open('ont_IC' , 'w') as file:
            for term in ontology.each():
                file.write(f"{term}\t{ontology.get_IC(term)}\n")

    if options.get('translate') == 'names':
        translate(ontology, 'names', options)  

    if len(options['childs'][0]) > 0:
        terms, modifiers = options['childs']
        all_childs = get_childs(ontology, terms, modifiers)
        for ac in all_childs:
            if 'r' in modifiers or 'CNS' in modifiers:
                print("\t".join(ac))
            else:
                print(ac)

    if options.get('output_file') != None and options.get('similarity') == None:
        with open(options['output_file'], 'w') as file:
            if options.get('out2cols') == True:
                for pr_id, terms in ontology.profiles.items(): 
                    for term in terms: 
                        file.write("\t".join([pr_id, term]) + "\n")
            else:
                for pr_id, terms in ontology.profiles.items():
                    file.write("\t".join([pr_id, options["external_separator"].join(terms)]) + "\n")

    if options.get('statistics'): 
        for stat in ontology.profile_stats():
            print("\t".join([str(el) for el in stat]))

    if options.get('list_term_attributes'):
        for t_attr in ontology.list_term_attributes():
            t_attr = [str(el) for el in t_attr]
            print("\t".join(t_attr))

    if options.get('keyword') != None:
        xref_translated = []
        dictio = ontology.dicts['tag'][options['xref_sense']]
        if len(data[0]) == 2: # TRanslate profiles
            for info in data:
                pr_id, prof = info
                xrefs = []
                for t in prof:
                    query = dictio.get(t)
                    if query != None:
                        xrefs.extend(query) 
                if len(xrefs) > 0:
                    xref_translated.append([pr_id, xrefs]) 
            with open(options['output_file'], 'w') as f:
                for pr_id, prof in xref_translated:
                    for t in prof:
                        f.write("\t".join([pr_id, t]) + "\n")
        else: # Get dict: term - xreference (We assume that data has only one column)
            with open(options['output_file'], 'w') as f:
                for t in data:
                    t = t[0]
                    query = dictio.get(t)
                    if query != None: 
                        for x in query:
                            f.write("\t".join([t, x]) + "\n")

    if options.get('keyword_search') != None:
        keyword, fields = options['keyword_search']
        hits = ontology.return_terms_by_keyword_match(keyword, fields)
        for hit in hits:
            if options['translate_keyword_search']: print(ontology.translate_id(hit))
            else: print(hit)
    
    if options["profiles_self_similarities"] != None:
      self_similarities = ontology.get_profile_similarities_from_profiles(sim_type = options["profiles_self_similarities"])
      with open(options["profiles_self_similarities_output"], "w") as f:
        for profile, similarity in self_similarities.items():
          f.write(f"{profile}\t{similarity}\n")

def main_strsimnet(options):
    texts2compare = load_table_file(input_file = options.input_file,
                                 splitChar = options.split_char,
                                 targetCol = options.cindex,
                                 filterCol = options.findex,
                                 filterValue = options.filter_value)
    # Obtain all Vs all
    similitudes_AllVsAll = similitude_network(texts2compare, charsToRemove = options.rm_char)

    # Iter and store
    with open(options.output_file, "w") as f:
      for item, item_similitudes in similitudes_AllVsAll.items():
        for item2, sim in item_similitudes.items():
          f.write("\t".join([item, item2 , str(sim)]) + "\n" )


def main_remote_retriever(opts):
    keywords = load_keywords(opts.input_file)
    if opts.source == 'pubmed':
        query_pubmed(keywords, opts.output_file)

def load_keywords(file):
    keywords = []
    with open(file) as f:
        for line in f:
            fields = line.rstrip().split("\t")
            if len(fields) == 2:
                id, keyword = fields
                keywords.append([id, keyword.lower()])
            else:
                id, keyword, alternatives = fields
                alternatives = alternatives.split(',')
                alternatives = [ a.lower() for a in alternatives ]
                keywords.append([id, keyword.lower(), alternatives])
    return keywords

def query_pubmed(keywords, file): 
    with open(file, 'w') as f:
        for keyword in keywords:
            if len(keyword) == 2:
                id , kw = keyword
            elif len(keyword) == 3:
                id , main_kw, synonims = keyword
                kw = r'\" OR \"'.join(list(set([main_kw] + synonims))) #inner list/set makes uniq keywords
            cmd = f"esearch -db pubmed -query \"\\\"{kw}\\\"\" 2> /dev/null | efetch -format uid 2> /dev/null"
            # query must be a string with "" to enclose putative ' characters. For this reason, when " is use to skip the NCBI translation is escaped with \\ (one additional \ to send the slash character to the terminal)
            query_sp = subprocess.run(cmd, input="", shell=True, capture_output=True, encoding='UTF-8')
            if query_sp.returncode != 0:
                print(cmd)
                print(f"{id} has returned the following non zero code error:{query_sp.returncode}") 
                break
            query = re.sub("\n", ",", query_sp.stdout.strip())
            f.write(f"{id}\t{query}\n") # write on the fly to avoid lose the previous queries for any incovenience
            time.sleep(1)

def main_get_sorted_suggestions(opts):
    options = vars(opts)

    ##### LOADING AND PRECOMPUTING ONTOLOGY AND LOADING AND CLEANING QUERY INPUT TERMS
    options['ontology_file'] = get_ontology_file(options['ontology_file'], ONTOLOGY_INDEX, ONTOLOGIES) 
    ontology = Ontology(file = options['ontology_file'], load_file = True)
    ontology.precompute()

    query_terms = flatten(CmdTabs.load_input_data(options["query_terms"]))
    query_terms = delete_duplicated_entries(query_terms) #checking for duplicates and removing them if so (and also warning the user)

    # Check wether to do a soft (obsolete or invalid) or hard cleaning (also cleaning parentals) of the query terms
    if not options["clean_query_terms"]:
        cleaned_query_terms, _ = ontology.check_ids(query_terms)
        removed_queries_self_parentals = []
    else:
        cleaned_query_terms = ontology.clean_profile_hard(query_terms)
        _, removed_queries_self_parentals = ontology.remove_ancestors_from_profile(query_terms)

    # IF FILTER_PARENTAL_TARGETS IS SET, CREATE A SET OF QUERY PARENTAL TERMS TO REMOVE CORRESPONDING TARGETS FROM THE RELATIONS FILE
    query_terms_parental_targets = set()
    if options["filter_parental_targets"]:
      for query_term in cleaned_query_terms: query_terms_parental_targets.update(ontology.get_ancestors(query_term))

    # IF BLACK_LIST IS SET, REMOVE TERMS IN THE BLACK LIST FROM THE TARGETS
    black_list = [] if options["black_list"] == None else set(flatten(CmdTabs.load_input_data(options["black_list"])))


    ##### LOADING ONLY TERMS RELATED WITH QUERY TERMS FROM RELATIONS FILE
    query_related_terms, blacklisted_terms = load_query_related_target_terms(options["term_relations"], cleaned_query_terms, black_list)

    ##### CALCULATING NUMBER OF HITS (RELATION WITH A QUERY) FOR EACH TARGET TERM AND SORTING THEM FROM HIGHEST TO LOWEST 
    related_terms_query = invert_nested_hash(query_related_terms)
    target_number_of_hits = {target: len(queries.values()) for target, queries in related_terms_query.items()}
    unfiltered_targets_heatmap_sort_list = [term_hits_pair[0] for term_hits_pair in sorted(target_number_of_hits.items(), key=lambda term_hits_pair: (term_hits_pair[1], term_hits_pair[0]), reverse=True)]
    targets_heatmap_sort_list = [hp for hp in unfiltered_targets_heatmap_sort_list if hp not in query_terms_parental_targets]

    ## LIMITING THE NUMBER OF TARGETS TO PLOT TO THE MAX_TARGETS PARAMETER (FOR PLOTTING PURPOSES) PROVIDED BY THE USER
    if options["max_targets"] == 0: options["max_targets"] = len(targets_heatmap_sort_list) ## Setting max_targets to whole targets list if max_targets is 0(default, so no value was provided)
    targets_to_plot = set(targets_heatmap_sort_list[:options["max_targets"]])
    deleted_query_parental_targets_terms = query_terms_parental_targets.intersection(set(unfiltered_targets_heatmap_sort_list[:options["max_targets"]]))

    ##Filtering out query terms with no targets_to_plot (all the queries whose targets hypergeometric scores are 0, so not relationship with any target)
    deleted_empty_query_terms = []
    for query in cleaned_query_terms:
        filtered_targets_to_plot = dict(filter( lambda target_value: target_value[0] in targets_to_plot, query_related_terms[query].items()))

        if sum(filtered_targets_to_plot.values()) == 0:
            deleted_empty_query_terms.append(query)
            del query_related_terms[query]

    #### CALCULATING THE MEAN OF THE TARGETS HYPERGEOMETRIC SCORE FOR EACH QUERY TERM AND SORTING THEM FROM HIGHEST TO LOWEST
    queries_mean_hypergeometric = {query: custom_mean_and_filter(targets.items(), targets_to_plot) for query, targets in query_related_terms.items()}
    queries_heatmap_sort_list = [term_mean_pair[0] for term_mean_pair in sorted(queries_mean_hypergeometric.items(), key=lambda term_mean_pair: term_mean_pair[1], reverse=True)]

    #### CHECKING WETHER TO TRANSLATE TERMS CODES TO HUMAN READABLE NAMES
    all_terms_list = set(query_related_terms.keys()).union(set(related_terms_query.keys())).union(set(black_list))
    if options["translate"] == "c": 
      code_to_name = { term: term for term in all_terms_list }
    elif options["translate"] == "c": 
      code_to_name = { term: ontology.translate_ids([term])[0][0] for term in all_terms_list }
    elif options["translate"] == "cn":
      code_to_name = { term: f"({term}) {ontology.translate_ids([term])[0][0]}" for term in all_terms_list }
    elif options["translate"] == "nc":
      code_to_name = { term: f"{ontology.translate_ids([term])[0][0]} ({term})" for term in all_terms_list }
    else:
      raise Exception("Invalid translate parameter value. Valid values are: c, n, cn, nc")

    #### PREPARING AND WRITTING THE OUTPUT TABLE AND DELETED QUERY TERMS FILE

    #Preparing query-targets hyper values heatmap table
    report_table_format = [  ["queries"] + [ code_to_name[term] for term in targets_heatmap_sort_list[:options["max_targets"]] ]  ]

    for query_term in queries_heatmap_sort_list:
        row = [code_to_name[query_term]]
        for target_term in targets_heatmap_sort_list[:options["max_targets"]]:
          value = None if query_related_terms[query_term][target_term] == 0 else query_related_terms[query_term][target_term]
          if options["transform_rel_values"] == "no" or value == None:
            row.append(value)
          elif options["transform_rel_values"] == "exp":
            row.append(10**(-value))
          elif options["transform_rel_values"] == "log10":
            row.append(-math.log10(value))
        report_table_format.append(row)

    #Preparing targets and number of hits datatable
    blacklisted_terms_inverted = invert_nested_hash(blacklisted_terms)
    blacklisted_terms_hits = {target: len(queries.values()) for target, queries in blacklisted_terms_inverted.items()}
    targets_and_n_hits = [["Target HP name", "Target HP Code", "Number of hits"]] + [
                          [ontology.translate_ids([target])[0][0], target, target_number_of_hits[target]] for target in unfiltered_targets_heatmap_sort_list]
    targets_and_n_hits += [[ontology.translate_ids([target])[0][0], target, blacklisted_terms_hits[target]] for target in blacklisted_terms_hits.keys()]

    #Writting output files
    output_file_dir = os.path.dirname(options["output_file"])
    sample_name = os.path.basename(options["output_file"]).replace(".txt", "")
    
    CmdTabs.write_output_data(report_table_format, options["output_file"])
    CmdTabs.write_output_data(targets_and_n_hits, os.path.join(output_file_dir, "targets_number_of_hits.txt"))
    
    if options["deleted_terms"] != None:
      CmdTabs.write_output_data([[ontology.translate_ids([term])[0][0]+f" ({term})"] for term in deleted_empty_query_terms], os.path.join(options["deleted_terms"], f"{sample_name}_deleted_empty_query_terms.txt"))
      CmdTabs.write_output_data([[ontology.translate_ids([term])[0][0]+f" ({term})"] for term in removed_queries_self_parentals], os.path.join(options["deleted_terms"], f"{sample_name}_deleted_query_self_parentals.txt"))
      CmdTabs.write_output_data([[ontology.translate_ids([term])[0][0]+f" ({term})"] for term in deleted_query_parental_targets_terms], os.path.join(options["deleted_terms"], f"{sample_name}_deleted_query_parental_targets.txt"))

#########################################################################
# FUNCTIONS
#########################################################################

def format_tabular_data(data, separator, id_col, terms_col):
  for i, row in enumerate(data): data[i] = [row[id_col], row[terms_col].split(separator)]

def store_profiles(file, ontology, load_hard_cleaned_profiles = False, options = {}):
  for t_id, terms in file: ontology.add_profile(t_id, terms, clean_hard = load_hard_cleaned_profiles, options = options)

def translate(ontology, mode, options, profiles = None):
  not_translated = {}
  if mode == 'names':
    for pr_id, terms in ontology.profiles.items():
      translation, untranslated = ontology.translate_ids(terms)
      ontology.profiles[pr_id] = translation  
      if len(untranslated) > 0: not_translated[pr_id] = untranslated
  elif mode == 'codes':
    for pr_id, terms in profiles.items():
      translation, untranslated = ontology.translate_names(terms)
      profiles[pr_id] = translation
      if len(untranslated) > 0: not_translated[pr_id] = untranslated
  if len(not_translated) > 0:
    with open(options['untranslated_path'], 'w') as file:
      for pr_id, terms in not_translated.items():
          file.write("\t".join([pr_id, ";".join(terms)]) + "\n")

def clean_profile(profile, ontology, options):
  cleaned_profile = ontology.clean_profile_hard(profile, options)	
  return cleaned_profile

def clean_profiles(profiles, ontology, options):
  removed_profiles = []
  for pr_id, terms in profiles.items():
    cleaned_profile = clean_profile(terms, ontology, options)
    if len(cleaned_profile) == 0:
      removed_profiles.append(pr_id)
    else:
      profiles[pr_id] = cleaned_profile
  for rp in removed_profiles: profiles.pop(rp)
  return removed_profiles

def write_similarity_profile_list(output, onto_obj, similarity_type, refs):
  profiles_similarity = onto_obj.compare_profiles(sim_type = similarity_type, external_profiles = refs)
  with open(output, 'w') as f:
    for profA, profB_and_sim in profiles_similarity.items():
      for profB, sim in profB_and_sim.items(): f.write(f"{profA}\t{profB}\t{sim}\n")

def download(source, key, output, ontologies_folder):
  source_list = dict(CmdTabs.load_input_data(source))
  if not os.path.exists(os.path.dirname(ontologies_folder)): os.mkdir(os.path.dirname(ontologies_folder))
  if not os.path.exists(ontologies_folder): os.mkdir(ontologies_folder)
  if key == 'list':
    for f in glob.glob(os.path.join(ontologies_folder,'*.obo')): print(f)
  else:
    url = source_list[key]
    if output != None:
      output_path = output
    else:
      file_name = key + '.obo'
      try:
        with open(os.path.join(ontologies_folder, file_name), 'w') as file:
          file.write('')
          file.close()
        output_path = os.path.join(ontologies_folder, file_name)
      except IOError as error: 
        output_path = file_name
    if url != None:
      r = requests.get(url, allow_redirects=True)
      open(output_path, 'wb').write(r.content)

def get_ontology_file(path, source, ontologies_folder):
  if not os.path.exists(path):
    ont_index = dict(CmdTabs.load_input_data(source))
    if ont_index.get(path) != None:
      path = os.path.join(ontologies_folder, path + '.obo')
    else:
      raise Exception("Input ontology file not exists")
  return path

def sort_terms_by_levels(terms, modifiers, ontology, all_childs):
  term_levels = ontology.get_terms_levels(all_childs)
  if 'a' in modifiers:
    term_levels.sort(key=lambda x: x[1], reverse = True)
  else:
    term_levels.sort(key=lambda x: x[1])
  all_childs = [ t[0] for t in term_levels ]
  return all_childs, term_levels

def get_childs(ontology, terms, modifiers):
  #modifiers
  # - a: get ancestors instead of decendants
  # - r: get parent-child relations instead of list descendants/ancestors
  # - hN: when list of relations, it is limited to N hops from given term
  # - n: give terms names instead of term codes
  # - s: sort terms by level in the ontology
  # - CNS: git list with Code, Name and synonims
  all_childs = []
  sort_by_level = True if 's' in modifiers else False
  for term in terms:
    childs = ontology.get_ancestors(term, sort_by_level) if 'a' in modifiers else ontology.get_descendants(term, sort_by_level) 
    all_childs = pxc.union(pxc.uniq(all_childs),pxc.uniq(childs)) 
  if 'r' in modifiers:
    relations = []
    all_childs = pxc.union(pxc.uniq(terms),pxc.uniq(all_childs)) if "s" in modifiers else pxc.union(pxc.uniq(all_childs),pxc.uniq(terms))# Add parents that generated child list
    target_hops = None
    matches = re.search(r"h([0-9]+)", modifiers) 
    if matches:
      target_hops = int(matches.group(1)) + 1 # take into account refernce term (parent/child) addition
      all_childs, term_levels = sort_terms_by_levels(terms, modifiers, ontology, all_childs)

    current_level = None
    hops = 0
    for i, term in enumerate(all_childs):
      if target_hops != None:
        level = term_levels[i][1]
        if level != current_level:
          current_level = level
          hops +=1
          if hops == target_hops + 1: break  # +1 take into account that we have detected a level change and we saved the last one entirely
      
      descendants = ontology.get_direct_ancentors(term) if 'a' in modifiers else ontology.get_direct_descendants(term)
      if descendants != None:
        for desc in descendants:
          relations.append([desc, term]) if 'a' in modifiers else relations.append([term, desc])
    all_childs = []
    for rel in relations: 
      if 'n' in modifiers: rel, _ = ontology.translate_ids(rel) 
      all_childs.append(rel)
  elif 'n' in modifiers:
    all_childs = [ ontology.translate_id(c) for c in all_childs ]
  elif 'CNS' in modifiers:
    all_childs = [ [c, ontology.translate_id(c), ','.join(ontology.get_synonims(c))] for c in all_childs ]    
  return all_childs

def format_data(data, options):
    if options.get('2cols') == True:
        data = CmdTabs.aggregate_column(data, options.get('subject_column'), options.get('annotations_column'), sep=options.get('separator'))
    if not options.get('simple_list'): format_tabular_data(data, options.get('separator'), options.get('subject_column'), options.get('annotations_column'))
    return data

def load_table_file(input_file, splitChar = "\t", targetCol = 1, filterCol = -1, filterValue = None):
    texts = []
    with open(input_file) as f:
        for line in f:
            row = line.rstrip().split(splitChar)
            if filterCol >= 0 and row[filterCol] != filterValue: continue 
            texts.append(row[targetCol]) 
        # Remove uniques
        texts = pxc.uniq(texts)
        return texts
    
def delete_duplicated_entries(query_terms, detailed=False):
  uniq_terms = list(set(query_terms))
  if detailed:
    terms_count = defaultdict(lambda: 0)
    for term in query_terms: terms_count[term] += 1
    duplicated_terms = [term for term, count in terms_count.items() if count > 1]
    if len(duplicated_terms) > 0: warnings.warn(f"Input query file contains duplicated terms: {duplicated_terms}")
  else:
    if len(uniq_terms) != len(query_terms): warnings.warn("Input query file contains duplicated terms")
  return uniq_terms

def load_ontology(external_data, ontology_file):
  ont_index_file = os.path.join(external_data, 'ontologies.txt')
  if ontology_file != None: ontology_file = get_ontology_file(ontology_file, ont_index_file)
  extra_dicts = []
  ontology = Ontology(file = ontology_file, load_file = True, extra_dicts = extra_dicts)
  ontology.precompute()
  return ontology

def load_query_related_target_terms(filename, cleaned_query_terms, black_list):
  query_related_terms = {query_term: defaultdict(lambda: 0) for query_term in cleaned_query_terms}
  blacklisted_terms = {query_term: defaultdict(lambda: 0) for query_term in cleaned_query_terms}
  with open(filename) as file:
   for line in file:
        term1, term2, value = line.strip().split("\t")
        value = float(value)
        if term1 not in cleaned_query_terms and term2 not in cleaned_query_terms: continue
        if term1 in cleaned_query_terms and term2 in cleaned_query_terms: continue

        if term1 in cleaned_query_terms:
            query = term1
            target = term2
        else:
            query = term2
            target = term1
        if target in black_list: 
            blacklisted_terms[query][target] = value
        else:
            query_related_terms[query][target] = value
  return query_related_terms, blacklisted_terms

def custom_mean(values, target_terms_subset):
   if len(values) == 0: return 0
   else: return sum(values)/len(target_terms_subset)
   
def custom_mean_and_filter(items, target_terms_subset):
   values_to_calc = [values for keys, values in items if keys in target_terms_subset]
   return custom_mean(values_to_calc, target_terms_subset)


######################## FUNCTIONS FOR ST ENGINE ########################
def show_gpu_information(options):
    if options["verbose"]:
      print("Information about the available GPUs:")
      print("GPU available: ", torch.cuda.is_available())
      print("Number of GPUs: ", torch.cuda.device_count())
      print("CUDA version: ", torch.version.cuda)
      print("Current CUDA device: ", torch.cuda.current_device())
      print("CUDA device object: ", torch.cuda.device(0))
      print("CUDA device name: ", torch.cuda.get_device_name(0), "\n")


def load_several_queries(options, embedded_queries_filenames):
    queries_content = {}
    for embedded_query_filename in embedded_queries_filenames:
        embedded_query_basename = os.path.splitext(os.path.basename(embedded_query_filename))[0]
        with open(embedded_query_filename, "rb") as fIn:
            if options["verbose"]: print(f"---Loading embedded query from {embedded_query_basename}")
            queries_content[embedded_query_basename] = pickle.load(fIn)
    return queries_content

def embedd_several_queries(options, embedder, queries_filenames):
    queries_content = {}
    for query_filename in queries_filenames:
        query_basename, query_ids, queries, query_embeddings = embedd_single_query(query_filename, embedder, options)
        queries_content[query_basename] = {'query_ids': query_ids, "queries": queries, "embeddings": query_embeddings}
        if options.get("query_embedded") != None:
            if options["verbose"]: print(f"---Saving embedded query in {query_basename}")
            with open(os.path.join(options["query_embedded"], query_basename) + '.pkl', "wb") as fOut:
                pickle.dump(queries_content[query_basename], fOut)
    return queries_content

def embedd_single_query(query_filename, embedder, options):
    query_basename = os.path.splitext(os.path.basename(query_filename))[0]
    if options["verbose"]: print(f"---Loading query from {query_basename}")
    keyword_index = load_keyword_index(query_filename) # keywords used in queries
    queries = []
    query_ids = []
    for kwdID, kwds in keyword_index.items():
        queries.extend(kwds)
        query_ids.extend([kwdID for i in range(0, len(kwds))])
    query_embeddings = embedd_text(queries, embedder, options)
    return [query_basename, query_ids, queries, query_embeddings]

def embedd_or_load_several_corpus(options, embedder, queries_content, corpus_filenames):
    for corpus_filename in corpus_filenames:
         ### LOAD AND EMBED CORPUS (AND SAVE IT options["corpus_embedded"] != None)
        if options.get("corpus") != None:
            corpus_basename, textIDs, corpus, corpus_embeddings = embedd_single_corpus(corpus_filename, embedder, options)
            corpus_info = {'textIDs': textIDs, "corpus": corpus, "embeddings": corpus_embeddings}        
            if options.get("corpus_embedded") != None:
                if options["verbose"]: print(f"---Saving embedded corpus in {corpus_basename}")
                with open(os.path.join(options["corpus_embedded"], corpus_basename) + '.pkl', "wb") as fOut:
                    pickle.dump(corpus_info, fOut)
        
        ### LOAD EMBEDDED CORPUS IF ALREADY EMBEDDED PREVIOUSLY
        elif options.get("corpus") == None and options.get("corpus_embedded") != None:
            embedded_corpus_filename = corpus_filename
            corpus_basename = os.path.splitext(os.path.basename(embedded_corpus_filename))[0]
            if options["verbose"]: print(f"---Loading embedded corpus from {os.path.basename(corpus_basename)}")
            with open(embedded_corpus_filename, "rb") as fIn:
                corpus_info = pickle.load(fIn)
        
        ### CALCULATE SIMILARITIES
        if options.get("output_file"):
            if options["verbose"]: print("-Calculating similarities:")
            for query_basename, query_info in queries_content.items():
                if options["verbose"]: print(f"---Calculating similarities between corpus {corpus_basename} and query {query_basename}")
                best_matches = calculate_similarities(query_info, corpus_info, options["top_k"])
                output_filename = os.path.join(options["output_file"],query_basename)
                if options["verbose"]: print(f"---Saving similarities in file {output_filename}\n")
                save_similarities(output_filename, best_matches)
  
def embedd_single_corpus(corpus_filename, embedder, options):
    corpus_basename = os.path.splitext(os.path.basename(corpus_filename))[0]
    if options["verbose"]: print(f"---Loading corpus of {corpus_basename}")
    pubmed_index = load_pubmed_index(corpus_filename) # abstracts
    textIDs = list(pubmed_index.keys())
    corpus = list(pubmed_index.values())
    corpus_embeddings = embedd_text(corpus, embedder, options)
    return [corpus_basename, textIDs, corpus, corpus_embeddings]

def embedd_text(text, embedder, options):
    start = time.time()
    if options["gpu_device"] != None:
        if len(options["gpu_device"]) > 1:
            pool = embedder.start_multi_process_pool(options["gpu_device"])
            text_embedding = embedder.encode_multi_process(text, pool = pool)
            embedder.stop_multi_process_pool(pool)
        elif len(options["gpu_device"]) == 1:
            text_embedding = embedder.encode(text, convert_to_numpy=True, show_progress_bar = options["verbose"], device= options["gpu_device"][0]) #convert_to_tensor=True
    else:
        text_embedding = embedder.encode(text, convert_to_numpy=True, show_progress_bar = options["verbose"]) #convert_to_tensor=True
    
    if options["verbose"]: print(f"---Embedding time with {0 if options.get('gpu_device') == None else len(options['gpu_device'])} GPUs: {time.time() - start} seconds")
    return text_embedding

def load_keyword_index(file):
    keywords = {}
    with open(file) as f:
        for line in f:
            fields = line.rstrip().split("\t")
            if len(fields) == 2:
                id, keyword = fields
                keywords[id] = [keyword.lower()]
            else:
                id, keyword, alternatives = fields
                alternatives = alternatives.split(',')
                alternatives.append(keyword)
                alternatives = [ a.lower() for a in alternatives ]
                kwrds = list(set(alternatives))
                keywords[id] = kwrds
    return keywords

def load_pubmed_index(file):
  pubmed_index = {}
  with gzip.open(file, "rt") as f:
    for line in f:
        try:
            id, text = line.rstrip().split("\t")
            pubmed_index[int(id)] = text
        except:
            warnings.warn(f"Error reading line in file {os.path.basename(file)}: {line}")
  return pubmed_index

def calculate_similarities(query_info, corpus_info, top_k):
  corpus_ids = corpus_info["textIDs"]
  corpus_embeddings = corpus_info["embeddings"]

  query_ids = query_info['query_ids']
  query_embeddings = query_info["embeddings"]
  search = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k)

  return find_best_matches(corpus_ids, query_ids, search)

def find_best_matches(corpus_ids, query_ids, search):
    best_matches = {}
    for i,query in enumerate(search):
      kwdID = query_ids[i]
      kwd = best_matches.get(kwdID)
      if kwd == None:
        kwd = {}
        best_matches[kwdID] = kwd

      for hit in query:
        textID = corpus_ids[hit['corpus_id']]
        score = hit['score']
        text_score = kwd.get(textID)
        if text_score == None or text_score < score :
          kwd[textID] = score
      #sentence = corpus_sentences[hit['corpus_id']]
    return best_matches

def save_similarities(filepath, best_matches):
    with open(filepath, 'a') as f:
      for kwdID, matches in best_matches.items():
        for textID, score in matches.items():
          f.write(f"{kwdID}\t{textID}\t{score}\n")


######################## FUNCTIONS FOR GET PUBMED INDEX ########################

def process_several_abstracts(options, filenames):
    if options["n_cpus"] == 1:
      for filename in filenames:
        process_single_abstract([options, filename])
    else:
      with ProcessPoolExecutor(max_workers=options["n_cpus"]) as executor:
        executor.map(process_single_abstract, [[options, filename] for filename in filenames])

def process_single_abstract(options_filename_pair):
    options, filename = options_filename_pair
    basename = os.path.basename(filename).replace(".xml.gz", "")
    abstract_index = get_abstract_index(filename)
    out_filename = os.path.join(options["output"], basename+".gz")
    with gzip.open(out_filename, 'wt') as f:
      for pmid, text in abstract_index:
        f.write(f"{pmid}\t{text}\n")

def process_several_custom_chunksize_abstracts(options, filenames):
    if options["n_cpus"] == 1:
      process_a_pack_of_custom_chunksize_abstracts([options, filenames, 0])
    else:
      filenames_lots = []
      lot_size = (len(filenames)+(len(filenames) % options["n_cpus"])) // options["n_cpus"]
      for index in range(0, options["n_cpus"]):
        filenames_lots.append(filenames[index*lot_size:(index+1)*lot_size])

      with ProcessPoolExecutor(max_workers=options["n_cpus"]) as executor:
        executor.map(process_a_pack_of_custom_chunksize_abstracts, [[options, filename_lot, idx] for idx,filename_lot in enumerate(filenames_lots)])

def process_a_pack_of_custom_chunksize_abstracts(options_filenames_counter_trio):
    options, filenames, sup_counter = options_filenames_counter_trio
    acummulative_abstracts = []
    counter = 0
    for filename in filenames:    
      abstract_index = get_abstract_index(filename)
      acummulative_abstracts.extend(abstract_index)
      while len(acummulative_abstracts) >= options["chunk_size"]:
        out_filename = os.path.join(options["output"], options["tag"]+f"{sup_counter}_{counter}.gz" )
        counter += 1
        with gzip.open(out_filename, 'wt') as f:
          for times in range(options["chunk_size"]):
            pmid, text = acummulative_abstracts.pop()
            f.write(f"{pmid}\t{text}\n")
    
    out_filename = os.path.join(options["output"], options["tag"]+f"{sup_counter}_{counter}.gz" )
    with gzip.open(out_filename, 'wt') as f:
      for pmid, text in acummulative_abstracts:
        f.write(f"{pmid}\t{text}\n")


def get_abstract_index(file): 
	texts = [] # aggregate all abstracts in XML file
	with gzip.open(file) as gz:
		mytree = ET.parse(gz)
		pubmed_article_set = mytree.getroot()
		for article in pubmed_article_set:
			pmid = None
			for data in article.find('MedlineCitation'):
				if data.tag == 'PMID':
					pmid = data.text
				abstract = data.find('Abstract')
				if abstract != None:
					for i in abstract:
						abstractText = abstract.find('AbstractText')
						if abstractText != None:
							txt = abstractText.text
							texts.append([pmid, txt])
	return texts