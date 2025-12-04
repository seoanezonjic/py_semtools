import sys, os, glob, math, re, subprocess, warnings, time, requests, site, copy, argparse
from collections import defaultdict
from importlib.resources import files
import pickle
import pandas as pd

from py_semtools.ontology import Ontology
from py_semtools.indexers.text_indexer import TextIndexer
from py_semtools.stEngine import STengine

import py_semtools # For external_data
from py_semtools.sim_handler import *
import py_exp_calc.exp_calc as pxc
from py_cmdtabs import CmdTabs
from py_exp_calc.exp_calc import invert_nested_hash, flatten
from py_report_html import Py_report_html

#For get_pubmed_index
import numpy as np

ONTOLOGY_INDEX = str(files('py_semtools.external_data').joinpath('ontologies.txt'))
REPORT_TEMPLATE = str(files('py_semtools.templates').joinpath('report.txt'))
#https://pypi.org/project/platformdirs/
ONTOLOGIES=os.path.join(site.USER_BASE, "semtools", 'ontologies')


#########################################################################
# MAIN FUNCTIONS 
#########################################################################

def main_get_corpus_index(opts: argparse.Namespace) -> None: 
  options = vars(opts)
  TextIndexer.build_index(options)


def main_stEngine(opts: argparse.Namespace) -> None: 
    options = vars(opts)
    if options["top_k"] == 0: options["top_k"] = np.inf
    stEngine = STengine(gpu_devices=options["gpu_device"])
    if options.get("gpu_device") and "cuda" in options["gpu_device"]: stEngine.show_gpu_information(verbose= options['verbose'])

    stEngine.init_model(options["model_name"], cache_folder = options["model_path"], verbose= options['verbose'])

    if options.get("query") != None: 
        stEngine.embedd_several_queries(options, expand_path(options["query"]), verbose= options['verbose'])
    elif options.get("query_embedded") != None:
        stEngine.load_several_queries(options, expand_path(options["query_embedded"]), verbose= options['verbose'])

    if options.get("corpus") != None:
      corpus_filenames = expand_path(options["corpus"])
    elif options.get("corpus") == None and options.get("corpus_embedded") != None:
      corpus_filenames = expand_path(options["corpus_embedded"])
    else: # Raw text or saved embedding for corpus not defined so we cannot embed o calculate similarities.
      exit()

    stEngine.process_corpus_get_similarities(corpus_filenames, options, options['verbose'])

    if options.get("gpu_device"): stEngine.show_gpu_information(verbose= options['verbose']) #A final check to GPU information

def main_stEngine_report(opts: argparse.Namespace) -> None: 
  options = vars(opts)
  pickle_path = os.path.join(options['output_file'].replace(".html", ""), 'tmp')
  pickle_filename = os.path.join(pickle_path, 'similitudes.pckl')
  if os.path.exists(pickle_filename) and options['use_pickle']:
      print("Pickle does exist, skipping calculations and only generating report")
      file = open(pickle_filename, 'rb')
      data = pickle.load(file)
      file.close()
      similarities = data['similarities']
      pubmed_ids_and_titles_dict = data['pubmed_ids_and_titles_dict']
      container = data['container']

  else:
      print("Pickle does not exist or --use_pickle not used, so doing similarity calculations")
      papers_hpo_sims = read_and_aggregate_sims(options['input_file'], aggregation = max)
      with open(options["pubmed_ids_and_titles"]) as f: pubmed_ids_and_titles = [line.rstrip().split("\t") for line in f]

      # Load ontology and precompute it
      options['ontology'] = get_ontology_file(options['ontology'], ONTOLOGY_INDEX, ONTOLOGIES)
      hpo = Ontology(file = options['ontology'], load_file = True)
      hpo.precompute()

      # Load and clean documents profiles
      documents_profiles = read_stEngine_profiles(options['input_file'], options['id_col'], options['ont_col'], options['separator'])
      clean_hard = not options['hard_check']
      for doc_id, terms in documents_profiles.items():
        hpo.add_profile(doc_id, terms=terms, clean_hard=clean_hard)
      clean_profiles = copy.deepcopy(hpo.profiles)

      #Load and clean reference profile
      ref_profile = hpo.clean_profile_hard(options["ref_prof"])
      hpo.load_profiles({"ref": ref_profile}, reset_stored= True)

      candidate_sim_matrix, _, candidates_ids, similarities, candidate_pr_cd_term_matches, candidate_terms_all_sims = hpo.calc_sim_term2term_similarity_matrix(ref_profile, "ref", clean_profiles, 
              term_limit = options["matrix_limits"][0], candidate_limit = options["matrix_limits"][-1], sim_type = options['sim'], bidirectional = False,
              string_format = True, header_id = "HP")

      candidate_terms_all_sims = {candidates_ids[candidate_idx]:canditate_terms_sims for candidate_idx, canditate_terms_sims in candidate_terms_all_sims.items()}
      negative_matrix, _ = hpo.get_negative_terms_matrix(candidate_terms_all_sims, string_format = True, header_id = "HP",
              term_limit = options["neg_matrix_limits"][0], candidate_limit = options["neg_matrix_limits"][-1])

      candidates_sims = [[str(candidate), str(value)] for candidate, value in similarities["ref"].items()]


      ref_profile_phenIDX_and_code = list(enumerate([hpo.translate_name(row[0]) for row in candidate_sim_matrix[1:]]))
      stEngine_sims_table = [[0] * (len(row)-1) for row in candidate_sim_matrix[1:]]

      for paper_index in candidate_pr_cd_term_matches.keys():
          paper_id = candidates_ids[paper_index]
          for prof_phen_idx, profile_phen_code in ref_profile_phenIDX_and_code:
              paper_matched_hpo = candidate_pr_cd_term_matches[paper_index].get(profile_phen_code, None)
              hpo_stengine_sim_raw = papers_hpo_sims[paper_id].get(paper_matched_hpo)
              hpo_stengine_sim = round(hpo_stengine_sim_raw, 2) if hpo_stengine_sim_raw else ""
              stEngine_sims_table[prof_phen_idx][paper_index] = hpo_stengine_sim

      stEngine_sims_table.insert(0, candidates_ids)
      for prof_phen_idx, row in enumerate(stEngine_sims_table): row.insert(0, candidate_sim_matrix[prof_phen_idx][0])

      # Sort candidates by similarity and optionally save the full sorted list
      candidates_sims.sort(key=lambda row: float(row[1]), reverse=True)
      if options['get_full_sim_sorted_list']:
        with open(options["output_file"].replace(".html", "_sims.txt"), 'w') as f:
          for candidate, value in sorted(similarities["ref"].items(), key=lambda pair: pair[1], reverse=True):
            f.write("\t".join([str(candidate), str(value)])+"\n") 
      
      ##### DOING SOME ADDITIONAL PREPROCESSING TO MAKE TOP50 TABLE BEFORE ADDING THE DATA TO CONTAINER
      upper_limit = options["matrix_limits"][-1]
      candidates_sims = candidates_sims[0:upper_limit]    
      pubmed_ids_and_sims_df = pd.DataFrame(candidates_sims, columns=["pubmed_id", "similarity"])
      pubmed_ids_and_titles_df = pd.DataFrame(pubmed_ids_and_titles, columns=["pubmed_id", "title", "article-type", "article-category"])

      pubmed_ids_sims_and_titles = pd.merge(pubmed_ids_and_sims_df, pubmed_ids_and_titles_df, on="pubmed_id", how="inner")
      pubmed_ids_sims_and_titles.drop(columns=["article-type", "article-category"], inplace=True) #convert back to a list of lists and add the header back
      pubmed_ids_sims_and_titles["similarity"] = pubmed_ids_sims_and_titles["similarity"].astype(float).round(2)
      pubmed_ids_sims_and_titles["pubmed_id"] = pubmed_ids_sims_and_titles["pubmed_id"].apply(lambda pmid: f"<a href=https://pubmed.ncbi.nlm.nih.gov/{pmid}>{pmid}</a>")
      supp_info = pubmed_ids_sims_and_titles.values.tolist()
      supp_info.insert(0, ["pubmed_id", "similarity", "title"])

      pubmed_ids_and_titles_dict = {row[0]: row[1] for row in pubmed_ids_and_titles}

      container = { "similarity_matrix": candidate_sim_matrix, 
                  "negative_matrix": negative_matrix, 
                  "stEngine_sims_table": stEngine_sims_table,
                  "ont_sim_method": options['sim'],
                  "supp_info": supp_info}
      
      pickle_obj = {'container': container,
                    'similarities': similarities,
                    'pubmed_ids_and_titles_dict': pubmed_ids_and_titles_dict}
      
      if options['use_pickle']:
        os.makedirs(pickle_path, exist_ok = True)
        file = open(pickle_filename, 'wb')
        pickle.dump(pickle_obj, file)
        file.close()

  #MAKING REPORT
  template = open(str(files('py_semtools.templates').joinpath('stEngine.txt'))).read()
  report = Py_report_html(container, f'stEngine report', data_from_files=True)
  report.build(template)
  report.write(options["output_file"])


def main_get_sorted_profs(opts: argparse.Namespace) -> None: 
    """
    Main function to get sorted profiles based on ontology term similarities.

    :param opts: Command-line options parsed by argparse.
    :type opts: argparse.Namespace

    :returns: None

    --------- DEBUG ---------
    We should reformat or wrap part of this functionallity in order to be used as a method when loading the library. 
    However, it is still needed to change how methods in Ontology class work (they need to save info in slots instead of returning it) 
    """
    options = vars(opts)

    options['ontology_file'] = get_ontology_file(options['ontology_file'], ONTOLOGY_INDEX, ONTOLOGIES) 
    ontology = Ontology(file = options['ontology_file'], load_file = True, removable_terms= options['excluded_terms'])
    ontology.precompute()

    data = CmdTabs.load_input_data(options['input_file'])
    data = format_data(data, options)
    for t_id, terms in data: ontology.add_profile(t_id, terms, clean_hard = options['hard_check'], options = options)
    clean_profiles = ontology.profiles

    if options.get("ref_prof"):
      ref_profile = ontology.clean_profile_hard(options["ref_prof"])
    else:
      ref_profile = ontology.get_general_profile(options["term_freq"])

    ontology.load_profiles({"ref": ref_profile}, reset_stored= True)

    candidate_sim_matrix, _, candidates_ids, similarities, candidate_pr_cd_term_matches, candidate_terms_all_sims = ontology.calc_sim_term2term_similarity_matrix(ref_profile, "ref", clean_profiles, 
          term_limit = options["matrix_limits"][0], candidate_limit = options["matrix_limits"][-1], sim_type = options['similarity'], bidirectional = False,
          string_format = True, header_id = "HP")
    
    candidate_terms_all_sims = {candidates_ids[candidate_idx]:canditate_terms_sims for candidate_idx, canditate_terms_sims in candidate_terms_all_sims.items()}
    negative_matrix, _ = ontology.get_negative_terms_matrix(candidate_terms_all_sims, 
            term_limit = options["matrix_limits"][0], candidate_limit = options["matrix_limits"][-1],
            string_format = True, header_id = options['header_id'])
    
    template = open(str(files('py_semtools.templates').joinpath('similarity_matrix.txt'))).read()
    container = { "similarity_matrix": candidate_sim_matrix, "negative_matrix": negative_matrix}
    report = Py_report_html(container, 'Similarity matrix')
    report.build(template)
    report.write(options["output_file"])

    with open(options["output_file"].replace('.html','') +'.txt', 'w') as f:
      for candidate, value in sorted(similarities["ref"].items(), key=lambda pair: pair[1], reverse=True):
        f.write("\t".join([str(candidate), str(value)])+"\n")

            
def main_semtools(opts: argparse.Namespace) -> None: 
    """
    Main function to handle various ontology-related operations based on command-line options.

    :param opts: Command-line options parsed by argparse.
    :type opts: argparse.Namespace

    :returns: None

    ------- DEBUG ------
    It is needed a whole refactor of the main semtools binary, as it has grown in a unsorted way
    """
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
        if options.get('list_translate') == None and options.get('filter_list') == None or options['keyword'] != None:
            data = format_data(data, options)
            if options.get('translate') != 'codes' and options.get('keyword') == None:
                #TODO: change add_profile for load_profiles method and add the clean_hard argument to the method
                for t_id, terms in data: ontology.add_profile(t_id, terms, clean_hard = options['load_hard_cleaned_profiles'], options = options)
    if options.get('list_translate') != None:
        for term in data:
            if options['list_translate'] == 'names':
                translation, untranslated = ontology.translate_ids(term)
            elif options['list_translate'] == 'codes':
                translation, untranslated = ontology.translate_names(term)
            print(f"{term[0]}\t{ '-' if len(translation) == 0 else translation[0]}")
    if options.get('filter_list') != None:
        negated_parentals = []
        afirmed_parentals = []
        for operation, ids in options.get('filter_list'):
            if operation == 'p': afirmed_parentals.extend(ids)
            if operation == 'n': negated_parentals.extend(ids)
        data = [ t[0] for t in data]        
        filt_data = ontology.filter_list(data, whitelist=afirmed_parentals, blacklist=negated_parentals)
        for term in filt_data: print(term)

    if options.get('translate') == 'codes':
        profiles = {}
        for info in data:
            pr_id, terms = info
            profiles[pr_id] = terms
        translate(ontology, 'codes', options, profiles)
        for t_id, terms in profiles.items(): ontology.add_profile(t_id, terms, clean_hard = options['load_hard_cleaned_profiles'], options = options)
           
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
                refs_removed_profiles = clean_profiles(refs, ontology, options) 
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

    if options["return_all_terms_with_user_defined_attributes"] != None:
        print("\t".join(options["return_all_terms_with_user_defined_attributes"]))
        for termID, attrs in ontology.each(att = True):
            text_to_print = ""
            for user_field in options["return_all_terms_with_user_defined_attributes"]:
                field_content = attrs.get(user_field)
                text_to_print += "None\t" if field_content == None else f"{field_content}\t"
            print(text_to_print)
    
    if options.get('output_report') != None:
      ontology.add_observed_terms_from_profiles(reset = True)
      if not ontology.dicts.get('term_stats'): ontology.get_profiles_terms_frequency()
      if not hasattr(ontology, 'profile_sizes') and not hasattr(ontology, 'parental_terms_per_profile'): ontology.get_profile_redundancy()
      if not ontology.dicts.get('prof_IC_struct') and not ontology.dicts.get('prof_IC_observ'):
        ontology.get_observed_ics_by_onto_and_freq()
        ontology.get_profiles_resnik_dual_ICs()
      if options.get('similarity_cluster_plot'): 
        tmp_path = os.path.join(os.path.dirname(options['output_report']), "tmp", "clustermap_tmp")
        os.makedirs(tmp_path, exist_ok=True)
        ontology.get_similarity_clusters(method_name=options['similarity_cluster_plot'], temp_folder=tmp_path, options=options)
      # Building report
      container = {"ontology": ontology, 
                   "root_term": options['root_term'], 
                   "ref_term":options['ref_term'], 
                   "ontoplot_mode": options['ontoplot_mode'], 
                   "onto_count_parent": options['count_parentals'],
                   "onto_min_freq": options['onto_min_freq'],
                   "similarity_cluster_plot": options['similarity_cluster_plot']}
      template = open(REPORT_TEMPLATE).read()
      report = Py_report_html(container, os.path.basename(options["output_report"]), True)
      report.data_from_files = False # We are sending the a ontology object not a raw table file loaded with report_html's I/O methods
      report.build(template)
      report.write(options['output_report'])

def main_strsimnet(options: argparse.Namespace) -> None: 
    texts2compare = load_table_file(input_file = options.input_file,
                                 splitChar = options.split_char,
                                 targetCol = options.cindex,
                                 filterCol = options.findex,
                                 filterValue = options.filter_value)
    # Obtain all Vs all if 1 column was given, or A vs B if 2 columns were given
    similitudes = similitude_network(texts2compare, charsToRemove = options.rm_char, algorithm = options.sim_algorithm)
    # Iter and store
    with open(options.output_file, "w") as f:
        for item, item2, sim in similitudes:
            f.write("\t".join([item, item2 , str(sim)]) + "\n" )


def main_remote_retriever(opts: argparse.Namespace) -> None: 
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

def main_get_sorted_suggestions(opts: argparse.Namespace) -> None: 
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

    #Writting output files and making report 
    output_file_dir = os.path.dirname(options["output_file"])
    sample_name = os.path.basename(options["output_file"]).replace(".txt", "")
    
    CmdTabs.write_output_data(report_table_format, options["output_file"])
    CmdTabs.write_output_data(targets_and_n_hits, os.path.join(output_file_dir, "targets_number_of_hits.txt"))

    if options["output_report"] != None:
      blacklisted_terms_and_names = [["Code", "Term"]] + sorted([[term, ontology.translate_id(term)] for term in black_list])
      deleted_empty_query_terms_and_names = [["Code", "Term"]] + sorted([[term, ontology.translate_id(term)] for term in deleted_empty_query_terms])
      removed_queries_self_parentals_and_names = [["Code", "Term"]] + sorted([[term, ontology.translate_id(term)] for term in removed_queries_self_parentals])
      deleted_query_parental_targets_terms_and_names = [["Code", "Term"]] + sorted([[term, ontology.translate_id(term)] for term in deleted_query_parental_targets_terms])
      container = {'suggs_table': report_table_format,
                  'targets_and_n_hits': targets_and_n_hits,
                  'blacklisted_terms': blacklisted_terms_and_names,
                  'deleted_empty_query_terms': deleted_empty_query_terms_and_names,
                  'removed_queries_self_parentals': removed_queries_self_parentals_and_names,
                  'deleted_query_parental_targets_terms': deleted_query_parental_targets_terms_and_names,
                  'filter_parental_targets': options["filter_parental_targets"],
                  'clean_query_terms': options["clean_query_terms"],
                  'heatmap_color_preset': options["heatmap_color_preset"]}
      template = open(str(files('py_semtools.templates').joinpath('comorb_sugg.txt'))).read()
      report = Py_report_html(container, f'stEngine report', data_from_files=True)
      report.build(template)
      report.write(options["output_report"])
    
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
    ontname_and_urls = source_list.items() if key == "all" else [[key, source_list[key]]]
    for ont_name, url in ontname_and_urls: 
      if output != None:
        output_path = output
      else:
        file_name = ont_name + '.obo'
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
    all_childs = [ [c, ontology.translate_id(c), '|'.join(ontology.get_synonims(c))] for c in all_childs ]    
  return all_childs

def format_data(data, options):
    if options.get('2cols') == True:
        data = CmdTabs.aggregate_column(data, options.get('subject_column'), options.get('annotations_column'), sep=options.get('separator'), agg_mode="concatenate")
    if not options.get('simple_list'): format_tabular_data(data, options.get('separator'), options.get('subject_column'), options.get('annotations_column'))
    return data

def load_table_file(input_file, splitChar = "\t", targetCol = [1], filterCol = -1, filterValue = None):
    texts = []
    with open(input_file) as f:
        for line in f:
            texts_to_add = []
            row = line.rstrip().split(splitChar)
            if filterCol >= 0 and row[filterCol] != filterValue: continue
            
            if len(targetCol) == 1:
              tcol = targetCol[0]
              texts.append(row[tcol])
            elif len(targetCol) == 2: 
              for tcol in targetCol:
                texts_to_add.append(row[tcol])
              texts.append(texts_to_add)

        # Remove uniques
        if len(targetCol) == 1:
          texts = pxc.uniq(texts)
          texts = [[text] for text in texts]
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

def read_and_aggregate_sims(input_filename, aggregation=max):
	data = {}
	with open(input_filename, 'r') as file:
		for line in file:
			row = line.strip().split('\t')
			paper_id = row[0]
			phenotypes = row[1].split(',')
			similarities = [list(map(float, sim.split(';'))) for sim in row[2].split(',')]

			data[paper_id] = {}
			for phenotype, similarity_list in zip(phenotypes, similarities):
				data[paper_id][phenotype] = aggregation(similarity_list)
	return data

def read_stEngine_profiles(input_filename, id_col, profile_col, terms_sep=","):
    profiles = {}
    with open(input_filename, 'r') as file:
        for line in file:
            row = line.strip().split('\t')
            profile_id = row[id_col]
            terms = row[profile_col].split(terms_sep)
            profiles[profile_id] = terms
    return profiles

def expand_path(path):
    expanded_paths = glob.glob(path)
    if len(expanded_paths) == 0: raise Exception(f"Path {path} does not match any file")
    return expanded_paths