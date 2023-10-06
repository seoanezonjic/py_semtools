import argparse
import sys
import os

from py_semtools.ontology import Ontology
import py_semtools # For external_data
from py_semtools.sim_handler import *


EXTERNAL_DATA=os.path.join(os.path.dirname(py_semtools.__file__), 'external_data')


## TYPES
def text_list(string): return string.split(',')

def childs(string):
    if '/' in string:
      modifiers, terms = string.split('/')
    else:
      modifiers = ''
      terms = string
    terms = terms.split(',')
    return [terms, modifiers]  

##############################################

def semtools(args = None):
    if args == None: args = sys.argv[1:]

	parser = argparse.ArgumentParser(description='Perform Ontology driven analysis ')

	parser.add_argument('-p', "--processes", dest="processes", default= 2, type=int,
	          help="Number of processes to parallelize calculations. Applied to: semantic similarity.")
	parser.add_argument("-d", "--download", dest="download", default= None, 
	          help="Download obo file from an official resource. MONDO, GO and HPO are possible values.")
	parser.add_argument("-i", "--input_file", dest="input_file", default= None, 
	          help="Filepath of profile data")
	parser.add_argument("-o", "--output_file", dest="output_file", default= None, 
	          help="Output filepath")
	parser.add_argument("-I", "--IC", dest="ic", default= None, 
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
	parser.add_argument('-n', "--statistics", dest="statistics", default= False, action='store_true', 
	          help="To obtain main statistical descriptors of the profiles file.")
	parser.add_argument('-l', "--list_translate", dest="list_translate", default= None, 
	          help="Translate to 'names' or to 'codes' input list.")
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
	parser.add_argument("--list", dest="simple_list", default= False, action='store_true', 
	          help="Input file is a simple list with one term/word/code per line, Only use to get a dictionaire with -k.")
	parser.add_argument("--2cols", dest="2cols", default= False, action='store_true', 
	          help="Input file is a two column table, first is an id and the second is a simgle ontology term.")
	parser.add_argument("--out2cols", dest="out2cols", default= False, action='store_true', 
	          help="Output file will be a two column table")
    opts =  parser.parse_args(args)
    main_semtools(opts)

def main_semtools(opts):
	options = vars(opts)
	ont_index_file = os.path.join(EXTERNAL_DATA, 'ontologies.txt')
	if options.get('download') != None:
	  download(ont_index_file, options['download'], options['output_file'])
	  exit()

	if options.get('ontology_file') != None:
	  options['ontology_file'] = get_ontology_file(options['ontology_file'], ont_index_file)

	extra_dicts = []
	if options.get('keyword') != None: extra_dicts.append(['xref', {'select_regex': eval('r"'+options['keyword']+'"'), 'store_tag': 'tag', 'multiterm': True}]) 
	ontology = Ontology(file = options['ontology_file'], load_file = True, extra_dicts = extra_dicts)
	ontology.precompute()
	ontology.threads = options['processes']

	if options['root'] != None: Ontology.mutate(options['root'], ontology, clone = False)  # TODO fix method and convert in class method

	if options['input_file'] != None:
	  data = load_tabular_file(options['input_file'])
	  if options.get('list_translate') == None or options['keyword'] != None:
	    data = format_data(data, options)
	    if options.get('translate') != 'codes' and options.get('keyword') == None: store_profiles(data, ontology) 
	if options.get('list_translate') != None:
	  for term in data:
	    if options['list_translate'] == 'names':
	      translation, untranslated = ontology.translate_ids(term)
	    elif options['list_translate'] == 'codes':
	      translation, untranslated = ontology.translate_names(term)
	    print(f"{term[0]}\t{ '-' if len(translation) == 0 else translation[0]}")
	  exit()

	if options.get('translate') == 'codes':
	  profiles = {}
	  for info in data:
	    pr_id, terms = info
	    profiles[pr_id] = terms.split(options['separator'])
	  translate(ontology, 'codes', options, profiles)
	  store_profiles(profiles, ontology)
	   
	if options.get('clean_profiles'):
	  removed_profiles = clean_profiles(ontology.profiles, ontology, options)	
	  if removed_profiles != None and len(removed_profiles) > 0:
	    with open(options['removed_path'], 'w') as f:
	      for profile in removed_profiles: f.write(profile + "\n")

	if options.get('expand_profiles') != None:
	  ontology.expand_profiles(options['expand_profiles'], unwanted_terms = options['unwanted_terms'])

	if options.get('similarity') != None:
	  refs = None
	  if options.get('reference_profiles') != None:
	    refs = load_tabular_file(options['reference_profiles'])
	    format_tabular_data(refs, options['separator'], 0, 1)
	    refs = dict(refs)
	    if options['clean_profiles']: refs = clean_profiles(ontology.profiles, ontology, options) 
	    if refs == None or len(refs) == 0: raise Exception('Reference profiles are empty after cleaning ')
	  write_similarity_profile_list(options['output_file'], ontology, options['similarity'], refs)


	if options.get('ic') == 'prof':
	  ontology.add_observed_terms_from_profiles()
	  by_ontology, by_freq = ontology.get_profiles_resnik_dual_ICs()
	  ic_file = os.path.splittext(os.path.basename(options['input_file']))[0]+'_IC_onto_freq'
	  with open(ic_file , 'w') as file:
	    for pr_id in ontology.profiles.keys():
	      file.write("\t".join([pr_id, by_ontology[pr_id], by_freq[pr_id]]) + "\n")
	elif options.get('ic') == 'ont':
	  with open('ont_IC' , 'w') as file:
	    for term in ontology.each():
	        file.write(f"{term}\t{ontology.get_IC(term)}\n")

	if options.get('translate') == 'names': translate(ontology, 'names', options)  

	if len(options['childs'][0]) > 0:
	  terms, modifiers = options['childs']
	  all_childs = get_childs(ontology, terms, modifiers)
	  for ac in all_childs:
	    if 'r' in modifiers:
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
	      for pr_id, terms in ontology.profiles.items(): file.write("\t".join([pr_id, "|".join(terms)]) + "\n")

	if options.get('statistics'): 
	  for stat in get_stats(ontology.profile_stats()): print("\t".join([str(el) for el in stat]))

	if options.get('list_term_attributes'):
	  for t_attr in ontology.list_term_attributes(): print("\t".join(t_attr))

	if options.get('keyword') != None:
	  xref_translated = []
	  dictio = ontology.dicts['tag'][options['xref_sense']]
	  if len(data[0]) == 2: # TRanslate profiles
	    for info in data:
	      pr_id, prof = info
	      xrefs = []
	      for t in prof:
	        query = dictio.get(t)
	        if query != None: xrefs.extend(query) 
	      if len(xrefs) > 0: xref_translated.append([pr_id, xrefs]) 
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
	          for x in query: f.write("\t".join([t, x]) + "\n")

def load_tabular_file(file):
  records = []
  if file == '-':
    input_data = sys.stdin
  else:
    with open(file, 'r') as f:
      input_data = f.readlines()
  for line in input_data: records.append(line.rstrip().split("\t"))
  return records

def format_tabular_data(data, separator, id_col, terms_col):
  for i, row in enumerate(data): data[i] = [row[id_col], row[terms_col].split(separator)]

def aggregate_tabular_data(data, id_col, terms_col):
  agg_data = {}
  for d in data:
    p_id = d[id_col]
    term = d[terms_col]
    query = agg_data.get(p_id)
    if query == None:
      agg_data[p_id] = [term]
    else:
      query.append(term)
  return [[p_id, terms] for p_id, terms in agg_data.items()]

def store_profiles(file, ontology):
  for t_id, terms in file: ontology.add_profile(t_id, terms)

def load_value(hash_to_load, key, value, unique = True):
    query = hash_to_load.get(key)
    if query == None:
       if type(value)  is not list: value = [value]
       hash_to_load[key] = value
    else:
        if type(value) is list:
            query.extend(value)
        else:
            query.append(value)
        if unique: 
          uniq_query = list(set(query))
          query.clear()
          query.extend(uniq_query)

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
      profiles[pr_id] = options['separator'].join(translation)
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

def download(source, key, output):
  source_list = dict(load_tabular_file(source))
  external_data = os.path.dirname(source)
  if key == 'list':
    for f in glob.glob(os.path.join(external_data,'*.obo')): print(f)
  else:
    url = source_list[key]
    if output != None:
      output_path = output
    else:
      file_name = key + '.obo'
      try:
        with open(os.path.join(external_data, file_name), 'w') as file:
          file.write('')
          file.close()
        output_path = os.path.join(external_data, file_name)
      except IOError as error: 
        output_path = file_name
    if url != None:
      r = requests.get(url, allow_redirects=True)
      open(output_path, 'wb').write(r.content)

def get_ontology_file(path, source):
  if not os.path.exists(path):
    ont_index = dict(load_tabular_file(source))
    if ont_index.get(path) != None:
      path = os.path.join(os.path.dirname(source), path + '.obo')
    else:
      raise Exception("Input ontology file not exists")
  return path

def get_stats(stats):
  report_stats = []
  report_stats.append(['Elements', stats['count']])
  report_stats.append(['Elements Non Zero', stats['countNonZero']])
  report_stats.append(['Non Zero Density', stats['countNonZero'] / stats['count']])
  report_stats.append(['Max', stats['max']])
  report_stats.append(['Min', stats['min']])
  report_stats.append(['Average', stats['average']])
  report_stats.append(['Variance', stats['variance']])
  report_stats.append(['Standard Deviation', stats['standardDeviation']])
  report_stats.append(['Q1', stats['q1']])
  report_stats.append(['Median', stats['median']])
  report_stats.append(['Q3', stats['q3']])
  return report_stats

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
  all_childs = []
  for term in terms:
    childs = ontology.get_ancestors(term) if 'a' in modifiers else ontology.get_descendants(term) 
    all_childs = list(set(all_childs) | set(childs))
  if 'r' in modifiers:
    relations = []
    all_childs = list(set(all_childs) | set(terms)) # Add parents that generated child list
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
  return all_childs

def format_data(data, options):
  if options.get('2cols') == True:
    data = aggregate_tabular_data(data, options.get('subject_column'), options.get('annotations_column'))
  else:
    if not options.get('simple_list'): format_tabular_data(data, options.get('separator'), options.get('subject_column'), options.get('annotations_column'))
  return data

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

def load_table_file(input_file, splitChar = "\t", targetCol = 1, filterCol = -1, filterValue = None):
	texts = []
	with open(input_file) as f:
		for line in f:
			row = line.rstrip().split(splitChar)
			if filterCol >= 0 and row[filterCol] != filterValue: continue 
			texts.append(row[targetCol]) 
		# Remove uniques
		texts = list(set(texts))
		return texts