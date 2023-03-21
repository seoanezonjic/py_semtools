#! /usr/bin/env python
import argparse
import sys
import os
import glob
import requests

ROOT_PATH=os.path.dirname(__file__)
EXTERNAL_DATA=os.path.join(ROOT_PATH, '..', 'external_data')
sys.path.insert(0, os.path.join(ROOT_PATH, '..'))
from semtools import Ontology

require 'down'

######################################################################################
## METHODS
######################################################################################
def load_tabular_file(file):
  records = []
  with open(file, 'r') as f:
    for line in f: records.append(line.rstrip().split("\t"))
  return records

def format_tabular_data(data, separator, id_col, terms_col):
  for i, row in enumerate(data): data[i] = [row[id_col], row[terms_col].split(separator)]

def store_profiles(file, ontology):
  for t_id, terms in file: ontology.add_profile(t_id, terms)

def load_value(hash_to_load, key, value, unique = true)
   	query = hash_to_load[key]
    if query.nil?
       value = [value] if value.class != Array
       hash_to_load[key] = value
    else
        if value.class == Array
            query.concat(value)
        else
            query << value
        end
        query.uniq! unless unique == nil
    end
end

def translate(ontology, type, options, profiles = nil)
  not_translated = {}
  if type == 'names'
    ontology.profiles.each do |id, terms|
      translation, untranslated = ontology.translate_ids(terms)
      ontology.profiles[id] = translation  
      not_translated[id] = untranslated unless untranslated.empty?
    end  
  elsif type == 'codes'
    profiles.each do |id,terms|
      translation, untranslated = ontology.translate_names(terms)
      profiles[id] = translation
      profiles[id] = profiles[id].join("#{options[:separator]}") 
      not_translated[id] = untranslated unless untranslated.empty?
    end    
  end
  if !not_translated.empty?
    File.open(options[:untranslated_path], 'w') do |file|
      not_translated.each do |id, terms|
          file.puts([id, terms.join(";")].join("\t"))
      end
    end
  end    
end  

def clean_profile(profile, ontology, options)
	cleaned_profile = ontology.clean_profile_hard(profile)	
	unless options[:term_filter].nil?
		cleaned_profile.select! {|term| ontology.get_ancestors(term).include?(options[:term_filter])}
	end	
	return cleaned_profile
end

def clean_profiles(profiles, ontology, options)
	removed_profiles = []
	profiles.each do |id, terms|
		cleaned_profile = clean_profile(terms, ontology, options)
		profiles[id] = cleaned_profile
		removed_profiles << id if cleaned_profile.empty?
	end
	removed_profiles.each{|rp| profiles.delete(rp)}
	return removed_profiles
end

def write_similarity_profile_list(output, onto_obj, similarity_type, refs)
  profiles_similarity = onto_obj.compare_profiles(sim_type: similarity_type, external_profiles: refs)
  File.open(output, 'w') do |f|
    profiles_similarity.each do |pairsA, pairsB_and_values|
      pairsB_and_values.each do |pairsB, values|
        f.puts "#{pairsA}\t#{pairsB}\t#{values}"
      end
    end
  end
end

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
      if os.access(external_data, os.W_OK) == 0:
        output_path = os.path.join(external_data, file_name)
      else:
        output_path = file_name
    if url != None:
      r = requests.get(url, allow_redirects=True)
      open(output_path, 'wb').write(r.content)

def get_ontology_file(path, source):
  if not os.path.exists(path):
    ont_index = dict(load_tabular_file(source))
    if ont_index.get(path) != None
      path = os.path.join(os.path.dirname(source), path + '.obo')
    else
      raise Exception("Input ontology file not exists")
  return path

def get_stats(stats)
  report_stats = []
  report_stats << ['Elements', stats[:count]]
  report_stats << ['Elements Non Zero', stats[:countNonZero]]
  report_stats << ['Non Zero Density', stats[:countNonZero].fdiv(stats[:count])]
  report_stats << ['Max', stats[:max]]
  report_stats << ['Min', stats[:min]]
  report_stats << ['Average', stats[:average]]
  report_stats << ['Variance', stats[:variance]]
  report_stats << ['Standard Deviation', stats[:standardDeviation]]
  report_stats << ['Q1', stats[:q1]]
  report_stats << ['Median', stats[:median]]
  report_stats << ['Q3', stats[:q3]]
  return report_stats
end

def sort_terms_by_levels(terms, modifiers, ontology, all_childs)
  term_levels = ontology.get_terms_levels(all_childs)
  if modifiers.include?('a')
    term_levels.sort!{|t1,t2| t2[1] <=> t1[1]}
  else
    term_levels.sort!{|t1,t2| t1[1] <=> t2[1]}
  end
  all_childs = term_levels.map{|t| t.first}
  return all_childs, term_levels
end

  def get_childs(ontology, terms, modifiers)
    #modifiers
    # - a: get ancestors instead of decendants
    # - r: get parent-child relations instead of list descendants/ancestors
    # - hN: when list of relations, it is limited to N hops from given term
    # - n: give terms names instead of term codes
    all_childs = []
    terms.each do |term|
      if modifiers.include?('a') 
        childs = ontology.get_ancestors(term)
      else
        childs = ontology.get_descendants(term)
      end
      all_childs = all_childs | childs
    end
    if modifiers.include?('r')
      relations = []
      all_childs = all_childs | terms # Add parents that generated child list
      target_hops = nil
      if /h([0-9]+)/ =~ modifiers
        target_hops = $1.to_i + 1 # take into account refernce term (parent/child) addition
        all_childs, term_levels = sort_terms_by_levels(terms, modifiers, ontology, all_childs)
      end

      current_level = nil
      hops = 0
      all_childs.each_with_index do |term, i|
        if !target_hops.nil?
          level = term_levels[i][1]
          if level != current_level
            current_level = level
            hops +=1
            break if hops == target_hops + 1 # +1 take into account that we have detected a level change and we saved the last one entirely
          end
        end
        if modifiers.include?('a')
          descendants = ontology.get_direct_ancentors(term)
        else
          descendants = ontology.get_direct_descendants(term)
        end
        if !descendants.nil?
          descendants.each do |desc|
            if modifiers.include?('a')
              relations << [desc, term]
            else
              relations << [term, desc]
            end
          end
        end
      end
      all_childs = []
      relations.each do |rel| 
        rel, _ = ontology.translate_ids(rel) if modifiers.include?('n')
        all_childs << rel
      end
    else
      all_childs.map!{|c| ontology.translate_id(c)} if modifiers.include?('n') 
    end
    return all_childs
  end



####################################################################################
## OPTPARSE
####################################################################################
def text_list(string): return string.split(',')

def childs(string):
    if '/' in string:
      modifiers, terms = string.split('/')
    else:
      modifiers = ''
      terms = string
    terms = terms.split(',')
    return [terms, modifiers]  

parser = argparse.ArgumentParser(description='Perform Ontology driven analysis ')

parser.add_argument("-d", "--download", dest="download", default= None, 
          help="Download obo file from official resource. MONDO, GO and HPO are possible values.")
parser.add_argument("-i", "--input_file", dest="input_file", default= None, 
          help="Filepath of profile data")
parser.add_argument("-o", "--output_file", dest="output_file", default= None, 
          help="Output filepath")
parser.add_argument("-I", "--IC", dest="ic", default= None, 
          help="Get IC. 'prof' for stored profiles or 'ont' for terms in ontology")
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
          help="Desired path to write removed profiles file.")
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
          help="The number of the column for the annotation ids.")
parser.add_argument('-R', "--root", dest="root", default= None, 
          help="Term id to be considered the new root of the ontology.")
parser.add_argument("--xref_sense", dest="xref_sense", default= 'byValue', action='store_const', const='byTerm',  
          help="Ontology-xref or xref-ontology. By default xref-ontology if set, ontology-xref")
parser.add_argument('-C', "--childs", dest="childs", default= [[], ''], type=childs,
          help="Term code list (comma separated) to generate child list")
opts = parser.parse_args()

####################################################################################
## MAIN
####################################################################################
options = vars(opts)
ont_index_file = os.path.join(EXTERNAL_DATA, 'ontologies.txt')
if options.get('download') != None:
  download(ont_index_file, options['download'], options['output_file'])
  exit()

if options.get('ontology_file') != None:
  options['ontology_file'] = get_ontology_file(options['ontology_file'], ont_index_file)

extra_dicts = []
if options.get('keyword') != None: extra_dicts.append(['xref', {'select_regex': f"({options['keyword']})", 'store_tag': 'tag', 'multiterm': True}]) 
ontology = Ontology(file = options['ontology_file'], load_file = True, extra_dicts = extra_dicts)

if options['root'] != None: Ontology.mutate(options['root'], ontology, clone = False)  # TODO fix method and convert in class method

if options['input_file'] != None:
  data = load_tabular_file(options['input_file'])
  if options.get('list_translate') == None or options['keyword'] != None:
    format_tabular_data(data, options.get('separator'), options.get('subject_column'), options.get('annotations_column'))
    if options.get('translate') != 'codes' and options.get('keyword') == None: store_profiles(data, ontology) 

if options.get('list_translate') != None
  for term in data:
    if options['list_translate'] == 'names':
      translation, untranslated = ontology.translate_ids(term)
    elif options['list_translate'] == 'codes':
      translation, untranslated = ontology.translate_names(term)
    print f"{term[0]}\t{ '-' if len(translation) == 0 else translation[0]}"
  exit()

if options.get('translate') == 'codes'
  profiles = {}
  data.each do |id, terms|
    load_value(profiles, id, terms)
    profiles[id] = terms.split(options[:separator])
  end
  translate(ontology, 'codes', options, profiles)
  store_profiles(profiles, ontology)
end
   
if options[:clean_profiles]
	removed_profiles = clean_profiles(ontology.profiles, ontology, options)	
	if !removed_profiles.nil? && !removed_profiles.empty?
      File.open(options[:removed_path], 'w') do |f|
          removed_profiles.each do |profile|
              f.puts profile
          end
      end
	end
end

if !options[:expand_profiles].nil?
  ontology.expand_profiles(options[:expand_profiles], unwanted_terms: options[:unwanted_terms])
end 

if !options[:similarity].nil?
  refs = nil
  if !options[:reference_profiles].nil?
    refs = load_tabular_file(options[:reference_profiles])
    format_tabular_data(refs, options[:separator], 0, 1)
    refs = refs.to_h
    refs = clean_profiles(ontology.profiles, ontology, options) if options[:clean_profiles]
    abort('Reference profiles are empty after cleaning ') if refs.nil? || refs.empty?
  end
  write_similarity_profile_list(options[:output_file], ontology, options[:similarity], refs)
end 


if options[:IC] == 'prof'
  ontology.add_observed_terms_from_profiles
  by_ontology, by_freq = ontology.get_profiles_resnik_dual_ICs
  ic_file = File.basename(options[:input_file], ".*")+'_IC_onto_freq'
  File.open(ic_file , 'w') do |file|
    ontology.profiles.keys.each do |id|
        file.puts([id, by_ontology[id], by_freq[id]].join("\t"))
    end       
  end
elsif options[:IC] == 'ont'
  File.open('ont_IC' , 'w') do |file|
    ontology.each do |term|
        file.puts "#{term}\t#{ontology.get_IC(term)}"
    end       
  end
end    

if options[:translate] == 'names'
  translate(ontology, 'names', options)  
end

if !options[:childs].first.empty?
  terms, modifiers = options[:childs]
  all_childs = get_childs(ontology, terms, modifiers)
  all_childs.each do |ac|
    if modifiers.include?('r')
      puts ac.join("\t")
    else
      puts ac
    end
  end
end

if !options[:output_file].nil? && options[:similarity].nil?
  File.open(options[:output_file], 'w') do |file|
    ontology.profiles.each do |id, terms|
      file.puts([id, terms.join("|")].join("\t"))
    end
  end         
end

if options[:statistics]
  get_stats(ontology.profile_stats).each do |stat|
    puts stat.join("\t")
  end
end

if options[:list_term_attributes]
  term_attributes = ontology.list_term_attributes
  term_attributes.each do |t_attr|
    t_attr[0] = t_attr[0].to_s
    puts t_attr.join("\t")
  end
end

if !options[:keyword].nil?
  xref_translated = []
  dict = ontology.dicts[:tag][options[:xref_sense]]
  data.each do |id, prof|
    xrefs = []
    prof.each do |t|
	if options[:xref_sense] == :byValue
	      query = dict[t.to_s]
	else
	      query = dict[t]
	end
      xrefs.concat(query) if !query.nil?
    end
    xref_translated << [id, xrefs] if !xrefs.empty?
  end
  File.open(options[:output_file], 'w') do |f|
    xref_translated.each do |id, prof|
      prof.each do |t|
        f.puts [id, t].join("\t")
      end
    end
  end
end
