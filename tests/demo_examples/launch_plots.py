#! /usr/bin/env python
import os, argparse
import py_report_html
from py_report_html import Py_report_html
from py_cmdtabs import CmdTabs
import py_semtools
from py_semtools import Ontology
from importlib.resources import files
import site

########################### GLOBAL VARIABLES ###########################
ONTOLOGY_INDEX = str(files('py_semtools.external_data').joinpath('ontologies.txt'))
ONTOLOGIES=os.path.join(site.USER_BASE, "semtools", 'ontologies')

########################### FUNCTIONS ################################

def get_ontology_file(path, source, ontologies_folder):
  if not os.path.exists(path):
    ont_index = dict(CmdTabs.load_input_data(source))
    if ont_index.get(path) != None:
      path = os.path.join(ontologies_folder, path + '.obo')
    else:
      raise Exception("Input ontology file not exists")
  return path

########################### OPTPARSE ###########################
parser = argparse.ArgumentParser(description=f'Usage: {os.path.basename(__file__)} [options]')
parser.add_argument("-p", "--profiles", dest="profiles", default= None, help="Path to the profiles file")
parser.add_argument("-s", "--terms_separator", dest="terms_separator", default= ",", help="Separator for terms in profiles file")
parser.add_argument("-O", "--ontology_file", dest="ontology_file", default= None, help="Path to ontology file")
parser.add_argument("-t", "--template", dest="template", default= None, help="Path to the template file")
parser.add_argument("-o", "--output", dest="output", default= None, help="Output filepath")
opts = parser.parse_args()
options = vars(opts)

########################### MAIN ###########################
# Loading data
template = open(options['template']).read()
ontology_file = get_ontology_file(options['ontology_file'], ONTOLOGY_INDEX, ONTOLOGIES)
profiles = {row[0]: row[1].split(options['terms_separator']) for row in CmdTabs.load_input_data(options['profiles'])}

# Loading ontology and getting profiles terms frequency
ontology = Ontology(file = ontology_file, load_file = True, extra_dicts = {})
ontology.load_profiles(profiles, reset_stored=True)
ontology.get_profiles_terms_frequency()
ontology.get_profile_redundancy()
ontology.get_observed_ics_by_onto_and_freq()
ontology.get_profiles_resnik_dual_ICs()
ontology.get_similarity_clusters(method_name="lin", options={'cl_size_factor': 1})

# Building report
container = {"ontology": ontology}
report = Py_report_html(container, os.path.basename(options["output"]), True)
report.data_from_files = False # We are sending the a ontology object not a raw table file loaded with report_html's I/O methods
report.build(template)
report.write(options['output'] + '.html')
