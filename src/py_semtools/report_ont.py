from py_report_html import Py_report_html
from importlib.resources import files
import numpy as np
from collections import defaultdict
from py_semtools.cons import Cons

########################################
## Monkey Patching Methods
########################################

Py_report_html.additional_templates.append(str(files(Cons.TEMPLATES).joinpath('')))

##### METHODS FOR ONTOPLOT

def append_values_to_arrays(self, arrays, values):
    for idx, value in enumerate(values):
        arrays[idx].append(value)

def get_arc_degree_and_radius_values(self, ontology, term, level_linspace, level_current_index, root_level):
    term_level = ontology.get_term_level(term) - root_level
    current_level_idx = level_current_index[term_level]
    current_level_arc_array = level_linspace[term_level]
    arc_term_ont = float(current_level_arc_array[current_level_idx])
    
    level_current_index[term_level] += 1
    return arc_term_ont, term_level

def prepare_ontoplot_data(self, ontology, hpo_stats_dict, root_node, reference_node):
    level_terms = ontology.get_ontology_levels()
    hps_to_filter_in = set()
    root_level = 0
    root_found = False
    levels_to_remove = []

    for level in sorted(level_terms.keys()): 
        for term in level_terms[level]:
            if term == root_node: 
                hps_to_filter_in.update(ontology.get_descendants(term)+[term])
                root_found = True
                root_level = level

        if root_found: break
        levels_to_remove.append(level)
    for level in levels_to_remove: del level_terms[level]


    cleaned_level_terms = {(level - root_level): [term for term in terms if term in hps_to_filter_in] for level, terms in level_terms.items()}
    level_linspace = {level: np.linspace(0, 2*np.pi, len(terms)) for level, terms in cleaned_level_terms.items()}
    level_current_index = {level: 0 for level in level_linspace.keys()}
    visited_terms = set(root_node)
    terms_to_visit = [term for term in ontology.get_direct_descendants(root_node) if term in hps_to_filter_in]

    black = (0.0, 0.0, 0.0, 1.0)
    grey =  (0.5, 0.5, 0.5, 1.0)
    color_palette = Py_report_html.get_color_palette(len([ term for term in ontology.get_direct_descendants(reference_node) if term  in hps_to_filter_in]))
    top_parental_colors = {term: color_palette.pop() for term in ontology.get_direct_descendants(reference_node) if term  in hps_to_filter_in}
    all_term_colors = defaultdict(lambda: black)
    all_term_colors.update(top_parental_colors)
    for parental in top_parental_colors.keys():
        for child in ontology.get_descendants(parental):
            if all_term_colors[child] == black: all_term_colors[child] = top_parental_colors[parental]

    color_legend = {value: ontology.translate_id(key) for key, value in top_parental_colors.items()}
    color_legend.update({grey: "Ontology", black: "Others"})

    colors, sizes, radius_values, arc_values, hp_names = [grey], [1], [0], [0], [ontology.translate_id(root_node)]
    while len(terms_to_visit) > 0:
        term = terms_to_visit.pop(0)
        if term in visited_terms: continue
        visited_terms.add(term)    
        childs = ontology.get_direct_descendants(term)
        if childs != None and len(childs) > 0: terms_to_visit = [term for term in childs if term  in hps_to_filter_in] + terms_to_visit

        arc_hp_ont, hp_level = self.get_arc_degree_and_radius_values(ontology, term, level_linspace, level_current_index, root_level)
        current_color = all_term_colors[term]
        self.append_values_to_arrays([colors, sizes, radius_values, arc_values, hp_names], [grey, 1, hp_level, arc_hp_ont, ontology.translate_id(term)])
        if hpo_stats_dict.get(term) != None: self.append_values_to_arrays([colors, sizes, radius_values, arc_values, hp_names], [current_color, 1 + hpo_stats_dict[term], hp_level + 0.3, arc_hp_ont, ontology.translate_id(term)])
    return [[colors, sizes, radius_values, arc_values, hp_names], color_legend]

def ontoplot(self, **user_options):
  dynamic = user_options.get('dynamic', False)
  ontology = self.hash_vars[user_options['ontology']]
  term_frequencies = {term: proportion*100 for term, proportion in ontology.dicts['term_stats'].items()}
  root_node = user_options['root_node']
  reference_node = user_options['reference_node']
  prepared_data, color_legend = self.prepare_ontoplot_data(ontology, term_frequencies, root_node, reference_node)
  colors, sizes, radius_values, arc_values, hp_names = prepared_data

  ontoplot_table_format = [["colors", "sizes", "radius_values", "arc_values", "hp_names"]]
  ontoplot_table_format = ontoplot_table_format + [[colors[i], sizes[i], radius_values[i], arc_values[i], hp_names[i]] for i in range(len(colors))]
  
  self.hash_vars["ontoplot_table_format"] = ontoplot_table_format
  user_options["dynamic"] = dynamic
  return self.renderize_child_template(self.get_internal_template('ontoplot.txt'), color_legend=color_legend, **user_options)

def ontodist(self, **user_options):
    ontology = self.hash_vars[user_options['ontology']]
    ontology_levels, distribution_percentage = ontology.get_profile_ontology_distribution_tables()
    ontology_levels.insert(0, ["level", "ontology", "cohort"])
    distribution_percentage.insert(0, ["level", "ontology", "weighted cohort", "uniq terms cohort"])
    self.hash_vars['ontology_levels'] = ontology_levels
    self.hash_vars['distribution_percentage'] = distribution_percentage
    return self.renderize_child_template(self.get_internal_template('ontodist.txt'), **user_options)

def ontoICdist(self, **user_options):
    ontology = self.hash_vars[user_options['ontology']]
    term_IC_struct, term_IC_observed = ontology.get_observed_ics_by_onto_and_freq() # IC for TERMS
    prof_IC_struct = ontology.dicts['prof_IC_struct']
    prof_IC_observ = ontology.dicts['prof_IC_observ']
    term_ics = [ list(p) for p in zip(list(term_IC_struct.values()),list(term_IC_observed.values())) ]
    profile_ics = [ list(p) for p in zip(list(prof_IC_struct.values()), list(prof_IC_observ.values())) ]
    self.hash_vars['term_ics'] = term_ics
    self.hash_vars['profile_ics'] = profile_ics
    return self.renderize_child_template(self.get_internal_template('ontoICdist.txt'), **user_options)

def plotProfRed(self, **user_options):
    ontology = self.hash_vars[user_options['ontology']]
    term_redundancy = sorted(list(zip(ontology.profile_sizes, ontology.parental_terms_per_profile)), reverse=True, 
        key=lambda i: i[0])
    term_redundancy = [ list(i) for i in term_redundancy]
    self.hash_vars['term_redundancy'] = term_redundancy
    return self.renderize_child_template(self.get_internal_template('plotProfRed.txt'), **user_options)

def makeTermFreqTable(self, **user_options):
    ontology = self.hash_vars[user_options['ontology']]
    term_stat_dict = ontology.dicts['term_stats']
    term_stats = [ [ontology.translate_id(term), freq * 100] for term, freq in term_stat_dict.items()] 
    self.hash_vars['term_stats'] = term_stats
    return self.renderize_child_template(self.get_internal_template('makeTermFreqTable.txt'), **user_options)

def plotClust(self, **user_options):
    ontology = self.hash_vars[user_options['ontology']]
    self.hash_vars['semantic_clust'] = ontology.clustering[user_options['method_name']]
    return self.renderize_child_template(self.get_internal_template('plotClust.txt'), **user_options)


#### METHODS FOR SIMILARITY MATRIX HEATMAP

def similarity_matrix_plot(self, **user_options):
    return self.renderize_child_template(self.get_internal_template('similarity_heatmap.txt'), **user_options)

#### LOADING ALL MONKEYPATCHED METHODS
Py_report_html.append_values_to_arrays = append_values_to_arrays
Py_report_html.get_arc_degree_and_radius_values = get_arc_degree_and_radius_values
Py_report_html.prepare_ontoplot_data = prepare_ontoplot_data
Py_report_html.ontodist = ontodist
Py_report_html.ontoplot = ontoplot
Py_report_html.ontoICdist = ontoICdist
Py_report_html.plotProfRed = plotProfRed
Py_report_html.plotClust = plotClust
Py_report_html.makeTermFreqTable = makeTermFreqTable
Py_report_html.similarity_matrix_plot = similarity_matrix_plot