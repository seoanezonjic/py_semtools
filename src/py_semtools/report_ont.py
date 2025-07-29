import sys
from collections import defaultdict
from importlib.resources import files

import numpy as np
import networkx as nx

import py_exp_calc.exp_calc as pxc
from py_report_html import Py_report_html

########################################
## Monkey Patching Methods
########################################
Py_report_html.additional_templates.append(str(files('py_semtools').joinpath('templates'))) # https://github.com/python/cpython/issues/106614

##### METHODS FOR ONTOPLOT

## UTILS FUNCTIONS
def _transform_value(self, value, method):
    if method == "cubic_root":
        return value**(1/3)
    elif method == "bins":
        return self._get_alpha_bin(value)
    elif method == "none":
        return value
    elif callable(method):
        return method(value)

def _get_alpha_bin(self, value):
    bins = {(0, 0.2): 0.2, (0.2, 0.4): 0.3, (0.4, 0.6): 0.5, (0.6, 0.8): 0.7, (0.8, 1): 0.9}
    for bins, binarized_value in bins.items():
        if value >= bins[0] and value < bins[1]: return binarized_value
    if value == 1: return value    
    else: raise Exception(f"Input value is not between the closed range of 0-1, value was: {value}")

def _append_values_to_arrays(self, arrays, values):
    for idx, value in enumerate(values):
        arrays[idx].append(value)

def _get_arc_degree_and_radius_values(self, ontology, term, level_linspace, level_current_index, term_level):
    current_level_idx = level_current_index[term_level]
    current_level_arc_array = level_linspace[term_level]
    arc_term_ont = float(current_level_arc_array[current_level_idx])
    
    level_current_index[term_level] += 1
    return arc_term_ont

def _get_plot_points_params(self, hpo_stats_dict, guide_lines, freq_by, mode, user_sizes={}):
    is_dynamic = True if mode == "dynamic" else False
    
    #Defining the distance of profile term points from their circle level (ont_to_prof_dist) according to the maximum frequency of a term
    #in the profiles (as that point will be the biggest one in the plot)
    base_dist = 0.2 
    dist_fact = 0.3 if guide_lines == "ont" else 0.2
    max_freq = max(hpo_stats_dict.values())
    ont_to_prof_dist = base_dist+(dist_fact*max_freq) if freq_by != "alpha" else base_dist
    
    # Defining point sizes for the ontology terms and profile terms (the latter will be calculated according to the frequency of the term 
    # in the profiles if freq_by is "size" or "both"). If custom_sizes is provided, it will be used to define the sizes of the points.
    ont_size= user_sizes.get('ont_size') or (3 if is_dynamic else 2)
    
    prof_base_size = 3 if is_dynamic else 2
    if freq_by not in ["size", "both"]: prof_base_size = user_sizes.get('prof_base_size') or (prof_base_size + ont_size)  
    prof_size_factor = user_sizes.get('prof_size_factor') or (10 if is_dynamic else 100)
    
    # Defining the alpha and frequency of the ontology terms points
    ont_alpha=0.7
    ont_freq=1 if mode != "canvas" else 0.1

    return ont_to_prof_dist, prof_base_size, prof_size_factor, ont_size, ont_alpha, ont_freq

def _get_user_root_recalculated_levels(self, ontology, user_root):
    # First step: we get terms levels and filter in only terms that are descendants from the user defined root
    user_root_desc = ontology.get_descendants(user_root)
    terms_to_filter_in = set([user_root]+user_root_desc)
    terms_levels = { term: level[0] for term, level in pxc.invert_hash(ontology.get_ontology_levels()).items() if term in terms_to_filter_in }
    user_root_lvl = terms_levels[user_root]
    
    # Second step: recalculate levels respect to the user_root
    terms_levels = { term: [level - user_root_lvl] for term, level in terms_levels.items() }

    # Third step: Iterate throught terms with level equal or less than user_root level
    # (this means descendant terms of user_root but with a shortest path than goes through other terms)
    # to recalculate shortest path with respect to user_root (and also to child terms of those terms)
    terms_to_recalculate = [ term for term, level in terms_levels.items() if level[0] <= 0 ]
    cached_levels = {}
    while len(terms_to_recalculate) > 0:
        term = terms_to_recalculate.pop(0)
        if cached_levels.get(term): continue
        cached_levels[term] = nx.shortest_path_length( ontology.dag, source = user_root, target = term )
        child_terms = ontology.get_direct_descendants(term)
        if child_terms: terms_to_recalculate.extend(child_terms) 
    
    # Fourth step: reasign the levels of conflicting terms
    for term, level in cached_levels.items():
        terms_levels[term] = [level]
    
    level_terms = pxc.invert_hash(terms_levels)
    return level_terms, user_root_lvl, terms_to_filter_in, terms_levels

def _get_user_ontology_root_and_reference_nodes(self, ontology_name, user_opts):
    default_root_and_ref = {"hp": "HP:0000118", "go": "GO:0008150", "mondo": "MONDO:0000001", "doid": "DOID:4", "efo": "EFO:0000001"}
    try:
        user_opts['root_node'] = user_opts['root_node'] if user_opts.get('root_node') else default_root_and_ref[ontology_name]
        user_opts['reference_node'] = user_opts['reference_node'] if user_opts.get('reference_node') else default_root_and_ref[ontology_name]
    except Exception:
        if ontology_name == "Ontology": raise ValueError("Ontology name is not defined in the obo file and no root and ref terms were provided")
        raise ValueError(f"Ontology name {ontology_name} has no default root and ref terms defined (and no root and ref terms were provided by user)")

## MAIN METHODS

def prepare_ontoplot_data(self, ontology, hpo_stats_dict, user_root, reference_node, freq_by, mode, fix_alpha, guide_lines, user_sizes):
    root_centered_level_terms, user_root_lvl, hps_to_filter_in, terms_levels = self._get_user_root_recalculated_levels(ontology, user_root)
    max_level = max(root_centered_level_terms.keys())
    #print(f"User root: {user_root} with level {user_root_lvl}, max level in the ontology: {max_level}")
    
    level_linspace = {level: np.linspace(0, 2*np.pi, len(terms)+1) for level, terms in root_centered_level_terms.items()}
    level_current_index = {level: 0 for level in level_linspace.keys()}
    visited_terms = set(user_root)
    terms_to_visit = [term for term in ontology.get_direct_descendants(user_root) if term in hps_to_filter_in]

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
    root_legend = f"User root original level: {user_root_lvl}\nDeepest level from user root:{max_level}"

    ont_to_prof_dist, base_size, size_factor, ont_size, ont_alpha, ont_freq = self._get_plot_points_params(hpo_stats_dict, guide_lines, freq_by, mode, user_sizes)
    ADD = 1 ##COMENTED BECAUSE IT ADDS A WEIRD BEHAVIOUR TO LEGEND ORDER, TODO fix later #if guide_lines == "ont" else  0 #Previous logic did not had the option to remove ontology guide points, so I thought of this shorcut to take it into account without changing previous code
    colors, sizes, radius_values, arc_values, hp_names, alphas, freqs = [grey]*ADD, [ont_size]*ADD, [0]*ADD, [0]*ADD, [ontology.translate_id(user_root)]*ADD, [ont_alpha]*ADD, [ont_freq]*ADD
    while len(terms_to_visit) > 0:
        term = terms_to_visit.pop(0)
        if term in visited_terms: continue
        visited_terms.add(term)    
        childs = ontology.get_direct_descendants(term)
        if childs != None and len(childs) > 0: terms_to_visit = [term for term in childs if term  in hps_to_filter_in] + terms_to_visit
        hp_level = terms_levels[term][0]
        arc_hp_ont = self._get_arc_degree_and_radius_values(ontology, term, level_linspace, level_current_index, hp_level)

        #ADDING THE POINT FOR THE ONTOLOGY TERM
        if guide_lines == "ont":
            self._append_values_to_arrays([colors, sizes, radius_values, arc_values, hp_names, alphas, freqs], [grey, ont_size, hp_level, arc_hp_ont, ontology.translate_id(term), ont_alpha, ont_freq])
        #IF THE TERM EXIST IN THE PROFILE, ADDING ALSO THE PROFILE TERM POINT
        if hpo_stats_dict.get(term) != None: 
            freq = hpo_stats_dict[term]
            current_color = all_term_colors[term]
            current_alpha = self._transform_value(freq, method = fix_alpha) if freq_by in ["alpha", "both"] else 1 #method = "bins","cubic_root","none" or function
            current_size = base_size + (freq*size_factor) if freq_by in ["size", "both"] else base_size
            current_radius = hp_level + ont_to_prof_dist #Making the profile term point a bit further with respect to the ontology point
            self._append_values_to_arrays([colors, sizes, radius_values, arc_values, hp_names, alphas, freqs], [current_color, current_size, current_radius, arc_hp_ont, ontology.translate_id(term), current_alpha, freq])

    #Adding an invisible point at level 16 to keep the number of levels always the same (at the deepest level of the)
    if mode != "canvas":
        self._append_values_to_arrays([colors, sizes, radius_values, arc_values, hp_names, alphas, freqs], [grey, ont_size/100, max_level, np.pi/3, "depth", 0, 0])
    return [[colors, sizes, radius_values, arc_values, hp_names, alphas, freqs], color_legend, root_legend, max_level]

def ontoplot(self, **user_options):
  ontology = self.hash_vars[user_options['ontology']]
  ONT_NAME = ontology.ont_name.upper() if hasattr(ontology, 'ont_name') and ontology.ont_name != None else 'Ontology'
  self._get_user_ontology_root_and_reference_nodes(ONT_NAME, user_options)
  default_opts = {"width": 800, "height": 800, "ONT_NAME": ONT_NAME, "mode": "static", "freq_by": "size", "fix_alpha": "none", 'guide_lines': "ont", 'title': f"",
                  "responsive": True, "dynamic_units_calc": True, "dpi": 100, 'plotly_layout': {}, 'config': {}, 'user_sizes': {}, 'count_parentals': False, 'onto_min_freq': 0}
  default_opts.update(user_options)
  freq_by, mode, fix_alpha, guide_lines, user_sizes = default_opts['freq_by'], default_opts['mode'], default_opts['fix_alpha'], default_opts['guide_lines'], default_opts['user_sizes']

  term_frequencies = ontology.dicts['term_stats'] if not default_opts['count_parentals'] else ontology.dicts['term_stats_with_parentals']
  max_freq = 1  #max(ontology.dicts['term_stats'].values()) This would make a max scaling, not the desired behaviour, but leave it here for reference
  term_frequencies = {term: proportion/max_freq for term, proportion in term_frequencies.items() if proportion >= default_opts['onto_min_freq']}
  user_root = default_opts['root_node']
  reference_node = default_opts['reference_node']
  prepared_data, color_legend, root_legend, max_level = self.prepare_ontoplot_data(ontology, term_frequencies, user_root, reference_node, freq_by, mode, fix_alpha, guide_lines, user_sizes)
  colors, sizes, radius_values, arc_values, hp_names, alphas, freqs = prepared_data

  ontoplot_table_format = [["colors", "sizes", "radius_values", "arc_values", "hp_names", "alphas", "freqs"]]
  ontoplot_table_format = ontoplot_table_format + [[colors[i], sizes[i], radius_values[i], arc_values[i], hp_names[i], alphas[i], freqs[i]] for i in range(len(colors))]
  
  self.hash_vars["ontoplot_table_format"] = ontoplot_table_format
  default_opts["max_level"] = max_level
  return self.renderize_child_template(self.get_internal_template('ontoplot.txt'), color_legend=color_legend, root_legend=root_legend, **default_opts)

def ontodist(self, **user_options):
    ontology = self.hash_vars[user_options['ontology']]
    ONT_NAME = ontology.ont_name.upper() if hasattr(ontology, 'ont_name') and ontology.ont_name != None else 'Ontology'
    default_opts = {"width": "600px", "height": "600px", "ONT_NAME": ONT_NAME}
    default_opts.update(user_options)        
    ontology_levels, distribution_percentage = ontology.get_profile_ontology_distribution_tables()
    ontology_levels.insert(0, ["level", "ontology", "cohort"])
    distribution_percentage.insert(0, ["level", "ontology", "weighted cohort", "uniq terms cohort"])
    self.hash_vars['ontology_levels'] = ontology_levels
    self.hash_vars['distribution_percentage'] = distribution_percentage
    return self.renderize_child_template(self.get_internal_template('ontodist.txt'), **default_opts)

def ontoICdist(self, **user_options):
    ontology = self.hash_vars[user_options['ontology']]
    ONT_NAME = ontology.ont_name.upper() if hasattr(ontology, 'ont_name') and ontology.ont_name != None else 'Ontology'
    default_opts = {"width": "600px", "height": "600px", "ONT_NAME": ONT_NAME}
    default_opts.update(user_options)    
    term_IC_struct, term_IC_observed = ontology.get_observed_ics_by_onto_and_freq() # IC for TERMS
    prof_IC_struct = ontology.dicts['prof_IC_struct']
    prof_IC_observ = ontology.dicts['prof_IC_observ']
    term_ics = [ list(p) for p in zip(list(term_IC_struct.values()),list(term_IC_observed.values())) ]
    profile_ics = [ list(p) for p in zip(list(prof_IC_struct.values()), list(prof_IC_observ.values())) ]
    self.hash_vars['term_ics'] = term_ics
    self.hash_vars['profile_ics'] = profile_ics
    return self.renderize_child_template(self.get_internal_template('ontoICdist.txt'), **default_opts)

def plotProfRed(self, **user_options):
    ontology = self.hash_vars[user_options['ontology']]
    ONT_NAME = ontology.ont_name.upper() if hasattr(ontology, 'ont_name') and ontology.ont_name != None else 'Ontology'
    default_opts = {"width": "600px", "height": "600px", "ONT_NAME": ONT_NAME}
    default_opts.update(user_options)
    profiles_ids = list(ontology.profiles.keys())
    term_redundancy = sorted(list(zip(profiles_ids, ontology.profile_sizes, ontology.parental_terms_per_profile)), reverse=True, 
        key=lambda i: i[1])
    term_redundancy = [ list(i) for i in term_redundancy]
    self.hash_vars['term_redundancy'] = term_redundancy
    return self.renderize_child_template(self.get_internal_template('plotProfRed.txt'), **default_opts)

def makeTermFreqTable(self, **user_options):
    ontology = self.hash_vars[user_options['ontology']]
    ONT_NAME = ontology.ont_name.upper() if hasattr(ontology, 'ont_name') and ontology.ont_name != None else 'Ontology'
    default_opts = {"width": "600px", "height": "600px", "ONT_NAME": ONT_NAME, "count_parentals": False}    
    default_opts.update(user_options)    
    term_stat_dict = ontology.dicts['term_stats'] if not default_opts['count_parentals'] else ontology.dicts['term_stats_with_parentals']
    term_stats = [ [ontology.translate_id(term), freq * 100] for term, freq in term_stat_dict.items()] 
    self.hash_vars['term_stats'] = term_stats
    return self.renderize_child_template(self.get_internal_template('makeTermFreqTable.txt'), **default_opts)

def plotClust(self, **user_options):
    ontology = self.hash_vars[user_options['ontology']]
    ONT_NAME = ontology.ont_name.upper() if hasattr(ontology, 'ont_name') and ontology.ont_name != None else 'Ontology'
    default_opts = {"width": "600px", "height": "600px", "ONT_NAME": ONT_NAME}    
    default_opts.update(user_options)
    self.hash_vars['semantic_clust'] = ontology.clustering[user_options['method_name']]
    return self.renderize_child_template(self.get_internal_template('plotClust.txt'), **default_opts)


#### METHODS FOR SIMILARITY MATRIX HEATMAP

def similarity_matrix_plot(self, **user_options):
    default_opts = {"width": "600px", "height": "600px", "x_label": "xaxis", "title": "title"}
    default_opts.update(user_options)
    return self.renderize_child_template(self.get_internal_template('similarity_heatmap.txt'), **default_opts)

#### LOADING ALL MONKEYPATCHED METHODS
Py_report_html._get_user_ontology_root_and_reference_nodes = _get_user_ontology_root_and_reference_nodes
Py_report_html._get_user_root_recalculated_levels = _get_user_root_recalculated_levels
Py_report_html._get_plot_points_params = _get_plot_points_params
Py_report_html._transform_value = _transform_value
Py_report_html._get_alpha_bin = _get_alpha_bin
Py_report_html._append_values_to_arrays = _append_values_to_arrays
Py_report_html._get_arc_degree_and_radius_values = _get_arc_degree_and_radius_values
Py_report_html.prepare_ontoplot_data = prepare_ontoplot_data
Py_report_html.ontodist = ontodist
Py_report_html.ontoplot = ontoplot
Py_report_html.ontoICdist = ontoICdist
Py_report_html.plotProfRed = plotProfRed
Py_report_html.plotClust = plotClust
Py_report_html.makeTermFreqTable = makeTermFreqTable
Py_report_html.similarity_matrix_plot = similarity_matrix_plot