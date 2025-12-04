import argparse, re, sys, inspect
from py_semtools.main_modules import *
from collections import defaultdict

###########################################################################
## TYPES
###########################################################################

def one_column_file(file): return [line.strip() for line in open(file).readlines()]

def text_list(string): return string.split(',')

def text_to_default_dict(string):
    d = defaultdict(lambda: False)
    for item in text_list(string):
        d[item] = True
    return d

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

def split_column_numbers(string):
    return [int(number) for number in string.split(",")]

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
    parser.add_argument("-c", "--columns", dest="cindex", default=[0], type=split_column_numbers, 
              help="Column(s) index(es) wich contains texts to be compared. If two columns are defined (e.g. 0,1), only items from column 0 vs 1 are compared. Default: 0.")
    parser.add_argument("-C", "--filter_column", dest="findex", default=-1, type=int, 
              help="[OPTIONAL] Column index wich contains to be used as filters. Default: -1.")
    parser.add_argument("-f", "--filter_value", dest="filter_value", default=None, 
              help="[OPTIONAL] Value to be used as filter.")
    parser.add_argument("-r", "--remove_chars", dest="rm_char", default="", 
              help="Chars to be excluded from comparissons.")
    parser.add_argument("-o", "--output_file", dest="output_file", default= None, 
              help="Output similitudes file.")
    parser.add_argument("--sim_algorithm", dest="sim_algorithm", default= 'white',
              help="Similarity algorithm")

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
    parser.add_argument("--return_all_terms_with_user_defined_attributes", dest="return_all_terms_with_user_defined_attributes", default=None,
                        help="Returns a table with all the terms in the ontology with the desired fields given by user in CSV style, like: 'id,name,xref'", type=text_list)
    parser.add_argument('-p', "--processes", dest="processes", default=2, type=int,
                        help="Number of processes to parallelize calculations. Applied to: semantic similarity.")
    parser.add_argument("-d", "--download", dest="download", default=None,
                        help="Download obo file from an official resource. MONDO, GO, HPO, EFO, DO, CL and UBERON are possible values. Use 'all' to download all. Use 'list' to list available ontologies.")
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
              help="Calculate similarity between profile IDs computed by 'resnik', 'lin' or 'jiang_conrath' methods. Recently added 'eric', 'neric', 'nweric' and 'erlin' methods.")
    parser.add_argument("--reference_profiles", dest="reference_profiles", default= None, 
              help="Path to file tabulated file with first column as id profile and second column with ontology terms separated by separator.")
    parser.add_argument("--disable_cleaning_profiles", dest="clean_profiles", default= True, action='store_false', 
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
              help="Set to give a list of term attributes: term, translation, level and synonymns, for all terms of an ontology.")
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
    parser.add_argument("--output_report", dest="output_report", default=None,
              help="Use to create a short quality report with the chosen ontology and input profiles, to be saved at the path given")
    parser.add_argument("--root_term", dest="root_term", default=None,
              help="For the output report, it sets the root term to show in center of ontoplot")
    parser.add_argument("--ref_term", dest="ref_term", default=None,
              help="For the output report, it sets the term whose childs will be in the color legend to indicate the branches of descendants")
    parser.add_argument("--similarity_cluster_plot", dest="similarity_cluster_plot", default=None,
              help="For output report, use to activate profiles clustering and clustermap plot with a similarity method (resnik', 'lin' or 'jiang_conrath). Not active by default")
    parser.add_argument("--cl_size_factor", dest="cl_size_factor", default=1.0, type=float,
              help="When using dinamyc clustering weigths the contribution of the cluster size in tree cut. For smaller clusters use values > 1 for greater clusters use values < 1.")
    parser.add_argument("--ontoplot_mode", dest="ontoplot_mode", default='static',
              help="For the semtools report, it is used to set the mode of the ontoplot. Options: 'static','dynamic' or 'canvas'. Default is 'static'.")            
    parser.add_argument("--skip_count_parentals", dest="count_parentals", default=True, action='store_false',
              help="For the ontoplot, use it to deactivate propagating frequency to parentals terms")
    parser.add_argument("--onto_min_freq", dest="onto_min_freq", default=0.005, type=float,
              help="For the ontoplot, it sets the minimum frequency of terms to be shown in the plot. Default is 0.5 percent. Set to 0 to show all terms.") 
    opts =  parser.parse_args(args)
    main_semtools(opts)

def get_sorted_suggestions(args = None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Perform Ontology driven analysis with a network of relationships between terms')

    parser.add_argument('-q', "--query_terms", dest="query_terms", default= None,
            help="Path to the input file with 1 column format (each term in a new row) to be used as the query")
    parser.add_argument("-O", "--ontology_file", dest="ontology_file", default= None, 
            help="Path to ontology file")
    parser.add_argument('-r', "--term_relations", dest="term_relations", default= None,
            help="Path to the term-term pairs file. Expected three files (1º term, 2º term, relationship value)")
    parser.add_argument("-o", "--output_file", dest="output_file", default= None, 
            help="Path to the output table")

    parser.add_argument("-m", "--max_targets", dest="max_targets", default= 0, type = int, 
            help="Parameter to set the limit of targets terms to retrieve")
    parser.add_argument("-t", "--translate", dest="translate", default="c",
            help="Use if you want to be returned human readable names (use n), codes (use c), or both (use cn or nc depending on the order you want them to be returned)")
    parser.add_argument("-b", "--black_list", dest="black_list", default= None, 
            help="Path to a file with a list of target terms to be excluded from the analysis (one column format)")
    parser.add_argument("-d", "--deleted_terms", dest="deleted_terms", default= None, 
            help="Path to the folder to write the different deleted term files")

    parser.add_argument("-f", "--filter_parental_targets", dest="filter_parental_targets", default=False, action="store_true",
            help="Use if you want to filter out parental terms of the query terms present in the targets")
    parser.add_argument("-c", "--clean_query_terms", dest="clean_query_terms", default=False, action="store_true",
            help="Use if you want to filter out parental terms of the query terms present in the queries")

    parser.add_argument("-T", "--transform_rel_values", dest="transform_rel_values", default= "no",
            help="Use to apply a transformation to relation values. Currently accepted arguments: no (no transformation), log10 (-log10 transformation), exp (log-inverse exponential transformation, 10^(-x))")
    
    parser.add_argument("--output_report", dest="output_report", default= None, 
            help="Path to the output report file. If not used, no report will be generated")
    parser.add_argument("--heatmap_color_preset", dest="heatmap_color_preset", default= "1",
            help="Use to set the color preset for the heatmap. Options: '1' (default) or '2'. Each preset has a different color scheme.")

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
    parser.add_argument('-t', "--threshold", dest="threshold", default= 0, type = float,
            help="Similarity threshold to filter results to write")    
    parser.add_argument('-v', "--verbose", dest="verbose", default= False, action='store_true',
            help="Toogle on to get verbose output")
    parser.add_argument('-g', "--gpu_device", dest="gpu_device", default= [], type=text_list,
            help="Use to specify the GPU device to be used for speed-up (if available). The format is like: 'cuda:0' or 'cuda:0,cuda:1' to use multiple GPUs or cpu,cpu to use multiple CPUs")    
    parser.add_argument("-b", "--batch_size", dest="batch_size", default= 32, type=int,
            help="Use to specify batch size for multi-GPU embedding step") 
    parser.add_argument("--use_gpu_for_sim_calculation", dest="use_gpu_for_sim_calculation", default= False, action='store_true',
            help="Toogle on if you want to use GPU not only for the embedding process, but also for calculating query-corpus similarities (then dot score if used instead of cosine similarity)")
    parser.add_argument('-s', "--split", dest="split", default= False, action='store_true',
            help="Use it if your corpus comes splitted in smaller parts as list of lists (embedded in a json)")
    parser.add_argument("--order", dest="order", default= "corpus-query",
            help="Order of the semantic search. Options: 'corpus-query' or 'query-corpus'")
    parser.add_argument("--chunk_size", dest="chunk_size", default=10000, type=int,
            help="Size to be accumulating corpora until a threshold is reached before proceeding to embedd")
    parser.add_argument("--print_relevant_pairs", dest="print_relevant_pairs", default= False, action='store_true',
            help="Use it to print the relevant pairs of query-corpus with their scores")    
    opts =  parser.parse_args(args)
    main_stEngine(opts)

def stEngine_report(args = None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Perform Ontology driven analysis with Sentence Transformer')
    parser.add_argument("-O", "--ontology", dest="ontology", default= None,
                        help="Path to ontology file")
    parser.add_argument("-r", "--ref_profile", dest="ref_prof", default= None, 
                        type = lambda file: [line.strip() for line in open(file).readlines()],
                        help="Path to reference profile. One term code per line")
    parser.add_argument("-i", "--input_file", dest="input_file", default= None,
                        help="Input file with ids and profiles to compare with the reference profile")
    parser.add_argument("-d", "--id_col", dest="id_col", default= 0, type=int,
                        help="0-based position of the column with the id")
    parser.add_argument("-p", "--profiles_col", dest="ont_col", default= 1, type=int,
                        help="0-based position of the column with the Ontology terms")
    parser.add_argument("-S", "--separator", dest="separator", default='|',
                        help="Set which character must be used to split the terms profile. Default '|'")
    parser.add_argument("-o", "--output_file", dest="output_file", default= 'report.html',
                        help="Output report")
    parser.add_argument("-L", "--matrix_limits", dest="matrix_limits", default= [20, 40], type= lambda data: [int(i) for i in data.split(",")],
                        help="Number of rows and columns to show in heatmap defined as 'Nrows,Ncols'. Default 20,40")
    parser.add_argument("-N", "--neg_matrix_limits", dest="neg_matrix_limits", default= [20, 40], type= lambda data: [int(i) for i in data.split(",")],
                        help="Number of rows and columns to show in the negative heatmap defined as 'Nrows,Ncols'. Default 20,40")
    parser.add_argument("--hard_check", dest="hard_check", default= True, action="store_false",
                        help="Set to disable hard check cleaning. Default true") 
    parser.add_argument("--pubmed_ids_and_titles", dest="pubmed_ids_and_titles", default= None,
                        help="Path to pubmed_ids_and_titles file")
    parser.add_argument("--sim", dest="sim", default= None,
                        help="Similarity method to use")
    parser.add_argument("--use_pickle", dest="use_pickle", default= False, action="store_true",
                        help="Use it if you want to save (if first time)/use the cached version (if executed before) and just launch the report")                                        
    parser.add_argument("--get_full_sim_sorted_list", dest="get_full_sim_sorted_list", default= False, action="store_true",
                        help="Use it to get the full sorted list of similarities between the reference profile and the input profiles")      
    opts = parser.parse_args(args)
    main_stEngine_report(opts)

def get_corpus_index(args = None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Extract abstracts from pubmed xml files')

    parser.add_argument('-i', "--input", dest="input", default= None,
            help="Path to the pubmed input file in xml format (gzip compressed) to extract. Wildcards are accepted to process multiple files")
    parser.add_argument('-o', "--output", dest="output", default= None,
            help="Output path to save the extracted data (gzip compressed).")
    parser.add_argument('-k', "--items_per_file", dest="items_per_file", default= 0, type = int,
            help="Number of abstracts to be accumulated and saved in a file. If not used, the same input-output files will be used")
    parser.add_argument('-t', "--tag", dest="tag", default= "file_",
            help="If a items_per_file is used, this tag will be used to name the output files")
    parser.add_argument('-c', "--n_cpus", dest="n_cpus", default= 1, type = int,
            help="Number of cpus to be used for parallel processing")
    parser.add_argument('-z', "--chunk_size", dest="chunk_size", default= 0, type = int,
            help="Number of files to be processed be each cpu")
    parser.add_argument('-b', "--text_balancing_size", dest="text_balancing_size", default= 0, type = int,
            help="When white texts to disk, sort by text length and distribute them in packages of the given size. These packages are writed to the same file.")
    parser.add_argument('-d', '--debugging_mode', dest="debugging_mode", default=False, action='store_true',
            help="Activate to output stats about the content of the xml as warnings")
    parser.add_argument('-s', "--split", dest="split", default= False, action='store_true',
            help="Use it to split your text into smaller text units (then returned as a json)")
    parser.add_argument("--split_output_files", dest="split_output_files", default= False, action='store_true',
            help="Use this option to split output files with 'text_balancing_size' texts per file")
    parser.add_argument('-p', "--parse", dest="parse", default = None,
            help="'Basic' for raw text files. 'PubmedAbstract' for pubmed files with abstracs. 'PubmedPaper' for full paper Pubmed(PMC) files ")	
    parser.add_argument('-e', "--equivalences_file", dest="equivalences_file", default= None,
            help="Path to a 2 columns file with PMC-PMID equivalences to use when a parsed papers only finds the PMC ID inside its content.")
    parser.add_argument('-f', "--filter_by_blacklist", dest="filter_by_blacklist", default= None,
            help="Path to a single column file with blacklisted words to filter out documents whose titles contains these words")
    parser.add_argument("--blacklisted_mode", dest="blacklisted_mode", default= 'partial',
            help="When using 'filter_by_blacklist' use this option to choose between 'exact' or 'partial' match. Default is 'partial'")
    parser.add_argument("--clean_type", dest="clean_type", default= 'hard',
            help="Use to set the type of cleaning to be performed on the text. Options: basic (do not clean at all), soft or hard (default).")
    parser.add_argument("--split_type", dest="split_type", default= 'classic',
            help="Use to set the type of splitting to be performed on the text. Options: classic (default, splitting by '\n\n', \n', '.', ';', ',') or space_overlap ('\n\n', '\n', ' ', '' with overlap to avoid losing information).")
    parser.add_argument("--extra_fields", dest="extra_fields", default= defaultdict(lambda: False), type=text_to_default_dict,
            help="Comma-separated list of extra fields to include in the processed corpus. Available: title and/or keywords. Default: none")
    parser.add_argument("--remain_empty_documents", dest="remain_empty_documents", default= False, action='store_true',
            help="Use this option to keep entries with (main text) empty documents in the processed corpus (if they have at least title or keywords)")
    opts =  parser.parse_args(args)
    main_get_corpus_index(opts)


def get_sorted_profs(args=None):
    """
    Parses command-line arguments and calls the main function to get sorted profiles based on ontology term similarities.

    :param args: List of command-line arguments. If None, sys.argv[1:] is used.
    :type args: list or None

    :returns: None
    """
    if args == None: args = sys.argv[1:]
    parser = argparse.ArgumentParser(description=f'Usage: {inspect.stack()[0][3]} [options]')
    parser.add_argument("-i", "--input_file", dest="input_file", default= None,
                        help="Input file with ids and profiles")
    parser.add_argument("-d", "--pat_id_col", dest="subject_column", default= 0, type=int,
                        help="0-based position of the column with the subject id")    
    parser.add_argument("-p", "--term_col", dest="annotations_column", default= 1, type=int,
                        help="0-based position of the column with the ontology terms")    
    parser.add_argument("-S", "--terms_separator", dest="separator", default='|',
                        help="Set which character must be used to split the terms profile. Default '|'")
    parser.add_argument("-r", "--ref_profile", dest="ref_prof", default= None, type = one_column_file,
                        help="Path to reference profile. One term code per line. If not used, a general one will be made from the union of all external profiles")    
    parser.add_argument("-O", "--ontology_file", dest="ontology_file", default= None, 
                        help="Path to ontology file or Ontology name to use")   
    parser.add_argument("-X", "--excluded_terms", dest="excluded_terms", default= [], type = one_column_file,
                        help="File with excluded terms. One term code per line.")
    parser.add_argument("--hard_check", dest="hard_check", default= True, action="store_false",
                        help="Set to disable hard check cleaning. Default true")
    parser.add_argument("-s", "--similarity_method", dest="similarity", default= 'lin', 
                        help="Calculate similarity between profile IDs computed by 'resnik', 'lin' or 'jiang_conrath' methods. Recently added 'eric', 'neric', 'nweric' and 'erlin' methods.")    
    parser.add_argument("-o", "--output_file", dest="output_file", default= 'report.html',
                        help="Output report file")
    parser.add_argument("-f", "--general_prof_freq", dest="term_freq", default= 0, type= float,
                        help="When reference profile is not given, a general ine is computed with all profiles. If a freq is defined (0-1), all terms with freq minor than limit are removed")
    parser.add_argument("-L", "--matrix_limits", dest="matrix_limits", default= [20, 40], type= lambda data: [int(i) for i in data.split(",")],
                        help="Number of rows and columns to show in heatmap defined as 'Nrows,Ncols'. Default 20,40")
    parser.add_argument("--header_id", dest="header_id", default="HP", help="Header ID to use in plotted heatmaps")
    opts =  parser.parse_args(args)
    main_get_sorted_profs(opts)