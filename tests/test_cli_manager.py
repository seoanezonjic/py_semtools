import pytest, shutil, sys, os 
from io import StringIO
from py_semtools import Ontology, JsonParser
import py_semtools
from py_semtools.main_modules import expand_path
from py_cmdtabs import CmdTabs
ROOT_PATH=os.path.dirname(__file__)
ONTOLOGY_PATH = os.path.join(ROOT_PATH, 'data')
DATA_TEST_PATH = os.path.join(ONTOLOGY_PATH, 'input_scripts')
REF_DATA_PATH=os.path.join(ONTOLOGY_PATH ,'ref_output_scripts')
GET_SORTED_SUGG_PATH = os.path.join(ROOT_PATH, 'data', 'get_sorted_suggestions')
GET_SORTED_PROFS_PATH = os.path.join(ROOT_PATH, 'data', 'get_sorted_profs')
ST_ENGINE_REPORT_PATH = os.path.join(ROOT_PATH, 'data', 'st_engine_report')

#THIS IS NOT THE PROPER WAY TO DO IT, BUT SOMETHING IS FAILING AT TRYING TO GET site.USER_BASE ontologies from the group, ask PSZ later
HPO="/mnt/home/soft/soft_bio_267/programs/x86_64/python_cache/semtools/ontologies/HPO.obo" 

@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp("./tmp_output")
    return fn

@pytest.fixture
def enrichment_ontology():
    return os.path.join(ONTOLOGY_PATH, 'enrichment_ontology.obo')

@pytest.fixture
def enrichment_ontology_3():
    return os.path.join(ONTOLOGY_PATH, 'enrichment_ontology3.obo')

@pytest.fixture
def profiles():
    return os.path.join(DATA_TEST_PATH, 'profiles')

@pytest.fixture
def omim_profiles():
    return os.path.join(ROOT_PATH, "demo_examples", "data", "profiles.txt")

@pytest.fixture
def abs_profiles():
    return os.path.join(ST_ENGINE_REPORT_PATH, "abs_profiles.txt")

@pytest.fixture
def pmids_and_titles():
    return os.path.join(ST_ENGINE_REPORT_PATH, "pmids_and_titles.txt")

@pytest.fixture
def ref_profile():
    return os.path.join(ROOT_PATH, "demo_examples", "data", "ref_profile.txt")

def capture_stdout(func):
    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        tmpfile = StringIO()
        sys.stdout = tmpfile
        returned = func(*args, **kwargs)
        printed = sys.stdout.getvalue()
        sys.stdout = original_stdout
        return returned, printed
    return wrapper

@capture_stdout
def pysemtools(lsargs):
    return py_semtools.semtools(lsargs)

@capture_stdout
def pystrsimnet(lsargs):
    return py_semtools.strsimnet(lsargs)

def strng2table(strng, fs="\t", rs="\n"):
	table = [row.split(fs) for row in strng.split(rs)][0:-1]
	return table

def sort_table(table, sort_by, transposed=False):
	if transposed: 
		table = list(map(list, zip(*table)))
		table = sorted(table, key= lambda row: row[sort_by])
		table = list(map(list, zip(*table)))
	else:
		table = sorted(table, key= lambda row: row[sort_by])
	return table


################################################## TESTING SEMTOOLS BINARY #######################################################

## Terms Operations

def test_get_ancestors_descendants(enrichment_ontology):
    args = f"-C ranh2/branchAChild1,branchAChild2,branchB -O {enrichment_ontology}".split(" ")
    _, printed = pysemtools(args)
    test_result = strng2table(printed)
    expected_result = CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'parental_from_terms'))
    assert expected_result == test_result

    args = f"-C anh2/branchAChild1,branchAChild2,branchB -O {enrichment_ontology}".split(" ")
    _, printed = pysemtools(args)
    test_result = strng2table(printed)
    test_result = sort_table(test_result, 0)
    expected_result = [["All"],["Child1"]]
    assert expected_result == test_result

    args = f"-C branchAChild1,branchAChild2,branchB,branchA -O {enrichment_ontology}".split(" ")
    _, printed = pysemtools(args)
    test_result = strng2table(printed)
    test_result = sort_table(test_result, 0)
    expected_result = [['branchAChild1'], ['branchAChild2']]
    assert expected_result == test_result

## Profile operations
### Modification
@pytest.mark.parametrize("filter_type, filename", [
    (
        "'p(branchA)'", 
        "whitelisted.txt"
    ),
    (
        "'n(branchA)'", 
        "blacklisted.txt" 
    ),
    (
        "'p(branchA)n(branchAChild2)'", 
        "white_and_blacklisted.txt"
    )
])
def test_filter_profiles(tmp_dir, enrichment_ontology_3, filter_type, filename): # --list -F
    input_file_terms = os.path.join(DATA_TEST_PATH, 'terms_list')
    expected_file = CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, filename))
    
    args = f"-i {input_file_terms} --list -O {enrichment_ontology_3} -F {filter_type}".split(" ")
    _, printed = pysemtools(args)
    returned = [[item] for item in printed.strip().split("\n")]
    assert expected_file == returned

def test_clean_profiles(tmp_dir,enrichment_ontology,profiles): # -c -T
    removed_profile = os.path.join(tmp_dir, 'removed_profiles')
    output_file = os.path.join(tmp_dir, 'cleaned_profiles')
    args = f"-i {profiles} -c -T branchA -O {enrichment_ontology} -o {output_file} -r {removed_profile}".split(" ")
    pysemtools(args)
    test_result = CmdTabs.load_input_data(output_file)
    expected_result = CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'cleaned_profiles'))
    assert expected_result == test_result
    test_result = CmdTabs.load_input_data(removed_profile)
    expected_removed = [['P4']]
    assert expected_removed == test_result

    # Checking the --out2cols option
    output_file_out2cols = os.path.join(tmp_dir, 'cleaned_profiles_2cols')
    args = f"-i {profiles} -c -T branchA -O {enrichment_ontology} --out2cols -o {output_file_out2cols} -r {removed_profile}".split(" ")
    pysemtools(args)
    test_result = CmdTabs.load_input_data(output_file_out2cols)
    expected_result_out2cols = CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'cleaned_profiles_2cols'))
    assert test_result == expected_result_out2cols

    # Checking the --2cols option
    input_file = os.path.join(DATA_TEST_PATH, 'profiles_2cols')
    args = f"-i {input_file} -c -T branchA -O {enrichment_ontology} --2cols -o {output_file} -r {removed_profile}".split(" ")
    pysemtools(args)
    test_result = CmdTabs.load_input_data(output_file)
    assert expected_result == test_result

    # get profs with no removed profiles
    args = f"-i {profiles} -c -T root -O {enrichment_ontology} -o {output_file} -r {removed_profile}".split(" ")
    pysemtools(args)
    test_result = CmdTabs.load_input_data(output_file)
    expected =  [['P1', 'branchAChild1'], ['P2', 'branchB;branchAChild1'], ['P3', 'branchAChild1;branchAChild2'], ['P4', 'branchA'], ['P5', 'branchAChild1']]
    assert expected == test_result

def test_profile_expansion(tmp_dir,enrichment_ontology,profiles): 
    output_file = os.path.join(tmp_dir, 'expanded_profiles')
    args = f"-i {profiles} -O {enrichment_ontology} -e parental -o {output_file}".split(" ")
    pysemtools(args)
    test_result =  CmdTabs.load_input_data(output_file)
    expected_result =  CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'expanded_profiles'))
    assert expected_result == test_result

def test_translate(tmp_dir,enrichment_ontology,profiles):         
    input_file_terms = os.path.join(DATA_TEST_PATH, 'terms')
    output_file = os.path.join(tmp_dir, 'translated_profiles')
    output_file_codes = os.path.join(tmp_dir, 'translated_profiles_codes')
    
    args = f"-i {profiles} -O {enrichment_ontology} -t names -o {output_file}".split(" ")
    pysemtools(args)
    test_result =  CmdTabs.load_input_data(output_file)
    expected_result =  CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'translated_profiles_names'))
    assert expected_result == test_result

    args = f"-i {output_file} -O {enrichment_ontology} -t codes -o {output_file_codes}".split(" ")
    pysemtools(args)
    test_result =  CmdTabs.load_input_data(output_file_codes)
    expected_result =  CmdTabs.load_input_data(profiles)
    assert expected_result == test_result

    args = f"-i {input_file_terms} -O {enrichment_ontology} -l codes".split(" ")
    _, printed = pysemtools(args)
    test_result = strng2table(printed)
    expected_result =  CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'translated_terms_codes'))
    assert expected_result == test_result

### Analysis
def test_get_ic(tmp_dir,enrichment_ontology,profiles): # -I
    args = f"-i {profiles} -O {enrichment_ontology} -I prof".split(" ")
    pysemtools(args)
    test_result =  CmdTabs.load_input_data("./profiles_IC_onto_freq")
    os.remove("./profiles_IC_onto_freq")
    expected_result =  CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'profiles_IC_onto_freq'))
    assert test_result == expected_result

    args = f"-O {enrichment_ontology} -I ont".split(" ")
    pysemtools(args)
    test_result =  CmdTabs.load_input_data("./ont_IC")
    os.remove("./ont_IC")
    expected_result = CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'expected_IC_ont'))
    assert test_result == expected_result

def test_statistics_profiler(enrichment_ontology,profiles): # -n    
    args = f"-i {profiles} -O {enrichment_ontology} -n".split(" ")
    _, printed = pysemtools(args)
    test_result =  strng2table(printed)
    expected_result =  CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'profile_stats'))
    assert test_result == expected_result

def test_semantic_similarity(tmp_dir,enrichment_ontology,profiles): 
    output_file = os.path.join(tmp_dir, 'similarity_profiles')
    args = f"-i {profiles} -O {enrichment_ontology} -o {output_file} -s lin".split(" ")
    pysemtools(args)
    test_result =  CmdTabs.load_input_data(output_file)
    expected_result =  [['P5', 'P1', '0.6204627667845211'], ['P5', 'P2', '0.5'], ['P5', 'P3', '0.5793484513785037'],
                        ['P5', 'P4', '0.48185106713808395'], ['P5', 'P5', '1.0'], ['P1', 'P5', '0.6204627667845211'],
                        ['P1', 'P4', '0.6204627667845211'], ['P1', 'P3', '0.8272836890460279'], ['P1', 'P2', '0.7469751778563474'],
                        ['P1', 'P1', '1.0'], ['P2', 'P5', '0.5'], ['P2', 'P4', '0.24092553356904198'], 
                        ['P2', 'P3', '0.7195656342523358'], ['P2', 'P1', '0.7469751778563474'], ['P2', 'P2', '1.0'],
                        ['P3', 'P5', '0.5793484513785037'], ['P3', 'P4', '0.36138830035356295'], ['P3', 'P1', '0.8272836890460279'],
                        ['P3', 'P2', '0.7195656342523358'], ['P3', 'P3', '1.0'], ['P4', 'P5', '0.48185106713808395'],
                        ['P4', 'P1', '0.6204627667845211'], ['P4', 'P2', '0.24092553356904198'], ['P4', 'P3', '0.36138830035356295'], ['P4', 'P4', '1.0']]
    assert test_result == expected_result

    # With reference profiles
    reference_file = os.path.join(DATA_TEST_PATH, 'profiles_with_removedTerms')
    args = f"-i {profiles} -O {enrichment_ontology} -o {output_file} -s lin --reference_profiles {reference_file}".split(" ")
    pysemtools(args)
    test_result =  CmdTabs.load_input_data(output_file)
    expected_result = [['P1', 'P1', '1.0'], ['P1', 'P2', '0.7469751778563474'], ['P1', 'P3', '0.8272836890460279'], 
    ['P2', 'P1', '0.7469751778563474'], ['P2', 'P2', '1.0'], ['P2', 'P3', '0.7195656342523358'], 
    ['P3', 'P1', '0.8272836890460279'], ['P3', 'P2', '0.7195656342523358'], ['P3', 'P3', '1.0'],
    ['P4', 'P1', '0.6204627667845211'], ['P4', 'P2', '0.24092553356904198'], ['P4', 'P3', '0.36138830035356295'], 
    ['P5', 'P1', '0.6204627667845211'], ['P5', 'P2', '0.5'], ['P5', 'P3', '0.5793484513785037']]
    assert expected_result == test_result

def test_xref(tmp_dir,enrichment_ontology,profiles):
    output_file = os.path.join(tmp_dir, 'xref_profile')
    input_file = os.path.join(DATA_TEST_PATH, 'terms_for_xref')
    
    args = f"-i {profiles} -O {enrichment_ontology} -o {output_file} --xref_sense -k wikipedia".split(" ")
    pysemtools(args)
    test_result = CmdTabs.load_input_data(output_file)
    expected_result = [['P1', 'wikipedia'], ['P2', 'wikipedia'], ['P3', 'wikipedia'], ['P5', 'wikipedia']]
    assert expected_result == test_result

    args = f"-i {input_file} --list -O {enrichment_ontology} -o {output_file} --xref_sense -k wikipedia".split(" ")
    pysemtools(args)
    test_result = CmdTabs.load_input_data(output_file)
    expected_result = [['branchAChild1', 'wikipedia']]
    assert test_result == expected_result

def test_download(tmp_dir, profiles):
    @capture_stdout
    def script2test(lsargs):
        with pytest.raises(SystemExit):
            return py_semtools.semtools(lsargs)
        
    output_file = os.path.join(tmp_dir, 'downloaded_ontology')
    args = f"-d HPO -o {output_file}".split(" ")
    script2test(args)
    assert os.path.getsize(output_file) > 0

    download_args = f"-d GO".split(" ")
    removed_profile = os.path.join(tmp_dir, 'removed_profiles')
    get_ont_args = f"-i {profiles} -c -O GO -r {removed_profile}".split(" ")
    script2test(download_args)# Talk with PSZ
    pysemtools(get_ont_args)
    test_result = CmdTabs.load_input_data(removed_profile)
    expected_result = [['P1'], ['P2'], ['P3'], ['P4'], ['P5']]
    assert expected_result == test_result

 # Testing list term attributes

def test_list_term_attributes(enrichment_ontology):
    args = f"-O {enrichment_ontology} --list_term_attributes".split(" ")
    _, printed = pysemtools(args)
    test_result = strng2table(printed)
    expected_result = CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'terms_attr'))
    assert test_result == expected_result
    
# Testing the semtools keyword based search

def test_retrieve_keyword_matched_hps(enrichment_ontology):
    args = f"-O {enrichment_ontology} --keyword_search child|name,synonym".split(" ")
    expected = "\n".join(["branchA", "branchAChild1", "branchAChild2", "branchB\n"])
    _, printed = pysemtools(args)
    assert expected == printed

    args.extend(["--translate_keyword_search"])
    expected = "\n".join(["Child1", "Child2" ,"Child4" ,"Child5\n"])
    _, printed = pysemtools(args)
    assert expected == printed

# Testing semtools quality report

def test_get_report(tmp_dir, enrichment_ontology_3, profiles):
    tmp_dir = "./tests/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    output_file = os.path.join(tmp_dir, 'quality_report.html')
    
    root = "root"
    args = f"-i {profiles} -O {enrichment_ontology_3} --output_report {output_file} --similarity_cluster_plot lin --root_term {root} --ref_term {root}".split(" ")
    pysemtools(args)
    assert os.path.exists(output_file)
    shutil.rmtree(output_file, ignore_errors=True)

################################################## TESTING STRSINMNET BINARY #######################################################

def test_strsimnet(tmp_dir):
    input_file = os.path.join(DATA_TEST_PATH, 'string_values')
    output_file1 = os.path.join(tmp_dir, 'strsimnet')
    args1 = f"-i {input_file} -c 0 -o {output_file1}".split(" ")
    output_file2 = os.path.join(tmp_dir, 'strsimnet_with_filter')
    args2 = f"-i {input_file} -c 0 -C 1 -f 2 -o {output_file2}".split(" ")
    
    pystrsimnet(args1)
    test_result = CmdTabs.load_input_data(output_file1)
    expected_result = CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'strsimnet'))
    assert expected_result == test_result

    pystrsimnet(args2)
    test_result = CmdTabs.load_input_data(output_file2)
    expected_result = CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'strsimnet_cutoff2'))
    assert expected_result == test_result

################################################## TESTING GET_SORTED_SUGGESTIONS BINARY #######################################################

def test_get_sorted_suggestions():
    relations_file = os.path.join(GET_SORTED_SUGG_PATH, 'input_data', 'relations.txt')
    query_hps_file = os.path.join(GET_SORTED_SUGG_PATH, 'input_data', 'query_hps.txt')
    ontology_file =  os.path.join(GET_SORTED_SUGG_PATH, 'input_data', 'enrichment_ontology3.obo')
    os.makedirs(os.path.join(GET_SORTED_SUGG_PATH, 'returned'), exist_ok=True)

    #Asserting base case without filter neither limits
    returned_file_no_filter_no_limit = os.path.join(GET_SORTED_SUGG_PATH, 'returned', 'no_filter_no_limit.txt')
    args = f"-q {query_hps_file} -r {relations_file} -O {ontology_file} -o {returned_file_no_filter_no_limit}".split()

    py_semtools.get_sorted_suggestions(args)
    expected = CmdTabs.load_input_data(os.path.join(GET_SORTED_SUGG_PATH, 'expected', 'no_filter_no_limit.txt'))
    returned = CmdTabs.load_input_data(returned_file_no_filter_no_limit)
    assert expected == returned

    #Asserting case without filter but limit to 2 targets
    returned_file_no_filter_limit_2 = os.path.join(GET_SORTED_SUGG_PATH, 'returned', 'no_filter_limit_2.txt')
    args2 = f"-q {query_hps_file} -r {relations_file} -O {ontology_file} -o {returned_file_no_filter_limit_2} --max_targets 2".split(" ")

    py_semtools.get_sorted_suggestions(args2)
    expected = CmdTabs.load_input_data(os.path.join(GET_SORTED_SUGG_PATH, 'expected', 'no_filter_limit_2.txt'))
    returned = CmdTabs.load_input_data(returned_file_no_filter_limit_2)
    assert expected == returned


    #Asserting case with target parentals filter
    returned_file_filter_target_parentals = os.path.join(GET_SORTED_SUGG_PATH, 'returned', 'filter_target_parentals.txt')
    args3 = f"-q {query_hps_file} -r {relations_file} -O {ontology_file} -o {returned_file_filter_target_parentals} -f"
    args_list3 = args3.split(" ")

    py_semtools.get_sorted_suggestions(args_list3)
    expected = CmdTabs.load_input_data(os.path.join(GET_SORTED_SUGG_PATH, 'expected', 'filter_target_parentals.txt'))
    returned = CmdTabs.load_input_data(returned_file_filter_target_parentals)
    assert expected == returned

    #Asserting case with query parentals filter
    returned_file_filter_query_parentals = os.path.join(GET_SORTED_SUGG_PATH, 'returned', 'filter_query_parentals.txt')
    args4 = f"-q {query_hps_file} -r {relations_file} -O {ontology_file} -o {returned_file_filter_query_parentals} -c"
    args_list4 = args4.split(" ")

    py_semtools.get_sorted_suggestions(args_list4)
    expected = CmdTabs.load_input_data(os.path.join(GET_SORTED_SUGG_PATH, 'expected', 'filter_query_parentals.txt'))
    returned = CmdTabs.load_input_data(returned_file_filter_query_parentals)
    assert expected == returned

    #Asserting case with both query and target parentals filter
    returned_file_filter_both_parentals = os.path.join(GET_SORTED_SUGG_PATH, 'returned', 'filter_target_and_query_parentals.txt')
    args5 = f"-q {query_hps_file} -r {relations_file} -O {ontology_file} -o {returned_file_filter_both_parentals} -f -c"
    args_list5 = args5.split(" ")

    py_semtools.get_sorted_suggestions(args_list5)
    expected = CmdTabs.load_input_data(os.path.join(GET_SORTED_SUGG_PATH, 'expected', 'filter_target_and_query_parentals.txt'))
    returned = CmdTabs.load_input_data(returned_file_filter_both_parentals)
    assert expected == returned

    #Asserting that html report is generated when using the --ouput_report option
    tmp_dir = "./tests/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    output_file = os.path.join(tmp_dir, 'sugg_report.html')
    
    returned_file_filter_both_parentals = os.path.join(GET_SORTED_SUGG_PATH, 'returned', 'filter_target_and_query_parentals.txt')
    args6 = f"-q {query_hps_file} -r {relations_file} -O {ontology_file} -o {returned_file_filter_both_parentals} -f -c --output_report {output_file}"
    args_list6 = args6.split(" ")
    py_semtools.get_sorted_suggestions(args_list6)
    assert os.path.exists(output_file)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    #Remove the returned files
    path_to_remove_files = os.path.join(GET_SORTED_SUGG_PATH, 'returned')
    for file in os.listdir(path_to_remove_files):
        os.remove(os.path.join(path_to_remove_files, file))


################################################## TESTING GET_SORTED_PROFS BINARY #######################################################

def test_get_sorted_profs(omim_profiles, ref_profile):
    os.makedirs(f"{os.path.join(GET_SORTED_PROFS_PATH)}", exist_ok=True)
    list_of_args = ["-i", omim_profiles,
                    "-r", ref_profile, 
                    "-S", ",", 
                    "-O", HPO,
                    "-o", f"{os.path.join(GET_SORTED_PROFS_PATH, 'report.html')}"]
    
    #Testing to check if calculations can be executed and report done without errors. Just checking that file exist
    py_semtools.get_sorted_profs(list_of_args)
    assert os.path.exists(f"{os.path.join(GET_SORTED_PROFS_PATH, 'report.html')}")
    assert os.path.exists(f"{os.path.join(GET_SORTED_PROFS_PATH, 'report.txt')}")

    shutil.rmtree(f"{os.path.join(GET_SORTED_PROFS_PATH)}")


################################################## TESTING ST_ENGINE_REPORT BINARY #######################################################

def test_stEngine_report(abs_profiles, pmids_and_titles, ref_profile):
    tmp_dir = "./tests/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    output_file = os.path.join(tmp_dir, 'report.html')
    args = f"-r {ref_profile} -i {abs_profiles} -d 0 -p 1 -S , -O {HPO} -L 35,50 -N 20,50 -o {output_file} --pubmed_ids_and_titles {pmids_and_titles} --sim nweric --get_full_sim_sorted_list --use_pickle"
    list_of_args = args.split(" ")
    print(list_of_args)

    #Testing to check if calculations can be executed and report done without errors. Just checking that file exist
    py_semtools.stEngine_report(list_of_args)

    assert os.path.exists(output_file)
    assert os.path.exists(output_file.replace('.html', '_sims.txt'))
    assert os.path.exists(output_file.replace('.html', '/tmp/similitudes.pckl'))

    shutil.rmtree(tmp_dir, ignore_errors=True)



################################################## TESTING AUXILIARY FUNCTIONS #######################################################

def test_expand_path():
    #Testing a path that matches several files
    path_with_content = os.path.join(ROOT_PATH, 'data', 'st_engine_report')
    expected_files = [os.path.join(path_with_content, f) for f in ["abs_profiles.txt", "pmids_and_titles.txt"]]
    returned_files = expand_path(os.path.join(path_with_content, '*'))
    assert set(expected_files) == set(returned_files)

    #Testing a path with no matching files
    path_with_no_content = os.path.join(ROOT_PATH, 'tmp', '*')
    with pytest.raises(Exception) as excinfo:
        expand_path(path_with_no_content)
    assert str(excinfo.value) == f"Path {path_with_no_content} does not match any file"