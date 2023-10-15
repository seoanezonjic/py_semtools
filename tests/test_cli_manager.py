import pytest
import sys
import os 
from io import StringIO
from py_semtools import Ontology, JsonParser
import py_semtools
from py_cmdtabs import CmdTabs
ROOT_PATH=os.path.dirname(__file__)
ONTOLOGY_PATH = os.path.join(ROOT_PATH, 'data')
DATA_TEST_PATH = os.path.join(ONTOLOGY_PATH, 'input_scripts')
REF_DATA_PATH=os.path.join(ONTOLOGY_PATH ,'ref_output_scripts')


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp("./tmp_output")
    return fn

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

## Terms Operations

def test_get_ancestors_descendants():
    input_file = os.path.join(DATA_TEST_PATH, 'profiles')
    ontology_file = os.path.join(ONTOLOGY_PATH, 'enrichment_ontology.obo')
    args = f"-C ranh2/branchAChild1,branchAChild2,branchB -O {ontology_file}".split(" ")
    @capture_stdout
    def script2test(lsargs):
        return py_semtools.semtools(lsargs)
    _, printed = script2test(args)
    print(printed)
    test_result = strng2table(printed)
    expected_result = CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'parental_from_terms'))
    assert expected_result == test_result

## Profile operations
### Modification
def test_clean_profiles(tmp_dir): # -c -T
    input_file = os.path.join(DATA_TEST_PATH, 'profiles')
    ontology_file = os.path.join(ONTOLOGY_PATH, 'enrichment_ontology.obo')
    output_file = os.path.join(tmp_dir, 'cleaned_profiles')
    args = f"-i {input_file} -c -T branchA -O {ontology_file} -o {output_file}".split(" ")
    @capture_stdout
    def script2test(lsargs):
        return py_semtools.semtools(lsargs)
    script2test(args)
    test_result = CmdTabs.load_input_data(output_file)
    expected_result = CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'cleaned_profiles'))
    os.remove("./rejected_profs")
    assert expected_result == test_result
    
def test_profile_expansion(tmp_dir): 
    input_file = os.path.join(DATA_TEST_PATH, 'profiles')
    ontology_file = os.path.join(ONTOLOGY_PATH, 'enrichment_ontology.obo')
    output_file = os.path.join(tmp_dir, 'expanded_profiles')
    args = f"-i {input_file} -O {ontology_file} -e parental -o {output_file}".split(" ")
    @capture_stdout
    def script2test(lsargs):
        return py_semtools.semtools(lsargs)
    _, printed = script2test(args)
    test_result =  CmdTabs.load_input_data(output_file)
    expected_result =  CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'expanded_profiles'))
    assert expected_result == test_result

def test_translate(tmp_dir): 
    input_file = os.path.join(DATA_TEST_PATH, 'profiles')
    input_file_terms = os.path.join(DATA_TEST_PATH, 'terms')
    ontology_file = os.path.join(ONTOLOGY_PATH, 'enrichment_ontology.obo')
    output_file = os.path.join(tmp_dir, 'translated_profiles')
    output_file_codes = os.path.join(tmp_dir, 'translated_profiles_codes')
    @capture_stdout
    def script2test(lsargs):
        return py_semtools.semtools(lsargs)
    
    args = f"-i {input_file} -O {ontology_file} -t names -o {output_file}".split(" ")
    _, printed = script2test(args)
    test_result =  CmdTabs.load_input_data(output_file)
    expected_result =  CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'translated_profiles_names'))
    assert expected_result == test_result

    args = f"-i {output_file} -O {ontology_file} -t codes -o {output_file_codes}".split(" ")
    _, printed = script2test(args)
    test_result =  CmdTabs.load_input_data(output_file_codes)
    expected_result =  CmdTabs.load_input_data(input_file)
    assert expected_result == test_result

# Talk with PSZ the exit()
"""     args = f"-i {input_file_terms} -O {ontology_file} -l codes -o {output_file_codes}".split(" ")
    _, printed = script2test(args)
    test_result = strng2table(printed)
    expected_result =  CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'translated_terms_codes'))
    assert expected_result == test_result """

### Analysis
def test_get_ic(tmp_dir): # -I
    input_file = os.path.join(DATA_TEST_PATH, 'profiles')
    ontology_file = os.path.join(ONTOLOGY_PATH, 'enrichment_ontology.obo')
    args = f"-i {input_file} -O {ontology_file} -I prof".split(" ")
    @capture_stdout
    def script2test(lsargs):
        return py_semtools.semtools(lsargs)
    _, printed = script2test(args)
    test_result =  CmdTabs.load_input_data("./profiles_IC_onto_freq")
    os.remove("./profiles_IC_onto_freq")
    expected_result =  CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'profiles_IC_onto_freq'))
    assert test_result == expected_result

def test_statistics_profiler(): # -n    
    input_file = os.path.join(DATA_TEST_PATH, 'profiles')
    ontology_file = os.path.join(ONTOLOGY_PATH, 'enrichment_ontology.obo')
    args = f"-i {input_file} -O {ontology_file} -n".split(" ")
    @capture_stdout
    def script2test(lsargs):
        return py_semtools.semtools(lsargs)
    _, printed = script2test(args)
    test_result =  strng2table(printed)
    expected_result =  CmdTabs.load_input_data(os.path.join(REF_DATA_PATH, 'profile_stats'))
    assert test_result == expected_result

def test_semantic_similarity(tmp_dir): 
    input_file = os.path.join(DATA_TEST_PATH, 'profiles')
    output_file = os.path.join(tmp_dir, 'similarity_profiles')
    ontology_file = os.path.join(ONTOLOGY_PATH, 'enrichment_ontology.obo')
    args = f"-i {input_file} -O {ontology_file} -o {output_file} -s lin".split(" ")
    @capture_stdout
    def script2test(lsargs):
        return py_semtools.semtools(lsargs)
    _, printed = script2test(args)
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