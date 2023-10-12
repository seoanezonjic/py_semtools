import pytest
import sys
import os 
from io import StringIO
from py_semtools import Ontology, JsonParser
import py_semtools
#from py_cmdtabs import CmdTabs
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

# semtools script
## Terms Operations

def test_translate_terms():
    pass 

def test_get_ancestors_descendants():
    pass 

def load_input_data(input_path, sep="\t", limit=-1, first_only=False):
	if limit > 0: # THis is due to ruby compute de cuts in other way and this fix enables ruby mode. Think if adapt to python way
		limit -= 1
	if input_path == '-':
		input_data = sys.stdin
	else:
		file = open(input_path, "r")
		input_data = file.readlines()
		file.close()
	input_data_arr = []
	for line in input_data:
		#print('---', file=sys.stderr)
		#print(repr(), file=sys.stderr)
		input_data_arr.append(line.rstrip().split(sep, limit))
		if first_only:
			break
	return input_data_arr

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
    _, printed = script2test(args)
    test_result = load_input_data(output_file)
    expected_result = load_input_data(os.path.join(REF_DATA_PATH, 'cleaned_profiles'))
    assert expected_result == test_result
    

def test_profile_expansion(): # 
    pass

def test_translate():
    pass

### Analysis
def test_get_ic(): # -I
    pass

def test_statistics_profiler(): # -n
    pass

def test_semantic_similarity(): # -s
    pass