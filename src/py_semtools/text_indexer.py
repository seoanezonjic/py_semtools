import sys
import os
import glob
import warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from py_exp_calc.exp_calc import invert_nested_hash, flatten
import json
import gzip

from py_semtools.parallelizer import Parallelizer
from py_semtools.text_pubmed_paper_parser import TextPubmedPaperParser

class TextIndexer:

    @classmethod
    def build_index(cls, options):
        filenames = glob.glob(options["input"])
        if len(filenames) == 0: 
            sys.stderr.write(f"ERROR: {options['input']} has not files\n")
            sys.exit(1)

        if options["n_cpus"] == 1:
            from loguru import logger
            logger.add(f"./logs/main.log", format="{level} : {time} : {message}: {process}")
            logger.info("Starting process")
            cls.process_files(options, filenames, 0, logger = logger)
            logger.success("Process finished succesfully")            
        else:
            manager = Parallelizer(options["n_cpus"], options["chunk_size"])
            if options["items_per_file"] == 0:
                items = [[cls.process_files, [[options, [filename], None], {}]] for filename in filenames] # filename is a list to assign a one item list to a worker (worker per file)
            else:
                chunks = manager.get_chunks(filenames)
                items = [[cls.process_files, [[options, chunk, idx], {}]] for idx,chunk in enumerate(chunks)]
            manager.execute(items)

    @classmethod
    def process_files(cls, options, filenames, sup_counter, logger = None):
        accumulated_texts = []
        counter = 0
        for filename in filenames:    
          text_index = cls.get_index(filename, options, logger = logger)
          if options["items_per_file"] == 0:
            basename = os.path.basename(filename).replace(".xml.gz", "").replace(".tar.gz", "")
            cls.write_texts(text_index, options["output"], basename)
          else:
              accumulated_texts.extend(text_index)
              while len(accumulated_texts) >= options["items_per_file"]:
                texts2save = [accumulated_texts.pop() for _times in range(options["items_per_file"])]
                cls.write_texts(texts2save, options["output"], options["tag"], a_suffix=sup_counter, b_suffix=counter)
                counter += 1    
        # For records saved in accumulated_texts that the loop has not writed
        cls.write_texts(accumulated_texts, options["output"], options["tag"], a_suffix=sup_counter, b_suffix=counter)

    @classmethod
    def write_texts(cls, texts, folder, name, a_suffix=None, b_suffix=None):
        if a_suffix == None: a_suffix = ""
        if b_suffix != None: 
            f"_{b_suffix}"
        else:
            b_suffix = ""
        out_filename = os.path.join(folder, name+f"{a_suffix}{b_suffix}.gz" )
        if len(texts) > 0:
          with gzip.open(out_filename, 'wt') as f:
            for pmid, text, original_filename, year, text_length, number_of_sentences, length_of_sentences, title, article_type, article_category in texts:
              f.write(f"{pmid}\t{text}\t{original_filename}\t{year}\t{text_length}\t{number_of_sentences}\t{length_of_sentences}\t{title}\t{article_type}\t{article_category}\n")

    @classmethod
    def get_index(cls, file, options, logger = None):
        if options["parse"] == 'PubmedPaper':
            return cls.get_paper_index(file, options, logger = logger)
        elif options["parse"] == 'PubmedAbstract':
            return cls.get_abstract_index(file, options, logger = logger)


    @classmethod
    def get_abstract_index(cls, file, options, logger = None): 
        texts = [] # aggregate all abstracts in XML file
        parsed_texts = TextPubmedAbstractParser.parse(file)
        for parsed_text in parsed_texts:
            pmid, text, year, title, article_type, article_category = parsed_text
            if pmid != None and text != "":
                pmid_content_and_stats = cls.prepare_indexes(text, pmid, file, year, title, article_type, article_category, options)
                texts.append(pmid_content_and_stats)

        if logger != None: logger.warning(f"stats:file={file},total={stats['total']},no_abstract={stats['no_abstract']},no_pmid={stats['no_pmid']}")
        return texts

    @classmethod
    def get_paper_index(cls, file_path, options, logger = None):   
        file_exist = "does exist" if os.path.exists(file_path) else "does not exist"
        if logger != None: logger.info(f"The file {file_path} {file_exist}")
        parsed_texts, stats = TextPubmedPaperParser.parse(file_path, logger= logger)
        
        PMC_PMID_dict = None
        if options["equivalences_file"] != None: PMC_PMID_dict = dict(CmdTabs.load_input_data(options["equivalences_file"]))
        texts = [] # aggregate all papers in XML file (Technically it is just one for each xml file, but it is emulating the original abstract part logic)
        for parsed_text in parsed_texts:
            pmid, pmc, filename, year, whole_content, title, article_type, article_category = parsed_text
            if pmid == None and PMC_PMID_dict != None: pmid = PMC_PMID_dict.get(pmc)

            if pmid != None and whole_content != "": # TODO: Check if always there is al least pmc and change to PMC the indexing
                #pmid_content_and_stats = cls.prepare_indexes(whole_content, pmc+"-"+pmid, filename, year, options)
                pmid_content_and_stats = cls.prepare_indexes(whole_content, pmid, filename, year, title, article_type, article_category, options)
                texts.append(pmid_content_and_stats)

        if logger != None: logger.warning(f"stats:file={file_path},total={stats['total']},no_abstract={stats['no_abstract']},no_pmid={stats['no_pmid']}")
        if logger != None: logger.warning(f"logs_errors:file={file_path},errors_number={stats['errors']}")
        return texts


    @classmethod
    def prepare_indexes(cls, text, pmid, file, year, title, article_type, article_category, options):
        pmid = pmid.replace("\n", "")
        file = file.replace("\n", "")
        year = str(year).replace("\n", "")

        abstract_length = str(len(text))
        if options["split"]:
          abstract_parts = cls.split_abstract(text, pmid)
          flattened_abstract = flatten(abstract_parts)
          number_of_sentences = str(len(flattened_abstract))
          length_of_sentences = ",".join([str(len(sentence)) for sentence in flattened_abstract])
          abstract_parts_json = json.dumps(abstract_parts)
          prepared_index = [pmid, abstract_parts_json, file, year, abstract_length, number_of_sentences, length_of_sentences, 
                            title, article_type, article_category] 
        else:
          cleaned_abstract = text.strip().strip().replace("\r", "\n").replace("&#13", "\n").replace("\t", " ").replace("\n", " ")
          prepared_index = [pmid, cleaned_abstract, file, year, abstract_length, 1, abstract_length, 
                            title, article_type, article_category]
        return prepared_index



    @classmethod
    def split_abstract(cls, text, pmid):
        #paragraph_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap  = 20, length_function = len, separators=[r"\n\n", r"\.\n?"], keep_separator=False, is_separator_regex=True)
        paragraph_splitter = RecursiveCharacterTextSplitter(chunk_size = 10, chunk_overlap  = 0, length_function = len, separators=["\n\n"], keep_separator=False)
        sentences_splitter = RecursiveCharacterTextSplitter(chunk_size = 10, chunk_overlap  = 0, length_function = len, separators=["\n",".", ";", ","], keep_separator=False, is_separator_regex=False)
        #sentences_splitter = RecursiveCharacterTextSplitter(chunk_size = 120, chunk_overlap  = 20, length_function = len, separators=["\n", " ", ""], keep_separator=False, is_separator_regex=False)
        paragraphs = paragraph_splitter.split_text(text)
        sentences = [ sentences_splitter.split_text(paragraph.replace("\n", "")) for paragraph in paragraphs ]
        
        formated = []       
        for paragraph in sentences:
            formated_paragraph = [] 
            for sentence in paragraph:
                if len(sentence) > 2:
                    if len(sentence) > 3000: warnings.warn(f"ERROR: The pmid {pmid} has an unusual sentence lenght even after splitting. Total length of characters: {len(sentence)}. Content: {sentence}")
                    else: formated_paragraph.append(sentence)
            if len(formated_paragraph) > 0: formated.append(formated_paragraph)
        
        return formated