from collections import defaultdict
from pydoc import text
import sys, os, glob, re, warnings, json, gzip
from langchain_text_splitters import RecursiveCharacterTextSplitter
from py_exp_calc.exp_calc import invert_nested_hash, flatten

from py_cmdtabs import CmdTabs
from py_semtools.parallelizer import Parallelizer
from py_semtools.parsers.text.text_basic_parser import TextBasicParser
from py_semtools.parsers.text.text_pubmed_abstract_parser import TextPubmedAbstractParser
from py_semtools.parsers.text.text_pubmed_paper_parser import TextPubmedPaperParser
from py_semtools.parsers.text.text_pubmed_parser import TextPubmedParser

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
                items = [[[options, [filename], None], {}] for filename in filenames] # filename is a list to assign a one item list to a worker (worker per file)
            else:
                chunks = manager.get_chunks(filenames, workload_balance='disperse_max', workload_function= lambda filename: os.stat(filename).st_size)
                items = [[[options, chunk, idx], {}] for idx,chunk in enumerate(chunks)]
            res_status= manager.execute(items, cls.process_files)
            if set(list(res_status)) == {True}: 
                print("All processes finished succesfully")
            else:
                print("Some processes finished with errors")

    @classmethod
    def process_files(cls, options, filenames, sup_counter, logger = None):
        accumulated_texts = []
        if options['text_balancing_size'] == 0:
            balancer = None
        else:
            balancer = Parallelizer(1, options['text_balancing_size'])
        counter = 0
        for filename in filenames:    
          text_index = cls.get_index(filename, options, logger = logger)
          if options["items_per_file"] == 0:
            basename = os.path.basename(filename).replace(".xml.gz", "").replace(".tar.gz", "")
            cls.write_indexes(text_index, options["output"], basename, balancer = balancer, split_output_files = options["split_output_files"], items_per_file = options["text_balancing_size"])
          else:
              accumulated_texts.extend(text_index)
              while len(accumulated_texts) >= options["items_per_file"]:
                texts2save = [accumulated_texts.pop() for _times in range(options["items_per_file"])]
                cls.write_indexes(texts2save, options["output"], options["tag"], a_suffix=sup_counter, b_suffix=counter, balancer = balancer, split_output_files = options["split_output_files"], items_per_file = options["text_balancing_size"])
                counter += 1    
        # For records saved in accumulated_texts that the loop has not writed
        cls.write_indexes(accumulated_texts, options["output"], options["tag"], a_suffix=sup_counter, b_suffix=counter, balancer = balancer, split_output_files = options["split_output_files"], items_per_file = options["text_balancing_size"])
        return True

    @classmethod
    def write_indexes(cls, indexes, folder, name, a_suffix=None, b_suffix=None, balancer = None, split_output_files = False, items_per_file = 0):
        a_suffix = "" if a_suffix == None else f"_{a_suffix}"
        b_suffix = "" if b_suffix == None else f"_{b_suffix}"

        if len(indexes) > 0:
            if balancer != None: indexes = balancer.balance_workload(indexes, workload_balance='disperse_max', workload_function= lambda idx: len(idx[1]))
            indexes.reverse()
            file_count = 0
            if split_output_files:
                out_filename = os.path.join(folder, name+f"{a_suffix}{b_suffix}_{file_count}.gz")
            else:
                out_filename = os.path.join(folder, name+f"{a_suffix}{b_suffix}.gz" )
            f = gzip.open(out_filename, 'wt')
            item_count = 0
            while indexes:
                pmid, text, original_filename, year, text_length, number_of_sentences, length_of_sentences, title, article_type, article_category, keywords  =  indexes.pop()
                if split_output_files and item_count >= items_per_file:
                    f.close()
                    file_count += 1
                    item_count = 0
                    f = gzip.open(os.path.join(folder, name+f"{a_suffix}{b_suffix}_{file_count}.gz" ), 'wt')
                f.write(f"{pmid}\t{text}\t{original_filename}\t{year}\t{text_length}\t{number_of_sentences}\t{length_of_sentences}\t{title}\t{article_type}\t{article_category}\t{keywords}\n")
                item_count += 1
            f.close()

    @classmethod
    def get_index(cls, file, options, logger = None):
        if options["parse"] == 'PubmedPaper':
            return cls.get_paper_index(file, options, logger = logger)
        elif options["parse"] == 'PubmedAbstract':
            return cls.get_abstract_index(file, options, logger = logger)
        elif options['parse'] == 'Basic':
            return cls.get_basic_index(file, options, logger = logger)

    @classmethod
    def get_basic_index(cls, file_path, options, logger = None):
        texts = [] # aggregate all texts
        file_exist = "does exist" if os.path.exists(file_path) else "does not exist"
        if logger != None: logger.info(f"The file {file_path} {file_exist}")
        parsed_texts, stats = TextBasicParser.parse(file_path, logger= logger, options=options)
        for parsed_text in parsed_texts:
            doc_id, file, text = parsed_text
            if doc_id == None or text == "": continue
            pmid_content_and_stats = cls.prepare_indexes(text, doc_id, file, "None", "None", "None", "None", options)
            texts.append(pmid_content_and_stats)
        if logger != None: logger.warning(f"stats:file={file_path},total={stats['total']}")
        if logger != None: logger.warning(f"logs_errors:file={file_path},errors_number={stats['errors']}")
        return texts

    @classmethod
    def get_abstract_index(cls, file, options, logger = None): 
        texts = [] # aggregate all abstracts in XML file
        file_exist = "does exist" if os.path.exists(file) else "does not exist"
        if logger != None: logger.info(f"The file {file} {file_exist}")
        parsed_texts, stats = TextPubmedAbstractParser.parse(file, logger= logger, options=options)
        for parsed_text in parsed_texts:
            pmid, text, year, title, article_type, article_category, keywords = parsed_text
            if pmid == "None": continue
            if text == "None" and title == "None" and len(keywords) == 0: continue

            if options['remain_empty_documents'] or text != "None":
                if options['filter_by_blacklist'] != None: #If a blacklist word file is given, check to filter out documents whose title or article_category contains any of the words
                    is_blacklisted = TextIndexer._check_if_blacklisted_pmid(pmid, title, article_category, options, logger)
                    if is_blacklisted: continue                       

                pmid_content_and_stats = cls.prepare_indexes(text, pmid, file, year, title, article_type, article_category, keywords, options)
                texts.append(pmid_content_and_stats)

        if logger != None: logger.warning(f"stats:file={file},total={stats['total']},no_abstract={stats['no_abstract']},no_pmid={stats['no_pmid']}")
        return texts

    @classmethod
    def get_paper_index(cls, file_path, options, logger = None):   
        file_exist = "does exist" if os.path.exists(file_path) else "does not exist"
        if logger != None: logger.info(f"The file {file_path} {file_exist}")
        parsed_texts, stats = TextPubmedPaperParser.parse(file_path, logger= logger, options=options)
        
        PMC_PMID_dict = None
        if options["equivalences_file"] != None: PMC_PMID_dict = dict(CmdTabs.load_input_data(options["equivalences_file"]))
        texts = [] # aggregate all papers in XML file (Technically it is just one for each xml file, but it is emulating the original abstract part logic)
        for parsed_text in parsed_texts:
            pmid, pmc, filename, year, whole_content, title, article_type, article_category, keywords = parsed_text
            pmid = cls._set_alternative_pmid_if_needed_and_available(pmid, pmc, PMC_PMID_dict, file_path, logger)           

            if pmid == "None": continue
            if whole_content == "None" and title == "None" and len(keywords) == 0: continue
            
            if options['remain_empty_documents'] or (whole_content != "None"):
                if options['filter_by_blacklist'] != None: #If a blacklist word file is given, check to filter out documents whose title or article_category contains any of the words
                    is_blacklisted = TextIndexer._check_if_blacklisted_pmid(pmid, title, article_category, options, logger)
                    if is_blacklisted: continue
                    
                #pmid_content_and_stats = cls.prepare_indexes(whole_content, pmc+"-"+pmid, filename, year, options)
                pmid_content_and_stats = cls.prepare_indexes(whole_content, pmid, filename, year, title, article_type, article_category, keywords, options)
                texts.append(pmid_content_and_stats)

        if logger != None: logger.warning(f"stats:file={file_path},total={stats['total']},no_abstract={stats['no_abstract']},no_pmid={stats['no_pmid']}")
        if logger != None: logger.warning(f"logs_errors:file={file_path},errors_number={stats['errors']}")
        return texts
      

    @classmethod
    def prepare_indexes(cls, text, pmid, file, year, title, article_type, article_category, keywords, options):
        extra_fields = options.get("extra_fields", defaultdict(lambda: False))
        pmid = pmid.replace("\n", "")
        file = file.replace("\n", "")
        year = str(year).replace("\n", "")
        document_length = str(len(text))

        cleaned_text = TextPubmedParser.perform_soft_cleaning(text, type=options["clean_type"])
        if text == "None":
          doc_to_jsonify = []
          document_length = "0"
          number_of_sentences = "0"
          length_of_sentences = "0"
        elif options["split"]:
          document_parts = cls.split_document(cleaned_text, pmid, split_type = options["split_type"])
          doc_to_jsonify = document_parts
          flattened_document = flatten(document_parts)
          number_of_sentences = str(len(flattened_document))
          length_of_sentences = ",".join([str(len(sentence)) for sentence in flattened_document])
        else:
          number_of_sentences = "1"
          length_of_sentences = document_length
          doc_to_jsonify = [[cleaned_text]]

        keywords_to_add = keywords if len(keywords) > 0 else ["None"]
        if extra_fields["keywords"]: doc_to_jsonify.insert(0, ["KEYWORDS"] + keywords_to_add)
        if extra_fields["title"]: doc_to_jsonify.insert(0, ["TITLE", title])
        
        document_json = json.dumps(doc_to_jsonify)
        prepared_index = [pmid, document_json, file, year, document_length, number_of_sentences, length_of_sentences, 
                            title, article_type, article_category, ",".join(keywords_to_add)]
        return prepared_index



    @classmethod
    def split_document(cls, text, pmid, split_type = "classic"):
        #paragraph_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap  = 20, length_function = len, separators=[r"\n\n", r"\.\n?"], keep_separator=False, is_separator_regex=True)
        paragraph_splitter = RecursiveCharacterTextSplitter(chunk_size = 10, chunk_overlap  = 0, length_function = len, separators=["\n\n"], keep_separator=False)
        if split_type == "classic":
            sentences_splitter = RecursiveCharacterTextSplitter(chunk_size = 10, chunk_overlap  = 0, length_function = len, separators=["\n",".", ";", ","], keep_separator=False, is_separator_regex=False)
        elif split_type == "space_overlap":
            sentences_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap  = 20, length_function = len, separators=["\n", " ", ""], keep_separator=False, is_separator_regex=False)
            #sentences_splitter = RecursiveCharacterTextSplitter(chunk_size = 20, chunk_overlap  = 2, length_function = cls._len_by_words_func, separators=["\n", " ", ""], keep_separator=False, is_separator_regex=False)
        #sentences_splitter = RecursiveCharacterTextSplitter(chunk_size = 120, chunk_overlap  = 20, length_function = len, separators=["\n", " ", ""], keep_separator=False, is_separator_regex=False)
        paragraphs = [paragraph.strip() for paragraph in paragraph_splitter.split_text(text)]
        paragraph_sentences = [ sentences_splitter.split_text(paragraph) for paragraph in paragraphs ]
        
        formated = []       
        for paragraph in paragraph_sentences:
            formated_paragraph = [] 
            for sentence in paragraph:
                if len(sentence) > 2:
                    if len(sentence) > 3000: warnings.warn(f"ERROR: The pmid {pmid} has an unusual sentence lenght even after splitting. Total length of characters: {len(sentence)}. Content: {sentence}")
                    else: formated_paragraph.append(sentence.strip())
            if len(formated_paragraph) > 0: formated.append(formated_paragraph)
        
        return formated
    

    ######### UTILS METHODS
    @classmethod
    def _len_by_words_func(cls, text):
        return text.count(" ")

    @classmethod
    def _set_alternative_pmid_if_needed_and_available(cls, pmid, pmc, PMC_PMID_dict, file_path, logger):
        #If there is no pmid in the paper or in the PMC_PMID_dict, we will use the pmc as the document identifier, specifying it is PMC in the index. 
        #If not PMC nor PMID, we will warn the user
        if pmid == "None" and PMC_PMID_dict != None: 
            pmid = PMC_PMID_dict.get(pmc, "None")
        if pmid == "None" and pmc == "None": 
            logger.warning(f"ERROR: The file {file_path} has a paper without PMID and PMC")
        elif pmid == "None" and pmc != "None": 
            pmid = pmc
        return pmid

    @classmethod
    def _check_if_blacklisted_pmid(cls, pmid, title, article_category, options, logger):
        if os.path.exists(options['filter_by_blacklist']):
            is_blacklisted = False
            blacklisted_words = [word.strip() for word in open(options['filter_by_blacklist']).readlines()]
            filtered_out, w, c, t = TextIndexer._check_to_filter_out(blacklisted_words, title, article_category, options['blacklisted_mode'])
            if filtered_out:
                if logger != None: logger.warning(f"Blacklisted PMID {pmid} for having word {w} in {c} with content: {t}")
                is_blacklisted = True
            return is_blacklisted
        else: 
            raise Exception("Blacklisted words filepath given does not exist")  


    @classmethod
    def _check_to_filter_out(cls, blacklisted_words, title, article_category, mode):
        title_l = title.lower()
        article_category_l = article_category.lower()
        for word in blacklisted_words:
            if bool(re.fullmatch(r'\s*', word)): continue # Ignore empty lines
            word_l = word.lower()
            if mode == "exact":
                if word_l == title_l: return True, word, "title", title
                if word_l == article_category_l: return True, word, "category", article_category
            elif mode == "partial":
                if word_l in title_l: return True, word, "title", title
                if word_l in article_category_l: return True, word, "category", article_category             
            else: 
                raise Exception("Wrong blacklisted words filtering out mode has been selected. Acepted modes: partial or exact")
        return False, None, None, None