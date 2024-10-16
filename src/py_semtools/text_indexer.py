import sys
import os
import glob
from loguru import logger
logger.remove(0)
from os import getpid
import warnings
import tarfile
import xml.etree.ElementTree as ET
import re
from concurrent.futures import ProcessPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from py_exp_calc.exp_calc import invert_nested_hash, flatten
import json
import gzip

class TextIndexer:

    @classmethod
    def build_index(cls, options):
        filenames = glob.glob(options["input"])
        if len(filenames) == 0: 
            sys.stderr.write(f"ERROR: {options['input']} has not files\n")
            sys.exit(1)

        if options["chunk_size"] == 0:
            cls.process_several_abstracts(options, filenames)
        else:
            cls.process_several_custom_chunksize_abstracts(options, filenames)

    @classmethod
    def process_several_abstracts(cls, options, filenames):
        if options["n_cpus"] == 1:
          for filename in filenames:
            cls.process_single_abstract([options, filename])
        else:
          with ProcessPoolExecutor(max_workers=options["n_cpus"]) as executor:
            for result in executor.map(cls.process_single_abstract, [[options, filename] for filename in filenames]): return result
    
    @classmethod
    def process_single_abstract(cls, options_filename_pair):
        options, filename = options_filename_pair

        pID = getpid()
        logger.add(f"./logs/{pID}.log", format="{level} : {time} : {message}: {process}", filter=lambda record: record["extra"]["task"] == f"{pID}")
        child_logger = logger.bind(task=f"{pID}")
        options["child_logger"] = child_logger
        child_logger.info("Starting to process papers")

        basename = os.path.basename(filename).replace(".xml.gz", "")
        abstract_index = cls.get_index(filename, options)
        out_filename = os.path.join(options["output"], basename+".gz")
        cls.save_abstracts(out_filename, abstract_index)

    @classmethod
    def process_several_custom_chunksize_abstracts(cls, options, filenames):
        if options["n_cpus"] == 1:
          cls.process_a_pack_of_custom_chunksize_abstracts([options, filenames, 0])
        else:
          filenames_lots = cls.distribute_files_workload(filenames, options["n_cpus"])
          distributed_work = [[options, filename_lot, idx] for idx,filename_lot in enumerate(filenames_lots) if filename_lot]
          with ProcessPoolExecutor(max_workers=options["n_cpus"]) as executor:
            for result in executor.map(cls.process_a_pack_of_custom_chunksize_abstracts, distributed_work): return result

    @classmethod
    def distribute_files_workload(cls, filenames, n_cpus):
        filenames_lots = []
        lot_size = (len(filenames)+(len(filenames) % n_cpus)) // n_cpus
        for index in range(0, n_cpus):
          if index == n_cpus-1:
            for idx, remaining_filename in enumerate(filenames[index*lot_size:]):
              filenames_lots[idx].append(remaining_filename)
          else:
            filenames_lots.append(filenames[index*lot_size:(index+1)*lot_size])
        return filenames_lots

    @classmethod
    def process_a_pack_of_custom_chunksize_abstracts(cls, options_filenames_counter_trio):
        options, filenames, sup_counter = options_filenames_counter_trio
        
        pID = getpid()
        logger.add(f"./logs/{pID}.log", format="{level} : {time} : {message}: {process}", filter=lambda record: record["extra"]["task"] == f"{pID}")
        child_logger = logger.bind(task=f"{pID}")
        options["child_logger"] = child_logger
        child_logger.info("Starting to process papers")
        
        acummulative_abstracts = []
        counter = 0
        for filename in filenames:    
          abstract_index = cls.get_index(filename, options)
          acummulative_abstracts.extend(abstract_index)
          while len(acummulative_abstracts) >= options["chunk_size"]:
            out_filename = os.path.join(options["output"], options["tag"]+f"{sup_counter}_{counter}.gz" )
            counter += 1
            abstracts_to_save = [acummulative_abstracts.pop() for _times in range(options["chunk_size"])]
            cls.save_abstracts(out_filename, abstracts_to_save)
        
        out_filename = os.path.join(options["output"], options["tag"]+f"{sup_counter}_{counter}.gz" )
        cls.save_abstracts(out_filename, acummulative_abstracts)
        child_logger.success("Proccess finished succesfully")

    @classmethod
    def save_abstracts(cls, out_filename, abstracts):
        if len(abstracts) > 0:
          with gzip.open(out_filename, 'wt') as f:
            for pmid, text, original_filename, year, abstract_length, number_of_sentences, length_of_sentences, title, article_type, article_category in abstracts:
              f.write(f"{pmid}\t{text}\t{original_filename}\t{year}\t{abstract_length}\t{number_of_sentences}\t{length_of_sentences}\t{title}\t{article_type}\t{article_category}\n")

    @classmethod
    def get_index(cls, file, options):
        if options["parse_paper"] == True:
            return cls.get_paper_index(file, options)
        else:
            return cls.get_abstract_index(file, options)

    ##### New functions to parse papers

    @classmethod
    def get_paper_index(cls, file_path, options):
        PMC_PMID_dict = None
        if options["equivalences_file"] != None: PMC_PMID_dict = dict(CmdTabs.load_input_data(options["equivalences_file"]))
        texts = [] # aggregate all papers in XML file (Technically it is just one for each xml file, but it is emulating the original abstract part logic)
        stats = {"no_abstract": 0, "no_pmid": 0}
        total = 0
        errors = 0
        
        file_exist = "does exist" if os.path.exists(file_path) else "does not exist"
        options["child_logger"].info(f"The file {file_path} {file_exist}")
        
        tar = tarfile.open(file_path, 'r:gz') 
        options["child_logger"].info(f"The file {file_path} has { len([member for member in tar.getmembers()]) } xml papers")
        for member in tar.getmembers():
            total += 1
            f=tar.extractfile(member)
            filename = os.path.join(file_path, member.path)
            try:            
                paper_xml_string=f.read()
                pmid, pmc, year, whole_content, title, article_type, article_category = cls.parse_paper(paper_xml_string, filename)

                if pmid == None and PMC_PMID_dict != None: pmid = PMC_PMID_dict.get(pmc)

                if pmid == None:
                    stats["no_pmid"] += 1
                    if options["debugging_mode"]: warnings.warn(f"Warning: Article without PMID found in file {filename}")
                elif whole_content == "":
                    stats["no_abstract"] += 1
                    if options["debugging_mode"]: warnings.warn(f"Warning: Article PDMID:{pmid} without abstract found in file {filename}")
                else:               
                    #pmid_content_and_stats = cls.prepare_indexes(whole_content, pmc+"-"+pmid, filename, year, options)
                    pmid_content_and_stats = cls.prepare_indexes(whole_content, pmid, filename, year, title, article_type, article_category, options)
                    texts.append(pmid_content_and_stats)
            except Exception as e:
                errors += 1
                options["child_logger"].error(f"There was a problem proccessing the file {filename} with the following error: {e}")
        tar.close()

        if options["debugging_mode"]: warnings.warn(f"stats:file={file_path},total={total},no_abstract={stats['no_abstract']},no_pmid={stats['no_pmid']}")
        if options["debugging_mode"]: warnings.warn(f"logs_errors:file={file_path},errors_number={errors}")
        return texts

    @classmethod
    def parse_paper(cls, paper_xml_string, filename):
        whole_content = ""
        year = 0
        pmc = None
        pmid = None
        article_root = ET.fromstring(paper_xml_string)

        #GETTING ARTICLE TITLE FIELD
        title = cls.do_recursive_find(article_root, ['front','article-meta','title-group','article-title'])
        title = cls.get_paper_body_content(title).strip().lower() if cls.check_not_none_or_empty(title) else "none"
        #GETTING article-type property from article tag and article category from article-categories tag
        article_type = article_root.get('article-type').lower() if article_root.get('article-type') != None else "none"
        article_category = cls.do_recursive_find(article_root, ['front','article-meta','article-categories', 'subj-group', 'subject'])
        article_category = article_category.text.strip().lower() if cls.check_not_none_or_empty(article_category) else "none"
        #GETTING PMC ID, PMID AND YEAR
        for id_tags in article_root.iter('article-id'):
            if id_tags.get('pub-id-type') == "pmid":
                pmid = id_tags.text 
            if id_tags.get('pub-id-type') == "pmc":
                pmc = id_tags.text

        for date_fields in article_root.iter("date"):
            if date_fields.get("date-type") == "accepted":
                year = date_fields.find("year").text
        if year == 0: #In case date-type is not available, we try getting year by using pmc-release field
            for date_fields in article_root.iter("pub-date"):
                if date_fields.get("pub-type") == "pmc-release":
                    year = date_fields.find("year").text
                if date_fields.get("pub-type") != "pmc-release" and year == 0:
                    year = date_fields.find("year").text

        #GETTING PAPER WHOLE CONTENT
        paper_root = article_root.find("body")
        if paper_root != None: whole_content = cls.perform_soft_cleaning(  cls.get_paper_body_content(paper_root).strip()  )
            
        return pmid, pmc, year, whole_content, title, article_type, article_category

    @classmethod
    def get_paper_body_content(cls, element):
        whole_content = ""
        if element.tag in ["table-wrap", "table", "fig", "fig-group"]: return whole_content
        if element.tag == "sec": whole_content += "\n\n"
        if element.tag == "p": whole_content += "\n\n"  
        # Content before nested element
        if element.tag not in ["xref", "sup"]:
            content = element.text
            if content != None: whole_content += " " + content.replace('\n', ' ') + " " 
        # Content of nested element
        for child in element: 
            whole_content += cls.get_paper_body_content(child)
        # Content after nested element
        tail = element.tail
        if tail != None: whole_content +=  " " + tail.replace('\n', ' ') + " "
        return re.sub(r'\s+', ' ', whole_content)



    ### Common functions for both (Parser Papers and Parser Abstracts)

    @classmethod
    def prepare_indexes(cls, abstract_content, pmid, file, year, title, article_type, article_category, options):
        pmid = pmid.replace("\n", "")
        file = file.replace("\n", "")
        year = str(year).replace("\n", "")

        abstract_length = str(len(abstract_content))
        if options["split"]:
          abstract_parts = cls.split_abstract(abstract_content, pmid, file)
          flattened_abstract = flatten(abstract_parts)
          number_of_sentences = str(len(flattened_abstract))
          length_of_sentences = ",".join([str(len(sentence)) for sentence in flattened_abstract])
          abstract_parts_json = json.dumps(abstract_parts)
          prepared_index = [pmid, abstract_parts_json, file, year, abstract_length, number_of_sentences, length_of_sentences, 
                            title, article_type, article_category] 
        else:
          cleaned_abstract = abstract_content.strip().strip().replace("\r", "\n").replace("&#13", "\n").replace("\t", " ").replace("\n", " ")
          prepared_index = [pmid, cleaned_abstract, file, year, abstract_length, 1, abstract_length, 
                            title, article_type, article_category]
        return prepared_index

    @classmethod
    def perform_soft_cleaning(cls, abstract):
            raw_abstract = abstract.strip().replace("\r", "\n").replace("&#13", "\n").replace("\t", " ")
            raw_abstract = re.sub(r"\\[a-z]+(\[.+\])?(\{.+\})", r" ", raw_abstract) #Removing latex commands
            raw_abstract = re.sub(r"[ ]+", r" ", raw_abstract) #Removing additional whitespaces between words
            raw_abstract = re.sub(r"([A-Za-z\(\)]+[ ]*)\n([ ]*[A-Z-a-z\(\)]+)", r"\1 \2", raw_abstract) #Removing nonsense newlines
            raw_abstract = re.sub(r"([0-9]+)[\.\,]([0-9]+)", r"\1'\2", raw_abstract) #Changing floating point numbers from 4.5 or 4,5 to 4'5
            raw_abstract = re.sub(r"i\.?e\.?", "ie", raw_abstract).replace("al.", "al ") #Changing i.e to ie and et al. to et al
            return raw_abstract

    @classmethod
    def split_abstract(cls, abstract, pmid, file):
        #paragraph_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap  = 20, length_function = len, separators=[r"\n\n", r"\.\n?"], keep_separator=False, is_separator_regex=True)
            paragraph_splitter = RecursiveCharacterTextSplitter(chunk_size = 10, chunk_overlap  = 0, length_function = len, separators=["\n\n"], keep_separator=False)
            sentences_splitter = RecursiveCharacterTextSplitter(chunk_size = 10, chunk_overlap  = 0, length_function = len, separators=["\n",".", ";", ","], keep_separator=False, is_separator_regex=False)
            #sentences_splitter = RecursiveCharacterTextSplitter(chunk_size = 120, chunk_overlap  = 20, length_function = len, separators=["\n", " ", ""], keep_separator=False, is_separator_regex=False)
            paragraphs = paragraph_splitter.split_text(abstract)
            sentences = [ sentences_splitter.split_text(paragraph.replace("\n", "")) for paragraph in paragraphs ]
            
            formated = []       
            for paragraph in sentences:
                formated_paragraph = [] 
                for sentence in paragraph:
                    if len(sentence) > 2:
                        if len(sentence) > 3000: warnings.warn(f"ERROR: The pmid {pmid} inside file {file} has an unusual sentence lenght even after splitting. Total length of characters: {len(sentence)}. Content: {sentence}")
                        else: formated_paragraph.append(sentence)
                if len(formated_paragraph) > 0: formated.append(formated_paragraph)
            
            return formated

    ##### Abstracts part functions
    @classmethod
    def get_abstract_index(cls, file, options): 
        texts = [] # aggregate all abstracts in XML file
        stats = {"no_abstract": 0, "no_pmid": 0}
        total = 0
        with gzip.open(file) as gz:
            mytree = ET.parse(gz)
            pubmed_article_set = mytree.getroot()
            for article in pubmed_article_set:
                total += 1
                pmid, abstract_content, year, title, article_type, article_category = parse_abstract(article)

                if pmid == None:
                    stats["no_pmid"] += 1
                    if options["debugging_mode"]: warnings.warn(f"Warning: Article without PMID found in file {file}")
                elif abstract_content == "":
                    stats["no_abstract"] += 1
                    if options["debugging_mode"]: warnings.warn(f"Warning: Article PDMID:{pmid} without abstract found in file {file}")
                else:
                    pmid_content_and_stats = cls.prepare_indexes(abstract_content, pmid, file, year, title, article_type, article_category, options)
                    texts.append(pmid_content_and_stats)
        
        if options["debugging_mode"]: warnings.warn(f"stats:file={file},total={total},no_abstract={stats['no_abstract']},no_pmid={stats['no_pmid']}")
        return texts

    @classmethod
    def parse_abstract(cls, article):
        pmid = None
        abstract_content = ""
        article_type = "none"
        article_category = "none"
        year = 0
        title = cls.do_recursive_find(article, ['MedlineCitation','Article','ArticleTitle'])
        title = cls.get_paper_body_content(title).strip().lower() if cls.check_not_none_or_empty(title) else "none"
        for data in article.find('MedlineCitation'):
            if data.tag == 'PMID':
                pmid = data.text
            if data.tag == 'DateCompleted':
                for fields in data:
                    if fields.tag == "Year": year = int(fields.text)                    
            abstract = data.find('Abstract')
            if abstract != None:
                for fields in abstract:     
                    if fields.tag == 'AbstractText':
                        abstractText = cls.get_paper_body_content(fields).strip()
                        if abstractText != None and abstractText != "":
                            #print(f"Text of abstract {pmid} in file {file}:")
                            #print(repr(abstractText), "\n\n")
                            raw_abstract = cls.perform_soft_cleaning(abstractText)                                                 
                            abstract_content += raw_abstract + "\n\n"
        return pmid, abstract_content, year, title, article_type, article_category
        

    ### AUXILIARY FUNCTIONS
    @classmethod
    def do_recursive_find(cls, initial_tag, subtags_list):
        if len(subtags_list) == 0:
            return initial_tag
        nested_tag = initial_tag.find(subtags_list[0])
        if nested_tag != None:
            return cls.do_recursive_find(nested_tag, subtags_list[1:])
        else:
            return None

    @classmethod
    def check_not_none_or_empty(cls, variable):
        if type(variable) != str: 
            condition = variable != None and variable.text != None and variable.text.strip().replace("&#x000a0;", "") != ""
        else:
            condition = variable != None and variable.strip().replace("&#x000a0;", "") != ""
        return condition
