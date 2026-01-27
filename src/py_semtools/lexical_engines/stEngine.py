import os
import time
import gc
import gzip, pickle
import json
import numpy as np
import warnings

class STengine:

    def __init__(self, gpu_devices = []):
        self.gpu_devices = gpu_devices
        self.model_name = None
        self.embedder = None
        self.queries_content = {}
        

    def show_gpu_information(self, verbose = False):
        import torch # Moving import here to avoid long import time
        devices = [int(device.replace("cuda:","")) for device in self.gpu_devices]
        if verbose:
          print("-"*30+"\nGeneral information about all the available GPUs:")
          self.show_general_global_gpu_information(torch)
          print("Specific information about each GPU device:")
          for device_number in devices:
              self.show_gpu_type_specific_information(device_number, torch)
          print("-"*30)

    def show_general_global_gpu_information(self, torch):
        print(f"LOG: Are there any GPU available: {torch.cuda.is_available()}")
        print(f"LOG: Number of GPUs available: {torch.cuda.device_count()}")
        print(f"LOG: GPUs UUIDs: {torch.cuda._raw_device_uuid_nvml()}")
        print(f"LOG: CUDA version: {torch.version.cuda}")
        print(f"LOG: Current CUDA device: {torch.cuda.current_device()}")

    def show_gpu_type_specific_information(self, device_number, torch):
        print(f"LOG: CUDA device Number: {device_number}")
        print(f"LOG: CUDA device ID: {torch.cuda._get_device_index(device_number)}")
        print(f"LOG: CUDA device name: {torch.cuda.get_device_name(device_number)}")
        print(f"LOG: CUDA device object: {torch.cuda.device(device_number)}")
        print(f"LOG: CUDA device properties: {torch.cuda.get_device_properties(device_number)}")
        self.show_gpu_specific_stats(device_number, torch)
        self.show_gpu_specific_memory_summary(device_number, torch)

    def show_gpu_specific_stats(self, device_number, torch):        
        self.show_gpu_specific_memory_stats(device_number, torch) 
        self.show_gpu_specific_usage(device_number, torch)

    def show_gpu_specific_usage(self, device_number, torch):
        print(f"LOG: GPU computation percentage: {torch.cuda.utilization(device_number)}")
        print(f"LOG: GPU currently active processes: {torch.cuda.list_gpu_processes(device_number)}")    

    def show_gpu_specific_memory_stats(self, device_number, torch):
        print(f"LOG: GPU memory usage: {torch.cuda.memory_usage(device_number)}")
        print(f"LOG: GPU memory allocated: {torch.cuda.memory_allocated(device_number)}")
        print(f"LOG: GPU memory reserved: {torch.cuda.memory_reserved(device_number)}")
        print(f"LOG: GPU memory max memory allocated: {torch.cuda.max_memory_allocated(device_number)}")
        print(f"LOG: GPU memory max memory reserved: {torch.cuda.max_memory_reserved(device_number)}")

    def show_gpu_specific_memory_summary(self, device_number, torch):
        print(f"LOG: GPU memory summary:\n{torch.cuda.memory_summary(device_number)}\n")

    def init_model(self, model_name, cache_folder = None, verbose = False):
        from sentence_transformers import SentenceTransformer #Moving import here due to long import time
        if model_name != None and cache_folder != None:
            if verbose: print(f"\n-Downloading or loading model {model_name} inside path {cache_folder}")
            self.model_name = model_name
            has_cached_model = os.path.exists(os.path.join(cache_folder, f'models--sentence-transformers--{model_name}')) #SentenceTransfomers now uses local_files_only to control internet access, even if the model is cached, and trying to download the model with this variable false gives error, so we have to control it
            self.embedder = SentenceTransformer(model_name, cache_folder = cache_folder, local_files_only=has_cached_model)

    def embed_save_corpus(self, options, corpus_basename, all_textIDs, all_corpus, total_papers):
        if options["verbose"]: print(f"---Embedding corpus of {corpus_basename} comprised by {total_papers} initial papers with {len(all_textIDs)} sentences, with {'GPU' if options.get('gpu_device') else 'CPU'}")
        corpus_embeddings = self.embedd_text(all_corpus, options)       
        corpus_info = {'textIDs': all_textIDs, "all_corpus": all_corpus, "embeddings": corpus_embeddings}
        if options.get("corpus_embedded") != None:
            if options["verbose"]: print(f"---Saving embedded corpus in {corpus_basename}")
            with open(os.path.join(options["corpus_embedded"], corpus_basename) + '.pkl', "wb") as fOut:
                pickle.dump(corpus_info, fOut)
        return corpus_info

    def calculate_similarities(self, options, corpus_info):
        from sentence_transformers import util # Moving import here to avoid long import time
        if options.get("output_file"):
          for query_basename, query_info in self.queries_content.items():
            best_matches = self.calculate_similarity(query_info, corpus_info, options, util=util)
            if options['print_relevant_pairs']: self.print_similarities(query_info, corpus_info, best_matches, options)
            output_filename = os.path.join(options["output_file"],query_basename)
            self.save_similarities(output_filename, best_matches, options)

    def print_similarities(self, query_info, corpus_info, best_matches, options):
        term_related_sentences = {}
        for textID, matches in best_matches.items():
            textIDX = corpus_info["textIDs"].index(textID)
            text = corpus_info["all_corpus"][textIDX]
            for kwdID, score in matches.items():
                if score >= options["threshold"]: 
                    kwIDXs = (idx for idx, char in enumerate(query_info['query_ids']) if char == kwdID) 
                    term = " -- ".join([query_info["queries"][kwIDX] for kwIDX in kwIDXs])
                    if term not in term_related_sentences: term_related_sentences[term] = {"term_id": kwdID, "sentences": []}
                    term_related_sentences[term]["sentences"].append((textID, text, score))
        print("-"*30)
        for term, data in term_related_sentences.items():
            print(f"Term: {term} (ID: {data['term_id']}) has {len(data['sentences'])} related sentences:")
            for textID, text, score in data["sentences"]:
                print(f"  - Text ID: {textID}, Score: {score}")
                print(f"    Text: {text}")
            print("-"*30)

    def load_several_queries(self, options, embedded_queries_filenames, verbose = False):
        if verbose: print("\n-Loading embedded queries:")
        for embedded_query_filename in embedded_queries_filenames:
            embedded_query_basename = os.path.splitext(os.path.basename(embedded_query_filename))[0]
            with open(embedded_query_filename, "rb") as fIn:
                if verbose: print(f"---Loading embedded query from {embedded_query_basename}")
                self.queries_content[embedded_query_basename] = pickle.load(fIn)

    def embedd_several_queries(self, options, queries_filenames, verbose = False):
        if verbose: print("\n-Loading and embedding queries:")
        for query_filename in queries_filenames:
            query_basename, query_ids, queries, query_embeddings = self.embedd_single_query(query_filename, options)
            self.queries_content[query_basename] = {'query_ids': query_ids, "queries": queries, "embeddings": query_embeddings}
            if options.get("query_embedded") != None:
                if verbose: print(f"---Saving embedded query in {query_basename}")
                with open(os.path.join(options["query_embedded"], query_basename) + '.pkl', "wb") as fOut:
                    pickle.dump(self.queries_content[query_basename], fOut)

    def embedd_single_query(self, query_filename, options):
        """Embeds a single query from a file and returns its basename, IDs, queries, and embeddings.
        Args:
            query_filename (str): Path to the file containing the query.
            options (dict): Options for embedding, including verbosity and GPU settings.
        Returns:
            list: A list containing the basename of the query, its IDs, queries, and embeddings
        """
        query_basename = os.path.splitext(os.path.basename(query_filename))[0]
        if options["verbose"]: print(f"---Loading query from {query_basename}")
        keyword_index = self.load_keyword_index(query_filename) # keywords used in queries
        queries = []
        query_ids = []
        for kwdID, kwds in keyword_index.items():
            queries.extend(kwds)
            query_ids.extend([kwdID for i in range(0, len(kwds))])
        query_embeddings = self.embedd_text(queries, options)
        return [query_basename, query_ids, queries, query_embeddings]

    def embedd_text(self, text, options):
        """General embedding function for queries and corpora. Embeds text using the embedder. If GPU devices are specified, it uses GPU for embedding; otherwise, it uses CPU.
        Args:
            text (list): The texts to be embedded.       
            options (dict): Options for embedding, including verbosity and GPU settings.
        Returns:
            np.ndarray: The embeddings of the input text.
        """
        if self.gpu_devices:
            text_embedding = self.embedd_text_gpu(text, options)
        else:
            text_embedding = self.embedd_text_cpu(text, options)
        return text_embedding

    def embedd_text_cpu(self, text, options):
        start = time.time()
        text_embedding = self.embedder.encode(text, convert_to_numpy=True, show_progress_bar = options["verbose"]) #convert_to_tensor=True
        if options["verbose"]: print(f"---Embedding time with {os.environ.get('MKL_NUM_THREADS') or os.environ.get('OMP_NUM_THREADS') or 1} CPUs: {time.time() - start} seconds")
        return text_embedding

    def embedd_text_gpu(self, text, options):
        start = time.time()
        if len(options["gpu_device"]) > 1:
                pool = self.embedder.start_multi_process_pool(options["gpu_device"])
                text_embedding = self.embedder.encode_multi_process(text, pool = pool, batch_size=options["batch_size"])
                self.embedder.stop_multi_process_pool(pool)
        elif len(options["gpu_device"]) == 1:
                text_embedding = self.embedder.encode(text, convert_to_numpy=True, show_progress_bar = options["verbose"], device= options["gpu_device"][0]) #convert_to_tensor=True 
        if options["verbose"]: print(f"---Embedding time with {0 if options.get('gpu_device') == None else len(options['gpu_device'])} GPUs: {time.time() - start} seconds")
        return text_embedding

    def load_keyword_index(self, file):
        """Load keyword index from a file. Columns are separated by tab. 
        The first column is the keyword ID, the second is the keyword name, and the third (optional) is a list of synonimns or alternative keywords names separated by '|'. 

        Args:
            file (str): Path to the file containing the keyword index.
        Returns:
            dict: A dictionary where keys are keyword IDs and values are lists of keywords names (including synonyms and alternatives).
        """
        keywords = {}
        with open(file) as f:
            for line in f:
                fields = line.rstrip().split("\t")
                if len(fields) == 2:
                    id, keyword = fields
                    keywords[id] = [keyword]
                elif len(fields) == 3:
                    id, keyword, alternatives = fields
                    alternatives = alternatives.split('|')
                    alternatives.append(keyword)
                    kwrds = list(set(alternatives))
                    keywords[id] = kwrds
                else:
                    warnings.warn(f"Error reading line in file {os.path.basename(file)}: {line}. Expected 2 or 3 fields, got {len(fields)} fields. Skipping line.")
                    continue
        return keywords

    def get_splitted_document(self, id, text):
        """
        Process a document that has been split by get_corpus_index. It is a json formmated list of lists, where each sublist is a paragraph and each element inside the sublist is a sentence.
        Each sentence is stored in a dictionary with the key being the ID of the document in the format "id_paragraphNumber_sentenceNumber" and the value being the sentence text.
        Args:
            id (str): The identifier for the document.
            text (str): The text of the document in JSON format, where each paragraph is a list of sentences.
        Returns:
            dict: A dictionary where keys are IDs in the format "id_paragraph_sentence" and values are the corresponding sentences.
        """
        pubmed_index = {}
        abstract_parts = json.loads(text)
        paragraph_number = 0
        for paragraph in abstract_parts:
            if paragraph[0] == "TITLE": paragraph_number = -2
            if paragraph[0] == "KEYWORDS": paragraph_number = -1
            sentence_number = 0
            for sentence in paragraph:
                if sentence in ["TITLE", "KEYWORDS", "None"]: 
                    sentence_number += 1
                    continue
                id_tag = f"{id}_{paragraph_number}_{sentence_number}"
                pubmed_index[id_tag] = sentence
                sentence_number += 1
            paragraph_number += 1
        return pubmed_index

    def load_pubmed_index(self, file, is_splitted):
        """
        Load a PubMed processed index file get by 'get_corpus_index' binary. The file is expected to be in a specific format, where each line contains an ID and the corresponding text (and other columns with possible metadata not loaded).
        Args:
            file (str): Path to the file containing the PubMed index.
            is_splitted (bool): If True, the text is expected to be in a JSON format with sentences split into paragraphs.
        Returns:
            tuple: A tuple containing a dictionary where keys are IDs in the format "id_paragraphNumber_sentenceNumber" and values are the corresponding sentences, and the total number of papers processed.
        """
        pubmed_index = {}
        n_papers = 0
        with gzip.open(file, "rt") as f:
            for line in f:
                try:
                    id, text, *_rest = line.rstrip().split("\t")
                    pubmed_index_iter = self.get_splitted_document(id, text)
                    pubmed_index.update(pubmed_index_iter)
                    n_papers += 1
                except Exception as e:
                    warnings.warn(f"Error reading line in file {os.path.basename(file)}: {line}.\n Error: {e}")
        return pubmed_index, n_papers

    def calculate_similarity(self, query_info, corpus_info, options, util):
        corpus_ids = corpus_info["textIDs"]
        corpus_embeddings = corpus_info["embeddings"]

        query_ids = query_info['query_ids']
        query_embeddings = query_info["embeddings"]

        if options["gpu_device"] != None and options["use_gpu_for_sim_calculation"]:
            search = self.calculate_similarity_gpu(query_embeddings, corpus_embeddings, options["top_k"], util, options['cuda'], options['from_numpy'], options["verbose"], options["order"])
        else:
            search = self.calculate_similarity_cpu(query_embeddings, corpus_embeddings, options["top_k"], util, options["verbose"], options["order"])

        if options["order"] == "corpus-query":
            matches = self.find_best_matches(corpus_ids, query_ids, search)
        else:
            matches = self.find_best_matches(query_ids, corpus_ids, search)
        return matches

    def calculate_similarity_cpu(self, query_embeddings, corpus_embeddings, top_k, util, verbose=False, order="corpus-query"):
      if verbose: print(f"----Calculating similarities using {os.environ.get('MKL_NUM_THREADS') or os.environ.get('OMP_NUM_THREADS') or 1} CPUs")
      start = time.time()
      results = self.make_single_similarity_calculation(corpus_embeddings, query_embeddings, top_k=top_k, util=util, gpu_calc=False, order=order)
      if verbose: print(f"----Time to calculate similarities with CPU: {time.time() - start} seconds")
      return results

    def calculate_similarity_gpu(self, query_embeddings, corpus_embeddings, top_k, util, cuda, from_numpy, verbose=False, order="corpus-query"):
      if verbose: print("----Calculating similarities with GPU")
      start = time.time()
      corpus_embeddings = from_numpy(corpus_embeddings).to("cuda")
      corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
      query_embeddings = from_numpy(query_embeddings).to("cuda")
      query_embeddings = util.normalize_embeddings(query_embeddings)
      results = self.make_single_similarity_calculation(corpus_embeddings, query_embeddings, top_k=top_k, util=util, gpu_calc=True, order=order)
      del corpus_embeddings; del query_embeddings; gc.collect(); cuda.empty_cache()
      if verbose: print(f"----Time to calculate similarities with GPU: {time.time() - start} seconds")
      return results

    def make_single_similarity_calculation(self, corpus_embeddings, query_embeddings, top_k, util, gpu_calc=False, order="corpus-query"):
      sim_function = util.dot_score if gpu_calc else util.cos_sim

      if order == "query-corpus":
        result = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k, score_function=sim_function)
      elif order == "corpus-query":
        result = util.semantic_search(corpus_embeddings, query_embeddings, top_k=top_k, score_function=sim_function)
      else:
        raise Exception("Invalid order parameter value. Valid values are: query-corpus or corpus-query")
      return result

    def find_best_matches(self, query_ids, corpus_ids, search):
        best_matches = {}
        for i,query in enumerate(search):
          kwdID = query_ids[i]
          kwd = best_matches.get(kwdID)
          if kwd == None:
            kwd = {}
            best_matches[kwdID] = kwd

          for hit in query:
            textID = corpus_ids[hit['corpus_id']]
            score = hit['score']
            text_score = kwd.get(textID)
            if text_score == None or text_score < score :
              kwd[textID] = score
          #sentence = corpus_sentences[hit['corpus_id']]
        return best_matches

    def save_similarities(self, filepath, best_matches, options):
        #with gzip.open(filepath, "a") as f: #TODO: add it later
        with open(filepath, 'a') as f:
          for kwdID, matches in best_matches.items():
            for textID, score in matches.items():
              if score >= options["threshold"]: 
                if options["order"] == "corpus-query":
                  f.write(f"{textID}\t{kwdID}\t{score}\n")
                else:
                  f.write(f"{kwdID}\t{textID}\t{score}\n")

    def process_corpus_get_similarities(self, corpus_filenames, options, verbose=False):
        from torch import cuda, from_numpy # Moving import here to avoid long import time
        options['cuda'] = cuda
        options['from_numpy'] = from_numpy
        count = 0
        corpus_info = None
        all_textIDs = []; all_corpus = []; total_papers = 0 # Text accumulation variables
        for corpus_filename in corpus_filenames:
          
          if options.get("corpus") != None: #LOAD RAW CORPUS AND EMBEDD (AND MAYBE SAVE)
            if verbose: print(f"---Loading corpus of {corpus_filename}")
            pubmed_index, n_papers = self.load_pubmed_index(corpus_filename, options["split"]) # abstracts
            total_papers += n_papers
            all_textIDs.extend(pubmed_index.keys())
            all_corpus.extend(pubmed_index.values())
            if total_papers >= options['chunk_size']:
              corpus_basename = f"corpus_{count}"
              count += 1
              corpus_info = self.embed_save_corpus(options, corpus_basename, all_textIDs, all_corpus, total_papers)
              all_textIDs = []; all_corpus = []; total_papers = 0 # Reset text accumulation variables
              if options.get("output_file") == None:
                  # If similarities won't be calculated delete last embedding because it's saved as pickle
                  del corpus_info; gc.collect(); cuda.empty_cache(); corpus_info = None # Delete CPU/GPU data from last chunk

          else: #LOAD EMBEDDED CORPUS
            if options["verbose"]: print(f"---Loading embedded corpus from {os.path.basename(corpus_filename)}")
            with open(corpus_filename, "rb") as fIn: corpus_info = pickle.load(fIn)
          
          if corpus_info != None: # CALCULATE SIMILARITIES
              self.calculate_similarities(options, corpus_info)
              del corpus_info; gc.collect(); cuda.empty_cache(); corpus_info = None # Delete CPU/GPU data from last chunk

        # When we aggregate several files we could get an uncompleted chunk and must be processed to avoid lose the last items.
        if all_textIDs and all_corpus: corpus_info = self.embed_save_corpus(options, f"corpus_{count}", all_textIDs, all_corpus, total_papers) 
        if options.get("corpus") != None and corpus_info != None: self.calculate_similarities(options, corpus_info)