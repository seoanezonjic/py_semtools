import os, time, gc, pickle

from py_semtools.lexical_engines.engine_baseclass import LexicalEngineBaseClass

class STengine(LexicalEngineBaseClass):

    def __init__(self, gpu_devices = []):
        self.gpu_devices = gpu_devices
        self.model_name = None
        self.embedder = None
        self.reranker = None
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
        full_model_path = os.path.join(cache_folder, model_name)
        if verbose: print(f"\n-Downloading or loading bi encoder model {model_name} inside path {full_model_path}")
        self.model_name = model_name
        has_cached_model = os.path.exists(os.path.join(full_model_path, f'models--{model_name.replace("/", "--")}')) #SentenceTransfomers now uses local_files_only to control internet access, even if the model is cached, and trying to download the model with this variable false gives error, so we have to control it
        if verbose: print(f"Seeking cached model (found:{has_cached_model}) in", os.path.join(full_model_path, f'models--{model_name.replace("/", "--")}'))
        self.embedder = SentenceTransformer(model_name, cache_folder = full_model_path, local_files_only=has_cached_model)

    def init_rerank_model(self, model_name, cache_folder = None, verbose = False):
        from sentence_transformers import CrossEncoder
        full_model_path = os.path.join(cache_folder, model_name)
        if verbose: print(f"\n-Downloading or loading cross-encoder model {model_name} inside path {full_model_path}")
        self.reranker_model_name = model_name
        has_cached_model = os.path.exists(os.path.join(full_model_path, f'models--{model_name.replace("/", "--")}')) #SentenceTransfomers now uses local_files_only to control internet access, even if the model is cached, and trying to download the model with this variable false gives error, so we have to control it
        if verbose: print(f"Seeking cached model (found:{has_cached_model}) in ", os.path.join(full_model_path, f'models--{model_name.replace("/", "--")}'))
        self.reranker = CrossEncoder(model_name_or_path = model_name, cache_folder = full_model_path, local_files_only=has_cached_model)
     
    def embed_save_corpus(self, options, corpus_basename, all_textIDs, all_corpus, total_papers):
        if options["verbose"]: print(f"---Embedding corpus of {corpus_basename} comprised by {total_papers} initial papers with {len(all_textIDs)} sentences, with {'GPU' if options.get('gpu_device') else 'CPU'}")
        corpus_embeddings = self.embedd_text(all_corpus, options)       
        corpus_info = {'textIDs': all_textIDs, "all_corpus": all_corpus, "embeddings": corpus_embeddings}
        if options.get("corpus_embedded") != None:
            if options["verbose"]: print(f"---Saving embedded corpus in {corpus_basename}")
            with open(os.path.join(options["corpus_embedded"], corpus_basename) + '.pkl', "wb") as fOut:
                pickle.dump(corpus_info, fOut)
        return corpus_info

    def load_embedded_queries(self, options, embedded_queries_filenames, verbose = False):
        if verbose: print("\n-Loading embedded queries:")
        for embedded_query_filename in embedded_queries_filenames:
            embedded_query_basename = os.path.splitext(os.path.basename(embedded_query_filename))[0]
            with open(embedded_query_filename, "rb") as fIn:
                if verbose: print(f"---Loading embedded query from {embedded_query_basename}")
                self.queries_content[embedded_query_basename] = pickle.load(fIn)

    def embedd_several_queries(self, options, queries_filenames, verbose = False):
        if verbose: print("\n-Loading and embedding queries:")
        for query_filename in queries_filenames:
            query_basename, query_ids, queries, query_embeddings = self.load_and_embedd_single_query(query_filename, options)
            #self.queries_content[query_basename] = {'query_ids': query_ids, "queries": queries, "embeddings": query_embeddings}
            self.queries_content[query_basename].update({"embeddings": query_embeddings})
            if options.get("query_embedded") != None:
                if verbose: print(f"---Saving embedded query in {query_basename}")
                with open(os.path.join(options["query_embedded"], query_basename) + '.pkl', "wb") as fOut:
                    pickle.dump(self.queries_content[query_basename], fOut)

    def load_and_embedd_single_query(self, query_filename, options):
        """Loads and embeds a single query from a file and returns its basename, IDs, queries, and embeddings.
        Args:
            query_filename (str): Path to the file containing the query.
            options (dict): Options for embedding, including verbosity and GPU settings.
        Returns:
            list: A list containing the basename of the query, its IDs, queries, and embeddings
        """
        queries, query_ids, query_basename = self.load_single_query(query_filename, options)
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
        glob_start = time.time()
        if len(options["gpu_device"]) > 1:
                start = time.time()
                pool = self.embedder.start_multi_process_pool(options["gpu_device"])
                end = time.time()
                print(f"---Time to start multi-process pool for embedding: {end - start} seconds")
                start = time.time()
                text_embedding = self.embedder.encode_multi_process(text, pool = pool, batch_size=options["batch_size"], chunk_size=options['single_worker_chunk_size'])
                end = time.time()
                print(f"---Time to embed with multi-process pool: {end - start} seconds")
                self.embedder.stop_multi_process_pool(pool)
        elif len(options["gpu_device"]) == 1:
                text_embedding = self.embedder.encode(text, convert_to_numpy=True, show_progress_bar = options["verbose"], device= options["gpu_device"][0]) #convert_to_tensor=True 
        if options["verbose"]: print(f"---Embedding time with {0 if options.get('gpu_device') == None else len(options['gpu_device'])} GPUs: {time.time() - glob_start} seconds")
        return text_embedding

    def calculate_similarity(self, query_info, corpus_info, options):
        from sentence_transformers import util # Moving import here to avoid long import time
        corpus_ids = corpus_info["textIDs"]
        corpus_embeddings = corpus_info["embeddings"]
        corpus_texts = corpus_info["all_corpus"]

        query_ids = query_info['query_ids']
        query_embeddings = query_info["embeddings"]
        query_texts = query_info["queries"]

        if options["gpu_device"] != None and options["use_gpu_for_sim_calculation"]:
            search = self.calculate_similarity_gpu(query_embeddings, corpus_embeddings, options["top_k"], util, options['cuda'], options['from_numpy'], options["verbose"], options["order"])
        else:
            search = self.calculate_similarity_cpu(query_embeddings, corpus_embeddings, options["top_k"], util, options["verbose"], options["order"])

        if self.reranker:
            if options["verbose"]: print("---Reranking results with cross-encoder")
            if options["order"] == "corpus-query":
                search = self.rerank_results(corpus_texts, query_texts, search, options)
            else:
                search = self.rerank_results(query_texts, corpus_texts, search, options)

        if options["order"] == "corpus-query":
            matches = self.find_best_matches(corpus_ids, query_ids, search)
        else:
            matches = self.find_best_matches(query_ids, corpus_ids, search)
        return matches

    def rerank_results(self, query_texts, corpus_texts, search, options):
        query_indexes = []
        corpus_indexes = []
        text_pairs = []
        results = []
        for query_idx, query_info in enumerate(search):
            results.append([])
            for corpus_info in query_info:
                query_indexes.append(query_idx)
                corpus_indexes.append(corpus_info['corpus_id'])
                text_pairs.append([query_texts[query_idx], corpus_texts[corpus_info['corpus_id']]])
        reranked_scores = self.make_rerank(text_pairs, options)
        
        for idx, score in enumerate(reranked_scores):
            query_idx = query_indexes[idx]
            corpus_idx = corpus_indexes[idx]
            results[query_idx].append({'score': score, 'corpus_id': corpus_idx})
        return results

    def make_rerank(self, sentence_pairs, options):
        import torch
        if options['verbose']: print(f"---Reranking {len(sentence_pairs)} sentence pairs with cross-encoder model")
        start = time.time()
        if len(options["gpu_device"]) > 1:
                pool = self.reranker.start_multi_process_pool(options["gpu_device"])
                scores = self.reranker.predict(sentence_pairs, pool=pool, batch_size=options["batch_size"], chunk_size=options['single_worker_chunk_size'], activation_fn=torch.nn.Sigmoid())
                self.reranker.stop_multi_process_pool(pool)
        elif len(options["gpu_device"]) == 1:
                scores = self.reranker.predict(sentence_pairs, device= options["gpu_device"][0], batch_size=options["batch_size"], activation_fn=torch.nn.Sigmoid()) #convert_to_tensor=True 
        else:
                scores = self.reranker.predict(sentence_pairs, batch_size=options["batch_size"], activation_fn=torch.nn.Sigmoid()) #convert_to_tensor=True 
                #raise Exception("You need to provide GPU to use reranker. It should be a list of GPU devices like: ['cuda:0'] or ['cuda:0', 'cuda:1']")                
        if options["verbose"]: print(f"---Reranking time with {0 if options.get('gpu_device') == None else len(options['gpu_device'])} GPUs: {time.time() - start} seconds")
        return scores

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
            pubmed_index, n_papers = self.load_pubmed_index(corpus_filename, split_level=options["split_level"]) # abstracts
            total_papers += n_papers
            all_textIDs.extend(pubmed_index.keys())
            all_corpus.extend(pubmed_index.values())
            
            n_items = total_papers if options['chunk_size_sentences'] == 0 else len(all_corpus)
            threshold = options['chunk_size'] if options['chunk_size_sentences'] == 0 else options['chunk_size_sentences']
            if n_items >= threshold:
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