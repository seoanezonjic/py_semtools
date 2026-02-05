from py_semtools.lexical_engines.engine_baseclass import LexicalEngineBaseClass

class FMengine(LexicalEngineBaseClass):
    def __init__(self, gpu_devices = []):
        self.model_name = None
        self.queries_content = {}

    def init_model(self, model_name: str, verbose: bool = False) -> None:
        if model_name not in ["bm25", "tfidf"]:
            raise Exception(f"Model {model_name} not supported in FMengine")
        if verbose: print(f"\n-Loading {model_name} model")
        self.model_name = model_name

    def load_several_queries(self, options, query_filenames, verbose=False):
        if verbose: print("\n-Loading and embedding queries:")
        for query_filename in query_filenames:
            queries, query_ids, query_basename = self.load_single_query(query_filename, options)
            if verbose: print(f"---Loaded {len(set(query_ids))} unique query IDs from {query_basename}")

    def load_several_corpora(self, options, corpus_filenames, verbose=False):
        all_textIDs = []; all_corpus = []; total_papers = 0 
        for corpus_filename in corpus_filenames:
            if verbose: print(f"---Loading corpus of {corpus_filename}")
            pubmed_index, n_papers = self.load_pubmed_index(corpus_filename, options["split"]) # abstracts
            if verbose: print(f"------Loaded {n_papers} documents with {len(pubmed_index)} sentences from {corpus_filename}")
            total_papers += n_papers
            all_textIDs.extend(pubmed_index.keys())
            all_corpus.extend(pubmed_index.values())
        corpus_info = {'textIDs': all_textIDs, "all_corpus": all_corpus}
        if verbose: print(f"---Loaded a total of {total_papers} documents with {len(all_corpus)} sentences from {len(corpus_filenames)} corpora")
        return corpus_info

    def calculate_similarity(self, query_info, corpus_info, options):
        corpus_ids = corpus_info["textIDs"]
        corpus_text = corpus_info["all_corpus"]

        query_ids = query_info['query_ids']
        query_text = query_info["queries"]

        if options["order"] == "corpus-query":
            corpus_ids, query_ids = query_ids, corpus_ids
            corpus_text, query_text = query_text, corpus_text

        if self.model_name == "bm25":
            results_indexes, scores = self.calculate_bm25_similarity(query_text, corpus_text, options)
        elif self.model_name == "tfidf":
            results_indexes, scores = self.calculate_tfidf_similarity(query_text, corpus_text, options)
        matches = self.find_best_matches(results_indexes, scores, query_ids, corpus_ids)
        return matches
    
    def calculate_bm25_similarity(self, query_text, corpus_text, options):
        import bm25s
        import numpy as np
        
        # Adjust top_k to corpus size if top_k is infinite, otherwise bm25s throws an error (althouhg it works with sentence_transformers)
        if options['top_k'] == np.inf: options['top_k'] = len(corpus_text)
        
        retriever = bm25s.BM25()

        tokenized_corpus = bm25s.tokenize(corpus_text)
        tokenized_query = bm25s.tokenize(query_text)
        
        retriever.index(tokenized_corpus, show_progress=options['verbose'])
        results_indexes, scores = retriever.retrieve(tokenized_query, k=options['top_k'], return_as="indices")
        return results_indexes, scores

    def calculate_tfidf_similarity(self, query_text, corpus_text, options):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
        import numpy as np

        vectorizer = TfidfVectorizer().fit(corpus_text)
        tfidf_matrix = vectorizer.transform(corpus_text)
        query_vecs = vectorizer.transform(query_text)        
        cosine_sim_matrix = linear_kernel(query_vecs, tfidf_matrix)

        n_queries = cosine_sim_matrix.shape[0]
        n_corpus = cosine_sim_matrix.shape[1]
        min_k = min(options['top_k'], n_corpus)
        batch_indices = np.empty((n_queries, min_k), dtype=int)
        batch_scores = np.empty((n_queries, min_k), dtype=float)
        
        for i in range(n_queries):
            sim_scores = cosine_sim_matrix[i]
            top_indices = sim_scores.argsort()[-min_k:][::-1]
            batch_indices[i] = top_indices
            batch_scores[i] = sim_scores[top_indices]
            
        return batch_indices, batch_scores

    def find_best_matches(self, results_indexes, scores, query_ids, corpus_ids):
        matches = {}
        for row_idx, res_idxs  in enumerate(results_indexes):
            query_id = query_ids[row_idx]
            matches[query_id] = {}
            for col_idx, res_idx in enumerate(res_idxs):
                corpus_id = corpus_ids[res_idx]
                matches[query_id][corpus_id] = scores[row_idx, col_idx]
        return matches