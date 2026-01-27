#JPG: THIS IS AN ABSTRACT BASE CLASS FOR LEXICAL ENGINES AND SHOULD NOT BE INSTANTIATED DIRECTLY

import os, warnings, json, gzip
from abc import ABC, abstractmethod

class LexicalEngineBaseClass(ABC):

    def __init__(self):
        self.embedder = None
        self.queries_content = {}

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def find_best_matches(self):
        pass

    def load_single_query(self, query_filename, options):
        query_basename = os.path.splitext(os.path.basename(query_filename))[0]
        if options["verbose"]: print(f"---Loading query from {query_basename}")
        keyword_index = self.load_keyword_index(query_filename) # keywords used in queries
        queries, query_ids = self.prepare_queries_from_index(keyword_index)
        self.queries_content[query_basename] = {'query_ids': query_ids, "queries": queries}
        return queries, query_ids, query_basename

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
    
    def prepare_queries_from_index(self, keyword_index):
        """
        Prepare queries and their corresponding IDs from a keyword index.
        Args:
            keyword_index (dict): A dictionary where keys are keyword IDs and values are lists of keyword names (including synonyms and alternatives).
        Returns:
            tuple: A tuple containing two lists - one with all queries and another with their corresponding keyword IDs.
        """
        queries = []
        query_ids = []
        for kwdID, kwds in keyword_index.items():
            queries.extend(kwds)
            query_ids.extend([kwdID for i in range(0, len(kwds))])
        return queries, query_ids

    
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

    def calculate_similarities(self, options, corpus_info):
        if options.get("output_file"):
          for query_basename, query_info in self.queries_content.items():
            best_matches = self.calculate_similarity(query_info, corpus_info, options)
            if options['print_relevant_pairs']: self.print_similarities(query_info, corpus_info, best_matches, options)
            output_filename = os.path.join(options["output_file"],query_basename)
            self.save_similarities(output_filename, best_matches, options)

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