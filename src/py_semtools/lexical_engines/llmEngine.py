import os, re, logging, requests, warnings
from py_semtools.lexical_engines.engine_baseclass import LexicalEngineBaseClass
from ontogpt.io.template_loader import get_template_details, get_template_path
from ontogpt.engines.spires_engine import SPIRESEngine

class LLMengine(LexicalEngineBaseClass):
    def __init__(self, gpu_devices = []):
        self.model_name = None
        self.embedder = None
        self.reranker = None
        self.queries_content = {}

    def init_model(self, model_name: str, model_provider: str = None, verbose: bool = False) -> None:
        if verbose: print(f"\n-Using {model_name} model with {model_provider} provider")
        self.model_name = model_name
        self.model_provider = model_provider

    def load_several_queries(self, options, query_filenames, verbose=False):
        if verbose: print("\n-Loading and embedding queries:")
        for query_filename in query_filenames:
            query_basename = os.path.splitext(os.path.basename(query_filename))[0]
            self.queries_content[query_basename] = {'template': options["template"]}
            if options["verbose"]: print(f"---Using query template {options['template']} for {query_basename}")

    def load_several_corpora(self, options, corpus_filenames, verbose=False):
        all_textIDs = []; all_corpus = []; total_papers = 0 
        for corpus_filename in corpus_filenames:
            if verbose: print(f"---Loading corpus of {corpus_filename}")
            pubmed_index, n_papers = self.load_pubmed_index(corpus_filename, split_level=options["split_level"]) # abstracts
            if verbose: print(f"------Loaded {n_papers} documents with {len(pubmed_index)} sentences from {corpus_filename}")
            total_papers += n_papers
            all_textIDs.extend(pubmed_index.keys())
            all_corpus.extend(pubmed_index.values())
        corpus_info = {'textIDs': all_textIDs, "all_corpus": all_corpus}
        if verbose: print(f"---Loaded a total of {total_papers} documents with {len(all_corpus)} sentences from {len(corpus_filenames)} corpora")
        return corpus_info

    def calculate_similarity(self, query_info, corpus_info, options):
        kwargs = {'recurse': True, 'auto_prefix': 'AUTO'}
        corpus_texts = corpus_info["all_corpus"]
        corpus_ids = corpus_info["textIDs"]
        template = options["template"]
        model = options["model_name"]
        address = options["listening_addresses"]
        
        if options["logging_level"]:
            loggin_levels = {"DEBUG": logging.DEBUG, "INFO":logging.INFO, "WARNING":logging.WARNING, "ERROR":logging.ERROR}
            logger = logging.getLogger()
            logger.setLevel(level=loggin_levels[options["logging_level"]])

        if options['verbose']: print(f"\n-Calculating similarities between queries and corpus using {model} model with {template} template:")

        if template:
            template_details = get_template_details(template=template)
        else:
            raise ValueError("No template specified. Use -t/--template option.")

        ke = SPIRESEngine(template_details=template_details, model=model, temperature=1,
                          api_base=address, api_version=None, model_provider=None, system_message=None, max_text_length=None,
                          **kwargs)

        best_matches = {}
        for idx, text in enumerate(corpus_texts):
            if options['verbose']: print(f"Extracting for text ID {corpus_ids[idx]}")  # Print the beginning of the text for context

            #results = ke.extract_from_text( text=text, cls=None, show_prompt=None)
            results = self._execute_with_retry( func=ke.extract_from_text, 
                exception_type=requests.exceptions.ReadTimeout, max_retries=3, text_id=corpus_ids[idx], 
                text=text, cls=None, show_prompt=None)
            
            if results != None:
                hpos = self._get_genuine_terms(results.named_entities, options["regex_tag"])
                best_matches[corpus_ids[idx]] = hpos

                if options['stream_write_results']:
                    backup_file = os.path.join(options["output_file"], self.current_query_basename + "_backup")
                    warnings.warn(f"Streaming mode enabled: writing {len(hpos)}terms for text ID {corpus_ids[idx]} immediately after processing (inside file {backup_file} ).")
                    self.save_similarities(backup_file, {corpus_ids[idx]: hpos}, options) # Write results in streaming mode if enabled
        
        return best_matches
    
    def find_best_matches(self, results_indexes, scores, query_ids, corpus_ids):
        pass


    def _get_genuine_terms(self, named_entities, regex_tag):
        if regex_tag:
            return {entity.id : "-" for entity in named_entities if re.match(regex_tag, entity.id) and len(entity.original_spans) > 0}
        else:
            return {entity.id : "-" for entity in named_entities if len(entity.original_spans) > 0}
        
    def _execute_with_retry(self, func, exception_type, max_retries, text_id=None, *args, **kwargs):
        tries = 0
        while True:
            try:
                return func(*args, **kwargs)                
            except exception_type as e:
                tries += 1
                warnings.warn(f"{exception_type.__name__} occurred for text ID {text_id}. Retrying... Error: {e}")
                if max_retries > 0 and tries >= max_retries:
                    warnings.warn(f"Failed to process text ID {text_id} after {max_retries} attempts. Skipping this text.")
                    return None  # Salimos devolviendo None tras agotar intentos