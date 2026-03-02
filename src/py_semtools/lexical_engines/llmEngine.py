import os, re
from py_semtools.lexical_engines.engine_baseclass import LexicalEngineBaseClass
from ontogpt.io.template_loader import get_template_details, get_template_path
from ontogpt.engines.spires_engine import SPIRESEngine

class LLMengine(LexicalEngineBaseClass):
    def __init__(self, gpu_devices = []):
        self.model_name = None
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

        if template:
            template_details = get_template_details(template=template)
        else:
            raise ValueError("No template specified. Use -t/--template option.")

        ke = SPIRESEngine(template_details=template_details, model=model, temperature=1,
                          api_base=None, api_version=None, model_provider=None, system_message=None, max_text_length=None,
                          **kwargs)

        counts = 0
        if options['verbose']: print(f"\n-Calculating similarities between queries and corpus using {model} model with {template} template:")
        
        best_matches = {}
        #          for kwdID, matches in best_matches.items():
        #    for textID, score in matches.items():
        for idx, text in enumerate(corpus_texts):        
            results = ke.extract_from_text( text=text, cls=None, show_prompt=None)
            hpos = self._get_genuine_hpos(results.named_entities)
            print("-"*30)
            print(f"Text ID: {corpus_ids[idx]}")
            print(f"Original Text: {text}")
            print(f"Named Entities: {results.named_entities}")
            print(f"Extracted HPOs: {hpos}")
            print("-"*30)
            counts += 1
            best_matches[corpus_ids[idx]] = hpos
            if counts > 20: 
                print(best_matches)
                break

        return best_matches
    
    def find_best_matches(self, results_indexes, scores, query_ids, corpus_ids):
        pass

    def _get_genuine_hpos(self, named_entities):
        return {entity.id : "-" for entity in named_entities if re.match(r"HP:\d{7}", entity.id) and len(entity.original_spans) > 0}