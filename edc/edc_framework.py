from edc.extract import Extractor
from edc.schema_definition import SchemaDefiner
from edc.schema_canonicalization import SchemaCanonicalizer
from edc.entity_extraction import EntityExtractor
import edc.utils.llm_utils as llm_utils
from typing import List
from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
from edc.schema_retriever import SchemaRetriever
from tqdm import tqdm
import os
import csv
import pathlib
from functools import partial
import copy
import logging
from sentence_transformers import SentenceTransformer
from importlib import reload
import random
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

reload(logging)
logger = logging.getLogger(__name__)


class EDC:
    def __init__(self, **edc_configuration) -> None:
        # OIE module settings
        self.oie_llm_name = edc_configuration["oie_llm"]
        self.oie_prompt_template_file_path = edc_configuration["oie_prompt_template_file_path"]
        self.oie_few_shot_example_file_path = edc_configuration["oie_few_shot_example_file_path"]

        # Schema Definition module settings
        self.sd_llm_name = edc_configuration["sd_llm"]
        self.sd_template_file_path = edc_configuration["sd_prompt_template_file_path"]
        self.sd_few_shot_example_file_path = edc_configuration["sd_few_shot_example_file_path"]

        # Schema Canonicalization module settings
        self.sc_llm_name = edc_configuration["sc_llm"]
        self.sc_embedder_name = edc_configuration["sc_embedder"]
        self.sc_template_file_path = edc_configuration["sc_prompt_template_file_path"]

        # Refinement settings
        self.sr_adapter_path = edc_configuration["sr_adapter_path"]

        self.sr_embedder_name = edc_configuration["sr_embedder"]
        self.oie_r_prompt_template_file_path = edc_configuration["oie_refine_prompt_template_file_path"]
        self.oie_r_few_shot_example_file_path = edc_configuration["oie_refine_few_shot_example_file_path"]

        self.ee_llm_name = edc_configuration["ee_llm"]
        self.ee_template_file_path = edc_configuration["ee_prompt_template_file_path"]
        self.ee_few_shot_example_file_path = edc_configuration["ee_few_shot_example_file_path"]

        self.em_template_file_path = edc_configuration["em_prompt_template_file_path"]

        self.initial_schema_path = edc_configuration["target_schema_path"]
        self.enrich_schema = edc_configuration["enrich_schema"]
        
        # Concurrency settings
        self.batch_size = edc_configuration.get("batch_size", 5)  # Default batch size of 5
        self.max_workers = edc_configuration.get("max_workers", 5)  # Default max workers of 5
        self.use_batch_processing = edc_configuration.get("use_batch_processing", False)  # Disabled by default

        if self.initial_schema_path is not None:
            reader = csv.reader(open(self.initial_schema_path, "r"))
            self.schema = {}
            for row in reader:
                relation, relation_definition = row
                self.schema[relation] = relation_definition
        else:
            self.schema = {}

        # Load the needed models and tokenizers
        self.needed_model_set = set(
            [self.oie_llm_name, self.sd_llm_name, self.sc_llm_name, self.sc_embedder_name, self.ee_llm_name]
        )

        self.loaded_model_dict = {}

        logging.basicConfig(level=edc_configuration["loglevel"])

        logger.info(f"Model used: {self.needed_model_set}")
        if self.use_batch_processing:
            logger.info(f"Batch processing enabled with batch size: {self.batch_size}, max workers: {self.max_workers}")

    def oie(
        self, input_text_list: List[str], previous_extracted_triplets_list: List[List[str]] = None, free_model=False
    ):
        if not llm_utils.is_model_openai(self.oie_llm_name):
            # Load the HF model for OIE
            oie_model, oie_tokenizer = self.load_model(self.oie_llm_name, "hf")
            extractor = Extractor(oie_model, oie_tokenizer)
        else:
            extractor = Extractor(openai_model=self.oie_llm_name)

        oie_triples_list = []
        entity_hint_list = None
        relation_hint_list = None

        if previous_extracted_triplets_list is not None:
            # Refined OIE
            logger.info("Running Refined OIE...")
            oie_refinement_prompt_template_str = open(self.oie_r_prompt_template_file_path).read()
            oie_refinement_few_shot_examples_str = open(self.oie_r_few_shot_example_file_path).read()

            logger.info("Putting together the refinement hint...")
            entity_hint_list, relation_hint_list = self.construct_refinement_hint(
                input_text_list, previous_extracted_triplets_list, free_model=free_model
            )

            assert len(previous_extracted_triplets_list) == len(input_text_list)
            
            if self.use_batch_processing and llm_utils.is_model_openai(self.oie_llm_name):
                # Process in batches for OpenAI models
                oie_triples_list = self._batch_process_oie_with_hints(
                    input_text_list, 
                    entity_hint_list, 
                    relation_hint_list, 
                    oie_refinement_few_shot_examples_str, 
                    oie_refinement_prompt_template_str
                )
            else:
                # Process sequentially
                for idx, input_text in enumerate(tqdm(input_text_list)):
                    input_text = input_text_list[idx]
                    entity_hint_str = entity_hint_list[idx]
                    relation_hint_str = relation_hint_list[idx]
                    refined_oie_triplets = extractor.extract(
                        input_text,
                        oie_refinement_few_shot_examples_str,
                        oie_refinement_prompt_template_str,
                        entity_hint_str,
                        relation_hint_str,
                    )
                    oie_triples_list.append(refined_oie_triplets)
        else:
            # Normal OIE
            entity_hint_list = ["" for _ in input_text_list]
            relation_hint_list = ["" for _ in input_text_list]
            logger.info("Running OIE...")
            oie_few_shot_examples_str = open(self.oie_few_shot_example_file_path).read()
            oie_few_shot_prompt_template_str = open(self.oie_prompt_template_file_path).read()
            
            if self.use_batch_processing and llm_utils.is_model_openai(self.oie_llm_name):
                # Process in batches for OpenAI models
                oie_triples_list = self._batch_process_oie(
                    input_text_list, 
                    oie_few_shot_examples_str, 
                    oie_few_shot_prompt_template_str
                )
            else:
                # Process sequentially
                for input_text in tqdm(input_text_list):
                    oie_triples = extractor.extract(
                        input_text, 
                        oie_few_shot_examples_str, 
                        oie_few_shot_prompt_template_str
                    )
                    oie_triples_list.append(oie_triples)
                    logger.debug(f"{input_text}\n -> {oie_triples}\n")

        logger.info("OIE finished.")

        if free_model:
            logger.info(f"Freeing model {self.oie_llm_name} as it is no longer needed")
            llm_utils.free_model(oie_model, oie_tokenizer)
            del self.loaded_model_dict[self.oie_llm_name]

        return oie_triples_list, entity_hint_list, relation_hint_list
    
    def _batch_process_oie(self, input_text_list, few_shot_examples_str, prompt_template_str):
        """Process OIE in batches for faster processing with OpenAI models"""
        logger.info(f"Processing OIE in batches with batch size {self.batch_size}")
        results = [None] * len(input_text_list)
        
        # Define the function to process a single item
        def process_single_text(idx, text):
            system_prompt = prompt_template_str
            user_prompt = f"Text: {text}\n\nFew shot examples:\n{few_shot_examples_str}"
            messages = [{"role": "user", "content": user_prompt}]
            
            response = llm_utils.openai_chat_completion_with_key(
                model=self.oie_llm_name,
                messages=messages,
                system_prompt=system_prompt
            )
            
            # Parse the response into triplets
            triplets = []
            try:
                triplets_str = response.strip().split("\n")
                for triplet_str in triplets_str:
                    if triplet_str.strip():
                        # Simple parsing - can be improved based on format
                        if '[' in triplet_str and ']' in triplet_str:
                            # Try to parse as a list
                            try:
                                triplet = eval(triplet_str)
                                if isinstance(triplet, list) and len(triplet) == 3:
                                    triplets.append(triplet)
                            except:
                                # Fallback to simple splitting
                                triple_parts = triplet_str.strip('[]').split(',')
                                if len(triple_parts) >= 3:
                                    triplets.append([p.strip(' "\'') for p in triple_parts[:3]])
                        else:
                            # Try simple splitting
                            triple_parts = triplet_str.split(',')
                            if len(triple_parts) >= 3:
                                triplets.append([p.strip(' "\'') for p in triple_parts[:3]])
            except Exception as e:
                logger.error(f"Error parsing triplets: {e}")
            
            return idx, triplets
        
        # Process in batches
        num_batches = math.ceil(len(input_text_list) / self.batch_size)
        for batch_idx in tqdm(range(num_batches), desc="OIE Batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(input_text_list))
            batch_texts = input_text_list[start_idx:end_idx]
            
            # Process batch concurrently
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(process_single_text, start_idx + i, text)
                    for i, text in enumerate(batch_texts)
                ]
                
                for future in as_completed(futures):
                    try:
                        idx, triplets = future.result()
                        results[idx] = triplets
                        logger.debug(f"Completed OIE for text {idx}")
                    except Exception as e:
                        logger.error(f"Error in OIE processing: {e}")
        
        return results
    
    def _batch_process_oie_with_hints(self, input_text_list, entity_hint_list, relation_hint_list, 
                                     few_shot_examples_str, prompt_template_str):
        """Process OIE with hints in batches for faster processing with OpenAI models"""
        logger.info(f"Processing OIE with hints in batches with batch size {self.batch_size}")
        results = [None] * len(input_text_list)
        
        # Define the function to process a single item
        def process_single_text_with_hints(idx, text, entity_hint, relation_hint):
            system_prompt = prompt_template_str
            user_prompt = f"Text: {text}\n\nEntity hint: {entity_hint}\n\nRelation hint: {relation_hint}\n\nFew shot examples:\n{few_shot_examples_str}"
            messages = [{"role": "user", "content": user_prompt}]
            
            response = llm_utils.openai_chat_completion_with_key(
                model=self.oie_llm_name,
                messages=messages,
                system_prompt=system_prompt
            )
            
            # Parse the response into triplets (same as in _batch_process_oie)
            triplets = []
            try:
                triplets_str = response.strip().split("\n")
                for triplet_str in triplets_str:
                    if triplet_str.strip():
                        if '[' in triplet_str and ']' in triplet_str:
                            try:
                                triplet = eval(triplet_str)
                                if isinstance(triplet, list) and len(triplet) == 3:
                                    triplets.append(triplet)
                            except:
                                triple_parts = triplet_str.strip('[]').split(',')
                                if len(triple_parts) >= 3:
                                    triplets.append([p.strip(' "\'') for p in triple_parts[:3]])
                        else:
                            triple_parts = triplet_str.split(',')
                            if len(triple_parts) >= 3:
                                triplets.append([p.strip(' "\'') for p in triple_parts[:3]])
            except Exception as e:
                logger.error(f"Error parsing triplets: {e}")
            
            return idx, triplets
        
        # Process in batches
        num_batches = math.ceil(len(input_text_list) / self.batch_size)
        for batch_idx in tqdm(range(num_batches), desc="OIE Batches with Hints"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(input_text_list))
            batch_texts = input_text_list[start_idx:end_idx]
            batch_entity_hints = entity_hint_list[start_idx:end_idx]
            batch_relation_hints = relation_hint_list[start_idx:end_idx]
            
            # Process batch concurrently
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        process_single_text_with_hints, 
                        start_idx + i, 
                        text, 
                        batch_entity_hints[i], 
                        batch_relation_hints[i]
                    )
                    for i, text in enumerate(batch_texts)
                ]
                
                for future in as_completed(futures):
                    try:
                        idx, triplets = future.result()
                        results[idx] = triplets
                        logger.debug(f"Completed OIE with hints for text {idx}")
                    except Exception as e:
                        logger.error(f"Error in OIE with hints processing: {e}")
        
        return results

    def load_model(self, model_name, model_type):
        assert model_type in ["sts", "hf"]  # Either a sentence transformer or a huggingface LLM
        if model_name in self.loaded_model_dict:
            logger.info(f"Model {model_name} is already loaded, reusing it.")
        else:
            logger.info(f"Loading model {model_name}")
            if model_type == "hf":
                model, tokenizer = (
                    AutoModelForCausalLM.from_pretrained(model_name, device_map="auto"),
                    AutoTokenizer.from_pretrained(model_name),
                )
                self.loaded_model_dict[model_name] = (model, tokenizer)
            elif model_type == "sts":
                model = SentenceTransformer(model_name, trust_remote_code=True)
                self.loaded_model_dict[model_name] = model
        return self.loaded_model_dict[model_name]

    def schema_definition(self, triplets_list, free_model=False):
        if not llm_utils.is_model_openai(self.sd_llm_name):
            # Load the HF model for Schema Definition
            sd_model, sd_tokenizer = self.load_model(self.sd_llm_name, "hf")
            schema_definer = SchemaDefiner(model=sd_model, tokenizer=sd_tokenizer)
        else:
            schema_definer = SchemaDefiner(openai_model=self.sd_llm_name)

        logger.info("Running Schema Definition...")
        sd_few_shot_examples_str = open(self.sd_few_shot_example_file_path).read()
        sd_template_str = open(self.sd_template_file_path).read()

        # Flatten the triplets list
        flat_triplets = [triplet for triplets in triplets_list for triplet in triplets]
        unique_relations = set([triplet[1] for triplet in flat_triplets])
        unique_relations = sorted(list(unique_relations))  # Sort for deterministic output

        logger.info(f"Found {len(unique_relations)} unique relations: {unique_relations}")
        relation_definition_dict = {}
        
        if self.use_batch_processing and llm_utils.is_model_openai(self.sd_llm_name):
            # Process in batches for OpenAI models
            relation_definition_dict = self._batch_process_schema_definition(
                unique_relations, 
                sd_few_shot_examples_str, 
                sd_template_str
            )
        else:
            # Process sequentially
            for relation in tqdm(unique_relations):
                definition = schema_definer.define(relation, sd_few_shot_examples_str, sd_template_str)
                relation_definition_dict[relation] = definition
                logger.debug(f"{relation} -> {definition}")

        logger.info("Schema Definition finished.")

        if free_model:
            logger.info(f"Freeing model {self.sd_llm_name} as it is no longer needed")
            llm_utils.free_model(sd_model, sd_tokenizer)
            del self.loaded_model_dict[self.sd_llm_name]

        # Convert the single dictionary to a list of dictionaries, one per input text
        sd_dict_list = []
        for triplets in triplets_list:
            text_relations = {triplet[1] for triplet in triplets}
            text_dict = {rel: relation_definition_dict.get(rel, "") for rel in text_relations}
            sd_dict_list.append(text_dict)
        
        return sd_dict_list
    
    def _batch_process_schema_definition(self, unique_relations, few_shot_examples_str, template_str):
        """Process schema definition in batches for faster processing with OpenAI models"""
        logger.info(f"Processing schema definition in batches with batch size {self.batch_size}")
        results = {}
        
        # Define the function to process a single relation
        def process_single_relation(relation):
            system_prompt = template_str
            user_prompt = f"Relation: {relation}\n\nFew shot examples:\n{few_shot_examples_str}"
            messages = [{"role": "user", "content": user_prompt}]
            
            response = llm_utils.openai_chat_completion_with_key(
                model=self.sd_llm_name,
                messages=messages,
                system_prompt=system_prompt
            )
            
            # Extract the definition from the response
            definition = response.strip()
            
            return relation, definition
        
        # Process in batches
        num_batches = math.ceil(len(unique_relations) / self.batch_size)
        for batch_idx in tqdm(range(num_batches), desc="Schema Definition Batches"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(unique_relations))
            batch_relations = unique_relations[start_idx:end_idx]
            
            # Process batch concurrently
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(process_single_relation, relation)
                    for relation in batch_relations
                ]
                
                for future in as_completed(futures):
                    try:
                        relation, definition = future.result()
                        results[relation] = definition
                        logger.debug(f"Completed schema definition for relation: {relation} -> {definition}")
                    except Exception as e:
                        logger.error(f"Error in schema definition processing: {e}")
        
        return results

    def schema_canonicalization(
        self,
        input_text_list: List[str],
        oie_triplets_list: List[List[str]],
        schema_definition_dict_list: List[dict],
        free_model=False,
    ):
        assert len(input_text_list) == len(oie_triplets_list) and len(input_text_list) == len(
            schema_definition_dict_list
        )
        logger.info("Running Schema Canonicalization...")

        sc_verify_prompt_template_str = open(self.sc_template_file_path).read()

        # if self.sc_embedder_name not in self.loaded_model_dict:
        #     logger.info(f"Loading model {self.sc_embedder_name}.")
        #     sc_embedder = SentenceTransformer(self.sc_embedder_name, trust_remote_code=True)
        #     self.loaded_model_dict[self.sc_embedder_name] = sc_embedder

        # else:
        #     logger.info(f"Model {self.sc_embedder_name} is already loaded, reusing it.")
        #     sc_embedder = self.loaded_model_dict[self.sc_embedder_name]
        
        # sc_embedder = self.load_model(self.sc_embedder_name, "sts")
        sc_embedder = "BGE-MODEL"
        

        if not llm_utils.is_model_openai(self.sc_llm_name):
            sc_verify_model, sc_verify_tokenizer = self.load_model(self.sc_llm_name, "sts")
            # if self.sc_llm_name not in self.loaded_model_dict:
            #     logger.info(f"Loading model {self.sc_llm_name}")
            #     sc_verify_model, sc_verify_tokenizer = (
            #         AutoModelForCausalLM.from_pretrained(self.sc_llm_name, device_map="auto"),
            #         AutoTokenizer.from_pretrained(self.sc_llm_name),
            #     )
            #     self.loaded_model_dict[self.sc_llm_name] = (sc_verify_model, sc_verify_tokenizer)
            # else:
            #     logger.info(f"Model {self.sc_llm_name} is already loaded, reusing it.")
            #     sc_verify_model, sc_verify_tokenizer = self.loaded_model_dict[self.sc_llm_name]
            schema_canonicalizer = SchemaCanonicalizer(self.schema, sc_embedder, sc_verify_model, sc_verify_tokenizer)
        else:
            schema_canonicalizer = SchemaCanonicalizer(self.schema, sc_embedder, verify_openai_model=self.sc_llm_name)

        canonicalized_triplets_list = []
        canon_candidate_dict_per_entry_list = []

        for idx, input_text in enumerate(tqdm(input_text_list)):
            oie_triplets = oie_triplets_list[idx]
            canonicalized_triplets = []
            sd_dict = schema_definition_dict_list[idx]
            canon_candidate_dict_list = []
            for oie_triplet in oie_triplets:
                canonicalized_triplet, canon_candidate_dict = schema_canonicalizer.canonicalize(
                    input_text, oie_triplet, sd_dict, sc_verify_prompt_template_str, self.enrich_schema
                )
                canonicalized_triplets.append(canonicalized_triplet)
                canon_candidate_dict_list.append(canon_candidate_dict)

            canonicalized_triplets_list.append(canonicalized_triplets)
            canon_candidate_dict_per_entry_list.append(canon_candidate_dict_list)

            logger.debug(f"{input_text}\n, {oie_triplets} ->\n {canonicalized_triplets}")
            logger.debug(f"Retrieved candidate relations {canon_candidate_dict}")
        logger.info("Schema Canonicalization finished.")

        if free_model:
            logger.info(f"Freeing model {self.sc_embedder_name, self.sc_llm_name} as it is no longer needed")
            llm_utils.free_model(sc_embedder)
            llm_utils.free_model(sc_verify_model, sc_verify_tokenizer)
            del self.loaded_model_dict[self.sc_llm_name]

        return canonicalized_triplets_list, canon_candidate_dict_per_entry_list

    def construct_refinement_hint(
        self,
        input_text_list: List[str],
        extracted_triplets_list: List[List[List[str]]],
        include_relation_example="self",
        relation_top_k=10,
        free_model=False,
    ):
        entity_extraction_few_shot_examples_str = open(self.ee_few_shot_example_file_path).read()
        entity_extraction_prompt_template_str = open(self.ee_template_file_path).read()

        entity_merging_prompt_template_str = open(self.em_template_file_path).read()

        entity_hint_list = []
        relation_hint_list = []

        # Initialize entity extractor
        if not llm_utils.is_model_openai(self.ee_llm_name):
            # Load the HF model for Schema Definition
            ee_model, ee_tokenizer = self.load_model(self.ee_llm_name, "hf")
            # if self.ee_llm_name not in self.loaded_model_dict:
            #     logger.info(f"Loading model {self.ee_llm_name}")
            #     ee_model, ee_tokenizer = (
            #         AutoModelForCausalLM.from_pretrained(self.ee_llm_name, device_map="auto"),
            #         AutoTokenizer.from_pretrained(self.ee_llm_name),
            #     )
            #     self.loaded_model_dict[self.ee_llm_name] = (ee_model, ee_tokenizer)
            # else:
            #     logger.info(f"Model {self.ee_llm_name} is already loaded, reusing it.")
            #     ee_model, ee_tokenizer = self.loaded_model_dict[self.ee_llm_name]
            entity_extractor = EntityExtractor(model=ee_model, tokenizer=ee_tokenizer)
        else:
            entity_extractor = EntityExtractor(openai_model=self.sd_llm_name)

        # Initialize schema retriever
        # if self.sr_embedder_name not in self.loaded_model_dict:
        #     logger.info(f"Loading model {self.sr_embedder_name}.")
        #     sr_embedding_model = SentenceTransformer(self.sr_embedder_name)
        #     self.loaded_model_dict[self.sr_embedder_name] = sr_embedding_model
        # else:
        #     sr_embedding_model = self.loaded_model_dict[self.sr_embedder_name]
        #     logger.info(f"Model {self.sr_embedder_name} is already loaded, reusing it.")
        sr_embedding_model = self.load_model(self.sr_embedder_name, "sts")

        schema_retriever = SchemaRetriever(
            self.schema,
            sr_embedding_model,
            None,
            finetuned_e5mistral=False,
        )

        relation_example_dict = {}
        if include_relation_example == "self":
            # Include an example of where this relation can be extracted
            for idx in range(len(input_text_list)):
                input_text_str = input_text_list[idx]
                extracted_triplets = extracted_triplets_list[idx]
                for triplet in extracted_triplets:
                    relation = triplet[1]
                    if relation not in relation_example_dict:
                        relation_example_dict[relation] = [{"text": input_text_str, "triplet": triplet}]
                    else:
                        relation_example_dict[relation].append({"text": input_text_str, "triplet": triplet})
        else:
            # Todo: allow to pass gold examples of relations
            pass

        for idx in tqdm(range(len(input_text_list))):
            input_text_str = input_text_list[idx]
            extracted_triplets = extracted_triplets_list[idx]

            previous_relations = set()
            previous_entities = set()

            for triplet in extracted_triplets:
                previous_entities.add(triplet[0])
                previous_entities.add(triplet[2])
                previous_relations.add(triplet[1])

            previous_entities = list(previous_entities)
            previous_relations = list(previous_relations)

            # Obtain candidate entities
            extracted_entities = entity_extractor.extract_entities(
                input_text_str, entity_extraction_few_shot_examples_str, entity_extraction_prompt_template_str
            )
            merged_entities = entity_extractor.merge_entities(
                input_text_str, previous_entities, extracted_entities, entity_merging_prompt_template_str
            )
            entity_hint_list.append(str(merged_entities))

            # Obtain candidate relations
            hint_relations = previous_relations

            retrieved_relations = schema_retriever.retrieve_relevant_relations(input_text_str)

            counter = 0

            for relation in retrieved_relations:
                if counter >= relation_top_k:
                    break
                else:
                    if relation not in hint_relations:
                        hint_relations.append(relation)

            candidate_relation_str = ""
            for relation_idx, relation in enumerate(hint_relations):
                if relation not in self.schema:
                    continue

                relation_definition = self.schema[relation]

                candidate_relation_str += f"{relation_idx+1}. {relation}: {relation_definition}\n"
                if include_relation_example == "self":
                    if relation not in relation_example_dict:
                        # candidate_relation_str += "Example: None.\n"
                        pass
                    else:
                        selected_example = None
                        if len(relation_example_dict[relation]) != 0:
                            selected_example = random.choice(relation_example_dict[relation])
                        # for example in relation_example_dict[relation]:
                        #     if example["text"] != input_text_str:
                        #         selected_example = example
                        #         break
                        if selected_example is not None:
                            candidate_relation_str += f"""For example, {selected_example['triplet']} can be extracted from "{selected_example['text']}"\n"""
                        else:
                            # candidate_relation_str += "Example: None.\n"
                            pass
            relation_hint_list.append(candidate_relation_str)

        if free_model:
            logger.info(f"Freeing model {self.sr_embedder_name, self.ee_llm_name} as it is no longer needed")
            llm_utils.free_model(sr_embedding_model)
            llm_utils.free_model(ee_model, ee_tokenizer)
            del self.loaded_model_dict[self.sr_embedder_name]
            del self.loaded_model_dict[self.ee_llm_name]
        return entity_hint_list, relation_hint_list

    def extract_kg(self, input_text_list: List[str], output_dir: str = None, refinement_iterations=0):
        if output_dir is not None:
            if os.path.exists(output_dir):
                logger.error(f"Output directory {output_dir} already exists! Quitting.")
                exit()
            for iteration in range(refinement_iterations + 1):
                pathlib.Path(f"{output_dir}/iter{iteration}").mkdir(parents=True, exist_ok=True)


        # EDC run
        logger.info("EDC starts running...")

        required_model_dict = {
            "oie": self.oie_llm_name,
            "sd": self.sd_llm_name,
            "sc_embed": self.sc_embedder_name,
            "sc_verify": self.sc_llm_name,
            "ee": self.ee_llm_name,
            "sr": self.sr_embedder_name,
        }

        triplets_from_last_iteration = None
        for iteration in range(refinement_iterations + 1):
            logger.info(f"Iteration {iteration}:")

            iteration_result_dir = f"{output_dir}/iter{iteration}"

            required_model_dict_current_iteration = copy.deepcopy(required_model_dict)

            del required_model_dict_current_iteration["oie"]
            oie_triplets_list, entity_hint_list, relation_hint_list = self.oie(
                input_text_list,
                free_model=self.oie_llm_name not in required_model_dict_current_iteration.values()
                and iteration == refinement_iterations,
                previous_extracted_triplets_list=triplets_from_last_iteration,
            )

            del required_model_dict_current_iteration["sd"]
            sd_dict_list = self.schema_definition(
                oie_triplets_list,
                free_model=self.sd_llm_name not in required_model_dict_current_iteration.values()
                and iteration == refinement_iterations,
            )

            del required_model_dict_current_iteration["sc_embed"]
            del required_model_dict_current_iteration["sc_verify"]
            canon_triplets_list, canon_candidate_dict_list = self.schema_canonicalization(
                input_text_list,
                oie_triplets_list,
                sd_dict_list,
                free_model=self.sc_llm_name not in required_model_dict_current_iteration.values()
                and iteration == refinement_iterations,
            )

            non_null_triplets_list = [
                [triple for triple in triplets if triple is not None] for triplets in canon_triplets_list
            ]
            # for triplets in canon_triplets_list:
            #     non_null_triplets = []
            #     for triple in triplets:
            #         if triple is not None:
            #             non_n
            triplets_from_last_iteration = non_null_triplets_list

            # Write results
            assert len(oie_triplets_list) == len(sd_dict_list) and len(sd_dict_list) == len(canon_triplets_list)

            json_results_list = []
            for idx in range(len(oie_triplets_list)):
                result_json = {
                    "index": idx,
                    "input_text": input_text_list[idx],
                    "entity_hint": entity_hint_list[idx],
                    "relation_hint": relation_hint_list[idx],
                    "oie": oie_triplets_list[idx],
                    "schema_definition": sd_dict_list[idx],
                    "canonicalization_candidates": str(canon_candidate_dict_list[idx]),
                    "schema_canonicalizaiton": canon_triplets_list[idx],
                }
                json_results_list.append(result_json)
            result_at_each_stage_file = open(f"{iteration_result_dir}/result_at_each_stage.json", "w")
            json.dump(json_results_list, result_at_each_stage_file, indent=4)

            final_result_file = open(f"{iteration_result_dir}/canon_kg.txt", "w")
            for idx, canon_triplets in enumerate(non_null_triplets_list):
                final_result_file.write(str(canon_triplets))
                if idx != len(canon_triplets_list) - 1:
                    final_result_file.write("\n")
                final_result_file.flush()

        return canon_triplets_list