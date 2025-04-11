#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np
import pandas as pd

from data.batch_manager import BatchManager
from models.edc.extractor import Extractor
from models.edc.definer import Definer
from models.edc.canonicalizer import Canonicalizer
from models.embedding.similarity import SimilarityCalculator
from models.clustering.relation_cluster import RelationCluster
from utils.logger import get_logger

logger = get_logger(__name__)


class Step1Processor:
    
    def __init__(
        self, 
        batch_size: int = 50, 
        similarity_threshold: float = 0.85,
        output_dir: str = "data/processed",
        extractor_config: Optional[Dict[str, Any]] = None,
        definer_config: Optional[Dict[str, Any]] = None,
        canonicalizer_config: Optional[Dict[str, Any]] = None
    ):

        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_manager = BatchManager(batch_size=batch_size)
        self.extractor = Extractor(**(extractor_config or {}))
        self.definer = Definer(**(definer_config or {}))
        self.canonicalizer = Canonicalizer(**(canonicalizer_config or {}))
        self.similarity_calculator = SimilarityCalculator()
        self.relation_cluster = RelationCluster()
        
        self.defined_relations = set() 
        self.relation_clusters = {}     
        
        logger.info(f"Initialized Step1Processor with batch size {batch_size} and "
                   f"similarity threshold {similarity_threshold}")
    
    def process(self, triplex_data: List[Dict[str, Any]]) -> Dict[str, Any]:

        logger.info("Starting Step 1 processing")
        
        relation_batches = self.batch_manager.create_batches(triplex_data)
        logger.info(f"Created {len(relation_batches)} batches of relations")
        
        processed_data = {
            "original_data": triplex_data,
            "processed_batches": [],
            "defined_relations": {},
            "relation_mappings": {}
        }
        
        for i, batch in enumerate(relation_batches):
            logger.info(f"Processing batch {i+1}/{len(relation_batches)}")
            batch_result = self._process_batch(batch, triplex_data)
            processed_data["processed_batches"].append(batch_result)
            
            processed_data["defined_relations"].update(batch_result["defined_relations"])
            processed_data["relation_mappings"].update(batch_result["relation_mappings"])
        
        self._save_processed_data(processed_data)
        
        logger.info("Finished Step 1 processing")
        return processed_data
    
    def _process_batch(self, batch: List[str], triplex_data: List[Dict[str, Any]]) -> Dict[str, Any]:

        batch_triples = self._extract_batch_triples(batch, triplex_data)
        

        extracted_triples = self.extractor.extract(batch_triples)
        
        defined_relations = self.definer.define(extracted_triples)
        

        relation_mappings = {}
        batch_defined_relations = {}
        
        for relation_name, relation_def in defined_relations.items():
            if not self.defined_relations:
                self.defined_relations.add(relation_name)
                batch_defined_relations[relation_name] = relation_def
                relation_mappings[relation_name] = relation_name
                self.relation_clusters[relation_name] = {relation_name}
            else:
                similarities = self.similarity_calculator.calculate_similarities(
                    relation_def, 
                    {r: defined_relations.get(r, "") for r in self.defined_relations}
                )
                
                most_similar = max(similarities.items(), key=lambda x: x[1])
                similar_relation, similarity_score = most_similar
                
                if similarity_score >= self.similarity_threshold:
                    relation_mappings[relation_name] = similar_relation
                    self.relation_clusters[similar_relation].add(relation_name)
                else:
                    self.defined_relations.add(relation_name)
                    batch_defined_relations[relation_name] = relation_def
                    relation_mappings[relation_name] = relation_name
                    self.relation_clusters[relation_name] = {relation_name}
        
        batch_result = {
            "batch_relations": batch,
            "extracted_triples": extracted_triples,
            "defined_relations": batch_defined_relations,
            "relation_mappings": relation_mappings,
            "relation_clusters": {k: list(v) for k, v in self.relation_clusters.items()}
        }
        
        return batch_result
    
    def _extract_batch_triples(self, batch: List[str], triplex_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        batch_triples = []
        
        for entry in triplex_data:
            batch_entry = {
                "text": entry["text"],
                "triples": [],
                "metadata": entry.get("metadata", {})
            }
            
            batch_entry_triples = [
                triple for triple in entry["triples"]
                if triple["relation"] in batch
            ]
            
            if batch_entry_triples:
                batch_entry["triples"] = batch_entry_triples
                batch_triples.append(batch_entry)
        
        return batch_triples
    
    def _save_processed_data(self, processed_data: Dict[str, Any]) -> None:
        output_path = self.output_dir / "step1_processed_data.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2)
        
        defined_relations_df = pd.DataFrame([
            {"relation": rel, "definition": defn}
            for rel, defn in processed_data["defined_relations"].items()
        ])
        defined_relations_path = self.output_dir / "defined_relations.csv"
        defined_relations_df.to_csv(defined_relations_path, index=False)
        
        relation_mappings_df = pd.DataFrame([
            {"original_relation": rel, "mapped_relation": mapped}
            for rel, mapped in processed_data["relation_mappings"].items()
        ])
        relation_mappings_path = self.output_dir / "relation_mappings.csv"
        relation_mappings_df.to_csv(relation_mappings_path, index=False)
        
        logger.info(f"Saved processed data to {self.output_dir}")
