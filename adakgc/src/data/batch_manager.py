#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Dict, List, Any, Set
import numpy as np

logger = logging.getLogger(__name__)


class BatchManager:
    """
    Manages batches of relations for processing in the pipeline.
    
    Handles creation and grouping of relation batches from triplex data.
    """
    
    def __init__(self, batch_size: int = 50):
        """
        Initialize the batch manager.
        
        Args:
            batch_size: Size of each batch
        """
        self.batch_size = batch_size
        logger.info(f"Initialized BatchManager with batch size {batch_size}")
    
    def create_batches(self, triplex_data: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Create batches of relations from triplex data.
        
        Args:
            triplex_data: List of triplex data entries
            
        Returns:
            List of relation batches, where each batch is a list of relation names
        """
        # Extract all unique relations from the triplex data
        relations = self._extract_unique_relations(triplex_data)
        logger.info(f"Found {len(relations)} unique relations in triplex data")
        
        # Create batches of relations
        relation_batches = self._create_relation_batches(list(relations))
        logger.info(f"Created {len(relation_batches)} batches of relations")
        
        return relation_batches
    
    def _extract_unique_relations(self, triplex_data: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract all unique relations from triplex data.
        
        Args:
            triplex_data: List of triplex data entries
            
        Returns:
            Set of unique relation names
        """
        relations = set()
        for entry in triplex_data:
            for triple in entry["triples"]:
                relations.add(triple["relation"])
        
        return relations
    
    def _create_relation_batches(self, relations: List[str]) -> List[List[str]]:
        """
        Create batches of relations.
        
        Args:
            relations: List of relation names
            
        Returns:
            List of relation batches
        """
        # Shuffle relations for random assignment to batches
        np.random.shuffle(relations)
        
        # Create batches
        batches = []
        for i in range(0, len(relations), self.batch_size):
            batch = relations[i:i+self.batch_size]
            batches.append(batch)
        
        return batches
    
    def get_batch_triplex_data(self, batch: List[str], triplex_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get triplex data entries that contain relations in the given batch.
        
        Args:
            batch: List of relation names in the batch
            triplex_data: Full triplex data
            
        Returns:
            List of triplex data entries containing only the relations in the batch
        """
        batch_triplex_data = []
        batch_set = set(batch)
        
        for entry in triplex_data:
            # Create a new entry with only the triples that contain relations in the batch
            filtered_triples = [
                triple for triple in entry["triples"]
                if triple["relation"] in batch_set
            ]
            
            if filtered_triples:
                batch_entry = {
                    "text": entry["text"],
                    "triples": filtered_triples,
                    "metadata": entry.get("metadata", {})
                }
                batch_triplex_data.append(batch_entry)
        
        logger.debug(f"Found {len(batch_triplex_data)} triplex entries for batch with {len(batch)} relations")
        return batch_triplex_data
    
    def get_batch_text_triples(self, batch: List[str], triplex_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get text and corresponding triples for each relation in the batch.
        
        Args:
            batch: List of relation names in the batch
            triplex_data: Full triplex data
            
        Returns:
            Dictionary mapping relation names to lists of (text, subject, object) tuples
        """
        relation_text_triples = {rel: [] for rel in batch}
        
        for entry in triplex_data:
            text = entry["text"]
            
            for triple in entry["triples"]:
                relation = triple["relation"]
                
                if relation in batch:
                    relation_text_triples[relation].append({
                        "text": text,
                        "subject": triple["subject"],
                        "object": triple["object"]
                    })
        
        # Log number of examples for each relation
        for relation, examples in relation_text_triples.items():
            logger.debug(f"Relation '{relation}' has {len(examples)} examples")
        
        return relation_text_triples
