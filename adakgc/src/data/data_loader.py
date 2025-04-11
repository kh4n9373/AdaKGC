#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class WebNLGDataLoader:
    """
    Loader for WebNLG dataset that handles text and triples from 159 distinct schema.
    """
    
    def __init__(self, data_dir: str = "data/raw", schema_file: Optional[str] = None):
        """
        Initialize the WebNLG data loader.
        
        Args:
            data_dir: Directory containing the WebNLG dataset
            schema_file: Path to the schema file (CSV format)
        """
        self.data_dir = Path(data_dir)
        self.schema_file = schema_file
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist. Creating it.")
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_triplex_data(self) -> List[Dict[str, Any]]:
        """
        Load triplex data (text/triples) from WebNLG dataset.
        
        Returns:
            List of triplex data entries containing text and associated triples
        """
        triplex_file = self.data_dir / "webnlg_triplex.json"
        
        if not triplex_file.exists():
            logger.error(f"Triplex file {triplex_file} not found. Please download the WebNLG dataset.")
            return []
        
        logger.info(f"Loading triplex data from {triplex_file}")
        with open(triplex_file, "r", encoding="utf-8") as f:
            triplex_data = json.load(f)
        
        # Process and validate the triplex data
        processed_data = self._process_triplex_data(triplex_data)
        logger.info(f"Loaded {len(processed_data)} triplex entries")
        
        return processed_data
    
    def load_schema(self) -> pd.DataFrame:
        """
        Load relation schema information.
        
        Returns:
            DataFrame containing schema information for the 159 distinct relations
        """
        if self.schema_file is None:
            schema_file = self.data_dir / "webnlg_schema.csv"
        else:
            schema_file = Path(self.schema_file)
        
        if not schema_file.exists():
            logger.error(f"Schema file {schema_file} not found.")
            return pd.DataFrame()
        
        logger.info(f"Loading schema from {schema_file}")
        schema_df = pd.read_csv(schema_file)
        logger.info(f"Loaded schema with {len(schema_df)} relations")
        
        return schema_df
    
    def _process_triplex_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw triplex data to ensure proper formatting.
        
        Args:
            raw_data: Raw triplex data from WebNLG
            
        Returns:
            Processed triplex data
        """
        processed_data = []
        
        for entry in raw_data:
            # Validate entry structure
            if "text" not in entry or "triples" not in entry:
                logger.warning(f"Invalid entry structure: {entry}")
                continue
            
            # Validate triples structure
            valid_triples = []
            for triple in entry["triples"]:
                if not isinstance(triple, list) or len(triple) != 3:
                    logger.warning(f"Invalid triple structure: {triple}")
                    continue
                
                subject, relation, obj = triple
                valid_triples.append({
                    "subject": subject,
                    "relation": relation,
                    "object": obj
                })
            
            if not valid_triples:
                logger.warning(f"Entry has no valid triples: {entry}")
                continue
            
            # Create processed entry
            processed_entry = {
                "text": entry["text"],
                "triples": valid_triples,
                "metadata": entry.get("metadata", {})
            }
            
            processed_data.append(processed_entry)
        
        return processed_data
    
    def create_relation_batches(self, triplex_data: List[Dict[str, Any]], batch_size: int = 50) -> List[List[Dict]]:
        """
        Create batches of relations from triplex data.
        
        Args:
            triplex_data: Processed triplex data
            batch_size: Size of each batch
            
        Returns:
            List of relation batches
        """
        # Extract all unique relations from the triplex data
        relations = set()
        for entry in triplex_data:
            for triple in entry["triples"]:
                relations.add(triple["relation"])
        
        # Convert to list and shuffle
        relation_list = list(relations)
        np.random.shuffle(relation_list)
        
        # Create batches
        batches = []
        for i in range(0, len(relation_list), batch_size):
            batch = relation_list[i:i+batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches of relations with batch size {batch_size}")
        
        return batches
