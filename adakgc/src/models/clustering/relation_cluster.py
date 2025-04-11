
import logging
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class RelationCluster:
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 2,
        max_cluster_size: int = 20
    ):

        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        
        self.clusters = {}  # Mapping from cluster ID to set of relation names
        self.relation_to_cluster = {}  # Mapping from relation name to cluster ID
        self.cluster_centroids = {}  # Mapping from cluster ID to centroid embedding
        
        logger.info(f"Initialized RelationCluster with similarity threshold {similarity_threshold}")
    
    def create_clusters(
        self, 
        relation_embeddings: Dict[str, np.ndarray],
        relation_definitions: Dict[str, str]
    ) -> Dict[str, Set[str]]:

        self.clusters = {}
        self.relation_to_cluster = {}
        self.cluster_centroids = {}
        
        similarities = self._calculate_pairwise_similarities(relation_embeddings)
        
        self._create_clusters_greedy(similarities, relation_embeddings)
        
        self._log_cluster_info(relation_definitions)
        
        return self.clusters
    
    def assign_relation(
        self, 
        relation: str, 
        embedding: np.ndarray,
        existing_embeddings: Dict[str, np.ndarray]
    ) -> Optional[str]:

        similarities = {}
        
        for cluster_id, centroid in self.cluster_centroids.items():
            similarity = self._calculate_cosine_similarity(embedding, centroid)
            similarities[cluster_id] = similarity
        
        if not similarities:
            return None
        
        most_similar_cluster, max_similarity = max(similarities.items(), key=lambda x: x[1])
        
        if max_similarity >= self.similarity_threshold:
            if len(self.clusters[most_similar_cluster]) < self.max_cluster_size:
                self.clusters[most_similar_cluster].add(relation)
                self.relation_to_cluster[relation] = most_similar_cluster
                
                self._update_centroid(most_similar_cluster, existing_embeddings)
                
                logger.debug(f"Assigned relation '{relation}' to cluster '{most_similar_cluster}' with similarity {max_similarity:.4f}")
                return most_similar_cluster
        
        cluster_id = f"cluster_{len(self.clusters) + 1}"
        self.clusters[cluster_id] = {relation}
        self.relation_to_cluster[relation] = cluster_id
        self.cluster_centroids[cluster_id] = embedding.copy()
        
        logger.debug(f"Created new cluster '{cluster_id}' for relation '{relation}'")
        return cluster_id
    
    def merge_clusters(
        self, 
        relation_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, Set[str]]:

        centroid_similarities = {}
        cluster_ids = list(self.clusters.keys())
        
        for i, cluster1 in enumerate(cluster_ids):
            for cluster2 in cluster_ids[i+1:]:
                similarity = self._calculate_cosine_similarity(
                    self.cluster_centroids[cluster1],
                    self.cluster_centroids[cluster2]
                )
                centroid_similarities[(cluster1, cluster2)] = similarity
        
        sorted_similarities = sorted(
            centroid_similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        merged = set()
        for (cluster1, cluster2), similarity in sorted_similarities:
            if cluster1 in merged or cluster2 in merged:
                continue
                
            if similarity >= self.similarity_threshold:
                self._merge_clusters(cluster1, cluster2, relation_embeddings)
                
                merged.add(cluster2)
                logger.debug(f"Merged cluster '{cluster2}' into '{cluster1}' with similarity {similarity:.4f}")
        
        for cluster_id in merged:
            if cluster_id in self.clusters:
                del self.clusters[cluster_id]
                if cluster_id in self.cluster_centroids:
                    del self.cluster_centroids[cluster_id]
        
        return self.clusters
    
    def get_relation_cluster(self, relation: str) -> Optional[str]:

        return self.relation_to_cluster.get(relation)
    
    def get_cluster_relations(self, cluster_id: str) -> Set[str]:

        return self.clusters.get(cluster_id, set())
    
    def get_cluster_representative(self, cluster_id: str) -> Optional[str]:

        if cluster_id not in self.clusters or not self.clusters[cluster_id]:
            return None

        return next(iter(self.clusters[cluster_id]))
    
    def _calculate_pairwise_similarities(
        self, 
        embeddings: Dict[str, np.ndarray]
    ) -> Dict[Tuple[str, str], float]:

        similarities = {}
        relations = list(embeddings.keys())
        
        for i, rel1 in enumerate(relations):
            for rel2 in relations[i+1:]:
                similarity = self._calculate_cosine_similarity(
                    embeddings[rel1],
                    embeddings[rel2]
                )
                similarities[(rel1, rel2)] = similarity
                similarities[(rel2, rel1)] = similarity
        
        return similarities
    
    def _create_clusters_greedy(
        self, 
        similarities: Dict[Tuple[str, str], float],
        embeddings: Dict[str, np.ndarray]
    ) -> None:

        sorted_similarities = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for (rel1, rel2), similarity in sorted_similarities:
            if similarity < self.similarity_threshold:
                break
                
            cluster1 = self.relation_to_cluster.get(rel1)
            cluster2 = self.relation_to_cluster.get(rel2)
            
            if cluster1 is None and cluster2 is None:
                cluster_id = f"cluster_{len(self.clusters) + 1}"
                self.clusters[cluster_id] = {rel1, rel2}
                self.relation_to_cluster[rel1] = cluster_id
                self.relation_to_cluster[rel2] = cluster_id
                
                self.cluster_centroids[cluster_id] = (embeddings[rel1] + embeddings[rel2]) / 2
                
            elif cluster1 is None:
                if len(self.clusters[cluster2]) < self.max_cluster_size:
                    self.clusters[cluster2].add(rel1)
                    self.relation_to_cluster[rel1] = cluster2
                    
                    self._update_centroid(cluster2, embeddings)
                
            elif cluster2 is None:
                if len(self.clusters[cluster1]) < self.max_cluster_size:
                    self.clusters[cluster1].add(rel2)
                    self.relation_to_cluster[rel2] = cluster1
                    
                    self._update_centroid(cluster1, embeddings)
            
            elif cluster1 != cluster2:
                if len(self.clusters[cluster1]) + len(self.clusters[cluster2]) <= self.max_cluster_size:
                    self._merge_clusters(cluster1, cluster2, embeddings)
        
        for relation in embeddings:
            if relation not in self.relation_to_cluster:
                cluster_id = f"cluster_{len(self.clusters) + 1}"
                self.clusters[cluster_id] = {relation}
                self.relation_to_cluster[relation] = cluster_id
                self.cluster_centroids[cluster_id] = embeddings[relation].copy()
    
    def _merge_clusters(
        self, 
        target_cluster: str, 
        source_cluster: str,
        embeddings: Dict[str, np.ndarray]
    ) -> None:

        self.clusters[target_cluster].update(self.clusters[source_cluster])
        
        for relation in self.clusters[source_cluster]:
            self.relation_to_cluster[relation] = target_cluster
        
        self._update_centroid(target_cluster, embeddings)
        
        self.clusters.pop(source_cluster, None)
        self.cluster_centroids.pop(source_cluster, None)
    
    def _update_centroid(self, cluster_id: str, embeddings: Dict[str, np.ndarray]) -> None:

        if cluster_id not in self.clusters:
            return
        
        cluster_embeddings = [
            embeddings[relation] for relation in self.clusters[cluster_id]
            if relation in embeddings
        ]
        
        if cluster_embeddings:
            self.cluster_centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
    
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:

        embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        return float(similarity)
    
    def _log_cluster_info(self, relation_definitions: Dict[str, str]) -> None:

        logger.info(f"Created {len(self.clusters)} clusters from {len(relation_definitions)} relations")
        
        cluster_sizes = [len(relations) for relations in self.clusters.values()]
        if cluster_sizes:
            avg_size = sum(cluster_sizes) / len(cluster_sizes)
            max_size = max(cluster_sizes)
            min_size = min(cluster_sizes)
            logger.info(f"Cluster sizes: min={min_size}, avg={avg_size:.2f}, max={max_size}")
        
        for cluster_id, relations in list(self.clusters.items())[:3]:
            logger.info(f"Cluster {cluster_id}: {', '.join(sorted(list(relations)[:5]))}")
    
    def save_clusters(self, file_path: str) -> None:

        clusters_json = {
            cluster_id: list(relations)
            for cluster_id, relations in self.clusters.items()
        }
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "clusters": clusters_json,
                    "relation_to_cluster": self.relation_to_cluster
                }, f, indent=2)
            
            logger.info(f"Saved {len(self.clusters)} clusters to {file_path}")
        except Exception as e:
            logger.error(f"Error saving clusters to {file_path}: {e}")
    
    def load_clusters(self, file_path: str) -> None:

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.clusters = {
                cluster_id: set(relations)
                for cluster_id, relations in data["clusters"].items()
            }
            self.relation_to_cluster = data["relation_to_cluster"]
            
            logger.info(f"Loaded {len(self.clusters)} clusters from {file_path}")
        except Exception as e:
            logger.error(f"Error loading clusters from {file_path}: {e}")
            self.clusters = {}
            self.relation_to_cluster = {}
