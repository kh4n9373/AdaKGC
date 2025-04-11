
import os
import argparse
import yaml
import logging
from pathlib import Path

from data.data_loader import WebNLGDataLoader
from pipeline.step1_processor import Step1Processor
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="AdaKGC: Adaptive Knowledge Graph Construction")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/raw",
        help="Path to the directory containing WebNLG data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=50,
        help="Batch size for relation processing"
    )
    parser.add_argument(
        "--similarity_threshold", type=float, default=0.85,
        help="Threshold for relation similarity"
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point for the AdaKGC pipeline."""
    args = parse_args()
    
    config = load_config(args.config)
    
    log_level = getattr(logging, args.log_level)
    setup_logger(log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting AdaKGC pipeline")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading WebNLG data from {args.data_dir}")
    data_loader = WebNLGDataLoader(data_dir=args.data_dir)
    triplex_data = data_loader.load_triplex_data()
    
    logger.info("Running Step 1: Relation processing and schema improvement")
    step1 = Step1Processor(
        batch_size=args.batch_size,
        similarity_threshold=args.similarity_threshold,
        output_dir=args.output_dir,
        **config.get("step1_config", {})
    )
    processed_data = step1.process(triplex_data)
    
    output_path = output_dir / "processed_relations.json"
    logger.info(f"Saving processed data to {output_path}")
    with open(output_path, "w") as f:
        import json
        json.dump(processed_data, f, indent=2)
    
    logger.info("Pipeline completed successfully")
    

if __name__ == "__main__":
    main()
