#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Set up global logging configuration
def setup_logger(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S"
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Path to the log file. If None, logs will be output to console only.
        log_format: Format for log messages
        date_format: Format for date/time in log messages
    """
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    handlers.append(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    # Suppress verbose logs from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized with level {logging.getLevelName(level)}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_section(logger: logging.Logger, title: str, level: int = logging.INFO) -> None:
    """
    Log a section header to visually separate sections in the log.
    
    Args:
        logger: Logger instance
        title: Section title
        level: Logging level
    """
    separator = "=" * 80
    message = f"\n{separator}\n{title}\n{separator}"
    logger.log(level, message)


# Initialize default logger if imported directly
if __name__ != "__main__":
    setup_logger()
