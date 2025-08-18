"""
Logging Configuration for HSTECH Estimation System

This module provides centralized logging configuration.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from ..models import LoggingConfig


def setup_logging(config: LoggingConfig, logger_name: str = "hstech_estimation") -> logging.Logger:
    """
    Setup logging configuration based on config.
    
    Args:
        config: LoggingConfig object
        logger_name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, config.level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if config.file:
        log_file = Path(config.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse max file size
        max_bytes = _parse_file_size(config.max_file_size)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=config.backup_count
        )
        file_handler.setLevel(getattr(logging, config.level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def _parse_file_size(size_str: str) -> int:
    """Parse file size string like '10MB' to bytes."""
    size_str = size_str.upper()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        # Assume bytes
        return int(size_str)


def get_logger(name: str = "hstech_estimation") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)
