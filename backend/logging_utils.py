"""Logging utilities for LLM Council."""

import logging
from datetime import datetime
from typing import Optional

# Configure logger
logger = logging.getLogger("llm_council")


def format_request_log(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    client_ip: Optional[str] = None,
    error: Optional[str] = None
) -> str:
    """
    Format a request log entry.

    Args:
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        client_ip: Optional client IP address
        error: Optional error message

    Returns:
        Formatted log string
    """
    timestamp = datetime.utcnow().isoformat()
    parts = [
        f"[{timestamp}]",
        f"{method}",
        f"{path}",
        f"-> {status_code}",
        f"({duration_ms:.2f}ms)"
    ]

    if client_ip:
        parts.insert(1, f"[{client_ip}]")

    if error:
        parts.append(f"ERROR: {error}")

    return " ".join(parts)


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    client_ip: Optional[str] = None,
    error: Optional[str] = None
):
    """
    Log a request.

    Args:
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        client_ip: Optional client IP address
        error: Optional error message
    """
    log_message = format_request_log(
        method, path, status_code, duration_ms, client_ip, error
    )

    if error:
        logger.error(log_message)
    elif status_code >= 400:
        logger.warning(log_message)
    else:
        logger.info(log_message)


def log_model_query(
    model: str,
    success: bool,
    duration_seconds: float,
    error: Optional[str] = None
):
    """
    Log a model query.

    Args:
        model: Model identifier
        success: Whether the query was successful
        duration_seconds: Query duration in seconds
        error: Optional error message
    """
    status = "SUCCESS" if success else "FAILED"
    timestamp = datetime.utcnow().isoformat()

    log_message = f"[{timestamp}] Model query: {model} -> {status} ({duration_seconds:.2f}s)"

    if error:
        log_message += f" ERROR: {error}"
        logger.error(log_message)
    elif not success:
        logger.warning(log_message)
    else:
        logger.info(log_message)


def log_council_stage(stage: int, status: str, details: Optional[str] = None):
    """
    Log a council stage completion.

    Args:
        stage: Stage number (1, 2, or 3)
        status: Status message
        details: Optional additional details
    """
    timestamp = datetime.utcnow().isoformat()
    log_message = f"[{timestamp}] Council Stage {stage}: {status}"

    if details:
        log_message += f" - {details}"

    logger.info(log_message)
