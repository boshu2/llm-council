"""Validation utilities for LLM Council."""

import json
from typing import Any, Dict


def validate_body_size(body: Any, max_size: int = 1048576) -> bool:
    """
    Validate that a request body is within the size limit.

    Args:
        body: The request body (dict or string)
        max_size: Maximum size in bytes (default 1MB)

    Returns:
        True if within limit, False otherwise
    """
    try:
        if isinstance(body, str):
            size = len(body.encode('utf-8'))
        elif isinstance(body, dict):
            size = len(json.dumps(body).encode('utf-8'))
        else:
            size = len(str(body).encode('utf-8'))

        return size <= max_size
    except Exception:
        return False


def validate_json_structure(data: Any, required_fields: list) -> tuple[bool, str]:
    """
    Validate that JSON data has required fields.

    Args:
        data: The data to validate
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Data must be a JSON object"

    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"

    return True, ""


def validate_string_field(
    value: Any,
    field_name: str,
    min_length: int = 0,
    max_length: int = 10000
) -> tuple[bool, str]:
    """
    Validate a string field.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages
        min_length: Minimum length
        max_length: Maximum length

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, str):
        return False, f"{field_name} must be a string"

    if len(value) < min_length:
        return False, f"{field_name} must be at least {min_length} characters"

    if len(value) > max_length:
        return False, f"{field_name} must be at most {max_length} characters"

    return True, ""


def validate_list_field(
    value: Any,
    field_name: str,
    min_items: int = 0,
    max_items: int = 100
) -> tuple[bool, str]:
    """
    Validate a list field.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages
        min_items: Minimum number of items
        max_items: Maximum number of items

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, list):
        return False, f"{field_name} must be a list"

    if len(value) < min_items:
        return False, f"{field_name} must have at least {min_items} items"

    if len(value) > max_items:
        return False, f"{field_name} must have at most {max_items} items"

    return True, ""
