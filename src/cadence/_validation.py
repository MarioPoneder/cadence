"""Small validation and context-codec helpers shared by Cortex and wiring."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from decimal import Decimal
from numbers import Integral, Real


def integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def boolean(value, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def strict_json(text: str, max_bytes: int):
    if not isinstance(text, str) or len(text.encode("utf-8")) > max_bytes:
        raise ValueError("Checkpoint exceeds its text-size limit")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate checkpoint field: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    try:
        return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)
    except (TypeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("Malformed checkpoint") from error


def context_key(value) -> tuple:
    """Own a bounded, type-preserving copy of a JSON-like context.

    Lists, tuples and mappings retain their type; mapping order is irrelevant.
    Numeric arrays exposing ``tolist`` are accepted without a NumPy dependency.
    Arbitrary objects and non-finite numbers are rejected before state changes.
    """
    remaining = 4096

    def encode(item, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 32:
            raise ValueError("Context exceeds 4096 items or 32 nesting levels")
        if item is None:
            return ("none",)
        if type(item) is bool:
            return ("bool", item)
        if isinstance(item, Integral):
            return ("int", int(item))
        if isinstance(item, (Real, Decimal)):
            return ("float", number(item, "context"))
        if isinstance(item, str):
            if len(item) > 16384:
                raise ValueError("Context string exceeds 16384 characters")
            return ("str", item)
        if isinstance(item, (list, tuple)):
            return (
                "list" if isinstance(item, list) else "tuple",
                tuple(encode(x, depth + 1) for x in item),
            )
        if isinstance(item, Mapping):
            pairs = [
                (encode(k, depth + 1), encode(v, depth + 1)) for k, v in item.items()
            ]
            return ("dict", tuple(sorted(pairs, key=lambda p: canonical(p[0]))))
        if callable(getattr(item, "tolist", None)):
            return encode(item.tolist(), depth + 1)
        raise ValueError(
            "Contexts must contain numbers, strings, lists, tuples or mappings"
        )

    return encode(value, 0)


def restore_key(data) -> tuple:
    """Validate a checkpoint's tagged context without evaluating any code."""
    remaining = 4096

    def parse(item, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 32 or not isinstance(item, list) or not item:
            raise ValueError("Malformed checkpoint context")
        tag = item[0]
        if tag == "none" and len(item) == 1:
            return ("none",)
        if len(item) != 2:
            raise ValueError("Malformed checkpoint context")
        value = item[1]
        if tag == "bool" and type(value) is bool:
            return (tag, value)
        if tag == "int" and type(value) is int:
            return (tag, value)
        if tag == "float" and type(value) is float and math.isfinite(value):
            return (tag, value)
        if tag == "str" and isinstance(value, str) and len(value) <= 16384:
            return (tag, value)
        if tag in ("list", "tuple") and isinstance(value, list):
            return (tag, tuple(parse(x, depth + 1) for x in value))
        if tag == "dict" and isinstance(value, list):
            pairs = []
            for pair in value:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("Malformed mapping context")
                pairs.append((parse(pair[0], depth + 1), parse(pair[1], depth + 1)))
            if len({p[0] for p in pairs}) != len(pairs):
                raise ValueError("Duplicate mapping context key")
            if pairs != sorted(pairs, key=lambda p: canonical(p[0])):
                raise ValueError("Noncanonical mapping context")
            return (tag, tuple(pairs))
        raise ValueError("Malformed checkpoint context")

    return parse(data, 0)
