"""Bounded public diagnostics; tracebacks and service paths stay in operator logs."""

import json
import re
import subprocess
from pathlib import Path


def public_error(error: Exception, source: Path | None = None) -> str:
    if isinstance(error, (FileNotFoundError, PermissionError)):
        if source is not None and error.filename:
            try:
                relative = Path(error.filename).resolve().relative_to(source.resolve()).as_posix()
            except (ValueError, OSError):
                pass
            else:
                # Only dataset-relative filenames belong in a caller's diagnostic.
                relative = re.sub(r"[\x00-\x1f\x7f]", "?", relative)[:300]
                label = (
                    "Missing dataset file"
                    if isinstance(error, FileNotFoundError)
                    else "Cannot read dataset file"
                )
                return f"{type(error).__name__}: {label}: {relative}"
        return f"{type(error).__name__}: A service file is unavailable; contact the service operator"
    if isinstance(error, json.JSONDecodeError):
        return f"Invalid JSON data at line {error.lineno}, column {error.colno}"
    if isinstance(error, ValueError):
        message = str(error)
        message = re.sub(r"(?i)bearer\s+\S+", "Bearer <redacted>", message)
        message = re.sub(
            r"(?i)\b(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*[^\s,;]+",
            r"\1=<redacted>",
            message,
        )
        message = re.sub(r"[a-zA-Z][a-zA-Z0-9+.-]*://\S+", "<redacted URL>", message)
        message = re.sub(r"(?<![\w])(?:/[\w.~-]|[A-Za-z]:\\)[^\s'\"<>]*", "<server path>", message)
        message = re.sub(r"[\x00-\x1f\x7f]", " ", message)
        return "ValueError: " + (message[:1000] or "Invalid input data")
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        return "TimeoutError: The current processing stage exceeded its time limit"
    if isinstance(error, subprocess.CalledProcessError):
        return f"A processing subprocess failed (exit code {error.returncode}); contact the service operator"
    if type(error).__name__ == "OutOfMemoryError":
        return "GPU out of memory; contact the service operator"
    return "Inference failed; contact the service operator with this job ID"
