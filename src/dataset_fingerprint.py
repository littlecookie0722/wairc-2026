"""Privacy-preserving fingerprints for competition dataset inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .data import load_index, resolve_iq_path


DATASET_FINGERPRINT_SCHEMA = "dataset-fingerprint-v1"
FINGERPRINT_ALGORITHM = "sha256"
_HASH_CHUNK_SIZE = 1024 * 1024


def _update_blob(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, byteorder="big"))
    hasher.update(value)


def _canonical_row(row: dict[str, Any]) -> bytes:
    metadata = {key: value for key, value in row.items() if key != "iq_npz_relpath"}
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def fingerprint_dataset(root: Path, *, has_labels: bool) -> dict[str, Any]:
    """Return a root-independent fingerprint without exposing dataset paths.

    The digest covers normalized index metadata and the content of every IQ
    file referenced by the index, in index order. The returned object contains
    only aggregate counts and digests; paths, filenames, and sample IDs are not
    included.
    """
    root = Path(root)
    rows = load_index(root, has_labels=has_labels)
    index_hasher = hashlib.sha256()
    content_hasher = hashlib.sha256()
    _update_blob(index_hasher, DATASET_FINGERPRINT_SCHEMA.encode("ascii"))
    _update_blob(index_hasher, json.dumps({"hasLabels": has_labels}, sort_keys=True).encode("ascii"))
    _update_blob(content_hasher, DATASET_FINGERPRINT_SCHEMA.encode("ascii"))
    _update_blob(content_hasher, json.dumps({"hasLabels": has_labels}, sort_keys=True).encode("ascii"))

    node_presence = {f"node{node}": 0 for node in range(3)}
    for row in rows:
        canonical_row = _canonical_row(row)
        _update_blob(index_hasher, canonical_row)
        _update_blob(content_hasher, canonical_row)
        file_digest = _file_digest(resolve_iq_path(root, row))
        _update_blob(content_hasher, file_digest.encode("ascii"))
        for node in range(3):
            node_presence[f"node{node}"] += int(row[f"has_node{node}"])

    return {
        "schemaVersion": DATASET_FINGERPRINT_SCHEMA,
        "algorithm": FINGERPRINT_ALGORITHM,
        "scope": "canonical-index-and-iq-file-content",
        "hasLabels": has_labels,
        "rowCount": len(rows),
        "referencedFileCount": len(rows),
        "nodePresence": node_presence,
        "indexDigest": index_hasher.hexdigest(),
        "contentDigest": content_hasher.hexdigest(),
    }
