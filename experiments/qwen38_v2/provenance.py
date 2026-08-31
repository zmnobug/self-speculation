from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .contract import ContractError

_IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        raise ContractError(f"fingerprint path does not exist: {root}")
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if candidate.suffix in {".pyc", ".pyo"}:
            continue
        if candidate.is_symlink():
            continue
        if candidate.is_file():
            yield candidate


def fingerprint(paths: Iterable[str | Path]) -> dict[str, Any]:
    roots = [Path(path).resolve() for path in paths]
    if not roots:
        raise ContractError("at least one fingerprint path is required")
    entries: list[dict[str, str]] = []
    overall = hashlib.sha256()
    for root in sorted(roots, key=str):
        if not root.exists():
            raise ContractError(f"fingerprint path does not exist: {root}")
        for path in _files(root):
            relative = path.name if root.is_file() else str(path.relative_to(root))
            label = f"{root}:{relative}"
            sha256 = _hash_file(path)
            entries.append({"path": label, "sha256": sha256})
            overall.update(label.encode("utf-8"))
            overall.update(b"\0")
            overall.update(sha256.encode("ascii"))
            overall.update(b"\n")
    return {
        "schema_version": "spork-input-fingerprint-v2",
        "overall_sha256": overall.hexdigest(),
        "entries": entries,
    }


def create_stage_marker(
    *, stage: str, artifact: str | Path, critical_paths: Iterable[str | Path]
) -> dict[str, Any]:
    artifact_path = Path(artifact).resolve()
    if not artifact_path.exists():
        raise ContractError(f"stage artifact does not exist: {artifact_path}")
    return {
        "schema_version": "spork-stage-marker-v2",
        "stage": stage,
        "artifact": str(artifact_path),
        "critical_inputs": fingerprint(critical_paths),
        "artifact_fingerprint": fingerprint([artifact_path]),
    }


def verify_stage_marker(
    marker: str | Path,
    *,
    stage: str,
    critical_paths: Iterable[str | Path],
) -> dict[str, Any]:
    marker_path = Path(marker)
    if not marker_path.is_file():
        raise ContractError(f"stage marker does not exist: {marker_path}")
    try:
        value = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContractError(f"{marker_path}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{marker_path}: marker must be a JSON object")
    if value.get("schema_version") != "spork-stage-marker-v2":
        raise ContractError(f"{marker_path}: unsupported marker schema")
    if value.get("stage") != stage:
        raise ContractError(
            f"{marker_path}: expected stage {stage!r}, got {value.get('stage')!r}"
        )
    artifact = value.get("artifact")
    if not isinstance(artifact, str):
        raise ContractError(f"{marker_path}: artifact path is missing")
    current_inputs = fingerprint(critical_paths)
    current_artifact = fingerprint([artifact])
    locked_inputs = value.get("critical_inputs")
    locked_artifact = value.get("artifact_fingerprint")
    if not isinstance(locked_inputs, dict) or (
        locked_inputs.get("overall_sha256") != current_inputs["overall_sha256"]
    ):
        raise ContractError(
            f"{marker_path}: code/protocol/manifest fingerprint changed; rerun {stage}"
        )
    if not isinstance(locked_artifact, dict) or (
        locked_artifact.get("overall_sha256") != current_artifact["overall_sha256"]
    ):
        raise ContractError(
            f"{marker_path}: stage artifact changed after marking; rerun {stage}"
        )
    return {
        "schema_version": "spork-marker-verification-v2",
        "stage": stage,
        "passed": True,
        "marker": str(marker_path.resolve()),
        "artifact": artifact,
        "critical_inputs_sha256": current_inputs["overall_sha256"],
        "artifact_sha256": current_artifact["overall_sha256"],
    }
