"""Change impact analysis for code review.

Maps git diffs to affected functions, flows, communities, and test coverage
gaps. Produces risk-scored, priority-ordered review guidance.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .constants import SECURITY_KEYWORDS as _SECURITY_KEYWORDS
from .flows import get_affected_flows
from .graph import GraphNode, GraphStore, _sanitize_name, node_to_dict
from .parser import CodeParser

logger = logging.getLogger(__name__)

_GIT_TIMEOUT = int(os.environ.get("CRG_GIT_TIMEOUT", "30"))  # seconds, configurable

_SAFE_GIT_REF = re.compile(r"^[A-Za-z0-9_.~^/@{}\-]+$")


# ---------------------------------------------------------------------------
# 1. parse_git_diff_ranges
# ---------------------------------------------------------------------------


def parse_git_diff_ranges(
    repo_root: str,
    base: str = "HEAD~1",
) -> dict[str, list[tuple[int, int]]]:
    """Run ``git diff --unified=0`` and extract changed line ranges per file.

    Args:
        repo_root: Absolute path to the repository root.
        base: Git ref to diff against (default: ``HEAD~1``).

    Returns:
        Mapping of file paths to lists of ``(start_line, end_line)`` tuples.
        Returns an empty dict on error.
    """
    if not _SAFE_GIT_REF.match(base):
        logger.warning("Invalid git ref rejected: %s", base)
        return {}
    try:
        result = subprocess.run(
            ["git", "diff", "--unified=0", base, "--"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=_GIT_TIMEOUT,
        )
        if result.returncode != 0:
            logger.warning("git diff failed (rc=%d): %s", result.returncode, result.stderr[:200])
            return {}
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git diff error: %s", exc)
        return {}

    return _parse_unified_diff(result.stdout)


def _parse_unified_diff(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse unified diff output into file -> line-range mappings.

    Handles the ``@@ -old,count +new,count @@`` hunk header format.
    """
    ranges: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None

    # Match "+++ b/path/to/file"
    file_pattern = re.compile(r"^\+\+\+ b/(.+)$")
    # Match "@@ ... +start,count @@" or "@@ ... +start @@"
    hunk_pattern = re.compile(r"^@@ .+? \+(\d+)(?:,(\d+))? @@")

    for line in diff_text.splitlines():
        file_match = file_pattern.match(line)
        if file_match:
            current_file = file_match.group(1)
            continue

        hunk_match = hunk_pattern.match(line)
        if hunk_match and current_file is not None:
            start = int(hunk_match.group(1))
            count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
            if count == 0:
                # Pure deletion hunk (no lines added); still note the position.
                end = start
            else:
                end = start + count - 1
            ranges.setdefault(current_file, []).append((start, end))

    return ranges


# ---------------------------------------------------------------------------
# 2. map_changes_to_nodes
# ---------------------------------------------------------------------------


def map_changes_to_nodes(
    store: GraphStore,
    changed_ranges: dict[str, list[tuple[int, int]]],
) -> list[GraphNode]:
    """Find graph nodes whose line ranges overlap the changed lines.

    Args:
        store: The graph store.
        changed_ranges: Mapping of file paths to ``(start, end)`` tuples.

    Returns:
        Deduplicated list of overlapping graph nodes.
    """
    seen: set[str] = set()
    result: list[GraphNode] = []

    for file_path, ranges in changed_ranges.items():
        # Try the path as-is, then also try all nodes to match relative paths.
        nodes = store.get_nodes_by_file(file_path)
        if not nodes:
            # The graph may store absolute paths; try a suffix match.
            matched_paths = store.get_files_matching(file_path)
            for mp in matched_paths:
                nodes.extend(store.get_nodes_by_file(mp))

        for node in nodes:
            if node.qualified_name in seen:
                continue
            if node.line_start is None or node.line_end is None:
                continue
            # Check overlap with any changed range.
            for start, end in ranges:
                if node.line_start <= end and node.line_end >= start:
                    result.append(node)
                    seen.add(node.qualified_name)
                    break

    return result


# ---------------------------------------------------------------------------
# 3. compute_risk_score
# ---------------------------------------------------------------------------


def compute_risk_score(store: GraphStore, node: GraphNode) -> float:
    """Compute a risk score (0.0 - 1.0) for a single node.

    Scoring factors:
      - Flow participation: 0.05 per flow membership, capped at 0.25
      - Community crossing: 0.05 per caller from a different community, capped at 0.15
      - Test coverage: 0.30 if no TESTED_BY edges, 0.05 if tested
      - Security sensitivity: 0.20 if name matches security keywords
      - Caller count: callers / 20, capped at 0.10
    """
    score = 0.0

    # --- Flow participation (cap 0.25) ---
    flow_count = store.count_flow_memberships(node.id)
    score += min(flow_count * 0.05, 0.25)

    # --- Community crossing (cap 0.15) ---
    callers = store.get_edges_by_target(node.qualified_name)
    caller_edges = [e for e in callers if e.kind == "CALLS"]

    cross_community = 0
    node_cid = store.get_node_community_id(node.id)

    if node_cid is not None and caller_edges:
        caller_qns = [edge.source_qualified for edge in caller_edges]
        cid_map = store.get_community_ids_by_qualified_names(caller_qns)
        for cid in cid_map.values():
            if cid is not None and cid != node_cid:
                cross_community += 1
    score += min(cross_community * 0.05, 0.15)

    # --- Test coverage ---
    tested_edges = store.get_edges_by_target(node.qualified_name)
    has_test = any(e.kind == "TESTED_BY" for e in tested_edges)
    score += 0.05 if has_test else 0.30

    # --- Security sensitivity ---
    name_lower = node.name.lower()
    qn_lower = node.qualified_name.lower()
    if any(kw in name_lower or kw in qn_lower for kw in _SECURITY_KEYWORDS):
        score += 0.20

    # --- Caller count (cap 0.10) ---
    caller_count = len(caller_edges)
    score += min(caller_count / 20.0, 0.10)

    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# 4. analyze_changes
# ---------------------------------------------------------------------------


def analyze_changes(
    store: GraphStore,
    changed_files: list[str],
    changed_ranges: dict[str, list[tuple[int, int]]] | None = None,
    repo_root: str | None = None,
    base: str = "HEAD~1",
) -> dict[str, Any]:
    """Analyze changes and produce risk-scored review guidance.

    Args:
        store: The graph store.
        changed_files: List of changed file paths.
        changed_ranges: Optional pre-parsed diff ranges. If not provided and
            ``repo_root`` is given, they are computed via git.
        repo_root: Repository root (for git diff).
        base: Git ref to diff against.

    Returns:
        Dict with ``summary``, ``risk_score``, ``changed_functions``,
        ``affected_flows``, ``test_gaps``, and ``review_priorities``.
    """
    # Compute changed ranges if not provided.
    if changed_ranges is None and repo_root is not None:
        changed_ranges = parse_git_diff_ranges(repo_root, base)

    # Map changes to nodes.
    if changed_ranges:
        changed_nodes = map_changes_to_nodes(store, changed_ranges)
    else:
        # Fallback: all nodes in changed files.
        changed_nodes = []
        for fp in changed_files:
            changed_nodes.extend(store.get_nodes_by_file(fp))

    # Filter to functions/tests for risk scoring (skip File nodes).
    changed_funcs = [
        n for n in changed_nodes
        if n.kind in ("Function", "Test", "Class")
    ]

    # Compute per-node risk scores.
    node_risks: list[dict[str, Any]] = []
    for node in changed_funcs:
        risk = compute_risk_score(store, node)
        node_risks.append({
            **node_to_dict(node),
            "risk_score": risk,
        })

    # Overall risk score: max of individual risks, or 0.
    overall_risk = max((nr["risk_score"] for nr in node_risks), default=0.0)

    # Affected flows.
    affected = get_affected_flows(store, changed_files)

    # Detect test gaps: changed functions without TESTED_BY edges.
    test_gaps: list[dict[str, Any]] = []
    for node in changed_funcs:
        if node.is_test:
            continue
        tested = store.get_edges_by_target(node.qualified_name)
        if not any(e.kind == "TESTED_BY" for e in tested):
            test_gaps.append({
                "name": _sanitize_name(node.name),
                "qualified_name": _sanitize_name(node.qualified_name),
                "file": node.file_path,
                "line_start": node.line_start,
                "line_end": node.line_end,
            })

    # Review priorities: top 10 by risk score.
    review_priorities = sorted(node_risks, key=lambda x: x["risk_score"], reverse=True)[:10]

    # Build summary.
    summary_parts = [
        f"Analyzed {len(changed_files)} changed file(s):",
        f"  - {len(changed_funcs)} changed function(s)/class(es)",
        f"  - {affected['total']} affected flow(s)",
        f"  - {len(test_gaps)} test gap(s)",
        f"  - Overall risk score: {overall_risk:.2f}",
    ]
    if test_gaps:
        gap_names = [g["name"] for g in test_gaps[:5]]
        summary_parts.append(f"  - Untested: {', '.join(gap_names)}")

    return {
        "summary": "\n".join(summary_parts),
        "risk_score": overall_risk,
        "changed_functions": node_risks,
        "affected_flows": affected["affected_flows"],
        "test_gaps": test_gaps,
        "review_priorities": review_priorities,
    }


# ---------------------------------------------------------------------------
# 5. Breaking change detection
# ---------------------------------------------------------------------------


def _get_file_from_git(repo_root: str, file_path: str, ref: str) -> bytes | None:
    """Get the content of a file at a specific git ref."""
    if not _SAFE_GIT_REF.match(ref):
        return None
    try:
        result = subprocess.run(
            ["git", "show", f"{ref}:{file_path}"],
            capture_output=True,
            cwd=repo_root,
            timeout=_GIT_TIMEOUT,
        )
        if result.returncode == 0:
            return result.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git show failed: %s", exc)
    return None


def _parse_params(params_str: str | None) -> list[str]:
    """Parse parameter string into a list of parameter names."""
    if not params_str:
        return []
    params = []
    for part in params_str.split(","):
        part = part.strip()
        if not part:
            continue
        name = part.split(":")[0].split("=")[0].strip()
        name = name.replace("*", "").strip()
        if name and name not in ("self", "cls"):
            params.append(name)
    return params


def _detect_signature_change(
    old_node: GraphNode,
    new_node: GraphNode,
) -> dict[str, Any] | None:
    """Detect if there's a breaking signature change between old and new node."""
    changes: list[dict[str, str]] = []

    old_params = _parse_params(old_node.params)
    new_params = _parse_params(new_node.params)

    if old_params != new_params:
        if len(old_params) != len(new_params):
            changes.append({
                "type": "param_count",
                "severity": "high",
                "detail": f"Parameter count changed from {len(old_params)} to {len(new_params)}",
            })
        else:
            for i, (old_p, new_p) in enumerate(zip(old_params, new_params)):
                if old_p != new_p:
                    changes.append({
                        "type": "param_renamed",
                        "severity": "medium",
                        "detail": f"Parameter {i}: '{old_p}' renamed to '{new_p}'",
                    })

    if old_node.return_type != new_node.return_type:
        if old_node.return_type and new_node.return_type:
            changes.append({
                "type": "return_type",
                "severity": "high",
                "detail": f"Return type changed from '{old_node.return_type}' to '{new_node.return_type}'",
            })
        elif new_node.return_type is None:
            changes.append({
                "type": "return_type_removed",
                "severity": "high",
                "detail": "Return type annotation removed",
            })

    if changes:
        return {
            "qualified_name": old_node.qualified_name,
            "file": old_node.file_path,
            "line": old_node.line_start,
            "changes": changes,
        }
    return None


def _is_public_api(node: GraphNode) -> bool:
    """Return True if the node represents a public API (not starting with _)."""
    if node.name.startswith("_") and not node.name.startswith("__"):
        return False
    return node.kind in ("Function", "Class", "Type")


def detect_breaking_changes(
    store: GraphStore,
    repo_root: str,
    changed_files: list[str],
    base: str = "HEAD~1",
) -> dict[str, Any]:
    """Detect breaking API changes between the current state and a git ref.

    Args:
        store: The graph store (current state).
        repo_root: Repository root path.
        changed_files: List of changed file paths.
        base: Git ref to compare against (default: HEAD~1).

    Returns:
        Dict with breaking_changes, signature_changes, removed_apis, and summary.
    """
    parser = CodeParser()
    breaking_changes: list[dict[str, Any]] = []
    signature_changes: list[dict[str, Any]] = []
    removed_apis: list[dict[str, Any]] = []
    potential_issues: list[dict[str, Any]] = []

    for file_path in changed_files:
        old_source = _get_file_from_git(repo_root, file_path, base)
        if old_source is None:
            continue

        old_nodes: list[GraphNode] = []
        try:
            nodes, _ = parser.parse_bytes(Path(file_path), old_source)
            for n in nodes:
                node = store.get_node(n.qualified_name)
                if node:
                    old_nodes.append(node)
        except Exception as exc:
            logger.warning("Failed to parse old version of %s: %s", file_path, exc)
            continue

        new_nodes = store.get_nodes_by_file(file_path)
        old_by_qn = {n.qualified_name: n for n in old_nodes if _is_public_api(n)}
        new_by_qn = {n.qualified_name: n for n in new_nodes if _is_public_api(n)}

        for qn, old_node in old_by_qn.items():
            if qn not in new_by_qn:
                removed_apis.append({
                    "qualified_name": qn,
                    "file": file_path,
                    "line": old_node.line_start,
                    "kind": old_node.kind,
                    "severity": "high",
                    "detail": f"Public {old_node.kind.lower()} '{old_node.name}' was removed",
                })
                breaking_changes.append({
                    "qualified_name": qn,
                    "file": file_path,
                    "severity": "high",
                    "change_type": "removed",
                })

        for qn, new_node in new_by_qn.items():
            if qn not in old_by_qn:
                continue

            old_node = old_by_qn[qn]
            sig_change = _detect_signature_change(old_node, new_node)
            if sig_change:
                signature_changes.append(sig_change)
                severity = max(c["severity"] for c in sig_change["changes"])
                breaking_changes.append({
                    "qualified_name": qn,
                    "file": file_path,
                    "severity": severity,
                    "change_type": "signature",
                    "changes": sig_change["changes"],
                })

    if signature_changes:
        for sc in signature_changes:
            for change in sc.get("changes", []):
                if change["type"] == "return_type_removed":
                    potential_issues.append({
                        "type": "type_erasure",
                        "severity": "medium",
                        "detail": f"'{sc['qualified_name']}' lost type information",
                        "file": sc["file"],
                        "line": sc["line"],
                    })

    summary_parts = [f"Analyzed {len(changed_files)} changed file(s) for breaking changes:"]
    if breaking_changes:
        high_severity = [c for c in breaking_changes if c["severity"] == "high"]
        summary_parts.append(f"  - {len(breaking_changes)} breaking change(s) detected")
        summary_parts.append(f"  - {len(high_severity)} high severity")
        summary_parts.append(f"  - {len(removed_apis)} removed public API(s)")
        summary_parts.append(f"  - {len(signature_changes)} signature change(s)")
    else:
        summary_parts.append("  - No breaking changes detected")

    return {
        "summary": "\n".join(summary_parts),
        "breaking_changes": breaking_changes,
        "removed_apis": removed_apis,
        "signature_changes": signature_changes,
        "potential_issues": potential_issues,
        "total_breaking": len(breaking_changes),
        "high_severity_count": len([c for c in breaking_changes if c["severity"] == "high"]),
    }
