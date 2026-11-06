"""Lazy aliases for pre-domain-layout imports and ``python -m`` entry points.

Internal code uses canonical domain paths. Aliases share the canonical module object,
so SQLAlchemy models, registries, caches and monkeypatches are never duplicated.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

_PREFIX = "lerobot.data_platform."
MODULE_ALIASES = {
    "acceptance": "ops.acceptance",
    "account_passwords": "access.passwords",
    "admin_auth": "access.local_admin",
    "admin_management": "control.admin",
    "agent": "runtime.agent",
    "agent_install": "ops.agent_install",
    "archive_cli": "artifacts.cli",
    "archive_worker": "artifacts.worker",
    "artifact_store": "artifacts.store",
    "browser_validation": "ops.browser_validation",
    "cache_files": "datasets.cache_files",
    "camera_layout": "datasets.camera_layout",
    "caption_catalog": "annotations.catalog",
    "caption_job_groups": "annotations.groups",
    "caption_job_storage": "annotations.storage",
    "caption_reads": "annotations.reads",
    "control_plane": "control.store",
    "control_plane_summaries": "control.summaries",
    "curation": "datasets.curation",
    "curation_execution": "operators.implementations.curation",
    "data_protection": "access.source_policy",
    "dataset_captions": "annotations.dataset",
    "dataset_metadata": "datasets.metadata",
    "dataset_references": "datasets.references",
    "dataset_results": "datasets.results",
    "dataset_scans": "datasets.scans",
    "deployment": "ops.deployment",
    "deployment_backup": "ops.backup",
    "deployment_network": "ops.network",
    "deployment_progress": "ops.progress",
    "dev_roles": "access.dev_roles",
    "environment": "ops.environment",
    "environment_cli": "ops.environment_cli",
    "environment_web": "web.environment",
    "episode_deletion_requests": "access.episode_deletion",
    "execution": "runtime.supervisor",
    "execution_pools": "runtime.pools",
    "execution_resources": "runtime.resources",
    "job_concurrency": "runtime.concurrency",
    "job_management": "runtime.jobs",
    "job_progress": "runtime.progress",
    "lifecycle": "datasets.lifecycle.service",
    "lifecycle_repository": "datasets.lifecycle.repository",
    "list_queries": "control.queries",
    "local_execution": "runtime.local",
    "maintenance": "ops.maintenance",
    "management_cli": "control.cli",
    "management_storage": "control.storage",
    "mcap_contract": "operators.implementations.mcap.mcap_contract",
    "mcap_import": "operators.implementations.mcap.mcap_import",
    "mcap_runtime": "operators.implementations.mcap.mcap_runtime",
    "mcap_worker": "operators.implementations.mcap.mcap_worker",
    "mcp_access": "access.mcp",
    "mem_export": "artifacts.mem_export",
    "merge_options": "datasets.merge_options",
    "operation_log": "artifacts.audit",
    "precompute": "operators.implementations",
    "promotion": "ops.promotion",
    "qwen": "ai_tools.qwen",
    "release_cli": "ops.release_cli",
    "release_validation": "ops.validation",
    "releases": "ops.releases",
    "report_archive": "artifacts.reports",
    "report_migration": "artifacts.migration",
    "rolling_deployment": "ops.rolling",
    "source_cleanup": "datasets.cleanup.service",
    "source_cleanup_agent": "datasets.cleanup.agent",
    "source_cleanup_files": "datasets.cleanup.files",
    "split_dataset_by_episodes": "operators.implementations.preprocess.split_cli",
    "task_catalog": "datasets.tasks.catalog",
    "task_suggestions": "datasets.tasks.suggestions",
    "task_text": "datasets.tasks.text",
    "temporal_caption": "operators.implementations.caption.core",
    "temporal_caption_demo": "operators.implementations.caption.demo",
    "temporal_caption_flora": "operators.implementations.caption.flora",
    "temporal_caption_flora_worker": "operators.implementations.caption.flora_worker",
    "temporal_caption_fusion": "operators.implementations.caption.fusion",
    "temporal_caption_jobs": "annotations.jobs",
    "temporal_caption_media": "operators.implementations.caption.media",
    "temporal_caption_mixed": "operators.implementations.caption.mixed",
    "temporal_caption_signals": "operators.implementations.caption.signals",
    "temporal_caption_snapshot": "operators.implementations.caption.snapshot",
    "umi_viewers": "web.umi_viewers",
}


def canonical_name(fullname: str) -> str | None:
    if not fullname.startswith(_PREFIX):
        return None
    relative = fullname[len(_PREFIX) :]
    head, separator, tail = relative.partition(".")
    target = MODULE_ALIASES.get(head)
    if target is None or (separator and head != "precompute"):
        return None
    return _PREFIX + target + (separator + tail if separator else "")


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, target: str):
        self.fullname = fullname
        self.target = target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Replacing the temporary alias preserves the target's __spec__/__package__.
        sys.modules[self.fullname] = importlib.import_module(self.target)

    def get_code(self, fullname):
        # runpy needs executable code, whereas normal imports need object identity.
        return compile(
            f"from runpy import run_module\nrun_module({self.target!r}, run_name='__main__')\n",
            f"<legacy entry point {fullname}>",
            "exec",
        )


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        canonical = canonical_name(fullname)
        if canonical is None:
            return None
        spec = importlib.util.find_spec(canonical)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(fullname, canonical),
            origin=spec.origin,
            is_package=spec.submodule_search_locations is not None,
        )


def install_aliases() -> None:
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        # Intercept old package children before PathFinder can load a second copy
        # from the canonical package's __path__.
        sys.meta_path.insert(0, _AliasFinder())


def legacy_attribute(name: str):
    canonical = canonical_name(_PREFIX + name)
    if canonical is None:
        raise AttributeError(f"module 'lerobot.data_platform' has no attribute {name!r}")
    return importlib.import_module(canonical)
