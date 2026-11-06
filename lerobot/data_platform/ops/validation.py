"""Release test inventory, worker options, and isolated pytest caches."""

import importlib.util
import os
import re
from pathlib import Path

VALIDATION_TESTS = [
    "tests/datasets/test_package_layout.py",
    "tests/datasets/test_evaluation.py",
    "tests/datasets/test_mcp.py",
    "tests/datasets/test_ai_assistant_runtime.py",
    "tests/datasets/test_ai_assistant.py",
    "tests/datasets/test_ai_assistant_context.py",
    "tests/datasets/test_ai_general_chat.py",
    "tests/datasets/test_ai_assistant_security.py",
    "tests/datasets/test_ai_assistant_actions.py",
    "tests/datasets/test_ai_assistant_cli.py",
    "tests/datasets/test_ai_personalization.py",
    "tests/datasets/test_ai_memory_followups.py",
    "tests/datasets/test_ai_experience.py",
    "tests/datasets/test_ai_strategies.py",
    "tests/datasets/test_ai_learning.py",
    "tests/datasets/test_strategy_hardening.py",
    "tests/datasets/test_cli_installer.py",
    "tests/datasets/test_mem_export.py",
    "tests/datasets/test_memory_annotation.py",
    "tests/datasets/test_memory_batch.py",
    "tests/datasets/test_memory_source_repair.py",
    "tests/datasets/test_memory_routes.py",
    "tests/datasets/test_memory_training_review.py",
    "tests/datasets/test_memory_export.py",
    "tests/datasets/test_artifact_store.py",
    "tests/datasets/test_report_archive.py",
    "tests/datasets/test_report_migration.py",
    "tests/datasets/test_list_pagination.py",
    "tests/datasets/test_caption_dataset_jobs.py",
    "tests/datasets/test_caption_catalog.py",
    "tests/datasets/test_acceptance.py",
    "tests/datasets/test_console_services.py",
    "tests/datasets/test_console_interactions.py",
    "tests/datasets/test_remote_dataset_merge.py",
    "tests/datasets/test_task_ui.py",
    "tests/datasets/test_value_edit.py",
    "tests/datasets/test_source_cleanup.py",
    "tests/datasets/test_operators.py",
    "tests/datasets/test_umi_processing.py",
    "tests/datasets/test_umi_native_viewers.py",
    "tests/datasets/test_processing.py",
    "tests/datasets/test_pose_interpolate.py",
    "tests/datasets/test_preprocess_ops.py",
    "tests/datasets/test_preprocess_routes.py",
    "tests/datasets/test_lifecycle.py",
    "tests/datasets/test_compare.py",
    "tests/datasets/test_task_suggestions.py",
    "tests/datasets/test_dataset_captions.py",
    "tests/datasets/test_temporal_caption_jobs.py",
    "tests/datasets/test_temporal_caption_flora.py",
    "tests/datasets/test_caption_throughput.py",
    "tests/datasets/test_execution_resources.py",
    "tests/datasets/test_dataset_results.py",
    "tests/datasets/test_viewer_cache_publication.py",
    "tests/datasets/test_mixed_episode_viewer.py",
    "tests/datasets/test_viewer_csv_compression.py",
    "tests/datasets/test_viewer_editing.py",
    "tests/datasets/test_viewer_chart_focus.py",
    "tests/datasets/test_dataset_scans.py",
    "tests/datasets/test_job_progress.py",
    "tests/datasets/test_console_layout.py",
    "tests/datasets/test_inspection.py",
    "tests/datasets/test_data_manager.py",
    "tests/datasets/test_data_profile.py",
    "tests/datasets/test_curation.py",
    "tests/datasets/test_curation_routes.py",
    "tests/datasets/test_quality_checks.py",
    "tests/datasets/test_quality_eef.py",
    "tests/datasets/test_execution_supervisor.py",
    "tests/datasets/test_local_execution.py",
    "tests/datasets/test_environments.py",
    "tests/datasets/test_releases.py",
    "tests/datasets/test_rolling_deployment.py",
    "tests/datasets/test_deployment_progress.py",
    "tests/datasets/test_control_plane.py",
    "tests/datasets/test_operation_log.py",
    "tests/datasets/test_job_management.py",
    "tests/datasets/test_usage_list_bounds.py",
    "tests/datasets/test_list_refresh.py",
    "tests/datasets/test_job_extra_quota.py",
    "tests/datasets/test_mcap_import.py",
    "tests/datasets/test_mcap_recovery.py",
    "tests/datasets/test_data_platform_agent.py",
    "tests/datasets/test_management_ui.py",
]


def validation_pytest_options() -> list[str]:
    """Run the same complete suite in isolated workers when xdist is installed."""
    requested = os.environ.get("DATA_PLATFORM_RELEASE_TEST_WORKERS", "8")
    try:
        workers = int(requested)
    except ValueError as exc:
        raise ValueError("DATA_PLATFORM_RELEASE_TEST_WORKERS must be an integer from 1 to 8") from exc
    if not 1 <= workers <= 8:
        raise ValueError("DATA_PLATFORM_RELEASE_TEST_WORKERS must be an integer from 1 to 8")
    options = ["--durations=20", "--durations-min=0.5"]
    if workers == 1:
        return options
    if importlib.util.find_spec("xdist") is None:
        print("pytest-xdist is not installed; running release validation serially", flush=True)
        return options
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return [
        *options,
        "-p",
        "lerobot.data_platform.ops.validation",
        "-n",
        str(min(workers, available)),
        "--dist=loadfile",
    ]


def pytest_configure(config):
    root = os.environ.get("DATA_PLATFORM_VALIDATION_CACHE_ROOT")
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not root or not worker:
        return
    if not re.fullmatch(r"gw[0-9]+", worker):
        raise ValueError("Invalid pytest worker identity")
    for key in ("HF_HOME", "HF_DATASETS_CACHE", "HF_LEROBOT_HOME", "UV_CACHE_DIR"):
        os.environ[key] = str(Path(root) / worker / key)
