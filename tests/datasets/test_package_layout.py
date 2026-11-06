"""Domain migration keeps imports, execution entry points and operator schemas compatible."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from lerobot.data_platform._compat import MODULE_ALIASES
from lerobot.data_platform._paths import PACKAGE_ROOT


def _python(code):
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    return result


def test_package_import_does_not_load_runtime_or_model_dependencies():
    _python(
        "import sys; import lerobot.data_platform; "
        "assert not {'torch', 'flask', 'sqlalchemy', 'pyarrow'} & set(sys.modules)"
    )


@pytest.mark.parametrize(
    "legacy, canonical",
    [
        ("execution_resources", "runtime.resources"),
        ("control_plane", "control.store"),
        ("precompute.preprocess.common", "operators.implementations.preprocess.common"),
        ("temporal_caption_flora", "operators.implementations.caption.flora"),
    ],
)
@pytest.mark.parametrize("legacy_first", [True, False])
def test_legacy_and_canonical_imports_share_state_and_spec(legacy, canonical, legacy_first):
    names = ["lerobot.data_platform." + name for name in (legacy, canonical)]
    if not legacy_first:
        names.reverse()
    _python(
        "import importlib; "
        f"a, b = [importlib.import_module(name) for name in {names!r}]; "
        "assert a is b; "
        "a._layout_probe = object(); assert b._layout_probe is a._layout_probe; "
        f"assert a.__spec__.name == {'lerobot.data_platform.' + canonical!r}; "
        f"assert a.__package__ == {'lerobot.data_platform.' + canonical.rsplit('.', 1)[0]!r}"
    )


def test_legacy_attribute_and_monkeypatch_use_canonical_module(monkeypatch):
    import lerobot.data_platform as platform
    from lerobot.data_platform.ops import releases

    assert platform.releases is releases
    marker = object()
    monkeypatch.setattr("lerobot.data_platform.releases.run", marker)
    assert releases.run is marker


@pytest.mark.parametrize("entry", ["agent", "release_cli", "mem_export"])
def test_legacy_module_commands_remain_executable(entry):
    for module in (entry, MODULE_ALIASES[entry]):
        result = subprocess.run(
            [sys.executable, "-m", "lerobot.data_platform." + module, "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_legacy_operator_entry_retains_parameter_schema():
    from dataclasses import replace

    from lerobot.data_platform.operators import get_operator
    from lerobot.data_platform.operators.schema import parameter_schema

    spec = get_operator("preprocess.smooth_action")
    old = replace(
        spec,
        entry_point=spec.entry_point.replace("operators.implementations", "precompute"),
    )
    assert parameter_schema(old) == parameter_schema(spec)
    assert parameter_schema(spec)["properties"]


def test_alias_targets_and_isolated_worker_assets_are_packaged():
    for target in MODULE_ALIASES.values():
        path = PACKAGE_ROOT.joinpath(*target.split("."))
        assert path.with_suffix(".py").is_file() or (path / "__init__.py").is_file(), target
    for relative in (
        "templates/visualize_dataset_temporal_caption.html",
        "static/environment.js",
        "operators/implementations/caption/flora_worker.py",
        "operators/implementations/mcap/mcap_worker.py",
        "operators/implementations/mcap/mcap_contract.py",
        "operators/implementations/umi/assets/review.js",
    ):
        assert (PACKAGE_ROOT / relative).is_file(), relative


def test_operator_discovery_does_not_import_implementations():
    result = _python(
        "import json, sys; from lerobot.data_platform.operators import list_operators; "
        "operators = list_operators(); "
        "assert not any(name.startswith('lerobot.data_platform.operators.implementations') "
        "for name in sys.modules); print(json.dumps([row.operator_id for row in operators]))"
    )
    assert "preprocess.smooth_action" in json.loads(result.stdout)


@pytest.mark.parametrize(
    "wrapper, module, arguments",
    [
        ("update-all.sh", "release_cli", ["--env", "dev", "--release", "fixture"]),
        ("environment.sh", "environment_cli", ["status", "--env", "dev"]),
    ],
)
def test_deployment_wrapper_can_fall_back_to_flat_installed_release(tmp_path, wrapper, module, arguments):
    source = PACKAGE_ROOT.parent.parent / "deploy/data-platform"
    scripts = tmp_path / "checkout/deploy/data-platform"
    scripts.mkdir(parents=True)
    shutil.copyfile(source / wrapper, scripts / wrapper)
    installed = tmp_path / "installed/dev/current"
    python = installed / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    legacy_module = installed / "lerobot/data_platform" / f"{module}.py"
    legacy_module.parent.mkdir(parents=True)
    legacy_module.touch()
    python.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        "assert sys.argv[1] == '-m'\n"
        "assert Path(*sys.argv[2].split('.')).with_suffix('.py').is_file()\n"
        "print(sys.argv[2])\n"
    )
    python.chmod(0o700)
    # Exercise the actual fallback in a temporary tree; no sudo or real installation access.
    runner = (source / "run-command.sh").read_text()
    runner = runner.replace("/opt/data-platform/", str(tmp_path / "installed") + "/")
    runner = runner.replace("if [[ $EUID -ne 0 ]]; then", "if false; then")
    (scripts / "run-command.sh").write_text(runner)
    result = subprocess.run(
        ["bash", str(scripts / wrapper), *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lerobot.data_platform." + module
