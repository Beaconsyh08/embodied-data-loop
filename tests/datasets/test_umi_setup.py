"""Installer boundaries: pinned sources, isolated interpreters and preserved existing data."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from lerobot.data_platform.operators.implementations.umi.runtime import LAUNDRY_COMMIT

SCRIPT = Path(__file__).resolve().parents[2] / "deploy/data-platform/setup-umi.py"
spec = importlib.util.spec_from_file_location("umi_setup", SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "file.txt").write_text("pinned source\n")
    subprocess.run(["git", "-C", str(root), "add", "file.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return root, commit


def test_installer_pin_matches_platform_contract():
    assert setup.LAUNDRY_COMMIT == LAUNDRY_COMMIT


def test_source_verification_rejects_wrong_commit_edits_and_untracked_code(source):
    root, commit = source
    setup.verify_source(root, commit)
    with pytest.raises(ValueError, match="pinned"):
        setup.verify_source(root, "0" * 40)
    (root / "file.txt").write_text("changed\n")
    with pytest.raises(ValueError, match="local changes"):
        setup.verify_source(root, commit)
    (root / "file.txt").write_text("pinned source\n")
    (root / "injected.py").write_text("print('unexpected code')")
    with pytest.raises(ValueError, match="local changes"):
        setup.verify_source(root, commit)


def test_source_must_be_the_checkout_root(source):
    root, commit = source
    child = root / "subdirectory"
    child.mkdir()
    with pytest.raises(ValueError, match="dedicated Git checkout"):
        setup.verify_source(child, commit)


def test_existing_installation_is_not_overwritten(tmp_path, monkeypatch):
    root = tmp_path / "existing"
    root.mkdir()
    marker = root / "important.json"
    marker.write_text('{"keep":true}')
    monkeypatch.setattr(setup, "run", lambda *a, **k: pytest.fail("Must fail before external commands"))
    with pytest.raises(FileExistsError):
        setup.install(root, "local-mirror", "3.11")
    assert marker.read_text() == '{"keep":true}'


@pytest.mark.parametrize("root", [Path("relative"), setup.PLATFORM / "third_party/laundry"])
def test_installer_keeps_external_sources_outside_platform(root):
    with pytest.raises(ValueError, match="outside"):
        setup.install(root, "local-mirror", "3.11")


@pytest.mark.parametrize(
    "repository",
    [
        "--config=unexpected",
        "https://secret@example.invalid/repo",
        "https://example.invalid/repo?token=secret",
    ],
)
def test_installer_rejects_credential_urls_before_creating_directories(tmp_path, repository):
    root = tmp_path / "new"
    with pytest.raises(ValueError, match="credential-free"):
        setup.install(root, repository, "3.11")
    assert not root.exists()


def test_probe_isolates_python_and_disables_implicit_gpu_installation(monkeypatch, tmp_path):
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        monkeypatch.setenv(key, "/foreign/python")
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:1234")
    env = setup.clean_env(tmp_path)
    assert env["PYTHONPATH"] == str(tmp_path)
    assert not any(
        key in env for key in ("PYTHONHOME", "LD_LIBRARY_PATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")
    )
    assert env["LAUNDRY_CUMOTION_NO_INSTALL"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["HTTPS_PROXY"] == "http://localhost:1234"


def test_visual_meshes_must_exist_and_not_be_lfs_pointers(tmp_path):
    model = tmp_path / setup.ROBOT
    model.mkdir(parents=True)
    (model / "H10W_DVT2_TekkenD_with_Camera_2.urdf").write_text(
        '<robot><link><visual><geometry><mesh filename="visual.stl"/></geometry></visual>'
        '<collision><geometry><mesh filename="unused-collision.stl"/></geometry></collision></link></robot>'
    )
    with pytest.raises(ValueError, match="Missing robot mesh"):
        setup.verify_robot(tmp_path)
    mesh = model / "visual.stl"
    mesh.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="Git LFS"):
        setup.verify_robot(tmp_path)
    mesh.write_bytes(b"fixture visual mesh")
    setup.verify_robot(tmp_path)


def test_viewer_config_with_wrong_pin_does_not_launch_interpreter(tmp_path, monkeypatch):
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps({"laundry_commit": "wrong"}))
    monkeypatch.setattr(setup, "probe", lambda *a, **k: pytest.fail("Do not launch an unverified source"))
    with pytest.raises(ValueError, match="supported data-laundry"):
        setup.check(config, "viewer")


def test_install_requires_explicit_repository(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "install", "--root", str(tmp_path / "new")])
    with pytest.raises(SystemExit) as error:
        setup.main()
    assert error.value.code == 2
    assert not (tmp_path / "new").exists()
