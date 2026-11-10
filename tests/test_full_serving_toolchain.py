import copy
from pathlib import Path

import pytest

from full_serving.install_toolchain import PINS, component_specs, download, merge_tree
from full_serving import check_toolchain as diagnostic


def test_workspace_rejects_shared_home_and_other_node_before_cuda_import(tmp_path, monkeypatch):
    monkeypatch.setenv("SKB_ROOT", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "4882284")
    monkeypatch.setenv("SLURMD_NODENAME", "kh074")
    monkeypatch.setattr(diagnostic.socket, "gethostname", lambda: "kh074")
    expected = tmp_path / "cache/flashinfer-jit/4882284/kh074"
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(tmp_path))
    with pytest.raises(RuntimeError, match="isolated job/node"):
        diagnostic.verified_workspace()
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(expected))
    assert diagnostic.verified_workspace() == ("kh074", expected)
    monkeypatch.setenv("SLURMD_NODENAME", "kh077")
    with pytest.raises(RuntimeError, match="identify this host"):
        diagnostic.verified_workspace()


def index():
    return {"release_label": "12.9.1", **{
        name: {"version": version, "license": "CUDA Toolkit", "license_path": name + "/LICENSE.txt",
               "linux-x86_64": {"sha256": digest, "size": str(size),
                                 "relative_path": f"{name}/linux-x86_64/{name}-linux-x86_64-{version}-archive.tar.xz"}}
        for name, (version, digest, size) in PINS.items()}}


def test_component_declaration_rejects_changed_digest_or_path():
    assert len(component_specs(index())) == 4
    assert component_specs(index())[-1]["installed_subset"] == "include plus license only"
    for key, value in [("sha256", "0" * 64), ("relative_path", "../untrusted.tar.xz")]:
        altered = copy.deepcopy(index())
        altered["cuda_nvcc"]["linux-x86_64"][key] = value
        with pytest.raises(ValueError, match="Unexpected NVIDIA"):
            component_specs(altered)


def test_preexisting_download_must_match_declared_bytes(tmp_path):
    target = tmp_path / "archive"
    target.write_bytes(b"wrong archive")
    with pytest.raises(ValueError, match="Existing download fails integrity"):
        download("https://invalid.example/never-requested", target, "0" * 64)


def test_merge_rejects_component_collisions_without_overwriting(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir(); target.mkdir()
    (source / "header.h").write_text("new")
    (target / "header.h").write_text("existing")
    with pytest.raises(ValueError, match="Conflicting component file"):
        merge_tree(source, target)
    assert (target / "header.h").read_text() == "existing"


def test_merge_preserves_relative_library_symlinks(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir(); target.mkdir()
    (source / "libcudart.so.12").write_text("library")
    (source / "libcudart.so").symlink_to("libcudart.so.12")
    merge_tree(source, target)
    assert (target / "libcudart.so").is_symlink()
    assert (target / "libcudart.so").read_text() == "library"
