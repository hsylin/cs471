"""Install pinned NVIDIA CUDA 12.9.1 compiler components and cuRAND headers.

This does not change Python packages, system CUDA or the shell environment.
Use CUDA_HOME=<prefix> and prepend <prefix>/bin for the intended child process.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
INDEX = "redistrib_12.9.1.json"
INDEX_SHA256 = "8335301010b0023ee1ff61eb11e2600ca62002d76780de4089011ad77e0c7630"
PINS = {
    "cuda_nvcc": ("12.9.86", "7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35", 81100244),
    "cuda_cudart": ("12.9.79", "1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99", 1514676),
    "cuda_cccl": ("12.9.27", "8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa", 997888),
    "libcurand": ("10.3.10.19", "48281b4caadb1cf790d44ac76b23c77d06f474c0b1799814f314aafec9258ad6", 89582256),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path, expected: str, size: int | None = None) -> None:
    if path.exists():
        if sha256(path) != expected or (size is not None and path.stat().st_size != size):
            raise ValueError(f"Existing download fails integrity: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as target:
        shutil.copyfileobj(source, target, 4 * 1024 * 1024)
    if sha256(partial) != expected or (size is not None and partial.stat().st_size != size):
        raise ValueError(f"Download fails declared integrity: {path.name}")
    partial.replace(path)


def component_specs(index: dict) -> list[dict]:
    if index.get("release_label") != "12.9.1":
        raise ValueError("Wrong CUDA release")
    specs = []
    for name, (version, digest, size) in PINS.items():
        component = index[name]
        archive = component["linux-x86_64"]
        expected_path = f"{name}/linux-x86_64/{name}-linux-x86_64-{version}-archive.tar.xz"
        if (component["version"], archive["sha256"], int(archive["size"]), archive["relative_path"]) != (version, digest, size, expected_path):
            raise ValueError(f"Unexpected NVIDIA component declaration: {name}")
        specs.append({"component": name, "version": version, "sha256": digest,
                      "bytes": size, "url": BASE + expected_path,
                      "filename": Path(expected_path).name, "license": component["license"],
                      "license_path": component["license_path"],
                      "installed_subset": "include plus license only" if name == "libcurand" else "complete component"})
    return specs


def merge_tree(source: Path, target: Path) -> None:
    """Merge component directories without silently replacing conflicting files."""
    for entry in sorted(source.iterdir()):
        dest = target / entry.name
        if entry.is_symlink():
            link = os.readlink(entry)
            if dest.is_symlink() and os.readlink(dest) == link:
                continue
            if dest.exists() or dest.is_symlink():
                raise ValueError(f"Conflicting component link: {dest}")
            dest.symlink_to(link)
        elif entry.is_dir():
            dest.mkdir(exist_ok=True)
            merge_tree(entry, dest)
        elif dest.exists():
            if sha256(entry) != sha256(dest):
                raise ValueError(f"Conflicting component file: {dest}")
        else:
            shutil.copy2(entry, dest)


def install(prefix: Path, cache: Path) -> dict:
    if prefix.exists():
        raise ValueError("Destination already exists; use its immutable manifest or a new prefix")
    cache.mkdir(parents=True, exist_ok=True)
    index_file = cache / INDEX
    download(BASE + INDEX, index_file, INDEX_SHA256)
    specs = component_specs(json.loads(index_file.read_text()))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=prefix.name + ".staging-", dir=prefix.parent) as staging:
        stage = Path(staging)
        assembled = stage / "toolkit"
        assembled.mkdir()
        license_records = []
        for spec in specs:
            archive = cache / spec["filename"]
            print(f"Downloading/verifying {spec['component']} {spec['version']}", flush=True)
            download(spec["url"], archive, spec["sha256"], spec["bytes"])
            extraction = stage / spec["component"]
            extraction.mkdir()
            with tarfile.open(archive) as handle:
                handle.extractall(extraction, filter="data")
            roots = list(extraction.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise ValueError("Expected one component archive root")
            component_root = roots[0]
            licenses = assembled / "licenses" / spec["component"]
            licenses.mkdir(parents=True)
            for entry in list(component_root.iterdir()):
                if entry.is_file() and ("license" in entry.name.lower() or "eula" in entry.name.lower()):
                    shutil.copy2(entry, licenses / entry.name)
                    license_records.append({"component": spec["component"],
                                            "path": str((licenses / entry.name).relative_to(assembled)),
                                            "sha256": sha256(entry)})
                    entry.unlink()
            if not list(licenses.iterdir()):
                raise ValueError(f"No retained license for {spec['component']}")
            if spec["component"] == "libcurand":
                # Sampling kernels use device RNG headers, not the host RNG
                # library. Keep the existing Python CUDA libraries untouched.
                merge_tree(component_root / "include", assembled / "include")
            else:
                merge_tree(component_root, assembled)
        if not (assembled / "lib64").exists() and (assembled / "lib").is_dir():
            (assembled / "lib64").symlink_to("lib")
        required = ["bin/nvcc", "bin/ptxas", "include/cuda_runtime.h", "include/cub/cub.cuh",
                    "include/curand.h", "include/curand_kernel.h"]
        for name in required:
            if not (assembled / name).is_file():
                raise ValueError(f"Missing required toolkit file {name}")
        version = subprocess.check_output([str(assembled / "bin/nvcc"), "--version"], text=True)
        if "V12.9.86" not in version:
            raise ValueError("Installed nvcc reports unexpected version")
        records = []
        for path in sorted(assembled.rglob("*")):
            if path.is_symlink():
                records.append({"path": str(path.relative_to(assembled)), "symlink": os.readlink(path)})
            elif path.is_file():
                records.append({"path": str(path.relative_to(assembled)), "bytes": path.stat().st_size,
                                "sha256": sha256(path)})
        report = {"schema_version": 1, "status": "installed_verified",
                  "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "prefix": str(prefix), "release": "12.9.1", "platform": "linux-x86_64",
                  "index": {"url": BASE + INDEX, "sha256": INDEX_SHA256},
                  "components": specs, "licenses": license_records, "files": records,
                  "nvcc_version": version, "installer_sha256": sha256(Path(__file__)),
                  "environment_changes": "none; child processes must set CUDA_HOME and PATH explicitly",
                  "scope": "Compiler, runtime headers/libraries, CCCL and cuRAND headers only; no Python package or system toolkit changes",
                  "gpu_diagnostic": "not_run_by_installer"}
        (assembled / "install-manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        shutil.copy2(index_file, assembled / INDEX)
        assembled.rename(prefix)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = install(args.prefix.resolve(), args.cache.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "prefix": report["prefix"], "files": len(report["files"])}))


if __name__ == "__main__":
    main()
