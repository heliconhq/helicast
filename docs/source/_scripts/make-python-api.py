import importlib
import inspect
import logging
import os
import pkgutil
import subprocess
from math import ceil
from pathlib import Path
from pprint import pprint
from types import ModuleType
from typing import List, Optional

import numpy as np
import pandas as pd

import helicast.data_processing
from helicast.data_processing import add_future_columns, add_lagged_columns


def get_git_repo_root() -> Optional[str]:
    """Returns the root of the git repo as a string. If the folder in which process is
    executed is not within a git repo, a ``RuntimeError`` is raised."""
    try:
        # Run the git command to get the top-level repo directory
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        return repo_root
    except subprocess.CalledProcessError:
        raise RuntimeError("Not in a  git repo")


def _recursively_list_submodules_and_subpackages(package: ModuleType) -> List[str]:
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = []
    for loader, name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        results.append(name)
        if is_pkg:
            # Recursively list submodules/subpackages of this subpackage
            subpackage = importlib.import_module(name)
            results.extend(_recursively_list_submodules_and_subpackages(subpackage))
    return results


def list_submodules_and_subpackages(
    package: ModuleType, include_private: bool = True
) -> List[str]:
    module_names = _recursively_list_submodules_and_subpackages(package)

    def is_private_module(module_name: str):
        parts = module_name.split(".")
        if parts[-1].startswith("_"):
            return True
        return False

    def is_public_module(module_name: str):
        return not is_private_module(module_name)

    if not include_private:
        module_names = list(filter(is_public_module, module_names))
    module_names = list(sorted(module_names))
    return module_names


def list_functions(module_name: str, full_path: bool = False):
    module = importlib.import_module(module_name)
    functions = []
    for name in dir(module):
        attribute = getattr(module, name)
        if inspect.isfunction(attribute):
            if full_path:
                name = module_name + "." + name
            functions.append(name)
    functions = list(sorted(functions))
    return functions


def list_classes(module_name: str, full_path: bool = False):
    module = importlib.import_module(module_name)
    classes = []
    for name in dir(module):
        attribute = getattr(module, name)
        if inspect.isclass(attribute):
            if full_path:
                name = module_name + "." + name
            classes.append(name)
    classes = list(sorted(classes))
    return classes


def write_files(registry: dict) -> List[str]:
    new_registry = {}

    source_folder = Path(get_git_repo_root(), "docs", "source")
    for module, members in registry.items():
        new_registry[module] = []
        folder = Path(source_folder, "python-api", module.split(".")[-1])
        os.makedirs(folder, exist_ok=True)
        for name in members["classes"]:
            filepath = Path(folder, name + ".rst")
            full_name = module + "." + name
            new_registry[module].append(filepath)
            with open(filepath, "w") as fhandle:
                fhandle.write(f"{name}\n")
                fhandle.write(f"{'-'*int(1.5 *len(name))}\n")
                fhandle.write(f"\n")
                fhandle.write(f".. autoclass:: {full_name}\n")
                fhandle.write(f"    :members:\n")
                fhandle.write(f"    :undoc-members:\n")
                fhandle.write(f"    :inherited-members:\n")

        for name in members["functions"]:
            filepath = Path(folder, name + ".rst")
            full_name = module + "." + name
            new_registry[module].append(filepath)
            with open(filepath, "w") as fhandle:
                fhandle.write(f"{name}\n")
                fhandle.write(f"{'-'*int(1.5 *len(name))}\n")
                fhandle.write(f"\n")
                fhandle.write(f".. autofunction:: {full_name}\n")

    return new_registry


if __name__ == "__main__":
    module_names = list_submodules_and_subpackages(helicast, include_private=False)

    registry = {i: {} for i in module_names}

    for module_name in module_names:
        registry[module_name]["functions"] = list_functions(module_name)
        registry[module_name]["classes"] = list_classes(module_name)

    source_folder = Path(get_git_repo_root(), "docs", "source")
    new_registry = write_files(registry)
    for k in new_registry.keys():
        new_registry[k] = [
            str(i).replace(str(source_folder), "").strip("/").strip("\\")
            for i in new_registry[k]
        ]

    def clean_name(x: str):
        x = x.split(".")[-1]
        x = " ".join([i.capitalize() for i in x.split("_")])
        return x

    new_registry = {clean_name(i): j for i, j in new_registry.items()}

    with open(Path(source_folder, "python-api.template"), "r") as fhandle:
        raw_file = "".join(fhandle.readlines())

    include = []
    for module_name, rst_filepath in new_registry.items():
        include.append(".. toctree::\n")
        include.append("   :titlesonly:\n")
        include.append(f"   :caption: {module_name}\n")
        include.append("\n")
        for f in rst_filepath:
            include.append(f"   {f}\n")
        include.append(f"\n")

    raw_file = raw_file.replace("<INCLUDE HERE>", "".join(include))
    print(raw_file)

    with open(Path(source_folder, "python-api.rst"), "w") as fhandle:
        fhandle.write(raw_file)

    pprint(new_registry)
