import importlib
from pathlib import Path
from typing import List

import pytest


def get_modules(package_name: str, include_root: bool = False) -> List[str]:
    """List all the modules within a package. The modules are files within the package
    root directory that end with ".py" and do not start with an underscore.

    Args:
        package_name: The name of the package to list modules for.
        include_root: If True, the package root will be included in the list. Defaults
            to False.

    Returns:
        The list of module names within the package.
    """
    package = importlib.import_module(package_name)
    package_dir = Path(package.__path__[0])

    modules = []
    for p in package_dir.iterdir():
        if p.is_file() and p.name.endswith(".py") and not p.name.startswith("_"):
            mod = p.relative_to(package_dir).name.replace(".py", "")
            mod = package_name + "." + mod
            modules.append(mod)

    if include_root:
        modules = [package_name] + modules

    return modules


def get_subpackages(package_name: str, include_root: bool = False) -> List[str]:
    """List all the subpackages within a package. The subpackages are directories within
    the package root directory that contain an "__init__.py" file (which is their
    entrypoint).

    Args:
        package_name: The name of the package to list modules for.
        include_root: If True, the package root will be included in the list. Defaults
            to False.

    Returns:
        The list of subpackages names within the package.
    """
    package = importlib.import_module(package_name)
    package_dir = Path(package.__path__[0])

    subpackages = []
    for p in package_dir.iterdir():
        if p.is_dir() and Path(p, "__init__.py").is_file():
            subpackages.append(package_name + "." + p.relative_to(package_dir).name)

    if include_root:
        subpackages = [package_name] + subpackages
    return subpackages


def get_modules_and_subpackages(
    package_name: str, include_root: bool = False
) -> List[str]:
    """Get all modules and subpackages within a package. See `get_modules` and
    `get_subpackages` for more details.

    Args:
        package_name: The name of the package to list modules for.
        include_root: If True, the package root will be included in the list. Defaults
            to False.

    Returns:
        The list of modules and subpackages names within the package.
    """
    modules = get_modules(package_name, include_root=include_root)
    subpackages = get_subpackages(package_name, include_root=False)
    return modules + subpackages


@pytest.mark.parametrize("module", get_modules_and_subpackages("helicast"))
def test_import_modules(module):
    """
    Test that each module can be imported without raising an exception.
    """
    try:
        importlib.import_module(module)
    except Exception as e:
        pytest.fail(f"Import failed for {module}: {e}")
