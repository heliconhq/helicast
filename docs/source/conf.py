# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Helicast"
copyright = "2024, Helicon"
author = "Helicon"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]


extensions = [
    "nbsphinx",
    "sphinx.ext.napoleon",
    # "sphinx_rtd_theme",
    # "sphinx_book_theme",
    # "pydata_sphinx_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    # "sphinx.ext.inheritance_diagram",
    "sphinx.ext.todo",
    "sphinx_design",
]

# Options for autodoc
# -------------------
autosummary_generate = True
add_module_names = True
todo_include_todos = True
autodoc_typehints = "both"
autodoc_typehints_format = "short"
autodoc_member_order = "alphabetical"


templates_path = ["_templates"]
exclude_patterns = []
# latex_elements = {
#    "papersize": "letterpaper",
#    "pointsize": "10pt",
#    "preamble": "",
#    "figure_align": "htbp",
# }

# These folders are copied to the documentation's HTML output
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

html_js_files = [
    "js/custom.js",
]

html_logo = "_static/logo.svg"


html_theme_options = {
    ### Toc options
    # "collapse_navigation": False,
    # "sticky_navigation": False,
    # "includehidden": True,
    # "titles_only": False,
    # "external_links": [
    #    {"name": "link-one-name", "url": "https://<link-one>"},
    #    {"name": "link-two-name", "url": "https://<link-two>"},
    # ],
    # "show_nav_level": 3,
    # "navigation_depth": 3,
    "show_toc_level": 2,
    # "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "collapse_navigation": True,
}


import os
import pkgutil
import inspect
import helicast
import sys


def generate_docs_for_module(module, docs_path):
    """
    Generates .rst files for each public class and function in the given module.
    """
    for importer, modname, ispkg in pkgutil.walk_packages(
        module.__path__, module.__name__ + "."
    ):
        if ispkg:
            continue  # Skip subpackages

        # Load the module
        __import__(modname)
        mod = sys.modules[modname]

        # Create a directory for the module
        dir_path = os.path.join(docs_path, mod.__name__.replace(".", os.sep))
        os.makedirs(dir_path, exist_ok=True)

        # Iterate through the module's members
        for member_name, member_obj in inspect.getmembers(mod):
            if inspect.isfunction(member_obj) or inspect.isclass(member_obj):
                # Filter private members
                if member_name.startswith("_"):
                    continue

                # Determine the file path
                file_path = os.path.join(dir_path, f"{member_name}.rst")
                print(file_path)

                # Write the .rst content
                with open(file_path, "w") as f:
                    if inspect.isclass(member_obj):
                        f.write(
                            f".. autoclass:: {modname}.{member_name}\n   :members:\n"
                        )
                    elif inspect.isfunction(member_obj):
                        f.write(f".. autofunction:: {modname}.{member_name}\n")


# Path where you want to save the generated .rst files
# docs_path = "/home/leandro/Documents/repos/helicast/docs"
# generate_docs_for_module(helicast, docs_path)
