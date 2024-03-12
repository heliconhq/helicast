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
]

# Options for autodoc
# -------------------
autosummary_generate = True
add_module_names = True
todo_include_todos = True
autodoc_typehints = "both"
autodoc_typehints_format = "short"


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

html_logo = "_static/logo.jpg"


html_theme_options = {
    ### Toc options
    # "collapse_navigation": False,
    # "sticky_navigation": False,
    # "navigation_depth": -1,
    # "includehidden": True,
    # "titles_only": False,
    # "external_links": [
    #    {"name": "link-one-name", "url": "https://<link-one>"},
    #    {"name": "link-two-name", "url": "https://<link-two>"},
    # ],
    # "show_nav_level": 0,
}
