# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
import os
from pathlib import Path
from typing import Any
from sphinx.application import Sphinx


sys.path.insert(0, str(Path("../src").resolve()))


# -- Project information -----------------------------------------------------

project = "simpletable"
copyright = "2007-2025, M. Fouesneau"
author = "M. Fouesneau"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_automodapi.automodapi",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinx_mdinclude",
    "myst_nb",
]
numpydoc_show_class_members = False

# myst_nb configurations
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
    ".md": "myst-nb",
}
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath"]
# auto-generate heading anchors down to this level
myst_heading_anchors = 3

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/test_**", "**/.locks"]

# hide input prompts in notebooks
nbsphinx_prompt_width = "0"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_theme_options = {
    "path_to_docs": "doc",
    "repository_url": "https://github.com/mfouesneau/simpletable",
    "repository_branch": "master",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",
        "notebook_interface": "jupyterlab",
    },
    "use_edit_page_button": False,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_source_button": True,
}
html_baseurl = "https://mfouesneau.github.io/simpletable/"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# hide coppy button on outputs
copybutton_selector = "div:not(.output) > div.highlight pre"

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# remove module name prefix from functions
add_module_names = False

autodoc_member_order = "alphabetical"

autodoc_typehints_description_target = "all"

autodoc_typehints_format = "short"

coverage_statistics_to_report = True

coverage_statistics_to_stdout = True

toc_object_entries_show_parents = "hide"


def shorten_autosummary_titles(app: Sphinx, *args: Any) -> None:
    """Remove module and class from the autosummary titles."""
    autosummary_dir = os.path.join(app.srcdir, "api", "_autosummary")
    if not os.path.exists(autosummary_dir):
        return

    for filename in os.listdir(autosummary_dir):
        if not filename.endswith(".rst"):
            continue

        path = os.path.join(autosummary_dir, filename)
        with open(path, "r") as f:
            lines = f.readlines()

        # skip if missing a title or if a module/class
        if not lines or lines[0].count(".") < 2:
            continue

        short = lines[0].strip().rsplit(".", 1)[-1]
        lines[0] = short + "\n"
        lines[1] = "=" * len(short) + "\n"
        with open(path, "w") as f:
            f.writelines(lines)


def setup(app: Sphinx) -> None:
    app.connect("env-before-read-docs", shorten_autosummary_titles)
