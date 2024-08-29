# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "dendromatics"
copyright = "2023, Carlos Cabo & Diego Laino"
author = "Carlos Cabo & Diego Laino"
release = "00.00.01"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    #   "sphinx_reference_rename",
    "sphinx.ext.intersphinx",
]

napoleon_google_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]


# Modifying qualified names

# sphinx_reference_rename_mapping = {
#     "dendromatics.primitives.voxel.voxelate": "voxelate",
#     "dendromatics.primitives.clustering.DBSCAN_clustering": "DBSCAN_clustering",
#     "dendromatics.ground.clean_cloth": "clean_cloth",
#     "dendromatics.ground.clean_ground": "clean_ground",
#     "dendromatics.ground.generate_dtm": "generate_dtm",
#     "dendromatics.ground.normalize_heights": "normalize_heights",
#     "dendromatics.stripe.verticality_clustering": "verticality_clustering",
#     "dendromatics.stripe.verticality_clustering_iteration": "verticality_clustering_iteration",
#     "dendromatics.individualize.compute_axes": "compute_axes",
#     "dendromatics.individualize.compute_heights": "compute_heights",
#     "dendromatics.individualize.individualize_trees": "individualize_trees",
#     "dendromatics.sections.compute_sections": "compute_sections",
#     "dendromatics.sections.fit_circle": "fit_circle",
#     "dendromatics.sections.fit_circle_check": "fit_circle_check",
#     "dendromatics.sections.inner_circle": "inner_circle",
#     "dendromatics.sections.point_clustering": "point_clustering",
#     "dendromatics.sections.sector_occupancy": "sector_occupancy",
#     "dendromatics.sections.tilt_detection": "tilt_detection",
#     "dendromatics.sections.tree_locator": "tree_locator",
#     "dendromatics.draw.draw_axes": "draw_axes",
#     "dendromatics.draw.draw_circles": "draw_circles",
# }

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/dendromatics_logo.png"
