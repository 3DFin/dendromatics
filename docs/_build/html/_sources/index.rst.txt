.. dendromatics documentation master file, created by
   sphinx-quickstart on Fri Mar  3 17:07:59 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dendromatics: Automatic dendrometry in terrestrial point clouds
===============================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

This package provides functionalities to implement an updated version of algorithm presented in [CABO2018]_ to detect the trees present in a terrestrial 3D point cloud from a forest plot, and compute individual tree parameters: tree height, tree location, diameters along the stem (including DBH), and stem axis.


Contents
==================

.. toctree::
   :maxdepth: 2
   
   installation

   algorithm
   
   executable
   
   cc_plugin


Indices and tables
==================

:ref:`genindex`

:ref:`modindex`

:ref:`search`


References
==========

.. [CABO2018] Cabo, C., Ordonez, C., Lopez-Sanchez, C. A., & Armesto, J. (2018). Automatic dendrometry: Tree detection, tree height and diameter estimation using terrestrial laser scanning. International Journal of Applied Earth Observation and Geoinformation, 69, 164–174. https://doi.org/10.1016/j.jag.2018.01.011
