.. dendromatics documentation master file, created by
   sphinx-quickstart on Fri Mar  3 17:07:59 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dendromatics: Automatic dendrometry in terrestrial point clouds
===============================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

This package provides functionalities to implement an updated version of the algorithm presented in [CABO2018]_. It detects the trees present in a terrestrial 3D point cloud from a forest plot, and compute individual tree parameters: tree height, tree location, diameters along the stem (including DBH), and stem axis.


Contents
==================

.. toctree::
   :maxdepth: 2
   
   installation

   algorithm

   dendromatics

   examples
   
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

.. [ESTE1996] Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. www.aaai.org

.. [PREN2021] Prendes, C., Cabo, C., Ordonez, C., Majada, J., & Canga, E. (2021). An algorithm for the automatic parametrization of wood volume equations from Terrestrial Laser Scanning point clouds: application in Pinus pinaster. GIScience and Remote Sensing, 58(7), 1130–1150. https://doi.org/10.1080/15481603.2021.1972712

.. [ZHAN2016] Zhang, W., Qi, J., Wan, P., Wang, H., Xie, D., Wang, X., & Yan, G. (2016). An easy-to-use airborne LiDAR data filtering method based on cloth simulation. Remote Sensing, 8(6). https://doi.org/10.3390/rs8060501