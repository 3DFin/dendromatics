#### IMPORTS ####

import dendroptimized

# -----------------------------------------------------------------------------------------------------------------------------------
# voxelate
# ----------------------------------------------------------------------------------------------------------------------------------------


def voxelate(
    cloud,
    resolution_xy,
    resolution_z,
    n_digits=5,
    X_field=0,
    Y_field=1,
    Z_field=2,
    with_n_points=True,
    silent=False,
):

    return dendroptimized.voxelate(
        cloud, resolution_xy, resolution_z, n_digits, X_field, Y_field, Z_field
    )
