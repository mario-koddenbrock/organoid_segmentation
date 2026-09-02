"""Canonical feature sets used by the morphology classification analyses."""

# Measurements that can be computed from one nucleus mask in isolation.  These
# describe only that nucleus' size or shape; none requires an organoid mask,
# another nucleus, or an organoid-level count/summary.
PURE_NUCLEUS_FEATURES = [
    "volume_um3",
    "surface_area_um2",
    "sphericity",
    "solidity",
    "ellipsoid_axis_major_um",
    "ellipsoid_axis_medium_um",
    "ellipsoid_axis_minor_um",
    "aspect_ratio_minor_per_medium",
    "aspect_ratio_medium_per_major",
    "aspect_ratio_minor_per_major",
    "prolate_ratio",
    "oblate_ratio",
]

# These columns describe a nucleus in relation to its parent organoid or to
# other nuclei.  They must not be used in a nucleus-only morphology analysis.
ORGANOID_CONTEXT_FEATURES = [
    "relative_z_position",
    "distance_to_organoid_center_um",
    "neighborhood_density",
    "nuc_count_per_organoid",
    "count",
]
