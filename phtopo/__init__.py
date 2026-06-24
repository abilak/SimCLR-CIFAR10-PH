"""phtopo: differentiable, GPU-friendly persistent-homology toolkit for PHSim."""
from .diffph import (
    vr_h0_persistence,
    vr_h0_persistence_batch,
    h0_diagram_from_deaths,
    sliced_wasserstein_pd,
    sliced_wasserstein_h0,
    sliced_wasserstein_h0_batch,
    total_persistence,
    persistence_entropy,
)
from .losses import (
    topo_separation_loss,
    raw_sw_separation_loss,
    topo_consistency_loss,
    raw_consistency_loss,
    feature_map_to_pointcloud,
    standardize_pointcloud_batch,
)
from .adv import pgd_ascent_on_loss
from .dual_bn import (
    DualBatchNorm2d,
    bn_route,
    convert_to_dual_bn,
    has_dual_bn,
    count_dual_bn,
    state_dict_is_dual_bn,
    load_state_dict_auto,
)

__all__ = [
    "vr_h0_persistence",
    "vr_h0_persistence_batch",
    "h0_diagram_from_deaths",
    "sliced_wasserstein_pd",
    "sliced_wasserstein_h0",
    "sliced_wasserstein_h0_batch",
    "total_persistence",
    "persistence_entropy",
    "topo_separation_loss",
    "raw_sw_separation_loss",
    "topo_consistency_loss",
    "raw_consistency_loss",
    "feature_map_to_pointcloud",
    "standardize_pointcloud_batch",
    "pgd_ascent_on_loss",
    "DualBatchNorm2d",
    "bn_route",
    "convert_to_dual_bn",
    "has_dual_bn",
    "count_dual_bn",
    "state_dict_is_dual_bn",
    "load_state_dict_auto",
]
