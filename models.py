import torch.nn as nn
import torch

# -------------------------
# Model
# -------------------------
class SimCLR(nn.Module):
    """
    Returns:
      h_map_small: (B, c_small, H, W)   feature map for PH (channel reduced)
      h:          (B, C)               pooled backbone feature
      rep:        (B, proj_dim)        projection head output
    """
    # Spatial resolution / channels of each stage on 32x32 CIFAR with
    # cifar_no_maxpool=True (conv1 stride 2): layer1 16x16/64ch, layer2 8x8/128ch,
    # layer3 4x4/256ch, layer4 2x2/512ch. layer4 gives only 4 points -- too few for
    # meaningful persistent homology -- so the PH source layer is configurable.
    _LAYER_CHANNELS = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}

    def __init__(
        self,
        base_encoder_fn,
        projection_dim=128,
        proj_hidden_dim=512,
        reduce_channels=8,
        cifar_no_maxpool: bool = True,
        ph_source_layer: str = "layer3",
        ph_extra_layers=(),
    ):
        super().__init__()
        backbone = base_encoder_fn(weights=None)

        if cifar_no_maxpool:
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        else:
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.feature_dim = backbone.fc.in_features  # 512 for resnet18/34

        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, projection_dim),
        )

        # PH map is taken from `ph_source_layer`. layer3 -> 4x4=16 points (default),
        # layer2 -> 8x8=64 points (richer, supports H1). The 1x1 conv reduces channels.
        assert ph_source_layer in self._LAYER_CHANNELS, f"bad ph_source_layer {ph_source_layer}"
        self.ph_source_layer = ph_source_layer
        in_ch = self._LAYER_CHANNELS[ph_source_layer]
        self.ph_reduce = nn.Conv2d(in_ch, reduce_channels, kernel_size=1, bias=False)

        # Optional extra layers for MULTISCALE PH (persistence at several network
        # depths). Kept in a separate ModuleDict so single-layer checkpoints (which
        # only have `ph_reduce`) still load with strict=True.
        self.ph_extra_layers = tuple(ph_extra_layers)
        self.ph_reduce_extra = nn.ModuleDict({
            L: nn.Conv2d(self._LAYER_CHANNELS[L], reduce_channels, kernel_size=1, bias=False)
            for L in self.ph_extra_layers
        })

    def _backbone_feats(self, x):
        x = self.stem(x)
        feats = {}
        x = self.layer1(x); feats["layer1"] = x
        x = self.layer2(x); feats["layer2"] = x
        x = self.layer3(x); feats["layer3"] = x
        x = self.layer4(x); feats["layer4"] = x
        return feats

    def forward(self, x):
        feats = self._backbone_feats(x)
        h_map_small = self.ph_reduce(feats[self.ph_source_layer])  # (B, c_small, Hs, Ws)
        h = torch.flatten(self.avgpool(feats["layer4"]), 1)        # (B, 512)
        rep = self.projector(h)                                    # (B, proj_dim)
        return h_map_small, h, rep

    def ph_maps(self, x):
        """
        Multiscale variant: returns (maps, h, rep) where `maps` is a list of
        reduced PH feature maps [primary] + [extra layers...]. Single forward.
        """
        feats = self._backbone_feats(x)
        maps = [self.ph_reduce(feats[self.ph_source_layer])]
        for L in self.ph_extra_layers:
            maps.append(self.ph_reduce_extra[L](feats[L]))
        h = torch.flatten(self.avgpool(feats["layer4"]), 1)
        rep = self.projector(h)
        return maps, h, rep



