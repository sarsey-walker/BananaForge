"""Shared types for 3MF export."""

from dataclasses import dataclass


@dataclass
class LayerMaterial:
    """Represents material assignment for a specific layer."""

    layer_index: int
    material_id: str
    transparency: float = 1.0
    layer_height: float = 0.2


@dataclass
class ThreeMFExportConfig:
    """Configuration for 3MF export options."""

    bambu_compatible: bool = False
    include_metadata: bool = True
    include_thumbnail: bool = False
    compress_xml: bool = True
    validate_output: bool = True
