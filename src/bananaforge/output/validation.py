"""Validation and recovery helpers for 3MF export."""

import math
import re
import zlib
from typing import Any, Dict, List, Tuple

import numpy as np


class MeshValidator:
    """Validates basic mesh topology for 3MF export."""

    def check_manifold(
        self,
        vertices: List[Tuple[float, float, float]],
        triangles: List[Tuple[int, int, int]],
    ) -> bool:
        """Return True when every mesh edge has exactly two incident faces."""
        return self._edge_counts_are_closed(vertices, triangles)

    def check_watertight(
        self,
        vertices: List[Tuple[float, float, float]],
        triangles: List[Tuple[int, int, int]],
    ) -> bool:
        """Return True when the triangle mesh has no open edges."""
        return self._edge_counts_are_closed(vertices, triangles)

    def _edge_counts_are_closed(
        self,
        vertices: List[Tuple[float, float, float]],
        triangles: List[Tuple[int, int, int]],
    ) -> bool:
        if not vertices or not triangles:
            return False

        edge_counts: Dict[Tuple[int, int], int] = {}
        vertex_count = len(vertices)

        for triangle in triangles:
            if len(set(triangle)) != 3:
                return False
            if any(index < 0 or index >= vertex_count for index in triangle):
                return False

            a, b, c = triangle
            for edge in ((a, b), (b, c), (c, a)):
                key = tuple(sorted(edge))
                edge_counts[key] = edge_counts.get(key, 0) + 1

        return bool(edge_counts) and all(count == 2 for count in edge_counts.values())


class NormalValidator:
    """Validates and computes triangle normals."""

    def calculate_normal(
        self,
        vertices: List[Tuple[float, float, float]],
        triangle: Tuple[int, int, int],
    ) -> Tuple[float, float, float]:
        """Calculate the normalized normal for a triangle."""
        v0 = np.asarray(vertices[triangle[0]], dtype=float)
        v1 = np.asarray(vertices[triangle[1]], dtype=float)
        v2 = np.asarray(vertices[triangle[2]], dtype=float)
        normal = np.cross(v1 - v0, v2 - v0)
        length = np.linalg.norm(normal)
        if length == 0:
            return (0.0, 0.0, 0.0)
        normalized = normal / length
        return tuple(float(value) for value in normalized)


class ColorValidator:
    """Validates material color strings."""

    def validate_color_format(self, color: Any) -> bool:
        """Return True for #RRGGBB hex colors."""
        return (
            isinstance(color, str)
            and re.fullmatch(r"#[0-9A-Fa-f]{6}", color) is not None
        )


class MaterialResourceValidator:
    """Validates material resource dictionaries."""

    def __init__(self):
        self.color_validator = ColorValidator()

    def validate_material(self, material: Dict[str, Any]) -> bool:
        """Return True when required material fields are present and valid."""
        if not isinstance(material, dict):
            return False
        if not all(material.get(field) for field in ("id", "name", "color")):
            return False
        return self.color_validator.validate_color_format(material["color"])


class TransparencyValidator:
    """Validates normalized transparency values."""

    def validate_transparency(self, value: Any) -> bool:
        """Return True for numeric transparency in [0.0, 1.0]."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return 0.0 <= float(value) <= 1.0


class LayerValidator:
    """Validates layer settings for 3MF metadata."""

    def validate_layer_height(self, height: Any) -> bool:
        """Return True for plausible 3D-printing layer heights in mm."""
        if isinstance(height, bool) or not isinstance(height, (int, float)):
            return False
        return 0.0 < float(height) < 1.0


class LayerSequenceValidator:
    """Validates continuity of layer metadata."""

    def validate_sequence(self, layers: List[Dict[str, Any]]) -> bool:
        """Return True when layer indices are contiguous from zero."""
        if not isinstance(layers, list) or not layers:
            return False
        layer_indices = [layer.get("layer") for layer in layers]
        if any(not isinstance(index, int) for index in layer_indices):
            return False
        return sorted(layer_indices) == list(range(len(layer_indices)))


class MaterialTransitionValidator:
    """Validates material transition records."""

    def validate_transition(self, transition: Dict[str, Any]) -> bool:
        """Return True when a transition has valid adjacent layer/material fields."""
        if not isinstance(transition, dict):
            return False
        required = ("from_layer", "to_layer", "from_material", "to_material")
        if not all(transition.get(field) is not None for field in required):
            return False
        return transition["to_layer"] > transition["from_layer"]


class CompressionOptimizer:
    """Provides simple compression helpers for 3MF XML payloads."""

    def compress_xml(self, xml_data: str) -> bytes:
        """Compress XML bytes with zlib."""
        return zlib.compress(xml_data.encode("utf-8"), level=9)


class FileSizeValidator:
    """Validates coarse 3MF file size budgets."""

    MAX_SIZE_BYTES = {
        "small": 10 * 1024 * 1024,
        "medium": 50 * 1024 * 1024,
        "large": 100 * 1024 * 1024,
    }

    def validate_file_size(self, file_size: int, model_size: str = "medium") -> bool:
        """Return True when file_size fits the configured size class."""
        limit = self.MAX_SIZE_BYTES.get(model_size, self.MAX_SIZE_BYTES["medium"])
        return isinstance(file_size, int) and 0 <= file_size <= limit


class MemoryEfficientProcessor:
    """Processes large geometry data without copying the full payload."""

    def process_large_dataset(self, dataset: Dict[str, Any]) -> Dict[str, int]:
        """Return lightweight counts for large geometry datasets."""
        return {
            "vertex_count": len(dataset.get("vertices", ())),
            "triangle_count": len(dataset.get("triangles", ())),
        }


class ErrorRecoveryManager:
    """Cleans corrupted geometry data before export."""

    def recover_geometry_data(self, geometry_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove invalid vertices and triangles with bad references."""
        vertices = geometry_data.get("vertices", [])
        valid_vertices = []
        index_map = {}

        for index, vertex in enumerate(vertices):
            if len(vertex) != 3:
                continue
            if all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in vertex
            ):
                index_map[index] = len(valid_vertices)
                valid_vertices.append(tuple(float(value) for value in vertex))

        valid_triangles = []
        for triangle in geometry_data.get("triangles", []):
            if len(triangle) != 3:
                continue
            if all(index in index_map for index in triangle):
                mapped = tuple(index_map[index] for index in triangle)
                if len(set(mapped)) == 3:
                    valid_triangles.append(mapped)

        return {"vertices": valid_vertices, "triangles": valid_triangles}


class PartialExportManager:
    """Builds partial export results from component status records."""

    def create_partial_export(
        self, export_status: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return a successful partial result for components that completed."""
        components = {
            name: status["data"]
            for name, status in export_status.items()
            if status.get("success") and "data" in status
        }
        warnings = [
            f"{name}: {status.get('error', 'failed')}"
            for name, status in export_status.items()
            if not status.get("success")
        ]
        return {
            "success": bool(components),
            "components": components,
            "warnings": warnings,
        }
