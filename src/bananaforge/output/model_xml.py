"""3MF model XML generation."""

import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from xml.dom import minidom

from .export_types import LayerMaterial
from .package_xml import ThreeMFNamespaceManager


class ModelXMLGenerator:
    """Generates 3D/3dmodel.model XML content."""

    def __init__(self):
        """Initialize model XML generator."""
        self.ns_manager = ThreeMFNamespaceManager()
        self.model_ns = self.ns_manager.get_namespace_uri("3mf")
        self.material_ns = self.ns_manager.get_namespace_uri("material")

    def generate(
        self,
        vertices: List[Tuple[float, float, float]],
        triangles: List[Tuple[int, int, int]],
        materials: Optional[Dict[str, Any]] = None,
        layer_materials: Optional[Dict[int, LayerMaterial]] = None,
    ) -> str:
        """Generate 3D model XML with proper XML declaration and Bambu Studio compatibility."""
        model = ET.Element("model")
        model.set("unit", "millimeter")
        model.set("xml:lang", "en-US")
        model.set("xmlns", "http://schemas.microsoft.com/3dmanufacturing/core/2015/02")
        model.set(
            "xmlns:m", "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
        )

        resources = ET.SubElement(model, "resources")

        if materials:
            self._add_materials_to_resources(resources, materials)

        obj = ET.SubElement(resources, "object")
        obj.set("id", "1")
        obj.set("type", "model")

        mesh = ET.SubElement(obj, "mesh")

        self._add_vertices_to_mesh(mesh, vertices)
        self._add_triangles_to_mesh(mesh, triangles, layer_materials)

        build = ET.SubElement(model, "build")
        item = ET.SubElement(build, "item")
        item.set("objectid", "1")

        rough_string = ET.tostring(model, encoding="unicode")
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'

        try:
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            lines = pretty_xml.split("\n")
            if lines[0].startswith("<?xml"):
                lines = lines[1:]
            return xml_declaration + "\n".join(lines)
        except Exception:
            return xml_declaration + rough_string

    def _add_materials_to_resources(
        self, resources: ET.Element, materials: Dict[str, Any]
    ) -> None:
        """Add material definitions to resources section."""
        if not materials:
            return

        basematerials = ET.SubElement(resources, "basematerials")

        for idx, (material_id, material_data) in enumerate(materials.items()):
            base = ET.SubElement(basematerials, "base")
            base.set("name", material_data.get("name", f"Material {idx}"))
            base.set("displaycolor", material_data.get("color", "#ffffff"))

    def _add_vertices_to_mesh(
        self, mesh: ET.Element, vertices: List[Tuple[float, float, float]]
    ) -> None:
        """Add vertices to mesh element."""
        vertices_elem = ET.SubElement(mesh, "vertices")

        for x, y, z in vertices:
            vertex = ET.SubElement(vertices_elem, "vertex")
            vertex.set("x", f"{x:.6f}")
            vertex.set("y", f"{y:.6f}")
            vertex.set("z", f"{z:.6f}")

    def _add_triangles_to_mesh(
        self,
        mesh: ET.Element,
        triangles: List[Tuple[int, int, int]],
        layer_materials: Optional[Dict[int, LayerMaterial]] = None,
    ) -> None:
        """Add triangles to mesh element with optional layer material assignments."""
        triangles_elem = ET.SubElement(mesh, "triangles")

        material_mapping = {}
        if layer_materials:
            unique_materials = list(
                set(lm.material_id for lm in layer_materials.values())
            )
            for idx, material_id in enumerate(unique_materials):
                material_mapping[material_id] = str(idx)

        for i, (v1, v2, v3) in enumerate(triangles):
            triangle = ET.SubElement(triangles_elem, "triangle")
            triangle.set("v1", str(v1))
            triangle.set("v2", str(v2))
            triangle.set("v3", str(v3))

            if layer_materials:
                layer_material = list(layer_materials.values())[
                    i % len(layer_materials)
                ]
                material_id = layer_material.material_id

                if material_id in material_mapping:
                    triangle.set("pid", material_mapping[material_id])
                else:
                    triangle.set("pid", str(i % len(layer_materials)))
