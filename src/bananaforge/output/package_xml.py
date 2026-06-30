"""3MF package XML helpers."""

import xml.etree.ElementTree as ET
from typing import Dict
from xml.dom import minidom


class ThreeMFNamespaceManager:
    """Manages XML namespaces for 3MF files."""

    NAMESPACES = {
        "3mf": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
        "material": "http://schemas.microsoft.com/3dmanufacturing/material/2015/02",
        "slice": "http://schemas.microsoft.com/3dmanufacturing/slice/2015/07",
        "opc_rel": "http://schemas.openxmlformats.org/package/2006/relationships",
        "opc_ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    }

    def __init__(self):
        """Initialize namespace manager."""
        for prefix, uri in self.NAMESPACES.items():
            ET.register_namespace(prefix, uri)

    def get_registered_namespaces(self) -> Dict[str, str]:
        """Get all registered namespaces."""
        return self.NAMESPACES.copy()

    def get_namespace_uri(self, prefix: str) -> str:
        """Get namespace URI for a given prefix."""
        return self.NAMESPACES.get(prefix, "")


class ContentTypesGenerator:
    """Generates [Content_Types].xml for 3MF package."""

    CONTENT_TYPES = {
        "rels": "application/vnd.openxmlformats-package.relationships+xml",
        "model": "application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
        "png": "image/png",
        "xml": "application/xml",
    }

    def generate(self) -> str:
        """Generate [Content_Types].xml content."""
        ns_manager = ThreeMFNamespaceManager()
        ct_ns = ns_manager.get_namespace_uri("opc_ct")

        root = ET.Element(f"{{{ct_ns}}}Types")

        for extension, content_type in self.CONTENT_TYPES.items():
            default_elem = ET.SubElement(root, f"{{{ct_ns}}}Default")
            default_elem.set("Extension", extension)
            default_elem.set("ContentType", content_type)

        override_elem = ET.SubElement(root, f"{{{ct_ns}}}Override")
        override_elem.set("PartName", "/3D/3dmodel.model")
        override_elem.set("ContentType", self.CONTENT_TYPES["model"])

        rough_string = ET.tostring(root, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ").split("\n", 1)[1]


class RelationshipsGenerator:
    """Generates _rels/.rels for 3MF package."""

    def generate(self) -> str:
        """Generate _rels/.rels content."""
        ns_manager = ThreeMFNamespaceManager()
        rel_ns = ns_manager.get_namespace_uri("opc_rel")

        root = ET.Element(f"{{{rel_ns}}}Relationships")

        rel_elem = ET.SubElement(root, f"{{{rel_ns}}}Relationship")
        rel_elem.set(
            "Type", "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
        )
        rel_elem.set("Target", "3D/3dmodel.model")
        rel_elem.set("Id", "rel-1")

        rough_string = ET.tostring(root, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ").split("\n", 1)[1]
