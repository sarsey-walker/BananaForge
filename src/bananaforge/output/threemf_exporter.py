#!/usr/bin/env python3
"""3MF (3D Manufacturing Format) export functionality for BananaForge.

This module implements comprehensive 3MF export capabilities including:
- Core 3MF file structure generation (ZIP container with XML)
- Per-layer material and color assignment 
- Material properties embedding
- Bambu Studio compatibility extensions
- Validation and quality assurance

Follows 3MF Core Specification v1.3:
https://github.com/3MFConsortium/spec_core/blob/master/3MF%20Core%20Specification.md
"""

import zipfile
import xml.etree.ElementTree as ET
from xml.dom import minidom
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import uuid
from dataclasses import dataclass
import struct

import torch
import numpy as np

from ..materials.database import MaterialDatabase
from ..utils.logging import get_logger

logger = get_logger(__name__)


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


class ThreeMFNamespaceManager:
    """Manages XML namespaces for 3MF files."""
    
    NAMESPACES = {
        "3mf": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
        "material": "http://schemas.microsoft.com/3dmanufacturing/material/2015/02",
        "slice": "http://schemas.microsoft.com/3dmanufacturing/slice/2015/07",
        "opc_rel": "http://schemas.openxmlformats.org/package/2006/relationships",
        "opc_ct": "http://schemas.openxmlformats.org/package/2006/content-types"
    }
    
    def __init__(self):
        """Initialize namespace manager."""
        # Register namespaces for XML parsing
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
        "xml": "application/xml"
    }
    
    def generate(self) -> str:
        """Generate [Content_Types].xml content."""
        ns_manager = ThreeMFNamespaceManager()
        ct_ns = ns_manager.get_namespace_uri("opc_ct")
        
        # Create root element
        root = ET.Element(f"{{{ct_ns}}}Types")
        
        # Add default content types
        for extension, content_type in self.CONTENT_TYPES.items():
            default_elem = ET.SubElement(root, f"{{{ct_ns}}}Default")
            default_elem.set("Extension", extension)
            default_elem.set("ContentType", content_type)
        
        # Override for specific files
        override_elem = ET.SubElement(root, f"{{{ct_ns}}}Override")
        override_elem.set("PartName", "/3D/3dmodel.model")
        override_elem.set("ContentType", self.CONTENT_TYPES["model"])
        
        # Convert to string with proper formatting
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ").split('\n', 1)[1]  # Remove first line


class RelationshipsGenerator:
    """Generates _rels/.rels for 3MF package."""
    
    def generate(self) -> str:
        """Generate _rels/.rels content."""
        ns_manager = ThreeMFNamespaceManager()
        rel_ns = ns_manager.get_namespace_uri("opc_rel")
        
        # Create root element
        root = ET.Element(f"{{{rel_ns}}}Relationships")
        
        # Add relationship to 3D model
        rel_elem = ET.SubElement(root, f"{{{rel_ns}}}Relationship")
        rel_elem.set("Type", "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel")
        rel_elem.set("Target", "3D/3dmodel.model")
        rel_elem.set("Id", "rel-1")
        
        # Convert to string with proper formatting
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ").split('\n', 1)[1]


class ModelXMLGenerator:
    """Generates 3D/3dmodel.model XML content."""
    
    def __init__(self):
        """Initialize model XML generator."""
        self.ns_manager = ThreeMFNamespaceManager()
        self.model_ns = self.ns_manager.get_namespace_uri("3mf")
        self.material_ns = self.ns_manager.get_namespace_uri("material")
    
    def generate(self, vertices: List[Tuple[float, float, float]], 
                 triangles: List[Tuple[int, int, int]],
                 materials: Optional[Dict[str, Any]] = None,
                 layer_materials: Optional[Dict[int, LayerMaterial]] = None) -> str:
        """Generate 3D model XML with proper XML declaration and Bambu Studio compatibility."""
        
        # Create root model element with proper namespaces
        model = ET.Element("model")
        model.set("unit", "millimeter")
        model.set("xml:lang", "en-US")
        model.set("xmlns", "http://schemas.microsoft.com/3dmanufacturing/core/2015/02")
        model.set("xmlns:m", "http://schemas.microsoft.com/3dmanufacturing/material/2015/02")
        
        # Add resources section
        resources = ET.SubElement(model, "resources")
        
        # Add materials if provided
        if materials:
            self._add_materials_to_resources(resources, materials)
        
        # Add object definition
        obj = ET.SubElement(resources, "object")
        obj.set("id", "1")
        obj.set("type", "model")
        
        # Add mesh
        mesh = ET.SubElement(obj, "mesh")
        
        # Add vertices
        self._add_vertices_to_mesh(mesh, vertices)
        
        # Add triangles
        self._add_triangles_to_mesh(mesh, triangles, layer_materials)
        
        # Add build section
        build = ET.SubElement(model, "build")
        item = ET.SubElement(build, "item")
        item.set("objectid", "1")
        
        # Convert to string with proper XML declaration
        rough_string = ET.tostring(model, encoding='unicode')
        
        # Add XML declaration and format properly
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        
        # Parse and pretty print for Bambu Studio compatibility
        try:
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            # Remove the auto-generated XML declaration from minidom and add ours
            lines = pretty_xml.split('\n')
            if lines[0].startswith('<?xml'):
                lines = lines[1:]
            formatted_xml = xml_declaration + '\n'.join(lines)
            return formatted_xml
        except Exception:
            # Fallback to basic formatting
            return xml_declaration + rough_string
    
    def _add_materials_to_resources(self, resources: ET.Element, materials: Dict[str, Any]) -> None:
        """Add material definitions to resources section."""
        if not materials:
            return
            
        basematerials = ET.SubElement(resources, "basematerials")
        
        # Create materials with consistent indexing
        for idx, (material_id, material_data) in enumerate(materials.items()):
            base = ET.SubElement(basematerials, "base")
            base.set("name", material_data.get('name', f'Material {idx}'))
            base.set("displaycolor", material_data.get('color', '#ffffff'))
    
    def _add_vertices_to_mesh(self, mesh: ET.Element, vertices: List[Tuple[float, float, float]]) -> None:
        """Add vertices to mesh element."""
        vertices_elem = ET.SubElement(mesh, "vertices")
        
        for x, y, z in vertices:
            vertex = ET.SubElement(vertices_elem, "vertex")
            vertex.set("x", f"{x:.6f}")
            vertex.set("y", f"{y:.6f}")
            vertex.set("z", f"{z:.6f}")
    
    def _add_triangles_to_mesh(self, mesh: ET.Element, triangles: List[Tuple[int, int, int]],
                              layer_materials: Optional[Dict[int, LayerMaterial]] = None) -> None:
        """Add triangles to mesh element with optional layer material assignments."""
        triangles_elem = ET.SubElement(mesh, "triangles")
        
        # Create material mapping if available
        material_mapping = {}
        if layer_materials:
            # Create a simple mapping from material_id to index
            unique_materials = list(set(lm.material_id for lm in layer_materials.values()))
            for idx, material_id in enumerate(unique_materials):
                material_mapping[material_id] = str(idx)
        
        for i, (v1, v2, v3) in enumerate(triangles):
            triangle = ET.SubElement(triangles_elem, "triangle")
            triangle.set("v1", str(v1))
            triangle.set("v2", str(v2))
            triangle.set("v3", str(v3))
            
            # Add material assignment - assign materials in a round-robin fashion to all triangles
            if layer_materials:
                # Use modulo to cycle through available materials for all triangles
                layer_material = list(layer_materials.values())[i % len(layer_materials)]
                material_id = layer_material.material_id
                
                # Use consistent material index instead of hash
                if material_id in material_mapping:
                    triangle.set("pid", material_mapping[material_id])
                else:
                    # Fallback: use modulo of material index
                    triangle.set("pid", str(i % len(layer_materials)))


class BambuProductionExporter:
    """3MF Production Extension exporter optimized for Bambu Studio compatibility."""
    
    def __init__(self, material_db: Optional[MaterialDatabase] = None):
        """Initialize Bambu Production Extension exporter."""
        self.material_db = material_db
        # Use fixed UUIDs matching the working template
        self.main_object_uuid = "00000001-61cb-4c03-9d28-80fed5dfa1dc"
        self.component_object_uuid = "00010000-b206-40ff-9872-83e8017abed1"
        self.build_uuid = "2c7c17d8-22b5-4d84-8835-1976022ea369"
        self.item_uuid = "00000002-b1ec-4553-aec9-835e5b724bb4"
    
    def create_3mf_container(self, geometry_data: Dict[str, Any], 
                           material_data: Dict[str, Any],
                           config: ThreeMFExportConfig,
                           optimization_results: Dict[str, Any]) -> bytes:
        """Create 3MF ZIP container with Production Extension format."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Add [Content_Types].xml
            zip_file.writestr("[Content_Types].xml", self._generate_content_types())
            
            # 2. Add _rels/.rels  
            zip_file.writestr("_rels/.rels", self._generate_main_relationships(config))
            
            # 3. Add main 3D/3dmodel.model (just references)
            zip_file.writestr("3D/3dmodel.model", self._generate_main_model(config))
            
            # 4. Add 3D/_rels/3dmodel.model.rels
            zip_file.writestr("3D/_rels/3dmodel.model.rels", self._generate_model_relationships())
            
            # 5. Add 3D/Objects/object_1.model (actual geometry)
            zip_file.writestr("3D/Objects/object_1.model", 
                            self._generate_object_model(geometry_data, material_data, config))
            
            # 6. Add Bambu Studio metadata files
            if config.bambu_compatible:
                self._add_bambu_metadata_files(zip_file, geometry_data, material_data, optimization_results)
            elif config.include_metadata:
                zip_file.writestr("Metadata/model_info.xml", 
                                self._generate_minimal_metadata(geometry_data, material_data))
        
        return zip_buffer.getvalue()
    
    def _generate_content_types(self) -> str:
        """Generate [Content_Types].xml for Production Extension."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Default Extension="gcode" ContentType="text/x.gcode"/>
</Types>'''
    
    def _generate_main_relationships(self, config: ThreeMFExportConfig) -> str:
        """Generate _rels/.rels with Bambu Studio compatibility."""
        if config.bambu_compatible:
            return '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
 <Relationship Target="/Metadata/plate_1.png" Id="rel-2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>
 <Relationship Target="/Metadata/plate_1.png" Id="rel-4" Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-middle"/>
<Relationship Target="/Metadata/plate_1_small.png" Id="rel-5" Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-small"/>
</Relationships>'''
        else:
            return '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''
    
    def _generate_main_model(self, config: ThreeMFExportConfig) -> str:
        """Generate main 3D/3dmodel.model with Production Extension format."""
        # Create XML with proper namespaces for Production Extension
        model_elem = ET.Element("model")
        model_elem.set("unit", "millimeter")
        model_elem.set("xml:lang", "en-US")
        model_elem.set("xmlns", "http://schemas.microsoft.com/3dmanufacturing/core/2015/02")
        
        if config.bambu_compatible:
            # Add Bambu Studio specific namespaces and metadata for full compatibility
            model_elem.set("xmlns:BambuStudio", "http://schemas.bambulab.com/package/2021")
            model_elem.set("xmlns:p", "http://schemas.microsoft.com/3dmanufacturing/production/2015/06")
            model_elem.set("requiredextensions", "p")
            
            # Add comprehensive Bambu Studio metadata matching reference file
            import datetime
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            
            metadata_entries = [
                ("Application", "BambuStudio-02.01.00.59"),
                ("BambuStudio:3mfVersion", "1"),
                ("Copyright", ""),
                ("CreationDate", current_date),
                ("Description", ""),
                ("Designer", ""),
                ("DesignerCover", ""),
                ("DesignerUserId", "1731431467"),
                ("License", ""),
                ("ModificationDate", current_date),
                ("Origin", ""),
                ("Title", "")
            ]
            
            for name, value in metadata_entries:
                metadata_elem = ET.SubElement(model_elem, "metadata")
                metadata_elem.set("name", name)
                metadata_elem.text = value
        else:
            # Standard 3MF metadata
            model_elem.set("xmlns:p", "http://schemas.microsoft.com/3dmanufacturing/production/2015/06")
            model_elem.set("requiredextensions", "p")
            
            # Add metadata for BananaForge
            metadata_elem = ET.SubElement(model_elem, "metadata")
            metadata_elem.set("name", "Application")
            metadata_elem.text = "BananaForge-1.0"
        
        # Resources section with component reference
        resources = ET.SubElement(model_elem, "resources")
        
        # Main object that references the component
        obj = ET.SubElement(resources, "object")
        obj.set("id", "2")
        obj.set("p:UUID", self.main_object_uuid)
        obj.set("type", "model")
        
        # Components section pointing to separate object file
        components = ET.SubElement(obj, "components")
        component = ET.SubElement(components, "component")
        component.set("p:path", "/3D/Objects/object_1.model")
        component.set("objectid", "1")
        component.set("p:UUID", self.component_object_uuid)
        component.set("transform", "1 0 0 0 1 0 0 0 1 0 0 0")  # Identity transform
        
        # Build section
        build = ET.SubElement(model_elem, "build")
        build.set("p:UUID", self.build_uuid)
        
        item = ET.SubElement(build, "item")
        item.set("objectid", "2")
        item.set("p:UUID", self.item_uuid)
        # Use identity transform since positioning is handled in coordinate transformation
        item.set("transform", "1 0 0 0 1 0 0 0 1 0 0 0")  # Identity transform  
        item.set("printable", "1")
        
        # Convert to string with proper formatting
        rough_string = ET.tostring(model_elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent=" ")
        
        # Remove auto-generated XML declaration and add proper one
        lines = pretty_xml.split('\n')
        if lines[0].startswith('<?xml'):
            lines = lines[1:]
        
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        return xml_declaration + '\n'.join(lines)
    
    def _generate_model_relationships(self) -> str:
        """Generate 3D/_rels/3dmodel.model.rels."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/Objects/object_1.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''
    
    def _generate_object_model(self, geometry_data: Dict[str, Any], 
                             material_data: Dict[str, Any], 
                             config: ThreeMFExportConfig) -> str:
        """Generate 3D/Objects/object_1.model using template with our geometry."""
        if config.bambu_compatible:
            return self._generate_object_model_from_template(geometry_data, material_data)
        
        # Create object model with Production Extension
        model_elem = ET.Element("model")
        model_elem.set("unit", "millimeter")
        model_elem.set("xml:lang", "en-US")
        model_elem.set("xmlns", "http://schemas.microsoft.com/3dmanufacturing/core/2015/02")
        model_elem.set("xmlns:m", "http://schemas.microsoft.com/3dmanufacturing/material/2015/02")
        
        if config.bambu_compatible:
            # Add Bambu Studio namespaces for full compatibility including slice extension
            model_elem.set("xmlns:BambuStudio", "http://schemas.bambulab.com/package/2021")
            model_elem.set("xmlns:p", "http://schemas.microsoft.com/3dmanufacturing/production/2015/06")
            model_elem.set("xmlns:s", "http://schemas.microsoft.com/3dmanufacturing/slice/2015/07")
            model_elem.set("xmlns:m", "http://schemas.microsoft.com/3dmanufacturing/material/2015/02")
            model_elem.set("requiredextensions", "p s m")
            
            # Add BananaForge metadata
            metadata_elem = ET.SubElement(model_elem, "metadata")
            metadata_elem.set("name", "BambuStudio:3mfVersion")
            metadata_elem.text = "1"
        else:
            model_elem.set("xmlns:p", "http://schemas.microsoft.com/3dmanufacturing/production/2015/06")
            model_elem.set("requiredextensions", "p")
        
        # Resources section
        resources = ET.SubElement(model_elem, "resources")
        
        # Add materials for both generic and Bambu compatibility
        if material_data.get('layer_materials'):
            self._add_materials_to_resources(resources, material_data)
            
        # Add slice stack for layer-based material assignments (Bambu Studio support)
        if config.bambu_compatible and material_data.get('layer_materials'):
            self._add_slice_stack_to_resources(resources, material_data)
        
        # Object with actual mesh
        obj = ET.SubElement(resources, "object")
        obj.set("id", "1")
        obj.set("p:UUID", self.component_object_uuid)
        obj.set("type", "model")
        
        # Reference slice stack for layer-based materials
        if config.bambu_compatible and material_data.get('layer_materials'):
            obj.set("s:slicestackid", "slicestack1")
        
        # Mesh element
        mesh = ET.SubElement(obj, "mesh")
        
        # Add vertices
        vertices_elem = ET.SubElement(mesh, "vertices")
        for x, y, z in geometry_data['vertices']:
            vertex = ET.SubElement(vertices_elem, "vertex")
            vertex.set("x", f"{x}")
            vertex.set("y", f"{y}")
            vertex.set("z", f"{z}")
        
        # Add triangles with or without material assignments based on bambu_compatible
        triangles_elem = ET.SubElement(mesh, "triangles")
        
        if config.bambu_compatible:
            # Add triangles WITHOUT material assignments for Bambu Studio (geometry only)
            for v1, v2, v3 in geometry_data['triangles']:
                triangle = ET.SubElement(triangles_elem, "triangle")
                triangle.set("v1", str(v1))
                triangle.set("v2", str(v2))
                triangle.set("v3", str(v3))
                # NO pid attribute - Bambu Studio handles materials at layer level
        else:
            # Add triangles WITH material assignments for generic 3MF
            layer_materials = material_data.get('layer_materials', {})
            if layer_materials:
                triangle_count = len(geometry_data['triangles'])
                material_count = len(layer_materials)
                
                for i, (v1, v2, v3) in enumerate(geometry_data['triangles']):
                    triangle = ET.SubElement(triangles_elem, "triangle")
                    triangle.set("v1", str(v1))
                    triangle.set("v2", str(v2))
                    triangle.set("v3", str(v3))
                    
                    # Assign material using round-robin distribution
                    if material_count > 0:
                        material_idx = i % material_count
                        material_id = list(layer_materials.keys())[material_idx]
                        triangle.set("pid", str(material_id + 1))  # Materials are 1-indexed
            else:
                # No materials - just geometry
                for v1, v2, v3 in geometry_data['triangles']:
                    triangle = ET.SubElement(triangles_elem, "triangle")
                    triangle.set("v1", str(v1))
                    triangle.set("v2", str(v2))
                    triangle.set("v3", str(v3))
        
        # Convert to string with proper formatting
        rough_string = ET.tostring(model_elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent=" ")
        
        # Remove auto-generated XML declaration and add proper one
        lines = pretty_xml.split('\n')
        if lines[0].startswith('<?xml'):
            lines = lines[1:]
            
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        return xml_declaration + '\n'.join(lines)
    
    def _add_slice_stack_to_resources(self, resources: ET.Element, material_data: Dict[str, Any]) -> None:
        """Add slice stack for layer-based material assignments (3MF Slice Extension)."""
        layer_materials = material_data.get('layer_materials', {})
        
        if not layer_materials:
            return
        
        # Create slice stack
        slice_stack = ET.SubElement(resources, "s:slicestack")
        slice_stack.set("id", "slicestack1")
        
        # Sort layers by index to ensure proper order
        sorted_layers = sorted(layer_materials.items(), key=lambda x: x[0])
        
        for layer_idx, layer_material in sorted_layers:
            if isinstance(layer_material, LayerMaterial):
                material_id = layer_material.material_id
                layer_height = layer_material.layer_height
            else:
                material_id = layer_material.get('material_id', f'material_{layer_idx}')
                layer_height = layer_material.get('layer_height', 0.08)
            
            # Calculate Z height for this layer
            z_height = (layer_idx + 1) * layer_height
            
            # Create slice for this layer
            slice_elem = ET.SubElement(slice_stack, "s:slice")
            slice_elem.set("ztop", f"{z_height:.3f}")
            
            # Add material reference
            # Map material_id to material index (1-indexed)
            material_index = self._get_material_index(material_id, sorted_layers)
            if material_index > 0:
                slice_elem.set("material", str(material_index))
            
            # Add geometry reference to the main object
            slice_elem.set("objectid", "1")
    
    def _get_material_index(self, material_id: str, sorted_layers: List[Tuple[int, Any]]) -> int:
        """Get 1-indexed material index for a material ID."""
        unique_materials = []
        for _, layer_material in sorted_layers:
            if isinstance(layer_material, LayerMaterial):
                mat_id = layer_material.material_id
            else:
                mat_id = layer_material.get('material_id', 'unknown')
            
            if mat_id not in unique_materials:
                unique_materials.append(mat_id)
        
        try:
            return unique_materials.index(material_id) + 1  # 1-indexed
        except ValueError:
            return 1  # Default to first material
    
    def _add_materials_to_resources(self, resources: ET.Element, material_data: Dict[str, Any]) -> None:
        """Add material definitions to resources section for Bambu Studio compatibility."""
        layer_materials = material_data.get('layer_materials', {})
        
        if not layer_materials:
            return
        
        # Create material group for Bambu Studio
        basematerials = ET.SubElement(resources, "m:basematerials")
        
        # Add each unique material
        for idx, (layer_idx, layer_material) in enumerate(layer_materials.items()):
            if isinstance(layer_material, LayerMaterial):
                material_id = layer_material.material_id
                transparency = layer_material.transparency
            else:
                material_id = layer_material.get('material_id', f'material_{layer_idx}')
                transparency = layer_material.get('transparency', 1.0)
            
            # Create base material entry
            base_elem = ET.SubElement(basematerials, "m:base")
            base_elem.set("m:id", str(idx + 1))  # 1-indexed
            base_elem.set("m:name", material_id)
            
            # Add color information (use default colors if not available)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
            color = colors[idx % len(colors)]
            base_elem.set("m:displaycolor", color)
            
            # Add transparency if not fully opaque
            if transparency < 1.0:
                base_elem.set("m:opacity", f"{transparency:.3f}")
            
            logger.info(f"Added material {idx + 1}: {material_id} (color: {color}, transparency: {transparency})")
    
    def _generate_minimal_metadata(self, geometry_data: Dict[str, Any], 
                                 material_data: Dict[str, Any]) -> str:
        """Generate minimal metadata for BananaForge."""
        metadata = {
            'generator': 'BananaForge',
            'version': '1.0',
            'vertices_count': len(geometry_data['vertices']),
            'triangles_count': len(geometry_data['triangles']),
            'layer_materials': len(material_data.get('layer_materials', {}))
        }
        
        root = ET.Element("metadata")
        for key, value in metadata.items():
            elem = ET.SubElement(root, key)
            elem.text = str(value)
        
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ").split('\n', 1)[1]
    
    def _add_bambu_metadata_files(self, zip_file: zipfile.ZipFile, 
                                 geometry_data: Dict[str, Any], 
                                 material_data: Dict[str, Any],
                                 optimization_results: Dict[str, Any]) -> None:
        """Add all required Bambu Studio metadata files."""
        
        # 1. Add project_settings.config with filament information
        zip_file.writestr("Metadata/project_settings.config", 
                         self._generate_project_settings(material_data))
        
        # 2. Add model_settings.config
        zip_file.writestr("Metadata/model_settings.config", 
                         self._generate_model_settings(geometry_data))
        
        # 3. Add slice_info.config  
        zip_file.writestr("Metadata/slice_info.config", 
                         self._load_template_file("slice_info.config", self._generate_slice_info()))
        
        # 4. Add plate_1.json with filament colors
        zip_file.writestr("Metadata/plate_1.json", 
                         self._generate_plate_json(material_data))
        
        # 5. Add dummy metadata files that Bambu expects
        zip_file.writestr("Metadata/cut_information.xml", 
                         self._load_template_file("cut_information.xml", self._generate_cut_information()))
        zip_file.writestr("Metadata/custom_gcode_per_layer.xml", 
                         self._generate_custom_gcode(material_data))
        
        # 6. Add proper thumbnails for Bambu Studio
        self._add_proper_thumbnails(zip_file, optimization_results)
    
    def _generate_project_settings(self, material_data: Dict[str, Any]) -> str:
        """Generate project_settings.config using the working Bambu template."""
        layer_materials = material_data.get('layer_materials', {})
        
        # Extract unique materials and get their actual colors from material database
        unique_materials = {}
        filament_colors = []
        filament_types = []
        filament_settings = []
        
        # Try to get colors from material database if available
        material_colors = {}
        if self.material_db:
            for material in self.material_db.get_all_materials():
                hex_color = material.color_hex
                # Ensure proper hex format
                if not hex_color.startswith('#'):
                    hex_color = '#FFFFFF'
                material_colors[material.name] = hex_color
        
        # Default colors as fallback
        default_colors = ["#FFFFFF", "#000000", "#FF6B6B", "#4ECDC4"]
        
        for i, (layer_idx, layer_material) in enumerate(layer_materials.items()):
            if isinstance(layer_material, LayerMaterial):
                material_id = layer_material.material_id
            else:
                material_id = layer_material.get('material_id', f'material_{i}')
            
            if material_id not in unique_materials:
                # Try to get color from material database first, then fallback to specific material mapping
                color = material_colors.get(material_id)
                if not color:
                    color = self._get_material_color(material_id)
                # Only use default colors if we don't have a specific mapping
                if not color:
                    color = default_colors[len(unique_materials) % len(default_colors)]
                
                unique_materials[material_id] = {
                    'color': color,
                    'index': len(unique_materials)
                }
                filament_colors.append(color)
                filament_types.append("PLA")
                filament_settings.append("Bambu PLA Basic @BBL A1M")
        
        # Ensure we have at least 4 filament slots (Bambu Studio expectation)
        while len(filament_colors) < 4:
            filament_colors.append("#FFFFFF")
            filament_types.append("PLA") 
            filament_settings.append("Bambu PLA Basic @BBL A1M")
        
        # Log the filament configuration
        logger.info(f"Setting project filament colors: {filament_colors[:4]}")
        logger.info(f"Unique materials mapping: {unique_materials}")
        
        # Load the working project settings template and modify it
        return self._load_bambu_template_with_materials(filament_colors[:4], filament_types[:4], filament_settings[:4])
    
    def _load_bambu_template_with_materials(self, filament_colors: List[str], 
                                          filament_types: List[str], 
                                          filament_settings: List[str]) -> str:
        """Load the working Bambu template and modify with our materials."""
        from pathlib import Path
        
        # Try multiple possible paths for the template
        possible_paths = [
            Path(__file__).parent / "bambu_template" / "Metadata" / "project_settings.config",
            Path(__file__).parent.parent.parent / "bambu_template" / "Metadata" / "project_settings.config",
            Path.cwd() / ".." / "bambu_template" / "Metadata" / "project_settings.config",
            Path("../bambu_template/Metadata/project_settings.config"),
            Path("bambu_template/Metadata/project_settings.config")
        ]
        
        template_path = None
        for path in possible_paths:
            if path.exists():
                template_path = path
                break
        
        if template_path is not None:
            try:
                with open(template_path, 'r') as f:
                    template_content = f.read()
                
                # Parse the template JSON
                template_config = json.loads(template_content)
                
                # Update with our specific filament settings
                template_config.update({
                    "filament_colour": filament_colors,
                    "filament_type": filament_types,
                    "filament_settings_id": filament_settings,
                    "version": "02.01.00.59"
                })
                
                # Return with 4-space indentation to match the template exactly
                return json.dumps(template_config, indent=4)
                
            except Exception as e:
                logger.warning(f"Could not load Bambu template: {e}")
        
        # Fallback to comprehensive settings if template not available
        project_config = self._get_comprehensive_bambu_settings()
        project_config.update({
            "filament_colour": filament_colors,
            "filament_type": filament_types, 
            "filament_settings_id": filament_settings,
            "version": "02.01.00.59"
        })
        
        return json.dumps(project_config, indent=4)
    
    def _load_template_file(self, filename: str, fallback_content: str) -> str:
        """Load a template file from bambu_template or use fallback."""
        from pathlib import Path
        
        # Try multiple possible paths for the template
        possible_paths = [
            Path(__file__).parent / "bambu_template" / "Metadata" / filename,
            Path(__file__).parent.parent.parent / "bambu_template" / "Metadata" / filename,
            Path.cwd() / ".." / "bambu_template" / "Metadata" / filename,
            Path(f"../bambu_template/Metadata/{filename}"),
            Path(f"bambu_template/Metadata/{filename}")
        ]
        
        for template_path in possible_paths:
            if template_path.exists():
                try:
                    with open(template_path, 'r') as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Could not load template {filename}: {e}")
                    break
        
        return fallback_content
    
    def _generate_object_model_from_template(self, geometry_data: Dict[str, Any], 
                                           material_data: Dict[str, Any]) -> str:
        """Generate object model by replacing mesh data in working template."""
        from pathlib import Path
        import xml.etree.ElementTree as ET
        
        # Try multiple possible paths for the template
        possible_paths = [
            Path(__file__).parent / "bambu_template" / "3D" / "Objects" / "object_1.model",
            Path(__file__).parent.parent.parent / "bambu_template" / "3D" / "Objects" / "object_1.model",
            Path.cwd() / ".." / "bambu_template" / "3D" / "Objects" / "object_1.model",
            Path("../bambu_template/3D/Objects/object_1.model"),
            Path("bambu_template/3D/Objects/object_1.model")
        ]
        
        template_path = None
        for path in possible_paths:
            if path.exists():
                template_path = path
                break
        
        if template_path is None:
            logger.warning("Bambu template object model not found, using fallback")
            return self._generate_object_model_fallback(geometry_data, material_data)
        
        try:
            # Load the working template
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Use string replacement to preserve exact template structure
            # Find the vertices and triangles sections and replace them
            
            # Generate vertices XML
            vertices = geometry_data.get('vertices', [])
            vertices_xml = []
            for x, y, z in vertices:
                vertices_xml.append(f'     <vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>')
            vertices_content = '\n'.join(vertices_xml)
            
            # Generate triangles XML  
            triangles = geometry_data.get('triangles', [])
            triangles_xml = []
            for v1, v2, v3 in triangles:
                triangles_xml.append(f'     <triangle v1="{v1}" v2="{v2}" v3="{v3}"/>')
            triangles_content = '\n'.join(triangles_xml)
            
            # Replace the mesh content using regex to preserve structure
            import re
            
            # Replace vertices section
            vertices_pattern = r'(<vertices>)(.*?)(</vertices>)'
            vertices_replacement = f'\\1\n{vertices_content}\n    \\3'
            template_content = re.sub(vertices_pattern, vertices_replacement, template_content, flags=re.DOTALL)
            
            # Replace triangles section
            triangles_pattern = r'(<triangles>)(.*?)(</triangles>)'
            triangles_replacement = f'\\1\n{triangles_content}\n    \\3'
            template_content = re.sub(triangles_pattern, triangles_replacement, template_content, flags=re.DOTALL)
            
            return template_content
            
        except Exception as e:
            logger.error(f"Failed to generate object model from template: {e}")
            return self._generate_object_model_fallback(geometry_data, material_data)
    
    def _generate_object_model_fallback(self, geometry_data: Dict[str, Any], 
                                      material_data: Dict[str, Any]) -> str:
        """Fallback object model generation."""
        # Create object model exactly like the working template
        model_elem = ET.Element("model")
        model_elem.set("unit", "millimeter")
        model_elem.set("xml:lang", "en-US")
        model_elem.set("xmlns", "http://schemas.microsoft.com/3dmanufacturing/core/2015/02")
        model_elem.set("xmlns:BambuStudio", "http://schemas.bambulab.com/package/2021")
        model_elem.set("xmlns:p", "http://schemas.microsoft.com/3dmanufacturing/production/2015/06")
        model_elem.set("requiredextensions", "p")
        
        # Add metadata exactly like template
        metadata_elem = ET.SubElement(model_elem, "metadata")
        metadata_elem.set("name", "BambuStudio:3mfVersion")
        metadata_elem.text = "1"
        
        # Resources with fixed UUID like template
        resources = ET.SubElement(model_elem, "resources")
        obj = ET.SubElement(resources, "object")
        obj.set("id", "1")
        obj.set("p:UUID", "00010000-81cb-4c03-9d28-80fed5dfa1dc")  # Fixed UUID like template
        obj.set("type", "model")
        
        # Add mesh
        mesh = ET.SubElement(obj, "mesh")
        
        # Add vertices
        vertices_elem = ET.SubElement(mesh, "vertices")
        vertices = geometry_data.get('vertices', [])
        for x, y, z in vertices:
            vertex = ET.SubElement(vertices_elem, "vertex")
            vertex.set("x", f"{x:.6f}")
            vertex.set("y", f"{y:.6f}")
            vertex.set("z", f"{z:.6f}")
        
        # Add triangles (NO pid attributes for Bambu compatibility)
        triangles_elem = ET.SubElement(mesh, "triangles")
        triangles = geometry_data.get('triangles', [])
        for v1, v2, v3 in triangles:
            triangle = ET.SubElement(triangles_elem, "triangle")
            triangle.set("v1", str(v1))
            triangle.set("v2", str(v2))
            triangle.set("v3", str(v3))
        
        # Convert to string
        rough_string = ET.tostring(model_elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent=" ")
        
        lines = pretty_xml.split('\n')
        if lines[0].startswith('<?xml'):
            lines = lines[1:]
        
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        return xml_declaration + '\n'.join(lines)
    
    def _get_comprehensive_bambu_settings(self) -> Dict[str, Any]:
        """Generate comprehensive Bambu Studio settings matching reference file structure."""
        return {
            "accel_to_decel_enable": "0",
            "accel_to_decel_factor": "50%",
            "activate_air_filtration": ["0", "0", "0", "0"],
            "additional_cooling_fan_speed": ["70", "70", "70", "70"],
            "apply_scarf_seam_on_circles": "1", 
            "auxiliary_fan": "0",
            "bed_custom_model": "",
            "bed_custom_texture": "",
            "bed_exclude_area": [],
            "bed_temperature_formula": "by_first_filament",
            "before_layer_change_gcode": "",
            "best_object_pos": "0.7,0.5",
            "bottom_color_penetration_layers": "7",
            "bottom_shell_layers": "1",
            "bottom_shell_thickness": "0",
            "bottom_surface_pattern": "monotonic",
            "bridge_angle": "0",
            "bridge_flow": "1",
            "bridge_no_support": "0",
            "bridge_speed": ["50"],
            "brim_object_gap": "0.1",
            "brim_type": "no_brim",
            "brim_width": "5",
            "chamber_temperatures": ["0", "0", "0", "0"],
            "change_filament_gcode": ";===== A1mini 20250206 =====\\nG392 S0\\nM1007 S0\\nM620 S[next_extruder]A\\nM204 S9000\\nG1 Z{max_layer_z + 3.0} F1200\\n\\nM400\\nM106 P1 S0\\nM106 P2 S0\\n{if old_filament_temp > 142 && next_extruder < 255}\\nM104 S[old_filament_temp]\\n{endif}\\n\\nG1 X180 F18000\\n\\n{if long_retractions_when_cut[previous_extruder]}\\nM620.11 S1 I[previous_extruder] E-{retraction_distances_when_cut[previous_extruder]} F1200\\n{else}\\nM620.11 S0\\n{endif}\\nM400\\n\\nM620.1 E F[old_filament_e_feedrate] T{nozzle_temperature_range_high[previous_extruder]}\\nM620.10 A0 F[old_filament_e_feedrate]\\nT[next_extruder]\\nM620.1 E F[new_filament_e_feedrate] T{nozzle_temperature_range_high[next_extruder]}\\nM620.10 A1 F[new_filament_e_feedrate] L[flush_length] H[nozzle_diameter] T[nozzle_temperature_range_high]\\n\\nG1 Y90 F9000\\n\\n{if next_extruder < 255}\\n\\n{if long_retractions_when_cut[previous_extruder]}\\nM620.11 S1 I[previous_extruder] E{retraction_distances_when_cut[previous_extruder]} F{old_filament_e_feedrate}\\nM628 S1\\nG92 E0\\nG1 E{retraction_distances_when_cut[previous_extruder]} F[old_filament_e_feedrate]\\nM400\\nM629 S1\\n{else}\\nM620.11 S0\\n{endif}\\n\\nM400\\nG92 E0\\nM628 S0\\n\\n{if flush_length_1 > 1}\\n; FLUSH_START\\n; always use highest temperature to flush\\nM400\\nM1002 set_filament_type:UNKNOWN\\nM109 S[nozzle_temperature_range_high]\\nM106 P1 S60\\n{if flush_length_1 > 23.7}\\nG1 E23.7 F{old_filament_e_feedrate} ; do not need pulsatile flushing for start part\\nG1 E{(flush_length_1 - 23.7) * 0.02} F50\\nG1 E{(flush_length_1 - 23.7) * 0.23} F{old_filament_e_feedrate}\\nG1 E{(flush_length_1 - 23.7) * 0.02} F50\\nG1 E{(flush_length_1 - 23.7) * 0.23} F{new_filament_e_feedrate}\\nG1 E{(flush_length_1 - 23.7) * 0.02} F50\\nG1 E{(flush_length_1 - 23.7) * 0.23} F{new_filament_e_feedrate}\\nG1 E{(flush_length_1 - 23.7) * 0.02} F50\\nG1 E{(flush_length_1 - 23.7) * 0.23} F{new_filament_e_feedrate}\\n{else}\\nG1 E{flush_length_1} F{old_filament_e_feedrate}\\n{endif}\\n; FLUSH_END\\nG1 E-[old_retract_length_toolchange] F1800\\nG1 E[old_retract_length_toolchange] F300\\nM400\\nM1002 set_filament_type:{filament_type[next_extruder]}\\n{endif}\\n\\n{if flush_length_1 > 45 && flush_length_2 > 1}\\n; WIPE\\nM400\\nM106 P1 S178\\nM400 S3\\nG1 X-3.5 F18000\\nG1 X-13.5 F3000\\nG1 X-3.5 F18000\\nG1 X-13.5 F3000\\nG1 X-3.5 F18000\\nG1 X-13.5 F3000\\nM400\\nM106 P1 S0\\n{endif}\\n\\n{if flush_length_2 > 1}\\nM106 P1 S60\\n; FLUSH_START\\nG1 E{flush_length_2 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_2 * 0.02} F50\\nG1 E{flush_length_2 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_2 * 0.02} F50\\nG1 E{flush_length_2 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_2 * 0.02} F50\\nG1 E{flush_length_2 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_2 * 0.02} F50\\nG1 E{flush_length_2 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_2 * 0.02} F50\\n; FLUSH_END\\nG1 E-[new_retract_length_toolchange] F1800\\nG1 E[new_retract_length_toolchange] F300\\n{endif}\\n\\n{if flush_length_2 > 45 && flush_length_3 > 1}\\n; WIPE\\nM400\\nM106 P1 S178\\nM400 S3\\nG1 X-3.5 F18000\\nG1 X-13.5 F3000\\nG1 X-3.5 F18000\\nG1 X-13.5 F3000\\nG1 X-3.5 F18000\\nG1 X-13.5 F3000\\nM400\\nM106 P1 S0\\n{endif}\\n\\n{if flush_length_3 > 1}\\nM106 P1 S60\\n; FLUSH_START\\nG1 E{flush_length_3 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_3 * 0.02} F50\\nG1 E{flush_length_3 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_3 * 0.02} F50\\nG1 E{flush_length_3 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_3 * 0.02} F50\\nG1 E{flush_length_3 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_3 * 0.02} F50\\nG1 E{flush_length_3 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_3 * 0.02} F50\\n; FLUSH_END\\nG1 E-[new_retract_length_toolchange] F1800\\nG1 E[new_retract_length_toolchange] F300\\n{endif}\\n\\n{if flush_length_3 > 45 && flush_length_4 > 1}\\n; WIPE\\nM400\\nM106 P1 S178\\nM400 S3\\nG1 X-3.5 F18000\\nG1 X-13.5 F3000\\nG1 X-3.5 F18000\\nG1 X-13.5 F3000\\nG1 X-3.5 F18000\\nG1 X-13.5 F3000\\nM400\\nM106 P1 S0\\n{endif}\\n\\n{if flush_length_4 > 1}\\nM106 P1 S60\\n; FLUSH_START\\nG1 E{flush_length_4 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_4 * 0.02} F50\\nG1 E{flush_length_4 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_4 * 0.02} F50\\nG1 E{flush_length_4 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_4 * 0.02} F50\\nG1 E{flush_length_4 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_4 * 0.02} F50\\nG1 E{flush_length_4 * 0.18} F{new_filament_e_feedrate}\\nG1 E{flush_length_4 * 0.02} F50\\n; FLUSH_END\\n{endif}\\n\\nM629\\n\\nM400\\nM106 P1 S60\\nM109 S[new_filament_temp]\\nG1 E5 F{new_filament_e_feedrate} ;Compensate for filament spillage during waiting temperature\\nM400\\nG92 E0\\nG1 E-[new_retract_length_toolchange] F1800\\nM400\\nM106 P1 S178\\nM400 S3\\nG1 X-3.5 F18000\\nG1 X-13.5 F3000\\nG1 X-3.5 F18000\\nG1 X-13.5 F3000\\nG1 X-3.5 F18000\\nG1 X-13.5 F3000\\nG1 X-3.5 F18000\\nG1 X-13.5 F3000\\nM400\\nG1 Z{max_layer_z + 3.0} F3000\\nM106 P1 S0\\n{if layer_z <= (initial_layer_print_height + 0.001)}\\nM204 S[initial_layer_acceleration]\\n{else}\\nM204 S[default_acceleration]\\n{endif}\\n{else}\\nG1 X[x_after_toolchange] Y[y_after_toolchange] Z[z_after_toolchange] F12000\\n{endif}\\n\\nM622.1 S0\\nM9833 F{outer_wall_volumetric_speed/2.4} A0.3 ; cali dynamic extrusion compensation\\nM1002 judge_flag filament_need_cali_flag\\nM622 J1\\n  G92 E0\\n  G1 E-[new_retract_length_toolchange] F1800\\n  M400\\n  \\n  M106 P1 S178\\n  M400 S7\\n  G1 X0 F18000\\n  G1 X-13.5 F3000\\n  G1 X0 F18000 ;wipe and shake\\n  G1 X-13.5 F3000\\n  G1 X0 F12000 ;wipe and shake\\n  G1 X-13.5 F3000\\n  G1 X0 F12000 ;wipe and shake\\n  M400\\n  M106 P1 S0 \\nM623\\n\\nM621 S[next_extruder]A\\nG392 S0\\n\\nM1007 S1\\n",
            "circle_compensation_manual_offset": "0",
            "circle_compensation_speed": ["200", "200", "200", "200"],
            "close_fan_the_first_x_layers": ["1", "1", "1", "1"],
            "complete_print_exhaust_fan_speed": ["70", "70", "70", "70"],
            "cool_plate_temp": ["35", "35", "35", "35"],
            "cool_plate_temp_initial_layer": ["35", "35", "35", "35"],
            "counter_coef_1": ["0", "0", "0", "0"],
            "counter_coef_2": ["0.008", "0.008", "0.008", "0.008"],
            "counter_coef_3": ["-0.041", "-0.041", "-0.041", "-0.041"],
            "counter_limit_max": ["0.033", "0.033", "0.033", "0.033"],
            "counter_limit_min": ["-0.035", "-0.035", "-0.035", "-0.035"],
            "default_acceleration": "500",
            "default_jerk": "9",
            "detect_overhang_wall": "1",
            "detect_thin_wall": "0",
            "draft_shield": "disabled",
            "elefant_foot_compensation": "0.15",
            "enable_arc_fitting": "0",
            "enable_overhang_bridge_fan": "1",
            "enable_overhang_speed": "1",
            "enable_prime_tower": "0",
            "enable_support": "0",
            "enforce_support_layers": "0",
            "ensure_vertical_shell_thickness": "1",
            "exclude_object": "1",
            "extra_perimeters_on_overhangs": "0",
            "extruder_clearance_height_to_lid": "140",
            "extruder_clearance_height_to_rod": "36",
            "extruder_clearance_radius": "65",
            "extruder_colour": ["", "", "", ""],
            "fan_speedup_time": ["0", "0", "0", "0"],
            "filament_colour": ["#FFFFFF", "#000000", "#FF6B6B", "#4ECDC4"],
            "filament_cooling_final_speed": ["3.4", "3.4", "3.4", "3.4"],
            "filament_cooling_initial_speed": ["2.2", "2.2", "2.2", "2.2"],
            "filament_cooling_moves": ["4", "4", "4", "4"],
            "filament_cost": ["20", "20", "20", "20"],
            "filament_density": ["1.24", "1.24", "1.24", "1.24"],
            "filament_diameter": ["1.75", "1.75", "1.75", "1.75"],
            "filament_flow_ratio": ["1", "1", "1", "1"],
            "filament_ids": ["GFL00", "GFL01", "GFL02", "GFL03"],
            "filament_loading_speed": ["28", "28", "28", "28"],
            "filament_loading_speed_start": ["3", "3", "3", "3"],
            "filament_minimal_purge_on_wipe_tower": ["15", "15", "15", "15"],
            "filament_ramming_parameters": ["120 100 6.6 6.8 7.2 7.6 7.9 8.2 8.7 9.4 9.9 10.0| 0.05 6.6 0.45 6.8 0.95 7.8 1.45 8.3 1.95 9.7 2.45 10 2.95 7.6 3.45 7.6 3.95 7.6 4.45 7.6 4.95 7.6", "120 100 6.6 6.8 7.2 7.6 7.9 8.2 8.7 9.4 9.9 10.0| 0.05 6.6 0.45 6.8 0.95 7.8 1.45 8.3 1.95 9.7 2.45 10 2.95 7.6 3.45 7.6 3.95 7.6 4.45 7.6 4.95 7.6", "120 100 6.6 6.8 7.2 7.6 7.9 8.2 8.7 9.4 9.9 10.0| 0.05 6.6 0.45 6.8 0.95 7.8 1.45 8.3 1.95 9.7 2.45 10 2.95 7.6 3.45 7.6 3.95 7.6 4.45 7.6 4.95 7.6", "120 100 6.6 6.8 7.2 7.6 7.9 8.2 8.7 9.4 9.9 10.0| 0.05 6.6 0.45 6.8 0.95 7.8 1.45 8.3 1.95 9.7 2.45 10 2.95 7.6 3.45 7.6 3.95 7.6 4.45 7.6 4.95 7.6"],
            "filament_settings_id": ["Bambu PLA Basic @BBL A1M", "Bambu PLA Basic @BBL A1M", "Bambu PLA Basic @BBL A1M", "Bambu PLA Basic @BBL A1M"],
            "filament_soluble": ["0", "0", "0", "0"],
            "filament_type": ["PLA", "PLA", "PLA", "PLA"],
            "filament_unloading_speed": ["90", "90", "90", "90"],
            "filament_unloading_speed_start": ["7", "7", "7", "7"],
            "filament_vendor": ["Bambu Lab", "Bambu Lab", "Bambu Lab", "Bambu Lab"],
            "filament_wipe": ["0", "0", "0", "0"],
            "filter_out_gap_fill": "0",
            "flush_into_infill": "0",
            "flush_into_objects": "0",
            "flush_into_support": "1",
            "flush_multiplier": "1",
            "flush_volumes_matrix": [["0", "280", "280", "280"], ["280", "0", "280", "280"], ["280", "280", "0", "280"], ["280", "280", "280", "0"]],
            "flush_volumes_vector": ["0", "0", "0", "0"],
            "force_flush_volume": ["0", "0", "0", "0"],
            "fuzzy_skin": "none",
            "fuzzy_skin_point_distance": "0.8",
            "fuzzy_skin_thickness": "0.3",
            "gap_infill_speed": ["250"],
            "gcode_add_line_number": "0",
            "gcode_comments": "0",
            "gcode_label_objects": "1",
            "hot_plate_temp": ["60", "60", "60", "60"],
            "hot_plate_temp_initial_layer": ["60", "60", "60", "60"],
            "independent_support_layer_height": "1",
            "infill_anchor": "2.5",
            "infill_anchor_max": "12",
            "infill_combination": "0",
            "infill_density": "15%",
            "infill_direction": "45",
            "infill_jerk": "9",
            "infill_wall_overlap": "25%",
            "initial_layer_acceleration": "500",
            "initial_layer_infill_speed": ["50"],
            "initial_layer_jerk": "9",
            "initial_layer_line_width": "0.5",
            "initial_layer_print_height": "0.16",
            "initial_layer_speed": ["50"],
            "initial_layer_travel_speed": ["120"],
            "inner_wall_acceleration": "5000",
            "inner_wall_jerk": "9",
            "inner_wall_line_width": "0.4",
            "inner_wall_speed": ["300"],
            "interface_shells": "0",
            "ironing_flow": "10%",
            "ironing_pattern": "zig-zag",
            "ironing_spacing": "0.1",
            "ironing_speed": ["15"],
            "ironing_type": "no ironing",
            "layer_height": "0.08",
            "line_width": "0.4",
            "machine_end_gcode": "M400 ; wait for buffer to clear\\nG92 E0 ; zero the extruder\\nG1 E-0.8 F1800 ; retract the filament\\nG91 ; relative positioning\\nG1 Z1 F3000 ; move z up little to prevent scratching of print\\nG90 ; absolute positioning\\nG1 X128 Y180 F4000 ; park print head\\n{if max_layer_z < 100}G1 Z100 F1600 ; Move print bed down{endif}\\n{if max_layer_z >= 100}G1 Z{max_layer_z + 10} F1600 ; Move print bed down{endif}\\nM140 S0 ; turn off heatbed\\nM104 S0 ; turn off temperature\\nM107 ; turn off fan\\nM221 S100 ; reset extrusion rate\\nM900 K0 ; reset LA\\n{if max_layer_z < 100}G1 Z100 F1600 ; Move print bed down{endif}\\n{if max_layer_z >= 100}G1 Z{max_layer_z + 10} F1600 ; Move print bed down{endif}\\nM220 S100\\n",
            "machine_max_acceleration_e": ["5000", "5000", "5000", "5000"],
            "machine_max_acceleration_extruding": ["20000", "20000", "20000", "20000"],
            "machine_max_acceleration_retracting": ["5000", "5000", "5000", "5000"],
            "machine_max_acceleration_travel": ["20000", "20000", "20000", "20000"],
            "machine_max_acceleration_x": ["20000", "20000", "20000", "20000"],
            "machine_max_acceleration_y": ["20000", "20000", "20000", "20000"],
            "machine_max_acceleration_z": ["500", "500", "500", "500"],
            "machine_max_jerk_e": ["2.5", "2.5", "2.5", "2.5"],
            "machine_max_jerk_x": ["9", "9", "9", "9"],
            "machine_max_jerk_y": ["9", "9", "9", "9"],
            "machine_max_jerk_z": ["0.4", "0.4", "0.4", "0.4"],
            "machine_max_speed_e": ["25", "25", "25", "25"],
            "machine_max_speed_x": ["500", "500", "500", "500"],
            "machine_max_speed_y": ["500", "500", "500", "500"],
            "machine_max_speed_z": ["12", "12", "12", "12"],
            "machine_pause_gcode": "M400 ; wait for buffer to clear\\nG60 S0 ; save current position\\nG91 ; relative positioning\\nG1 Z1 F300 ; move z up little to prevent scratching of print\\nG90 ; absolute positioning\\nG1 X128 Y180 F4000 ; park print head\\nM104 S0 ; turn off temperature\\nM107 ; turn off fan\\nM125 ; park\\nM25 ; pause print",
            "machine_start_gcode": ";===== A1mini 20250206 =====\\nG392 S0\\nM1007 S0\\nM9833.2\\nM1002 gcode_claim_action : 0\\nM1002 set_filament_type:UNKNOWN\\nM1004 S0\\nG90\\nG28\\nG1 Z5 F1200\\nG1 X128 Y90 F12000\\nG1 Z0.3 F1200\\n{if filament_type[initial_extruder]==\"PLA\"}\\nM1002 set_filament_type:PLA\\nM1004 S1\\n{elsif filament_type[initial_extruder]==\"PETG\"}\\nM1002 set_filament_type:PETG\\nM1004 S1\\n{elsif filament_type[initial_extruder]==\"ABS\"}\\nM1002 set_filament_type:ABS\\nM1004 S1\\n{elsif filament_type[initial_extruder]==\"PC\"}\\nM1002 set_filament_type:PC\\nM1004 S1\\n{elsif filament_type[initial_extruder]==\"PA\"}\\nM1002 set_filament_type:PA\\nM1004 S1\\n{elsif filament_type[initial_extruder]==\"PAHT\"}\\nM1002 set_filament_type:PAHT\\nM1004 S1\\n{elsif filament_type[initial_extruder]==\"PET\"}\\nM1002 set_filament_type:PET\\nM1004 S1\\n{endif}\\nM1002 judge_flag build_volume_detect_flag\\nM622 J1\\n  ; bed volume detect\\n  M1002 judge_flag g29_before_print_flag\\n  M622 J0\\n    M1002 judge_flag g29_level_flag\\n    M622 J1\\n      {if leveling_first_time}\\n      ; Home again after hard home\\n      M1002 judge_flag xy_hole_abs_flag\\n      M622 J1\\n        G91\\n        G1 Z-3 F1200\\n        G90\\n      M623\\n      G28 X Y F12000\\n      {endif}\\n      G29.2 S0 ; turn on ABL\\n      G29.2 Z{+0.0} ; set z offset\\n      G29.1 X128 Y90 F12000\\n      M400 P200\\n      G29.2 S1 ; turn off ABL\\n    M623\\n  M623\\nM623\\nG1 X128 Y90 F12000\\nG29.2 S1 ; turn off ABL\\nM104 S[nozzle_temperature_initial_layer] ; set extruder temp to turn on during probing\\nM140 S[bed_temperature_initial_layer] ; set bed temp\\nM190 S[bed_temperature_initial_layer] ; wait for bed temp\\nM109 S[nozzle_temperature_initial_layer] ; wait for extruder temp\\nG1 X-3 Y1 F12000\\nG1 Z0.3 F1200\\nM83 ; extruder relative mode\\nM1002 judge_flag g29_before_print_flag\\nM622 J1\\n  M1002 judge_flag corner_bed_leveling_flag\\n  M622 J1\\n    G29.1 X-3 Y1 F12000\\n    M400 P200\\n  M623\\nM623\\nG1 E2 F300 ; intro line\\nG1 X-3 Y15 E2.5  F1500 ; intro line\\nG1 X-3 Y50 E3 F3000 ; intro line\\nG1 X-3 Y80 E4 F3000 ; intro line\\nG1 X-3 Y95 E1.5 F1500 ; intro line\\nG92 E0\\nG1 E-0.8 F1800 ; retract to avoid oozing\\nG1 Z5 F1200 ; lift z\\nM400\\nG392 S1\\nM1007 S1",
            "max_bridge_length": "10",
            "max_layer_height": ["0.28", "0.28", "0.28", "0.28"],
            "max_travel_detour_distance": "0",
            "max_volumetric_speed": ["21", "21", "21", "21"],
            "min_layer_height": ["0.08", "0.08", "0.08", "0.08"],
            "minimum_sparse_infill_area": "15",
            "mmu_segmented_region_interlocking_depth": "0",
            "mmu_segmented_region_max_width": "0",
            "nozzle_diameter": ["0.4"],
            "nozzle_height": "4",
            "nozzle_hrc": "0",
            "nozzle_temperature": ["220", "220", "220", "220"],
            "nozzle_temperature_initial_layer": ["220", "220", "220", "220"],
            "nozzle_temperature_range_high": ["240", "240", "240", "240"],
            "nozzle_temperature_range_low": ["190", "190", "190", "190"],
            "nozzle_type": "hardened_steel",
            "nozzle_volume": "0",
            "only_one_wall_top": "1",
            "ooze_prevention": "0",
            "outer_wall_acceleration": "5000",
            "outer_wall_jerk": "9",
            "outer_wall_line_width": "0.4",
            "outer_wall_speed": ["200"],
            "overhang_fan_speed": ["100", "100", "100", "100"], 
            "overhang_fan_threshold": ["50%", "50%", "50%", "50%"],
            "overhang_reverse": "0",
            "overhang_reverse_internal_only": "0",
            "overhang_reverse_threshold": "70%",
            "overhang_speed_classic": ["50%", "50%", "50%", "50%"],
            "prime_tower_brim_width": "3",
            "prime_tower_width": "35",
            "print_flow_ratio": "1",
            "print_sequence": "by_layer",
            "print_settings_id": "0.08mm High Quality @BBL A1M",
            "printer_model": "Bambu Lab A1 mini",
            "printer_settings_id": "Bambu Lab A1 mini 0.4 nozzle",
            "printer_structure": "corexy",
            "printer_technology": "FFF",
            "printer_variant": "0.4",
            "printhost_authorization_type": "key",
            "raft_contact_distance": "0.1",
            "raft_expansion": "1.5",
            "raft_first_layer_density": "90%",
            "raft_first_layer_expansion": "2",
            "raft_layers": "0",
            "reduce_crossing_wall": "1",
            "reduce_fan_stop_start_freq": ["1", "1", "1", "1"],
            "reduce_infill_retraction": "1",
            "resolution": "0.01",
            "retract_before_wipe": ["70%", "70%", "70%", "70%"],
            "retract_lift_above": ["0", "0", "0", "0"],
            "retract_lift_below": ["199", "199", "199", "199"],
            "retract_lift_enforce": ["0", "0", "0", "0"],
            "retract_restart_extra": ["0", "0", "0", "0"],
            "retract_restart_extra_toolchange": ["0", "0", "0", "0"],
            "retract_when_changing_layer": ["1", "1", "1", "1"],
            "retraction_distances_when_cut": ["18", "18", "18", "18"],
            "retraction_length": ["0.8", "0.8", "0.8", "0.8"],
            "retraction_minimum_travel": ["1", "1", "1", "1"],
            "retraction_speed": ["30", "30", "30", "30"],
            "role_based_wipe_speed": "1",
            "scan_first_layer": "1",
            "seam_position": "aligned",
            "single_extruder_multi_material": "1",
            "skirt_distance": "2",
            "skirt_height": "1",
            "skirt_loops": "0",
            "slice_closing_radius": "0.049",
            "slow_down_for_layer_cooling": ["1", "1", "1", "1"],
            "slow_down_layer_time": ["8", "8", "8", "8"],
            "slow_down_min_speed": ["10", "10", "10", "10"],
            "solid_infill_below_area": "0",
            "solid_infill_filament": "-1",
            "sparse_infill_density": "15%",
            "sparse_infill_filament": "-1",
            "sparse_infill_pattern": "grid",
            "spiral_mode": "0",
            "standby_temperature_delta": "-5",
            "support_air_filtration": "0",
            "support_angle": "0",
            "support_base_pattern": "default",
            "support_base_pattern_spacing": "2.5",
            "support_bottom_interface_spacing": "0.2",
            "support_bottom_z_distance": "0.2",
            "support_critical_regions_only": "0",
            "support_expansion": "0",
            "support_filament": "-1",
            "support_interface_bottom_layers": "2",
            "support_interface_filament": "-1",
            "support_interface_loop_pattern": "0",
            "support_interface_spacing": "0.2",
            "support_interface_speed": ["80"],
            "support_interface_top_layers": "2",
            "support_line_width": "0.4",
            "support_material_auto": "0",
            "support_object_xy_distance": "0.35",
            "support_on_build_plate_only": "0",
            "support_remove_small_overhang": "1",
            "support_speed": ["150"],
            "support_style": "default",
            "support_threshold_angle": "30",
            "support_top_z_distance": "0.2",
            "support_type": "normal(auto)",
            "temperature_vitrification": ["45", "45", "45", "45"],
            "template_custom_gcode": "",
            "textured_plate_temp": ["60", "60", "60", "60"],
            "textured_plate_temp_initial_layer": ["60", "60", "60", "60"],
            "thick_bridges": "0",
            "thick_internal_bridges": "0",
            "thin_wall_line_width": "0.4",
            "thin_wall_speed": ["80"],
            "timelapse_type": "0",
            "top_bottom_acceleration": "500",
            "top_shell_layers": "15",
            "top_shell_thickness": "0",
            "top_solid_infill_flow_ratio": "1",
            "top_surface_acceleration": "500",
            "top_surface_jerk": "9",
            "top_surface_line_width": "0.4",
            "top_surface_pattern": "monotonic",
            "top_surface_speed": ["200"],
            "travel_acceleration": "10000",
            "travel_jerk": "12",
            "travel_speed": ["300"],
            "travel_speed_z": "12",
            "tree_support_adaptive_layer_height": "1",
            "tree_support_angle_slow": "25",
            "tree_support_auto_brim": "1",
            "tree_support_branch_angle": "45",
            "tree_support_branch_diameter": "2",
            "tree_support_branch_distance": "5",
            "tree_support_top_rate": "15%",
            "tree_support_wall_count": "1",
            "upward_compatible_machine": ["Bambu Lab A1 mini 0.4 nozzle"],
            "wall_distribution_count": "1",
            "wall_filament": "-1",
            "wall_generator": "arachne",
            "wall_loops": "3",
            "wall_transition_angle": "10",
            "wall_transition_filter_deviation": "25%",
            "wall_transition_length": "100%",
            "wipe": ["1", "1", "1", "1"],
            "wipe_on_loops": "0",
            "wipe_speed": ["80%", "80%", "80%", "80%"],
            "wipe_tower_no_sparse_layers": "0",
            "xy_contour_compensation": "0",
            "xy_hole_compensation": "0",
            "z_hop": ["0.4", "0.4", "0.4", "0.4"],
            "z_hop_types": ["Normal Lift", "Normal Lift", "Normal Lift", "Normal Lift"]
        }
    
    def _generate_model_settings(self, geometry_data: Dict[str, Any]) -> str:
        """Generate model_settings.config with object information."""
        face_count = len(geometry_data.get('triangles', []))
        
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="2">
    <metadata key="name" value="bananaforge_model.stl"/>
    <metadata key="extruder" value="1"/>
    <metadata face_count="{face_count}"/>
    <part id="1" subtype="normal_part">
      <metadata key="name" value="bananaforge_model.stl"/>
      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>
      <metadata key="source_file" value="bananaforge_model.stl"/>
      <metadata key="source_object_id" value="0"/>
      <metadata key="source_volume_id" value="0"/>
      <mesh_stat face_count="{face_count}" edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>
    </part>
  </object>
  <plate>
    <metadata key="plater_id" value="1"/>
    <metadata key="filament_map_mode" value="Auto For Flush"/>
    <metadata key="filament_maps" value="1 1 1 1"/>
    <model_instance>
      <metadata key="object_id" value="2"/>
      <metadata key="instance_id" value="0"/>
    </model_instance>
  </plate>
  <assemble>
   <assemble_item object_id="2" instance_id="0" transform="1 0 0 0 1 0 0 0 1 0 0 0" offset="0 0 0" />
  </assemble>
</config>'''
    
    def _generate_slice_info(self) -> str:
        """Generate slice_info.config."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<config>
  <header>
    <header_item key="X-BBL-Client-Type" value="slicer"/>
    <header_item key="X-BBL-Client-Version" value="02.01.00.59"/>
  </header>
</config>'''
    
    def _generate_plate_json(self, material_data: Dict[str, Any]) -> str:
        """Generate plate_1.json with filament information."""
        layer_materials = material_data.get('layer_materials', {})
        
        # Extract filament IDs and colors
        filament_ids = []
        filament_colors = []
        
        for layer_material in layer_materials.values():
            if isinstance(layer_material, LayerMaterial):
                material_id = layer_material.material_id
            else:
                material_id = layer_material.get('material_id', 'PLA')
            
            if material_id not in filament_ids:
                filament_ids.append(material_id)
        
        plate_config = {
            "filament_ids": filament_ids[:4],  # Limit to 4 filaments
            "filament_colors": filament_colors,
            "first_extruder": 1,
            "bed_type": "textured_plate",
            "nozzle_diameter": 0.4,
            "version": 2
        }
        
        return json.dumps(plate_config)
    
    def _generate_cut_information(self) -> str:
        """Generate minimal cut_information.xml."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<config/>
'''
    
    def _generate_custom_gcode(self, material_data: Dict[str, Any]) -> str:
        """Generate custom_gcode_per_layer.xml with material swap instructions."""
        layer_materials = material_data.get('layer_materials', {})
        
        if not layer_materials:
            return '''<?xml version="1.0" encoding="utf-8"?>
<custom_gcodes_per_layer>
<plate>
<plate_info id="1"/>
<mode value="MultiAsSingle"/>
</plate>
</custom_gcodes_per_layer>
'''

        # Build layer swap instructions
        gcode_layers = []
        
        # Get unique materials in order they appear
        unique_materials = []
        sorted_layers = sorted(layer_materials.items(), key=lambda x: x[0])
        
        for _, layer_material in sorted_layers:
            if isinstance(layer_material, LayerMaterial):
                material_id = layer_material.material_id
            else:
                material_id = layer_material.get('material_id', 'unknown')
            
            if material_id not in unique_materials:
                unique_materials.append(material_id)
        
        # Create material to extruder mapping (1-indexed)
        material_to_extruder = {mat_id: idx + 1 for idx, mat_id in enumerate(unique_materials)}
        
        prev_material = None
        for layer_idx, layer_material in sorted_layers:
            if isinstance(layer_material, LayerMaterial):
                material_id = layer_material.material_id
                layer_height = layer_material.layer_height
            else:
                material_id = layer_material.get('material_id', f'material_{layer_idx}')
                layer_height = layer_material.get('layer_height', 0.08)
                
            # Calculate Z height matching SwapInstructionGenerator logic
            # Layer 0 = initial_layer_height, Layer 1+ = initial_layer_height + (layer_idx * layer_height)
            initial_layer_height = 0.16  # From instructions
            if layer_idx == 0:
                z_height = initial_layer_height
            else:
                z_height = initial_layer_height + (layer_idx * layer_height)
            
            # Check if material changed
            if prev_material and material_id != prev_material:
                # Find material color and extruder mapping
                material_color = self._get_material_color(material_id)
                extruder_id = material_to_extruder.get(material_id, 1)
                
                # Bambu Studio expects the swap command one layer earlier than where the new material appears
                # So if layer_idx should have new material, we put the swap at the end of layer_idx-1
                if layer_idx == 0:
                    # Can't swap before layer 0, skip this case
                    continue
                    
                swap_layer_idx = layer_idx - 1
                if swap_layer_idx == 0:
                    swap_z = initial_layer_height  # End of first layer
                else:
                    swap_z = initial_layer_height + (swap_layer_idx * layer_height)
                
                logger.info(f"Layer swap at z={swap_z:.3f}: {prev_material} -> {material_id} (extruder {extruder_id}, color {material_color}) - for layer {layer_idx + 1}")
                
                gcode_layers.append(
                    f'<layer top_z="{swap_z:.6f}" type="2" extruder="{extruder_id}" color="{material_color}" extra="" gcode="tool_change"/>'
                )
            
            prev_material = material_id
        
        # Generate XML
        layers_xml = '\n'.join(gcode_layers)
        
        return f'''<?xml version="1.0" encoding="utf-8"?>
<custom_gcodes_per_layer>
<plate>
<plate_info id="1"/>
{layers_xml}
<mode value="MultiAsSingle"/>
</plate>
</custom_gcodes_per_layer>
'''
    
    def _get_material_color(self, material_id: str) -> str:
        """Get hex color for a material ID from material database."""
        try:
            material = self.material_db.get_material(material_id)
            if material and hasattr(material, 'color_hex'):
                return material.color_hex
        except:
            pass
        
        # Fallback colors based on material name
        color_mapping = {
            'bambu_pla_gray': '#808080',
            'bambu_pla_light_blue': '#ADD8E6', 
            'bambu_pla_white': '#FFFFFF',
            'bambu_pla_black': '#000000',
            'bambu_pla_pink': '#FFC0CB'
        }
        return color_mapping.get(material_id, '#FFFFFF')
    
    def _get_extruder_for_material(self, material_id: str, sorted_layers: List[Tuple[int, Any]]) -> int:
        """Map material to extruder number (1-indexed)."""
        unique_materials = []
        for _, layer_material in sorted_layers:
            if isinstance(layer_material, LayerMaterial):
                mat_id = layer_material.material_id
            else:
                mat_id = layer_material.get('material_id', 'unknown')
            
            if mat_id not in unique_materials:
                unique_materials.append(mat_id)
        
        try:
            return unique_materials.index(material_id) + 1  # 1-indexed
        except ValueError:
            return 1  # Default to extruder 1
    
    def _add_proper_thumbnails(self, zip_file: zipfile.ZipFile, optimization_results: Dict[str, Any]) -> None:
        """Generate proper 512x512 PNG thumbnails for Bambu Studio."""
        try:
            from PIL import Image, ImageDraw
            import io
            import numpy as np
            import os
            
            # Try to get the original image for preview generation
            source_image_path = optimization_results.get('source_image_path')
            if source_image_path and os.path.exists(source_image_path):
                # Load and resize the original image for preview
                with Image.open(source_image_path) as img:
                    img = img.convert('RGBA')
                    img = img.resize((512, 512), Image.Resampling.LANCZOS)
                    
                    # Create different views
                    self._add_image_thumbnail(zip_file, "Metadata/pick_1.png", img)
                    self._add_image_thumbnail(zip_file, "Metadata/plate_1.png", img)
                    
                    # Create smaller version for plate_1_small.png
                    small_img = img.resize((256, 256), Image.Resampling.LANCZOS)
                    small_img = small_img.resize((512, 512), Image.Resampling.NEAREST)  # Upscale with pixelation
                    self._add_image_thumbnail(zip_file, "Metadata/plate_1_small.png", small_img)
                    
                    # Create versions with different lighting/effects
                    dark_img = Image.new('RGBA', (512, 512), (40, 40, 40, 255))
                    dark_img.paste(img, (0, 0), img)
                    self._add_image_thumbnail(zip_file, "Metadata/plate_no_light_1.png", dark_img)
                    self._add_image_thumbnail(zip_file, "Metadata/top_1.png", img)
                    
                    return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Could not generate proper thumbnails: {e}")
            
        # Fallback to placeholders if image generation fails
        self._add_placeholder_thumbnails(zip_file)
    
    def _add_image_thumbnail(self, zip_file: zipfile.ZipFile, path: str, image: 'Image.Image') -> None:
        """Add a PIL image as a thumbnail to the ZIP file."""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        zip_file.writestr(path, buffer.getvalue())
    
    def _add_placeholder_thumbnails(self, zip_file: zipfile.ZipFile) -> None:
        """Add proper sized placeholder PNG thumbnails (512x512)."""
        try:
            from PIL import Image
            import io
            
            # Create a 512x512 gray placeholder image
            placeholder = Image.new('RGBA', (512, 512), (128, 128, 128, 255))
            
            # Add a simple grid pattern to make it recognizable
            from PIL import ImageDraw
            draw = ImageDraw.Draw(placeholder)
            for i in range(0, 512, 64):
                draw.line([(i, 0), (i, 512)], fill=(100, 100, 100, 255), width=1)
                draw.line([(0, i), (512, i)], fill=(100, 100, 100, 255), width=1)
            
            # Add text
            try:
                draw.text((256, 256), "BananaForge", fill=(255, 255, 255, 255), anchor="mm")
            except:
                pass  # Skip text if font not available
            
            # Save to buffer and add to all thumbnail locations
            buffer = io.BytesIO()
            placeholder.save(buffer, format='PNG', optimize=True)
            png_data = buffer.getvalue()
            
            zip_file.writestr("Metadata/plate_1.png", png_data)
            zip_file.writestr("Metadata/plate_1_small.png", png_data)
            zip_file.writestr("Metadata/plate_no_light_1.png", png_data)
            zip_file.writestr("Metadata/top_1.png", png_data)
            zip_file.writestr("Metadata/pick_1.png", png_data)
            
        except ImportError:
            # Ultimate fallback: minimal 1x1 PNG if PIL not available
            png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
            
            zip_file.writestr("Metadata/plate_1.png", png_data)
            zip_file.writestr("Metadata/plate_1_small.png", png_data)
            zip_file.writestr("Metadata/plate_no_light_1.png", png_data)
            zip_file.writestr("Metadata/top_1.png", png_data)
            zip_file.writestr("Metadata/pick_1.png", png_data)


class ThreeMFExporter:
    """Main 3MF export class that coordinates all export functionality."""
    
    def __init__(self, device: str = "cpu", material_db: Optional[MaterialDatabase] = None):
        """Initialize 3MF exporter."""
        self.device = device
        self.material_db = material_db
        self.production_exporter = BambuProductionExporter(material_db)
        
        logger.info("Initialized ThreeMFExporter with Production Extension support")
    
    def export(self, optimization_results: Dict[str, Any], 
               output_path: Union[str, Path],
               config: Optional[ThreeMFExportConfig] = None) -> Dict[str, Any]:
        """Export optimization results to 3MF format."""
        if config is None:
            config = ThreeMFExportConfig()
        
        try:
            logger.info(f"Starting 3MF export to {output_path}")
            
            # Extract geometry and materials from optimization results
            geometry_data = self._extract_geometry_data(optimization_results)
            material_data = self._extract_material_data(optimization_results)
            
            # Create 3MF container using Production Extension format
            threemf_data = self.production_exporter.create_3mf_container(
                geometry_data, material_data, config, optimization_results)
            
            # Write to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(threemf_data)
            
            file_size = len(threemf_data)
            logger.info(f"3MF export completed successfully. File size: {file_size} bytes")
            
            return {
                'success': True,
                'output_file': str(output_path),
                'file_size': file_size,
                'materials_count': len(material_data.get('materials', {})),
                'format': 'Production Extension'
            }
            
        except Exception as e:
            logger.error(f"3MF export failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_geometry_data(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract geometry data using STL generator for consistency."""
        heightmap = optimization_results.get('heightmap')
        if heightmap is None:
            raise ValueError("No heightmap found in optimization results")
        
        # Convert heightmap to mesh geometry using STL generator's method
        if isinstance(heightmap, torch.Tensor):
            heightmap_torch = heightmap
        else:
            heightmap_torch = torch.tensor(heightmap)

        max_triangles = self._max_supported_triangles()
        estimated_triangles = self._estimate_heightmap_triangles(heightmap_torch)
        stl_path = optimization_results.get('stl_path')

        if stl_path:
            stl_triangle_count = self._estimate_stl_triangle_count(stl_path)
            if stl_triangle_count is not None:
                estimated_triangles = stl_triangle_count

        if estimated_triangles > max_triangles:
            raise ValueError(
                "3MF export skipped because the mesh is too large for the current "
                f"in-memory XML exporter ({estimated_triangles:,} triangles; limit "
                f"{max_triangles:,}). Reduce --physical-size, increase "
                "--nozzle-diameter, or raise BANANAFORGE_MAX_3MF_TRIANGLES if you "
                "have enough RAM."
            )

        if stl_path:
            return self._extract_geometry_from_stl(stl_path, heightmap_torch)
        
        # Use STL generator to create the same mesh geometry
        from .stl_generator import STLGenerator
        stl_gen = STLGenerator()
        
        # Create temporary mesh to extract geometry
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
            try:
                # Generate mesh using STL generator's algorithm with identical settings to STL export
                mesh = stl_gen.generate_stl(
                    height_map=heightmap_torch,
                    output_path=tmp_file.name,
                    physical_size=optimization_results.get('optimization_metadata', {}).get('physical_size', 100.0),
                    smooth_mesh=True,  # Use same smoothing as STL export
                    add_base=True     # Ensure solid base like STL
                )
                
                # Extract vertices and triangles from the trimesh object
                raw_vertices = [(float(v[0]), float(v[1]), float(v[2])) for v in mesh.vertices]
                triangles = [(int(f[0]), int(f[1]), int(f[2])) for f in mesh.faces]
                
                # Apply Bambu Studio coordinate transformation if needed
                # Reference coordinates: X: -90 to -80, Y: 50-60, Z: 0.5-0.6
                # Our coordinates: X: 0-100, Y: 112-113, Z: 0.8
                vertices = self._transform_coordinates_for_bambu(raw_vertices)
                
                logger.info(f"3MF mesh extracted: {len(vertices)} vertices, {len(triangles)} triangles")
                
                return {
                    'vertices': vertices,
                    'triangles': triangles,
                    'heightmap_shape': heightmap_torch.shape
                }
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)

    def _extract_geometry_from_stl(
        self, stl_path: Union[str, Path], heightmap_torch: torch.Tensor
    ) -> Dict[str, Any]:
        """Extract geometry data from an already generated STL file."""
        import trimesh

        mesh = trimesh.load(str(stl_path), file_type="stl")
        raw_vertices = [(float(v[0]), float(v[1]), float(v[2])) for v in mesh.vertices]
        triangles = [(int(f[0]), int(f[1]), int(f[2])) for f in mesh.faces]
        vertices = self._transform_coordinates_for_bambu(raw_vertices)

        logger.info(
            "3MF mesh extracted from existing STL: %s vertices, %s triangles",
            len(vertices),
            len(triangles),
        )

        return {
            'vertices': vertices,
            'triangles': triangles,
            'heightmap_shape': heightmap_torch.shape
        }

    def _max_supported_triangles(self) -> int:
        """Return the maximum triangle count allowed for in-memory 3MF export."""
        import os

        raw_limit = os.getenv("BANANAFORGE_MAX_3MF_TRIANGLES", "2000000")
        try:
            return max(1, int(raw_limit))
        except ValueError:
            logger.warning(
                "Invalid BANANAFORGE_MAX_3MF_TRIANGLES=%r; using 2000000",
                raw_limit,
            )
            return 2000000

    def _estimate_heightmap_triangles(self, heightmap: torch.Tensor) -> int:
        """Estimate triangles generated by the STL heightmap mesh."""
        h, w = heightmap.shape[-2:]
        if h < 2 or w < 2:
            return 0

        surface_triangles = 4 * (h - 1) * (w - 1)
        side_triangles = 4 * ((h - 1) + (w - 1))
        return int(surface_triangles + side_triangles)

    def _estimate_stl_triangle_count(self, stl_path: Union[str, Path]) -> Optional[int]:
        """Read binary STL triangle count without loading the whole mesh."""
        import os

        path = Path(stl_path)
        if not path.exists() or path.stat().st_size < 84:
            return None

        try:
            with open(path, "rb") as f:
                f.seek(80)
                triangle_count = struct.unpack("<I", f.read(4))[0]
        except OSError:
            return None

        expected_binary_size = 84 + (triangle_count * 50)
        actual_size = os.path.getsize(path)
        if expected_binary_size == actual_size:
            return int(triangle_count)

        return None
    
    def _transform_coordinates_for_bambu(self, vertices: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Transform coordinates to center the model properly for Bambu Studio.
        
        This preserves the physical size of the model while centering it on the build plate.
        Bambu Studio build plate center is approximately at (128, 128).
        
        Args:
            vertices: List of (x, y, z) vertex coordinates
            
        Returns:
            Transformed vertices compatible with Bambu Studio
        """
        if not vertices:
            return vertices
        
        # Calculate current coordinate ranges
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        z_coords = [v[2] for v in vertices]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        z_min, z_max = min(z_coords), max(z_coords)
        
        # Calculate model center and size
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        
        # Debug: Log coordinate ranges
        logger.info(f"Original coordinates - X: {x_min:.2f} to {x_max:.2f} (center: {x_center:.2f})")
        logger.info(f"Original coordinates - Y: {y_min:.2f} to {y_max:.2f} (center: {y_center:.2f})")
        logger.info(f"Original coordinates - Z: {z_min:.2f} to {z_max:.2f} (center: {z_center:.2f})")
        
        # Bambu Studio build plate center coordinates
        target_x_center = 90.0  # Center of build plate
        target_y_center = 90.0  # Center of build plate
        target_z_base = 0.0     # On build plate surface
        
        # Calculate translation (no scaling - preserve physical size)
        x_offset = target_x_center - x_center
        y_offset = target_y_center - y_center
        z_offset = target_z_base - z_min  # Place base at target Z
        
        # Debug: Log offsets
        logger.info(f"Calculated offsets - X: {x_offset:.2f}, Y: {y_offset:.2f}, Z: {z_offset:.2f}")
        logger.info(f"Target position - X: {target_x_center:.2f}, Y: {target_y_center:.2f}, Z: {target_z_base:.2f}")
        
        transformed_vertices = []
        for x, y, z in vertices:
            # Translate to center position while preserving size
            new_x = x + x_offset
            new_y = y + y_offset
            new_z = z + z_offset
            
            transformed_vertices.append((new_x, new_y, new_z))
        
        return transformed_vertices
    
    def _extract_material_data(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract material data for layer-based color changes."""
        layer_materials = optimization_results.get('layer_materials', {})
        
        # Convert to LayerMaterial objects
        layer_material_objects = {}
        for layer_idx, material_info in layer_materials.items():
            if isinstance(material_info, str):
                # Simple material ID
                layer_material_objects[layer_idx] = LayerMaterial(
                    layer_index=layer_idx,
                    material_id=material_info
                )
            elif isinstance(material_info, dict):
                # Detailed material info with transparency
                layer_material_objects[layer_idx] = LayerMaterial(
                    layer_index=layer_idx,
                    material_id=material_info.get('material_id', f'material_{layer_idx}'),
                    transparency=material_info.get('transparency', 1.0),
                    layer_height=material_info.get('layer_height', 0.2)
                )
        
        return {
            'layer_materials': layer_material_objects,
            'material_swaps': self._extract_material_swaps(optimization_results)
        }
    
    def _extract_material_swaps(self, optimization_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract material swap information for layer-based changes."""
        layer_materials = optimization_results.get('layer_materials', {})
        swaps = []
        
        # Convert layer materials to swap instructions
        sorted_layers = sorted(layer_materials.keys())
        for i, layer_idx in enumerate(sorted_layers):
            if i == 0:
                continue  # Skip first layer (no swap needed)
            
            prev_layer = sorted_layers[i-1]
            prev_material = layer_materials[prev_layer]
            curr_material = layer_materials[layer_idx]
            
            # Check if material changed
            prev_id = prev_material.get('material_id') if isinstance(prev_material, dict) else prev_material
            curr_id = curr_material.get('material_id') if isinstance(curr_material, dict) else curr_material
            
            if prev_id != curr_id:
                swaps.append({
                    'layer': layer_idx,
                    'height': layer_idx * 0.08,  # Assuming 0.08mm layer height
                    'from_material': prev_id,
                    'to_material': curr_id
                })
        
        return swaps


# Additional classes for advanced functionality

class ThreeMFLayerProcessor:
    """Processes per-layer material assignments for 3MF export."""
    
    def create_material_resources(self, layer_assignments: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create unique material resources based on layer assignments."""
        unique_materials = {}
        
        for layer_idx, assignment in layer_assignments.items():
            material_id = assignment['material_id']
            transparency = assignment.get('transparency', 1.0)
            
            # Create unique key for material + transparency combination
            unique_key = f"{material_id}_t{transparency:.2f}"
            
            if unique_key not in unique_materials:
                unique_materials[unique_key] = {
                    'id': unique_key,
                    'base_material': material_id,
                    'transparency': transparency,
                    'layers': [layer_idx]
                }
            else:
                unique_materials[unique_key]['layers'].append(layer_idx)
        
        return list(unique_materials.values())


class ThreeMFSliceGenerator:
    """Generates layer/slice information for 3MF files."""
    
    def generate_layer_metadata(self, layer_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate layer metadata for 3MF slice extension."""
        layer_heights = layer_info.get('layer_heights', [])
        total_layers = layer_info.get('total_layers', len(layer_heights))
        
        slice_data = {
            'layer_heights': layer_heights,
            'total_layers': total_layers,
            'slice_refs': []
        }
        
        # Generate slice references
        for i, height in enumerate(layer_heights):
            slice_data['slice_refs'].append({
                'slice_id': i,
                'layer_height': height,
                'z_position': sum(layer_heights[:i+1])
            })
        
        return slice_data


class ThreeMFSwapInstructionGenerator:
    """Generates material swap instructions for slicers."""
    
    def generate_swap_instructions(self, layer_materials: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate material change instructions."""
        swap_instructions = []
        previous_material = None
        
        for layer_idx in sorted(layer_materials.keys()):
            current_material = layer_materials[layer_idx]['material_id']
            
            if previous_material and current_material != previous_material:
                swap_instructions.append({
                    'layer': layer_idx,
                    'from_material': previous_material,
                    'to_material': current_material,
                    'swap_type': 'material_change'
                })
            
            previous_material = current_material
        
        return swap_instructions
