#!/usr/bin/env python3
"""3MF validation testing for file format compliance and quality assurance.

This file focuses on testing the validation of 3MF files against the official
3MF specification and ensuring compatibility with various 3D printing slicers.
"""

import xml.etree.ElementTree as ET

import pytest


class Test3MFSpecificationCompliance:
    """Test compliance with official 3MF specification."""

    def test_3mf_namespace_validation(self):
        """Test that 3MF XML uses correct namespaces."""
        # Drives implementation of proper 3MF namespace handling
        expected_namespaces = {
            "3mf": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
            "material": "http://schemas.microsoft.com/3dmanufacturing/material/2015/02",
            "slice": "http://schemas.microsoft.com/3dmanufacturing/slice/2015/07",
        }

        from bananaforge.output.threemf_exporter import ThreeMFNamespaceManager

        manager = ThreeMFNamespaceManager()
        namespaces = manager.get_registered_namespaces()

        for prefix, uri in expected_namespaces.items():
            assert prefix in namespaces
            assert namespaces[prefix] == uri

    def test_content_types_xml_structure(self):
        """Test [Content_Types].xml follows 3MF specification."""
        expected_content_types = {
            "rels": "application/vnd.openxmlformats-package.relationships+xml",
            "model": "application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
        }

        from bananaforge.output.threemf_exporter import ContentTypesGenerator

        generator = ContentTypesGenerator()
        content_types_xml = generator.generate()

        # Parse and validate structure
        root = ET.fromstring(content_types_xml)
        assert (
            root.tag
            == "{http://schemas.openxmlformats.org/package/2006/content-types}Types"
        )

        # Check required content types
        default_elements = root.findall(
            ".//{http://schemas.openxmlformats.org/package/2006/content-types}Default"
        )
        assert len(default_elements) >= len(expected_content_types)

    def test_relationships_xml_structure(self):
        """Test _rels/.rels follows OPC specification."""
        from bananaforge.output.threemf_exporter import RelationshipsGenerator

        generator = RelationshipsGenerator()
        rels_xml = generator.generate()

        # Parse and validate structure
        root = ET.fromstring(rels_xml)
        assert (
            root.tag
            == "{http://schemas.openxmlformats.org/package/2006/relationships}Relationships"
        )

        # Should have relationship to 3dmodel.model
        relationships = root.findall(
            ".//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"
        )
        model_rel = next(
            (r for r in relationships if r.get("Target") == "3D/3dmodel.model"),
            None,
        )
        assert model_rel is not None

    def test_3d_model_xml_structure(self):
        """Test 3D/3dmodel.model follows 3MF core specification."""
        from bananaforge.output.threemf_exporter import ModelXMLGenerator

        # Mock geometry data
        mock_vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
        mock_triangles = [(0, 1, 2), (1, 3, 2)]

        generator = ModelXMLGenerator()
        model_xml = generator.generate(mock_vertices, mock_triangles)

        # Parse and validate structure
        root = ET.fromstring(model_xml)
        assert root.tag.endswith("model")

        # Check required elements
        assert (
            root.find(
                ".//{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}resources"
            )
            is not None
        )
        assert (
            root.find(
                ".//{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}build"
            )
            is not None
        )


class Test3MFGeometryValidation:
    """Test validation of 3D geometry in 3MF files."""

    def test_mesh_manifold_validation(self):
        """Test that generated meshes are manifold."""
        from bananaforge.output.threemf_exporter import MeshValidator

        # Valid manifold mesh (cube)
        valid_vertices = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ]
        valid_triangles = [
            # Bottom face
            (0, 1, 2),
            (0, 2, 3),
            # Top face
            (4, 6, 5),
            (4, 7, 6),
            # Side faces
            (0, 4, 5),
            (0, 5, 1),
            (1, 5, 6),
            (1, 6, 2),
            (2, 6, 7),
            (2, 7, 3),
            (3, 7, 4),
            (3, 4, 0),
        ]

        validator = MeshValidator()
        is_manifold = validator.check_manifold(valid_vertices, valid_triangles)
        assert is_manifold is True

        # Invalid non-manifold mesh
        invalid_triangles = [(0, 1, 2)]  # Single triangle
        is_manifold = validator.check_manifold(valid_vertices, invalid_triangles)
        assert is_manifold is False

    def test_mesh_watertight_validation(self):
        """Test that meshes are watertight (no holes)."""
        from bananaforge.output.threemf_exporter import MeshValidator

        validator = MeshValidator()

        # Mock complete mesh data
        complete_vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        complete_triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]  # Tetrahedron

        is_watertight = validator.check_watertight(
            complete_vertices, complete_triangles
        )
        assert is_watertight is True

        # Mesh with hole
        incomplete_triangles = [(0, 1, 2)]  # Missing triangles
        is_watertight = validator.check_watertight(
            complete_vertices, incomplete_triangles
        )
        assert is_watertight is False

    def test_normal_vector_validation(self):
        """Test that triangle normal vectors are correctly oriented."""
        from bananaforge.output.threemf_exporter import NormalValidator

        validator = NormalValidator()

        # Triangle with correct winding order (counter-clockwise)
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        triangle = (0, 1, 2)

        normal = validator.calculate_normal(vertices, triangle)
        expected_normal = (0, 0, 1)  # Pointing up (positive Z)

        # Allow for floating point precision
        assert abs(normal[0] - expected_normal[0]) < 1e-6
        assert abs(normal[1] - expected_normal[1]) < 1e-6
        assert abs(normal[2] - expected_normal[2]) < 1e-6


class Test3MFMaterialValidation:
    """Test validation of material definitions in 3MF files."""

    def test_material_resource_validation(self):
        """Test that material resources are properly defined."""
        from bananaforge.output.threemf_exporter import MaterialResourceValidator

        validator = MaterialResourceValidator()

        # Valid material definition
        valid_material = {
            "id": "material_1",
            "name": "PLA Red",
            "color": "#FF0000",
            "properties": {
                "nozzle_temp": 210,
                "bed_temp": 60,
            },
        }

        is_valid = validator.validate_material(valid_material)
        assert is_valid is True

        # Invalid material (missing required fields)
        invalid_material = {"name": "PLA Red"}  # Missing ID and color

        is_valid = validator.validate_material(invalid_material)
        assert is_valid is False

    def test_color_format_validation(self):
        """Test validation of color formats in materials."""
        from bananaforge.output.threemf_exporter import ColorValidator

        validator = ColorValidator()

        # Valid color formats
        valid_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFFFF"]
        for color in valid_colors:
            assert validator.validate_color_format(color) is True

        # Invalid color formats
        invalid_colors = ["FF0000", "#GG0000", "red", "#FF"]
        for color in invalid_colors:
            assert validator.validate_color_format(color) is False

    def test_transparency_validation(self):
        """Test validation of transparency values."""
        from bananaforge.output.threemf_exporter import TransparencyValidator

        validator = TransparencyValidator()

        # Valid transparency values (0.0 to 1.0)
        valid_values = [0.0, 0.33, 0.67, 1.0]
        for value in valid_values:
            assert validator.validate_transparency(value) is True

        # Invalid transparency values
        invalid_values = [-0.1, 1.1, "0.5", None]
        for value in invalid_values:
            assert validator.validate_transparency(value) is False


class Test3MFLayerValidation:
    """Test validation of layer information in 3MF files."""

    def test_layer_height_validation(self):
        """Test validation of layer height values."""
        from bananaforge.output.threemf_exporter import LayerValidator

        validator = LayerValidator()

        # Valid layer heights
        valid_heights = [0.1, 0.15, 0.2, 0.25, 0.3]
        for height in valid_heights:
            assert validator.validate_layer_height(height) is True

        # Invalid layer heights
        invalid_heights = [0.0, -0.1, 1.0, "0.2"]
        for height in invalid_heights:
            assert validator.validate_layer_height(height) is False

    def test_layer_sequence_validation(self):
        """Test validation of layer sequence and continuity."""
        from bananaforge.output.threemf_exporter import LayerSequenceValidator

        validator = LayerSequenceValidator()

        # Valid layer sequence
        valid_layers = [
            {"layer": 0, "height": 0.2, "material": "pla_red"},
            {"layer": 1, "height": 0.2, "material": "pla_blue"},
            {"layer": 2, "height": 0.2, "material": "pla_red"},
        ]

        is_valid = validator.validate_sequence(valid_layers)
        assert is_valid is True

        # Invalid sequence (missing layer)
        invalid_layers = [
            {"layer": 0, "height": 0.2, "material": "pla_red"},
            {"layer": 2, "height": 0.2, "material": "pla_blue"},  # Missing layer 1
        ]

        is_valid = validator.validate_sequence(invalid_layers)
        assert is_valid is False

    def test_material_transition_validation(self):
        """Test validation of material transitions between layers."""
        from bananaforge.output.threemf_exporter import MaterialTransitionValidator

        validator = MaterialTransitionValidator()

        # Valid transitions
        transitions = [
            {
                "from_layer": 0,
                "to_layer": 1,
                "from_material": "pla_red",
                "to_material": "pla_blue",
            },
            {
                "from_layer": 1,
                "to_layer": 2,
                "from_material": "pla_blue",
                "to_material": "pla_white",
            },
        ]

        for transition in transitions:
            is_valid = validator.validate_transition(transition)
            assert is_valid is True


class Test3MFCompressionAndSize:
    """Test file compression and size optimization."""

    def test_zip_compression_efficiency(self):
        """Test that 3MF files are efficiently compressed."""
        from bananaforge.output.threemf_exporter import CompressionOptimizer

        optimizer = CompressionOptimizer()

        # Mock large XML data
        large_xml_data = "<mesh>" + "<vertex x='1' y='2' z='3'/>" * 10000 + "</mesh>"

        compressed_data = optimizer.compress_xml(large_xml_data)
        compression_ratio = len(compressed_data) / len(large_xml_data.encode())

        # Should achieve good compression ratio (<0.3 for repetitive XML)
        assert compression_ratio < 0.3

    def test_file_size_limits(self):
        """Test that 3MF files stay within reasonable size limits."""
        from bananaforge.output.threemf_exporter import FileSizeValidator

        validator = FileSizeValidator()

        # Mock file size data
        small_model_size = 5 * 1024 * 1024  # 5MB
        large_model_size = 150 * 1024 * 1024  # 150MB

        assert validator.validate_file_size(small_model_size, "small") is True
        assert (
            validator.validate_file_size(large_model_size, "large") is False
        )  # Too large

    def test_memory_efficient_processing(self):
        """Test that 3MF generation is memory efficient."""
        from bananaforge.output.threemf_exporter import MemoryEfficientProcessor

        processor = MemoryEfficientProcessor()

        # Process a large dataset without copying the geometry payload.
        # Mock large dataset
        large_dataset = {
            "vertices": [(i, i + 1, i + 2) for i in range(100000)],
            "triangles": [(i, i + 1, i + 2) for i in range(50000)],
        }

        result = processor.process_large_dataset(large_dataset)

        assert result["vertex_count"] == 100000
        assert result["triangle_count"] == 50000


class Test3MFErrorRecovery:
    """Test error recovery and graceful failure handling."""

    def test_corrupted_data_recovery(self):
        """Test recovery from corrupted input data."""
        from bananaforge.output.threemf_exporter import ErrorRecoveryManager

        manager = ErrorRecoveryManager()

        # Corrupted geometry data
        corrupted_data = {
            "vertices": [(float("nan"), 0, 0), (1, float("inf"), 0)],
            "triangles": [(0, 1, 2)],  # Invalid triangle reference
        }

        # Should attempt to clean and recover data
        cleaned_data = manager.recover_geometry_data(corrupted_data)

        # Should remove invalid vertices and triangles
        assert len(cleaned_data["vertices"]) == 0  # All invalid
        assert len(cleaned_data["triangles"]) == 0  # Invalid references removed

    def test_partial_export_on_error(self):
        """Test partial export when some components fail."""
        from bananaforge.output.threemf_exporter import PartialExportManager

        manager = PartialExportManager()

        # Mock scenario where geometry succeeds but materials fail
        export_status = {
            "geometry": {"success": True, "data": "valid_mesh"},
            "materials": {"success": False, "error": "Invalid material data"},
            "metadata": {"success": True, "data": "valid_metadata"},
        }

        partial_result = manager.create_partial_export(export_status)

        # Should create export with available components
        assert partial_result["success"] is True
        assert "geometry" in partial_result["components"]
        assert "materials" not in partial_result["components"]
        assert "warnings" in partial_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
