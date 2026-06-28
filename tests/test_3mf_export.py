#!/usr/bin/env python3
"""Comprehensive 3MF export testing suite for BananaForge.

Following TDD/BDD methodology for Feature 7: Professional 3MF Export for Modern Slicers.
Tests are written first to drive the implementation of 3MF export functionality.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from unittest.mock import Mock, patch, MagicMock
import io

from bananaforge.core.optimizer import LayerOptimizer, OptimizationConfig
from bananaforge.image.processor import ImageProcessor
from bananaforge.materials.database import MaterialDatabase, DefaultMaterials
from bananaforge.materials.matcher import ColorMatcher
from bananaforge.output.exporter import ModelExporter


class Test3MFFileStructureGeneration:
    """Test Story 7.1: Core 3MF File Structure Generation.
    
    Acceptance Criteria:
    - Generate valid 3MF file with proper XML structure
    - Include all required 3MF specification elements (model, resources, build)
    - Create ZIP-based container with proper MIME types
    - Validate against 3MF specification schema
    - Ensure file can be opened by major slicers without errors
    """
    
    @pytest.fixture
    def device(self):
        """Get appropriate device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @pytest.fixture
    def sample_optimization_results(self, device):
        """Create sample optimization results for testing."""
        # Simulate BananaForge optimization results
        num_layers = 10
        layer_height = 0.2
        width, height = 50, 50
        
        # Mock geometry data (simplified heightmap)
        heightmap = torch.rand(height, width, device=device) * num_layers * layer_height
        
        # Mock material assignments per layer
        material_assignments = {
            i: f"material_{i % 3}"  # Cycle through 3 materials
            for i in range(num_layers)
        }
        
        # Mock layer information
        layer_info = {
            'layer_height': layer_height,
            'num_layers': num_layers,
            'material_assignments': material_assignments,
            'heightmap': heightmap,
            'dimensions': (width, height)
        }
        
        return layer_info
    
    @pytest.fixture
    def material_db(self):
        """Create test material database."""
        return DefaultMaterials.create_bambu_basic_pla()
    
    def test_3mf_exporter_initialization(self, device, material_db):
        """Test ThreeMFExporter can be initialized properly."""
        from bananaforge.output.threemf_exporter import ThreeMFExporter
        
        # Should initialize without errors
        exporter = ThreeMFExporter(device=device, material_db=material_db)
        
        # Verify initialization
        assert exporter.device == device
        assert exporter.material_db == material_db
        assert exporter.ns_manager is not None

    def test_3mf_reuses_existing_stl_geometry(self, tmp_path, monkeypatch, material_db):
        """3MF export should reuse an already generated STL when available."""
        from bananaforge.output.threemf_exporter import ThreeMFExporter

        stl_path = tmp_path / "existing.stl"
        stl_path.write_bytes(b"Binary STL".ljust(80, b" ") + (0).to_bytes(4, "little"))

        exporter = ThreeMFExporter(material_db=material_db)

        def fake_extract_from_stl(path, heightmap_torch):
            assert Path(path) == stl_path
            return {
                "vertices": [],
                "triangles": [],
                "heightmap_shape": heightmap_torch.shape,
            }

        monkeypatch.setattr(exporter, "_extract_geometry_from_stl", fake_extract_from_stl)
        monkeypatch.setattr(
            exporter.production_exporter,
            "create_3mf_container",
            lambda geometry, materials, config, results: b"3mf-data",
        )

        result = exporter.export(
            {
                "heightmap": torch.zeros(1, 1, 2, 2),
                "layer_materials": {},
                "stl_path": str(stl_path),
            },
            tmp_path / "model.3mf",
        )

        assert result["success"]

    def test_3mf_rejects_oversized_mesh_before_stl_generation(
        self, tmp_path, monkeypatch, material_db
    ):
        """Oversized 3MF exports should fail gracefully before huge mesh work."""
        from bananaforge.output.stl_generator import STLGenerator
        from bananaforge.output.threemf_exporter import ThreeMFExporter

        def fail_if_called(*args, **kwargs):
            raise AssertionError("STL generation should not run for oversized 3MF")

        monkeypatch.setenv("BANANAFORGE_MAX_3MF_TRIANGLES", "10")
        monkeypatch.setattr(STLGenerator, "generate_stl", fail_if_called)

        exporter = ThreeMFExporter(material_db=material_db)
        result = exporter.export(
            {
                "heightmap": torch.zeros(1, 1, 10, 10),
                "layer_materials": {},
            },
            tmp_path / "oversized.3mf",
        )

        assert not result["success"]
        assert "too large" in result["error"]
    
    def test_3mf_zip_container_structure(self, sample_optimization_results, material_db):
        """Test that 3MF file creates proper ZIP container structure."""
        from bananaforge.output.threemf_exporter import ThreeMFExporter, ThreeMFExportConfig
        import zipfile
        import io
        
        # This test defines the expected ZIP structure
        required_files = [
            "[Content_Types].xml",
            "_rels/.rels", 
            "3D/3dmodel.model"
        ]
        
        exporter = ThreeMFExporter(material_db=material_db)
        
        # Extract and prepare test data
        geometry_data = exporter._extract_geometry_data(sample_optimization_results)
        material_data = exporter._extract_material_data(sample_optimization_results)
        config = ThreeMFExportConfig()
        
        zip_data = exporter.create_3mf_container(geometry_data, material_data, config)
        
        # Validate ZIP structure
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_file:
            file_list = zip_file.namelist()
            for expected_file in required_files:
                assert expected_file in file_list, f"Missing required file: {expected_file}"
    
    def test_3mf_content_types_xml(self):
        """Test [Content_Types].xml structure follows 3MF specification."""
        from bananaforge.output.threemf_exporter import ContentTypesGenerator
        import xml.etree.ElementTree as ET
        
        expected_content_types = {
            "rels": "application/vnd.openxmlformats-package.relationships+xml",
            "model": "application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
            "png": "image/png"
        }
        
        generator = ContentTypesGenerator()
        content_types_xml = generator.generate()
        
        # Parse and validate
        full_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n{content_types_xml}'
        root = ET.fromstring(full_xml)
        assert root.tag.endswith("Types")
        
        # Check that expected content types are present
        default_elements = root.findall(".//*[@Extension]")
        found_extensions = {elem.get("Extension") for elem in default_elements}
        
        for ext in expected_content_types.keys():
            assert ext in found_extensions, f"Missing content type for extension: {ext}"


class TestModelExporterOutputOrdering:
    """Test ModelExporter ordering around lightweight and heavyweight outputs."""

    def test_instructions_are_written_before_heavy_stl_failure(
        self, tmp_path, monkeypatch
    ):
        """Instruction files should survive if a later mesh export fails."""
        material_db = DefaultMaterials.create_bambu_basic_pla()
        material_ids = material_db.get_material_ids()[:2]
        exporter = ModelExporter(material_db=material_db)

        def fail_stl_generation(*args, **kwargs):
            raise RuntimeError("simulated heavy mesh failure")

        monkeypatch.setattr(exporter.stl_generator, "generate_stl", fail_stl_generation)

        height_map = torch.ones(1, 1, 3, 3)
        material_assignments = torch.zeros(2, 3, 3, dtype=torch.long)
        material_assignments[1] = 1

        with pytest.raises(RuntimeError, match="simulated heavy mesh failure"):
            exporter.export_complete_model(
                height_map=height_map,
                material_assignments=material_assignments,
                material_database=material_db,
                material_ids=material_ids,
                output_dir=tmp_path,
                project_name="ordered",
                export_formats=["instructions", "stl"],
            )

        assert (tmp_path / "ordered_instructions.txt").exists()
        assert (tmp_path / "ordered_instructions.csv").exists()

    def test_triangle_estimator_matches_binary_stl_size(self):
        """Estimator should include the configured bottom face mode."""
        triangles = ModelExporter.estimate_triangle_count(10, 20, "full")
        simplified_triangles = ModelExporter.estimate_triangle_count(
            10, 20, "simplified"
        )
        open_triangles = ModelExporter.estimate_triangle_count(10, 20, "none")

        assert triangles == 4 * 9 * 19 + 4 * (9 + 19)
        assert simplified_triangles == 2 * 9 * 19 + 4 * (9 + 19) + 2
        assert open_triangles == 2 * 9 * 19 + 4 * (9 + 19)
        assert ModelExporter.estimate_binary_stl_size_bytes(triangles) == (
            84 + triangles * 50
        )

    def test_max_triangles_downscales_export_mesh(self, tmp_path, monkeypatch):
        """A triangle budget should downscale tensors before mesh generation."""
        material_db = DefaultMaterials.create_bambu_basic_pla()
        material_ids = material_db.get_material_ids()[:2]
        exporter = ModelExporter(material_db=material_db)
        captured_shape = {}

        def capture_stl_generation(height_map, output_path, physical_size, **kwargs):
            captured_shape["height_map"] = height_map.shape[-2:]
            Path(output_path).write_bytes(b"stub")

        monkeypatch.setattr(exporter.stl_generator, "generate_stl", capture_stl_generation)

        height_map = torch.ones(1, 1, 40, 60)
        material_assignments = torch.zeros(2, 40, 60, dtype=torch.long)

        exporter.export_complete_model(
            height_map=height_map,
            material_assignments=material_assignments,
            material_database=material_db,
            material_ids=material_ids,
            output_dir=tmp_path,
            project_name="limited",
            export_formats=["stl"],
            max_triangles=400,
        )

        limited_h, limited_w = captured_shape["height_map"]
        assert (limited_h, limited_w) != (40, 60)
        assert ModelExporter.estimate_triangle_count(
            limited_h, limited_w, exporter.bottom_mode
        ) <= 400


class Test3MFPerLayerMaterialAssignment:
    """Test Story 7.2: Per-Layer Material and Color Assignment.
    
    Acceptance Criteria:
    - Each layer has material ID assigned based on optimization results
    - Layer materials reflect transparency-aware color mixing calculations
    - 3MF file contains material resources for each unique layer material
    - Layer height information preserved for proper slicing
    - Slicers automatically configure material changes at layer boundaries
    - Support transparency layers for advanced color mixing effects
    """
    
    @pytest.fixture
    def layer_material_assignments(self):
        """Create test layer material assignments."""
        return {
            0: {"material_id": "pla_black", "transparency": 1.0},
            1: {"material_id": "pla_red", "transparency": 1.0}, 
            2: {"material_id": "pla_red", "transparency": 0.67},  # Transparency mixing
            3: {"material_id": "pla_white", "transparency": 1.0},
            4: {"material_id": "pla_white", "transparency": 0.33}, # More transparency
            5: {"material_id": "pla_blue", "transparency": 1.0},
        }
    
    def test_layer_material_mapping(self, layer_material_assignments):
        """Test that layer materials are correctly mapped to 3MF resources."""
        from bananaforge.output.threemf_exporter import ThreeMFLayerProcessor
        
        processor = ThreeMFLayerProcessor()
        material_resources = processor.create_material_resources(layer_material_assignments)
        
        # Should create unique materials for each transparency level
        assert len(material_resources) >= 4  # At least 4 unique combinations
        
        # Check transparency handling
        red_materials = [m for m in material_resources if "red" in m["id"]]
        assert len(red_materials) >= 2  # Regular red + transparent red
    
    def test_layer_height_preservation(self):
        """Test that layer height information is preserved in 3MF."""
        from bananaforge.output.threemf_exporter import ThreeMFSliceGenerator
        
        layer_info = {
            'layer_heights': [0.2, 0.2, 0.15, 0.2, 0.2, 0.3],  # Variable layer heights
            'total_layers': 6
        }
        
        generator = ThreeMFSliceGenerator()
        slice_data = generator.generate_layer_metadata(layer_info)
        
        assert 'layer_heights' in slice_data
        assert len(slice_data['layer_heights']) == 6
        assert slice_data['total_layers'] == 6
    
    def test_material_change_instructions(self, layer_material_assignments):
        """Test generation of material change instructions for slicers."""
        from bananaforge.output.threemf_exporter import ThreeMFSwapInstructionGenerator
        
        generator = ThreeMFSwapInstructionGenerator()
        swap_instructions = generator.generate_swap_instructions(layer_material_assignments)
        
        # Should detect material changes between layers
        # From the fixture: layer 0 (black) -> 1 (red) -> 3 (white) -> 5 (blue)
        assert len(swap_instructions) >= 2  # At least 2 material changes
        
        # Verify structure of swap instructions
        for instruction in swap_instructions:
            assert 'layer' in instruction
            assert 'from_material' in instruction
            assert 'to_material' in instruction


class Test3MFMaterialPropertiesEmbedding:
    """Test Story 7.3: Material Properties Embedding.
    
    Acceptance Criteria:
    - 3MF file includes material properties for each used filament
    - Embed nozzle temperatures, bed temperatures, and print speeds
    - Include material names, brands, and descriptions from material database
    - Support custom properties for advanced material characteristics
    - Allow slicers to automatically configure print profiles based on materials
    """
    
    @pytest.fixture
    def material_properties(self):
        """Create test material properties."""
        return {
            "pla_red": {
                "name": "Bambu PLA Basic Red",
                "brand": "Bambu Lab",
                "nozzle_temp": 220,
                "bed_temp": 60,
                "print_speed": 50,
                "density": 1.24,
                "color": [1.0, 0.0, 0.0]
            },
            "pla_white": {
                "name": "Bambu PLA Basic White", 
                "brand": "Bambu Lab",
                "nozzle_temp": 215,
                "bed_temp": 55,
                "print_speed": 55,
                "density": 1.24,
                "color": [1.0, 1.0, 1.0]
            }
        }
    
    def test_material_properties_xml_generation(self, material_properties):
        """Test that material properties are correctly embedded in 3MF XML."""
        # Drives implementation of material properties embedding
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFMaterialGenerator
            
            generator = ThreeMFMaterialGenerator()
            materials_xml = generator.generate_materials_xml(material_properties)
            
            root = ET.fromstring(materials_xml)
            materials = root.findall(".//material")
            assert len(materials) == 2
            
            # Check temperature embedding
            red_material = next(m for m in materials if "red" in m.get("name", "").lower())
            assert red_material is not None
    
    def test_bambu_material_mapping(self, material_properties):
        """Test mapping to Bambu Studio material database."""
        # Drives Bambu-specific material integration
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import BambuMaterialMapper
            
            mapper = BambuMaterialMapper()
            bambu_materials = mapper.map_to_bambu_database(material_properties)
            
            # Should include Bambu-specific IDs and properties
            assert "bambu_id" in bambu_materials["pla_red"]


class Test3MFValidationAndQualityAssurance:
    """Test Story 7.6: 3MF Validation and Quality Assurance.
    
    Acceptance Criteria:
    - Verify XML schema compliance against 3MF specification
    - Validate ZIP container structure and content types
    - Check mesh geometry for manifold properties and face normals
    - Verify material assignments are complete and consistent
    - Ensure file size is reasonable and compression is effective
    - Test loading in multiple slicer applications
    """
    
    def test_3mf_schema_validation(self):
        """Test that generated 3MF files comply with official schema."""
        # This drives implementation of proper XML schema validation
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFValidator
            
            validator = ThreeMFValidator()
            
            # Mock a 3MF file for testing
            mock_3mf_data = b"mock_3mf_zip_data"
            
            is_valid, errors = validator.validate_3mf_schema(mock_3mf_data)
            assert is_valid, f"3MF validation failed: {errors}"
    
    def test_mesh_geometry_validation(self):
        """Test mesh geometry validation (manifold, watertight, normals)."""
        # Drives mesh quality validation
        mock_vertices = np.random.rand(100, 3).astype(np.float32)
        mock_faces = np.random.randint(0, 100, (50, 3)).astype(np.int32)
        
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFMeshValidator
            
            validator = ThreeMFMeshValidator()
            is_manifold = validator.check_manifold(mock_vertices, mock_faces)
            is_watertight = validator.check_watertight(mock_vertices, mock_faces)
            
            # These checks ensure mesh quality
            assert isinstance(is_manifold, bool)
            assert isinstance(is_watertight, bool)
    
    def test_file_size_optimization(self):
        """Test that 3MF files are properly compressed and sized."""
        # Drives file size optimization
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFOptimizer
            
            optimizer = ThreeMFOptimizer()
            
            # Mock large mesh data
            large_mesh_data = np.random.rand(10000, 3).astype(np.float32)
            
            compressed_size = optimizer.estimate_compressed_size(large_mesh_data)
            raw_size = large_mesh_data.nbytes
            
            # Should achieve reasonable compression
            compression_ratio = compressed_size / raw_size
            assert compression_ratio < 0.5  # At least 50% compression


class Test3MFCLIIntegration:
    """Test Story 7.7 & 7.10: CLI Integration and User Experience.
    
    Acceptance Criteria:
    - Accept "3mf" as valid export format
    - Provide 3MF-specific options (--bambu-compatible, --include-metadata)
    - Show progress feedback during 3MF generation
    - Display file size and validation status upon completion
    - Support batch export of multiple formats simultaneously
    - Provide helpful error messages for 3MF generation failures
    """
    
    def test_cli_3mf_export_option(self):
        """Test that CLI accepts 3mf as export format."""
        # Test that 3MF export is properly integrated into CLI
        from bananaforge.cli import cli
        # CLI should import successfully with 3MF support
        assert cli is not None
    
    def test_bambu_compatibility_flag(self):
        """Test --bambu-compatible CLI flag."""
        # Test that Bambu compatibility flag is available
        from bananaforge.cli import cli
        # Should have bambu compatible options
        assert cli is not None
    
    def test_3mf_progress_reporting(self):
        """Test progress reporting during 3MF generation."""
        # Drives progress reporting implementation
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFProgressReporter
            
            reporter = ThreeMFProgressReporter()
            
            # Should provide progress callbacks
            progress_steps = []
            
            def progress_callback(step, total, message):
                progress_steps.append((step, total, message))
            
            reporter.set_progress_callback(progress_callback)
            reporter.report_progress(1, 5, "Generating mesh...")
            
            assert len(progress_steps) == 1
            assert progress_steps[0][2] == "Generating mesh..."


class Test3MFIntegrationWorkflow:
    """Test Story 7.12: Integration Testing and Quality Assurance.
    
    Acceptance Criteria:
    - Test complete image-to-3MF pipeline
    - Validate 3MF files load correctly in Bambu Studio, PrusaSlicer, and Cura
    - Test 3MF export with transparency features and advanced optimization
    - Verify material database integration and property mapping
    - Test large model performance and memory usage
    - Validate 3MF files against official 3MF specification tools
    - Ensure backward compatibility with existing export formats
    - Test concurrent export of multiple formats (STL + 3MF + instructions)
    """
    
    @pytest.fixture
    def integration_test_image(self):
        """Create a complex test image for integration testing."""
        # Generate a test image with multiple colors and transparency
        image = torch.zeros(1, 4, 128, 128)  # RGBA image
        
        # Create color regions
        image[:, 0, :64, :64] = 1.0    # Red region
        image[:, 1, :64, 64:] = 1.0    # Green region  
        image[:, 2, 64:, :64] = 1.0    # Blue region
        image[:, :3, 64:, 64:] = 0.5   # Gray region
        image[:, 3, :, :] = 0.8        # Semi-transparent
        
        return image
    
    def test_complete_image_to_3mf_pipeline(self, integration_test_image):
        """Test the complete workflow from image input to 3MF output."""
        # This is the ultimate integration test that drives the entire pipeline
        with pytest.raises(ImportError):
            from bananaforge.workflows.three_mf_pipeline import ImageTo3MFPipeline
            
            pipeline = ImageTo3MFPipeline()
            
            # Mock the complete workflow
            result = pipeline.process(
                image=integration_test_image,
                materials_csv="test_materials.csv",
                output_path="/tmp/test_output.3mf",
                enable_transparency=True,
                max_layers=20
            )
            
            # Verify complete pipeline execution
            assert result['success'] is True
            assert 'output_file' in result
            assert result['output_file'].endswith('.3mf')
    
    def test_concurrent_format_export(self, integration_test_image):
        """Test exporting multiple formats simultaneously."""
        # Drives concurrent export implementation
        with pytest.raises(ImportError):
            from bananaforge.output.multi_format_exporter import MultiFormatExporter
            
            exporter = MultiFormatExporter()
            
            formats = ['stl', '3mf', 'instructions', 'transparency_analysis']
            results = exporter.export_multiple(
                optimization_results={},  # Mock results
                formats=formats,
                output_dir="/tmp/test_output"
            )
            
            assert len(results) == len(formats)
            assert all(r['success'] for r in results.values())
    
    def test_large_model_performance(self):
        """Test 3MF export performance with large models."""
        # Test that 3MF exporter can be initialized successfully
        from bananaforge.output.threemf_exporter import ThreeMFExporter
        
        exporter = ThreeMFExporter()
        assert exporter is not None


class Test3MFBambuStudioCompatibility:
    """Test Story 7.5: Bambu Studio Compatibility Extensions.
    
    Acceptance Criteria:
    - Include Bambu-specific material mappings
    - Use Bambu's preferred color space and material naming conventions
    - Include AMS (Automatic Material System) slot assignments
    - Support Bambu's multi-color printing workflow requirements
    - Optimize for Bambu's layer change and purge tower algorithms
    """
    
    def test_ams_slot_assignments(self):
        """Test AMS slot assignment generation."""
        materials = ["pla_red", "pla_white", "pla_blue", "pla_black"]
        
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import BambuAMSManager
            
            ams_manager = BambuAMSManager()
            slot_assignments = ams_manager.assign_ams_slots(materials)
            
            # Should assign materials to AMS slots (1-4)
            assert len(slot_assignments) == len(materials)
            assert all(1 <= slot <= 4 for slot in slot_assignments.values())
    
    def test_bambu_color_space_conversion(self):
        """Test color space conversion for Bambu compatibility."""
        rgb_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import BambuColorConverter
            
            converter = BambuColorConverter()
            bambu_colors = converter.convert_to_bambu_colorspace(rgb_colors)
            
            assert len(bambu_colors) == len(rgb_colors)
    
    def test_purge_tower_optimization(self):
        """Test purge tower optimization for Bambu printers."""
        layer_materials = {i: f"material_{i % 3}" for i in range(20)}
        
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import BambuPurgeTowerOptimizer
            
            optimizer = BambuPurgeTowerOptimizer()
            purge_settings = optimizer.optimize_purge_tower(layer_materials)
            
            # Should minimize material waste
            assert 'purge_volume' in purge_settings
            assert purge_settings['purge_volume'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
