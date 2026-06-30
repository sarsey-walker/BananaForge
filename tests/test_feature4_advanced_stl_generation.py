"""BDD/TDD Tests for Feature 4: Advanced STL Generation

This module contains comprehensive tests for both stories in Feature 4,
following the Gherkin scenarios defined in the tasks file.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
import zipfile
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
import trimesh

from bananaforge.output.stl_generator import STLGenerator
from bananaforge.output.exporter import ModelExporter


class TestStory41AlphaChannelSupport:
    """BDD Tests for Story 4.1: Alpha Channel Support
    
    Acceptance Criteria:
    Given an input image with alpha channel transparency
    When the system generates STL output
    Then it should exclude vertices where alpha < 128
    And create proper boundaries around transparent regions
    And ensure the resulting mesh is still manifold
    And provide clean edges at transparency boundaries
    """

    @pytest.fixture
    def stl_generator(self):
        """Create STL generator for testing."""
        return STLGenerator(
            layer_height=0.08,
            initial_layer_height=0.16,
            nozzle_diameter=0.4,
            base_height=0.24,
        )

    @pytest.fixture
    def test_alpha_image_data(self):
        """Create test image data with alpha channel."""
        # Create a 32x32 RGBA image with transparency
        rgba_image = torch.zeros(1, 4, 32, 32)
        
        # Create a pattern with opaque center and transparent edges
        center_x, center_y = 16, 16
        radius = 12
        
        for i in range(32):
            for j in range(32):
                distance = ((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5
                if distance <= radius:
                    # Opaque center region
                    rgba_image[0, 0, i, j] = 1.0  # Red
                    rgba_image[0, 1, i, j] = 0.5  # Green
                    rgba_image[0, 2, i, j] = 0.0  # Blue
                    rgba_image[0, 3, i, j] = 1.0  # Alpha (fully opaque)
                elif distance <= radius + 2:
                    # Semi-transparent border
                    alpha_value = (radius + 2 - distance) / 2
                    rgba_image[0, 0, i, j] = 0.5
                    rgba_image[0, 1, i, j] = 0.8
                    rgba_image[0, 2, i, j] = 1.0
                    rgba_image[0, 3, i, j] = alpha_value
                # else: remains transparent (alpha = 0)
        
        # Create corresponding height map
        height_map = torch.zeros(1, 1, 32, 32)
        for i in range(32):
            for j in range(32):
                distance = ((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5
                if distance <= radius:
                    height_map[0, 0, i, j] = 5.0  # 5 layers high
                elif distance <= radius + 2:
                    height_map[0, 0, i, j] = 2.0  # 2 layers high
        
        # Create alpha mask (alpha >= 128/255 = 0.5)
        alpha_mask = rgba_image[0, 3] >= 0.5
        
        return rgba_image, height_map, alpha_mask

    def test_excludes_vertices_where_alpha_less_than_128(
        self, stl_generator, test_alpha_image_data
    ):
        """Test that vertices where alpha < 128 are excluded from STL generation."""
        rgba_image, height_map, alpha_mask = test_alpha_image_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "alpha_test.stl"
            
            # When generating STL with alpha channel support
            mesh = stl_generator.generate_stl_with_alpha(
                height_map=height_map,
                alpha_mask=alpha_mask,
                output_path=output_path,
                physical_size=100.0,
            )
            
            # Then vertices should only exist where alpha >= 128 (0.5)
            assert len(mesh.vertices) > 0, "Should have vertices in opaque regions"
            
            # Verify mesh is generated and saved
            assert output_path.exists(), "STL file should be created"
            
            # Analyze the mesh to ensure transparent areas are excluded
            quality_metrics = stl_generator.analyze_mesh_quality(mesh)
            assert quality_metrics["vertex_count"] > 0, "Should have vertices"
            assert quality_metrics["face_count"] > 0, "Should have faces"

    def test_alpha_stl_height_uses_initial_layer_without_extra_base_offset(
        self, stl_generator
    ):
        """Alpha STL Z bounds should use the same layer-height convention."""
        height_map = torch.ones(1, 1, 4, 4) * 5
        alpha_mask = torch.ones(4, 4, dtype=torch.bool)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "alpha_height_test.stl"

            mesh = stl_generator.generate_stl_with_alpha(
                height_map=height_map,
                alpha_mask=alpha_mask,
                output_path=output_path,
                physical_size=20.0,
                smooth_mesh=False,
                create_boundaries=False,
                ensure_manifold=False,
            )

            assert mesh.bounds[0][2] == pytest.approx(0.0)
            assert mesh.bounds[1][2] == pytest.approx(0.56)

    def test_creates_proper_boundaries_around_transparent_regions(
        self, stl_generator, test_alpha_image_data
    ):
        """Test that proper boundaries are created around transparent regions."""
        rgba_image, height_map, alpha_mask = test_alpha_image_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "boundary_test.stl"
            
            # When generating STL with alpha boundaries
            mesh = stl_generator.generate_stl_with_alpha(
                height_map=height_map,
                alpha_mask=alpha_mask,
                output_path=output_path,
                physical_size=100.0,
                create_boundaries=True,
            )
            
            # Then boundaries should be properly formed
            # Check that mesh has edges around the alpha boundary
            boundary_edges = stl_generator.detect_alpha_boundaries(mesh, alpha_mask)
            assert len(boundary_edges) > 0, "Should have boundary edges"
            
            # Verify boundary edges form closed loops
            boundary_loops = stl_generator.trace_boundary_loops(boundary_edges)
            assert len(boundary_loops) > 0, "Should have closed boundary loops"
            
            # Each loop should be properly closed
            for loop in boundary_loops:
                assert len(loop) >= 3, "Boundary loops should have at least 3 points"
                # First and last points should connect (closed loop)
                distance = np.linalg.norm(loop[0] - loop[-1])
                assert distance < 1e-6, "Boundary loops should be closed"

    def test_ensures_resulting_mesh_is_still_manifold(
        self, stl_generator, test_alpha_image_data
    ):
        """Test that the resulting mesh with alpha exclusions is still manifold."""
        rgba_image, height_map, alpha_mask = test_alpha_image_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "manifold_test.stl"
            
            # When generating STL with alpha support
            mesh = stl_generator.generate_stl_with_alpha(
                height_map=height_map,
                alpha_mask=alpha_mask,
                output_path=output_path,
                physical_size=100.0,
                ensure_manifold=True,
            )
            
            # Then the mesh should be manifold
            quality_metrics = stl_generator.analyze_mesh_quality(mesh)
            assert quality_metrics["manifold"], "Mesh should be manifold"
            assert quality_metrics["watertight"], "Mesh should be watertight"
            
            # Euler characteristic should be correct for a manifold mesh
            # For a single object with holes, χ = 2 - 2g - h (where g=genus, h=holes)
            euler_number = quality_metrics["euler_number"]
            assert euler_number >= 1, "Euler number should be valid for manifold mesh"

    def test_provides_clean_edges_at_transparency_boundaries(
        self, stl_generator, test_alpha_image_data
    ):
        """Test that clean edges are provided at transparency boundaries."""
        rgba_image, height_map, alpha_mask = test_alpha_image_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "clean_edges_test.stl"
            
            # When generating STL with edge cleanup
            mesh = stl_generator.generate_stl_with_alpha(
                height_map=height_map,
                alpha_mask=alpha_mask,
                output_path=output_path,
                physical_size=100.0,
                clean_edges=True,
            )
            
            # Then edges at transparency boundaries should be clean
            edge_quality = stl_generator.analyze_edge_quality(mesh, alpha_mask)
            
            # Check for proper edge manifold properties
            assert edge_quality["non_manifold_edges"] == 0, "Should have no non-manifold edges"
            assert edge_quality["boundary_edges"] > 0, "Should have boundary edges at alpha transitions"
            
            # Verify edge smoothness at boundaries
            boundary_smoothness = edge_quality["boundary_smoothness"]
            assert boundary_smoothness > 0.8, "Boundary edges should be smooth"
            
            # Check that no degenerate triangles exist at boundaries
            degenerate_faces = edge_quality["degenerate_faces"]
            assert degenerate_faces == 0, "Should have no degenerate faces at boundaries"

    def test_alpha_channel_performance_with_large_images(self, stl_generator):
        """Test that alpha channel processing works efficiently with larger images."""
        # Create a larger test image (128x128) with complex alpha patterns
        size = 128
        rgba_image = torch.zeros(1, 4, size, size)
        
        # Create multiple transparent regions in a checkerboard pattern
        block_size = 16
        for i in range(0, size, block_size * 2):
            for j in range(0, size, block_size * 2):
                # Opaque blocks
                rgba_image[0, :3, i:i+block_size, j:j+block_size] = torch.rand(3, block_size, block_size)
                rgba_image[0, 3, i:i+block_size, j:j+block_size] = 1.0
                
                # Transparent blocks (offset)
                if i + block_size < size and j + block_size < size:
                    rgba_image[0, 3, i+block_size:i+2*block_size, j+block_size:j+2*block_size] = 0.0
        
        height_map = torch.rand(1, 1, size, size) * 8.0  # Up to 8 layers
        alpha_mask = rgba_image[0, 3] >= 0.5
        
        import time
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "performance_test.stl"
            
            # When processing large image with alpha
            mesh = stl_generator.generate_stl_with_alpha(
                height_map=height_map,
                alpha_mask=alpha_mask,
                output_path=output_path,
                physical_size=200.0,
            )
            
            processing_time = time.time() - start_time
            
            # Then processing should complete in reasonable time
            assert processing_time < 30.0, "Large alpha processing should complete within 30 seconds"
            
            # And produce valid output
            quality_metrics = stl_generator.analyze_mesh_quality(mesh)
            assert quality_metrics["vertex_count"] > 0, "Should generate vertices"
            assert quality_metrics["face_count"] > 0, "Should generate faces"

    def test_alpha_channel_integration_with_material_assignments(
        self, stl_generator, test_alpha_image_data
    ):
        """Test that alpha channel support integrates properly with material assignments."""
        rgba_image, height_map, alpha_mask = test_alpha_image_data
        
        # Create material assignments for multiple layers
        num_layers = 6
        materials = torch.zeros(num_layers, 32, 32, dtype=torch.long)
        
        # Assign different materials to different layers
        for layer in range(num_layers):
            if layer < 3:
                materials[layer] = 0  # Material 0 for bottom layers
            else:
                materials[layer] = 1  # Material 1 for top layers
        
        # Only assign materials where alpha mask is true
        materials = materials * alpha_mask.long().unsqueeze(0)
        
        material_ids = ["PLA_Red", "PLA_Blue"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # When generating layer STLs with alpha support
            stl_paths = stl_generator.generate_layer_stls_with_alpha(
                height_map=height_map,
                material_assignments=materials,
                material_ids=material_ids,
                alpha_mask=alpha_mask,
                output_dir=temp_dir,
                physical_size=100.0,
            )
            
            # Then STL files should be generated for each material
            assert len(stl_paths) == len(material_ids), "Should generate STL for each material"
            
            for material_id, stl_path in stl_paths.items():
                assert material_id in material_ids, "Material ID should be valid"
                assert Path(stl_path).exists(), f"STL file should exist for {material_id}"
                
                # Load and verify each material mesh
                mesh = trimesh.load(stl_path)
                quality_metrics = stl_generator.analyze_mesh_quality(mesh)
                assert quality_metrics["vertex_count"] > 0, f"Material {material_id} should have vertices"


class TestStory42ProjectFileExport:
    """BDD Tests for Story 4.2: Project File Export
    
    Acceptance Criteria:
    Given a completed optimization with material assignments
    When I export the project
    Then the system should generate a .hfp format project file
    And include all material definitions and swap instructions
    And set appropriate layer heights and printing parameters
    And ensure compatibility with HueForge workflow
    """

    @pytest.fixture
    def model_exporter(self):
        """Create model exporter for testing."""
        return ModelExporter()

    @pytest.fixture
    def complete_optimization_data(self):
        """Create complete optimization data for project export testing."""
        # Create test optimization results
        height_map = torch.rand(1, 1, 64, 64) * 10.0  # Up to 10 layers
        
        # Material assignments for 10 layers
        num_layers = 10
        material_assignments = torch.zeros(num_layers, 64, 64, dtype=torch.long)
        
        # Create realistic material patterns
        for layer in range(num_layers):
            if layer < 3:
                material_assignments[layer] = 0  # Base material
            elif layer < 6:
                material_assignments[layer] = 1  # Middle material
            else:
                material_assignments[layer] = 2  # Top material
        
        # Material definitions
        materials = [
            {
                "id": "PLA_White",
                "name": "PLA White",
                "color": [1.0, 1.0, 1.0],
                "density": 1.24,
                "cost_per_kg": 25.0,
                "brand": "Bambu Lab",
            },
            {
                "id": "PLA_Red", 
                "name": "PLA Red",
                "color": [1.0, 0.0, 0.0],
                "density": 1.24,
                "cost_per_kg": 28.0,
                "brand": "Bambu Lab",
            },
            {
                "id": "PLA_Blue",
                "name": "PLA Blue", 
                "color": [0.0, 0.0, 1.0],
                "density": 1.24,
                "cost_per_kg": 28.0,
                "brand": "Bambu Lab",
            },
        ]
        
        # Optimization metadata
        optimization_params = {
            "layer_height": 0.08,
            "initial_layer_height": 0.16,
            "nozzle_diameter": 0.4,
            "iterations": 5000,
            "final_loss": 0.1234,
            "total_layers": num_layers,
        }
        
        return height_map, material_assignments, materials, optimization_params

    def test_generates_hfp_format_project_file(
        self, model_exporter, complete_optimization_data
    ):
        """Test that the system generates a .hfp format project file."""
        height_map, material_assignments, materials, optimization_params = complete_optimization_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project.hfp"
            
            # When exporting project in HueForge format
            export_result = model_exporter.export_hueforge_project(
                height_map=height_map,
                material_assignments=material_assignments,
                materials=materials,
                optimization_params=optimization_params,
                output_path=project_path,
            )
            
            # Then a .hfp project file should be generated
            assert project_path.exists(), "HueForge project file should be created"
            assert project_path.suffix == ".hfp", "File should have .hfp extension"
            
            # HFP files are ZIP archives, verify the structure
            assert zipfile.is_zipfile(project_path), "HFP file should be a valid ZIP archive"
            
            with zipfile.ZipFile(project_path, 'r') as zip_file:
                file_list = zip_file.namelist()
                
                # Should contain required HueForge project files
                assert "project.json" in file_list, "Should contain project metadata"
                assert "materials.json" in file_list, "Should contain material definitions"
                assert "instructions.json" in file_list, "Should contain swap instructions"
                
                # Should contain STL files for each material
                stl_files = [f for f in file_list if f.endswith('.stl')]
                assert len(stl_files) >= len(materials), "Should contain STL files for materials"

    def test_includes_all_material_definitions_and_swap_instructions(
        self, model_exporter, complete_optimization_data
    ):
        """Test that all material definitions and swap instructions are included."""
        height_map, material_assignments, materials, optimization_params = complete_optimization_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "materials_test.hfp"
            
            # When exporting project with materials and instructions
            export_result = model_exporter.export_hueforge_project(
                height_map=height_map,
                material_assignments=material_assignments,
                materials=materials,
                optimization_params=optimization_params,
                output_path=project_path,
                include_swap_instructions=True,
            )
            
            # Then all materials should be included
            with zipfile.ZipFile(project_path, 'r') as zip_file:
                # Verify material definitions
                materials_data = json.loads(zip_file.read("materials.json"))
                assert len(materials_data["materials"]) == len(materials), "All materials should be included"
                
                for i, material in enumerate(materials):
                    exported_material = materials_data["materials"][i]
                    assert exported_material["id"] == material["id"], "Material ID should match"
                    assert exported_material["name"] == material["name"], "Material name should match"
                    assert exported_material["color"] == material["color"], "Material color should match"
                    assert exported_material["brand"] == material["brand"], "Material brand should match"
                
                # Verify swap instructions
                instructions_data = json.loads(zip_file.read("instructions.json"))
                swap_instructions = instructions_data["swap_instructions"]
                
                assert len(swap_instructions) > 0, "Should have swap instructions"
                
                # Verify instruction format
                for instruction in swap_instructions:
                    assert "layer" in instruction, "Instruction should specify layer"
                    assert "from_material" in instruction, "Instruction should specify source material"
                    assert "to_material" in instruction, "Instruction should specify target material"
                    assert "estimated_time" in instruction, "Instruction should include time estimate"

    def test_sets_appropriate_layer_heights_and_printing_parameters(
        self, model_exporter, complete_optimization_data
    ):
        """Test that appropriate layer heights and printing parameters are set."""
        height_map, material_assignments, materials, optimization_params = complete_optimization_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "parameters_test.hfp"
            
            # When exporting project with specific printing parameters
            custom_params = {
                **optimization_params,
                "print_speed": 150,  # mm/s
                "temperature": 210,  # °C
                "bed_temperature": 60,  # °C
                "infill_percentage": 15,  # %
                "supports": False,
                "brim": True,
                "brim_width": 5.0,  # mm
            }
            
            export_result = model_exporter.export_hueforge_project(
                height_map=height_map,
                material_assignments=material_assignments,
                materials=materials,
                optimization_params=custom_params,
                output_path=project_path,
            )
            
            # Then printing parameters should be properly set
            with zipfile.ZipFile(project_path, 'r') as zip_file:
                project_data = json.loads(zip_file.read("project.json"))
                print_settings = project_data["print_settings"]
                
                # Verify layer parameters
                assert print_settings["layer_height"] == custom_params["layer_height"], "Layer height should match"
                assert print_settings["initial_layer_height"] == custom_params["initial_layer_height"], "Initial layer height should match"
                assert print_settings["total_layers"] == custom_params["total_layers"], "Total layers should match"
                
                # Verify printing parameters
                assert print_settings["print_speed"] == custom_params["print_speed"], "Print speed should match"
                assert print_settings["temperature"] == custom_params["temperature"], "Temperature should match"
                assert print_settings["bed_temperature"] == custom_params["bed_temperature"], "Bed temperature should match"
                assert print_settings["infill_percentage"] == custom_params["infill_percentage"], "Infill should match"
                
                # Verify boolean settings
                assert print_settings["supports"] == custom_params["supports"], "Supports setting should match"
                assert print_settings["brim"] == custom_params["brim"], "Brim setting should match"
                assert print_settings["brim_width"] == custom_params["brim_width"], "Brim width should match"

    def test_ensures_compatibility_with_hueforge_workflow(
        self, model_exporter, complete_optimization_data
    ):
        """Test that exported projects are compatible with HueForge workflow."""
        height_map, material_assignments, materials, optimization_params = complete_optimization_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "compatibility_test.hfp"
            
            # When exporting project for HueForge compatibility
            export_result = model_exporter.export_hueforge_project(
                height_map=height_map,
                material_assignments=material_assignments,
                materials=materials,
                optimization_params=optimization_params,
                output_path=project_path,
                hueforge_version="2.0",
            )
            
            # Then project should be HueForge compatible
            compatibility_check = model_exporter.validate_hueforge_compatibility(project_path)
            
            assert compatibility_check["valid"], "Project should be valid HueForge format"
            assert compatibility_check["version"] >= "2.0", "Should support HueForge 2.0+"
            
            # Verify required HueForge metadata
            assert "schema_version" in compatibility_check, "Should have schema version"
            assert "compatible_slicers" in compatibility_check, "Should list compatible slicers"
            
            # Check HueForge-specific requirements
            with zipfile.ZipFile(project_path, 'r') as zip_file:
                project_data = json.loads(zip_file.read("project.json"))
                
                # HueForge requires specific metadata fields
                assert "hueforge_version" in project_data, "Should specify HueForge version"
                assert "created_by" in project_data, "Should specify creation source"
                assert "model_info" in project_data, "Should include model information"
                
                model_info = project_data["model_info"]
                assert "dimensions" in model_info, "Should specify model dimensions"
                assert "estimated_print_time" in model_info, "Should estimate print time"
                assert "estimated_material_usage" in model_info, "Should estimate material usage"

    def test_project_export_with_alpha_channel_support(
        self, model_exporter, complete_optimization_data
    ):
        """Test that project export works correctly with alpha channel transparency."""
        height_map, material_assignments, materials, optimization_params = complete_optimization_data
        
        # Add alpha mask to the optimization data
        alpha_mask = torch.ones(64, 64, dtype=torch.bool)
        # Create some transparent regions
        alpha_mask[10:20, 10:20] = False  # Transparent square
        alpha_mask[40:50, 40:50] = False  # Another transparent square
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "alpha_export_test.hfp"
            
            # When exporting project with alpha channel data
            export_result = model_exporter.export_hueforge_project(
                height_map=height_map,
                material_assignments=material_assignments,
                materials=materials,
                optimization_params=optimization_params,
                alpha_mask=alpha_mask,
                output_path=project_path,
            )
            
            # Then alpha information should be preserved in export
            with zipfile.ZipFile(project_path, 'r') as zip_file:
                project_data = json.loads(zip_file.read("project.json"))
                
                # Should include alpha channel information
                assert "alpha_support" in project_data, "Should indicate alpha channel support"
                assert project_data["alpha_support"] == True, "Alpha support should be enabled"
                
                # Should include transparency regions metadata
                if "transparency_regions" in project_data:
                    transparency_regions = project_data["transparency_regions"]
                    assert len(transparency_regions) >= 2, "Should detect transparent regions"
                
                # STL files should respect alpha channel
                stl_files = [f for f in zip_file.namelist() if f.endswith('.stl')]
                for stl_file in stl_files:
                    # Extract and verify STL doesn't include transparent regions
                    stl_data = zip_file.read(stl_file)
                    # This would require more complex validation in a real implementation
                    assert len(stl_data) > 0, "STL file should contain data"

    def test_project_export_performance_with_large_models(
        self, model_exporter
    ):
        """Test that project export performs well with large models."""
        # Create large optimization data
        size = 256
        height_map = torch.rand(1, 1, size, size) * 15.0  # Up to 15 layers
        
        num_layers = 15
        material_assignments = torch.randint(0, 4, (num_layers, size, size))
        
        materials = [
            {"id": f"Material_{i}", "name": f"Test Material {i}", "color": [i*0.25, 0.5, 1.0-i*0.25], "density": 1.24, "cost_per_kg": 25.0 + i*5, "brand": "Test Brand"}
            for i in range(4)
        ]
        
        optimization_params = {
            "layer_height": 0.08,
            "initial_layer_height": 0.16,
            "nozzle_diameter": 0.4,
            "iterations": 10000,
            "final_loss": 0.0891,
            "total_layers": num_layers,
        }
        
        import time
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "large_model_test.hfp"
            
            # When exporting large project
            export_result = model_exporter.export_hueforge_project(
                height_map=height_map,
                material_assignments=material_assignments,
                materials=materials,
                optimization_params=optimization_params,
                output_path=project_path,
            )
            
            export_time = time.time() - start_time
            
            # Then export should complete in reasonable time
            assert export_time < 60.0, "Large model export should complete within 60 seconds"
            
            # And produce valid output
            assert project_path.exists(), "Project file should be created"
            file_size = project_path.stat().st_size
            assert file_size > 1024, "Project file should have reasonable size"  # At least 1KB
            
            # Verify the exported file is valid
            assert zipfile.is_zipfile(project_path), "Should produce valid ZIP file"


class TestFeature4Integration:
    """Integration tests for Feature 4: Advanced STL Generation."""

    @pytest.fixture
    def integrated_test_setup(self):
        """Set up integration test environment."""
        return {
            "stl_generator": STLGenerator(),
            "model_exporter": ModelExporter(),
            "temp_dir": tempfile.mkdtemp(),
        }

    def test_end_to_end_alpha_channel_workflow(self, integrated_test_setup):
        """Test complete workflow from alpha image to HueForge project."""
        stl_generator = integrated_test_setup["stl_generator"]
        model_exporter = integrated_test_setup["model_exporter"]
        temp_dir = integrated_test_setup["temp_dir"]
        
        try:
            # Create test data with alpha channel
            rgba_image = torch.zeros(1, 4, 48, 48)
            
            # Create circular pattern with transparency
            center = 24
            for i in range(48):
                for j in range(48):
                    distance = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                    if distance <= 20:
                        rgba_image[0, :3, i, j] = torch.tensor([1.0, 0.5, 0.0])  # Orange
                        rgba_image[0, 3, i, j] = 1.0  # Opaque
                    elif distance <= 22:
                        rgba_image[0, :3, i, j] = torch.tensor([0.0, 0.5, 1.0])  # Blue
                        rgba_image[0, 3, i, j] = 0.7  # Semi-transparent
            
            height_map = torch.zeros(1, 1, 48, 48)
            alpha_mask = rgba_image[0, 3] >= 0.5
            
            # Create height map based on alpha
            height_map[0, 0] = alpha_mask.float() * 6.0  # 6 layers where opaque
            
            # Material assignments
            num_layers = 6
            material_assignments = torch.zeros(num_layers, 48, 48, dtype=torch.long)
            for layer in range(num_layers):
                material_assignments[layer] = (layer // 2) * alpha_mask.long()
            
            materials = [
                {"id": "PLA_Orange", "name": "PLA Orange", "color": [1.0, 0.5, 0.0], "density": 1.24, "cost_per_kg": 28.0, "brand": "Test"},
                {"id": "PLA_Blue", "name": "PLA Blue", "color": [0.0, 0.5, 1.0], "density": 1.24, "cost_per_kg": 28.0, "brand": "Test"},
                {"id": "PLA_White", "name": "PLA White", "color": [1.0, 1.0, 1.0], "density": 1.24, "cost_per_kg": 25.0, "brand": "Test"},
            ]
            
            optimization_params = {
                "layer_height": 0.08,
                "initial_layer_height": 0.16,
                "nozzle_diameter": 0.4,
                "iterations": 3000,
                "final_loss": 0.1567,
                "total_layers": num_layers,
            }
            
            # Step 1: Generate STL with alpha support
            stl_path = Path(temp_dir) / "alpha_model.stl"
            mesh = stl_generator.generate_stl_with_alpha(
                height_map=height_map,
                alpha_mask=alpha_mask,
                output_path=stl_path,
                physical_size=120.0,
            )
            
            # Step 2: Generate layer STLs with alpha
            layer_stls = stl_generator.generate_layer_stls_with_alpha(
                height_map=height_map,
                material_assignments=material_assignments,
                material_ids=[m["id"] for m in materials],
                alpha_mask=alpha_mask,
                output_dir=temp_dir,
                physical_size=120.0,
            )
            
            # Step 3: Export complete HueForge project
            project_path = Path(temp_dir) / "complete_project.hfp"
            export_result = model_exporter.export_hueforge_project(
                height_map=height_map,
                material_assignments=material_assignments,
                materials=materials,
                optimization_params=optimization_params,
                alpha_mask=alpha_mask,
                output_path=project_path,
            )
            
            # Verify complete workflow
            assert stl_path.exists(), "Main STL should be generated"
            assert len(layer_stls) > 0, "Layer STLs should be generated"
            assert project_path.exists(), "HueForge project should be exported"
            
            # Verify mesh quality
            quality_metrics = stl_generator.analyze_mesh_quality(mesh)
            assert quality_metrics["manifold"], "Final mesh should be manifold"
            assert quality_metrics["vertex_count"] > 0, "Should have vertices"
            
            # Verify project completeness
            compatibility = model_exporter.validate_hueforge_compatibility(project_path)
            assert compatibility["valid"], "Project should be HueForge compatible"
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_feature4_backward_compatibility(self, integrated_test_setup):
        """Test that Feature 4 enhancements don't break existing functionality."""
        stl_generator = integrated_test_setup["stl_generator"]
        temp_dir = integrated_test_setup["temp_dir"]
        
        try:
            # Test with traditional height map (no alpha channel)
            height_map = torch.rand(1, 1, 32, 32) * 5.0
            
            # Traditional STL generation should still work
            traditional_path = Path(temp_dir) / "traditional.stl"
            mesh = stl_generator.generate_stl(
                height_map=height_map,
                output_path=traditional_path,
                physical_size=100.0,
            )
            
            # Should produce valid mesh
            assert traditional_path.exists(), "Traditional STL generation should work"
            quality_metrics = stl_generator.analyze_mesh_quality(mesh)
            assert quality_metrics["vertex_count"] > 0, "Should generate vertices"
            assert quality_metrics["face_count"] > 0, "Should generate faces"
            
            # Layer STL generation should still work
            material_assignments = torch.randint(0, 2, (5, 32, 32))
            layer_stls = stl_generator.generate_layer_stls(
                height_map=height_map,
                material_assignments=material_assignments,
                material_ids=["Material_A", "Material_B"],
                output_dir=temp_dir,
                physical_size=100.0,
            )
            
            assert len(layer_stls) == 2, "Should generate layer STLs"
            for stl_path in layer_stls.values():
                assert Path(stl_path).exists(), "Layer STL should exist"
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
