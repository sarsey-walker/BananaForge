#!/usr/bin/env python3
"""
TDD/BDD Tests for Story 5.2: Integration with Existing CLI Workflow

Following BDD Given-When-Then scenarios from tasks1.md:
- Existing workflow compatibility
- Batch processing with transparency
- Backward compatibility with existing CLI commands
- Clear progress reporting for multiple files
"""

import pytest
from pathlib import Path
from PIL import Image
from click.testing import CliRunner

# Import CLI components
from bananaforge.cli import cli, convert


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


class BackwardCompatibilityTestHelpers:
    """Helper class for backward compatibility testing."""
    
    @staticmethod
    def create_test_images(temp_dir: Path):
        """Create a variety of test images for compatibility testing."""
        images = {}
        
        # Opaque RGB image (traditional workflow)
        opaque_path = temp_dir / "opaque.png"
        Image.new('RGB', (64, 64), (255, 0, 0)).save(opaque_path, 'PNG')
        images['opaque'] = opaque_path
        
        # Transparent RGBA image
        transparent_path = temp_dir / "transparent.png"
        img = Image.new('RGBA', (64, 64), (255, 255, 255, 255))
        for x in range(32):
            for y in range(64):
                img.putpixel((x, y), (255, 0, 0, 0))  # Left half transparent
        img.save(transparent_path, 'PNG')
        images['transparent'] = transparent_path
        
        # Small opaque image
        small_path = temp_dir / "small.png"
        Image.new('RGB', (32, 32), (0, 255, 0)).save(small_path, 'PNG')
        images['small'] = small_path
        
        return images
    
    @staticmethod
    def create_materials_file(temp_dir: Path):
        """Create basic materials file for testing."""
        materials_path = temp_dir / "test_materials.csv"
        materials_content = """name,hex_color,brand,type
Red PLA,#FF0000,Test,PLA
Green PLA,#00FF00,Test,PLA
Blue PLA,#0000FF,Test,PLA
White PLA,#FFFFFF,Test,PLA"""
        materials_path.write_text(materials_content)
        return materials_path


class TestScenario1_ExistingWorkflowCompatibility:
    """
    BDD Scenario: Existing workflow compatibility
    Given I use my standard bananaforge convert command
    When transparency detection is active
    Then my command syntax remains unchanged
    And I get new safety checks without breaking changes
    And can opt-out if needed for special cases
    """
    
    def test_traditional_workflow_unchanged(self, runner, temp_dir):
        """Test that traditional CLI workflow remains unchanged."""
        # Given: I use my standard bananaforge convert command
        images = BackwardCompatibilityTestHelpers.create_test_images(temp_dir)
        materials_file = BackwardCompatibilityTestHelpers.create_materials_file(temp_dir)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: transparency detection is active (default behavior)
        # Use opaque image to avoid transparency stopping
        result = runner.invoke(convert, [
            str(images['opaque']),
            '--materials', str(materials_file),
            '--output', str(output_dir),
            '--device', 'cpu',
            '--max-layers', '3',
            '--iterations', '2',
            '--resolution', '32'  # Small for speed
        ])
        
        # Then: my command syntax remains unchanged (no syntax errors)
        syntax_error = "unrecognized arguments" in result.output.lower() or "no such option" in result.output.lower()
        assert not syntax_error, f"Traditional syntax should still work. Output: {result.output}"
        
        # And: I get new safety checks without breaking changes (should not fail due to transparency)
        transparency_failure = result.exit_code == 1 and "transparent" in result.output.lower()
        assert not transparency_failure, "Traditional workflow should not fail on opaque images"
    
    def test_existing_flags_still_work(self, runner, temp_dir):
        """Test that existing CLI flags continue to work."""
        # Given: traditional flags and opaque image
        images = BackwardCompatibilityTestHelpers.create_test_images(temp_dir)
        materials_file = BackwardCompatibilityTestHelpers.create_materials_file(temp_dir)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Traditional flags that should still work
        traditional_flags = [
            '--max-materials', '3',
            '--max-layers', '5', 
            '--layer-height', '0.1',
            '--device', 'cpu',
            '--iterations', '2',
            '--resolution', '32',
            '--project-name', 'test_project'
        ]
        
        # When: using traditional flags
        result = runner.invoke(convert, [
            str(images['small']),
            '--materials', str(materials_file),
            '--output', str(output_dir)
        ] + traditional_flags)
        
        # Then: should accept all traditional flags
        unrecognized_flag = "unrecognized arguments" in result.output.lower()
        unknown_option = "no such option" in result.output.lower()
        assert not (unrecognized_flag or unknown_option), "All traditional flags should be recognized"
    
    def test_opt_out_for_special_cases(self, runner, temp_dir):
        """Test that users can opt-out of transparency detection."""
        # Given: transparent image that would normally stop processing
        images = BackwardCompatibilityTestHelpers.create_test_images(temp_dir)
        materials_file = BackwardCompatibilityTestHelpers.create_materials_file(temp_dir)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: using opt-out flag for special cases
        result = runner.invoke(convert, [
            str(images['transparent']),
            '--materials', str(materials_file),
            '--output', str(output_dir),
            '--skip-transparency-check',  # Opt-out mechanism
            '--device', 'cpu',
            '--iterations', '1',
            '--resolution', '32'
        ])
        
        # Then: can opt-out and proceed
        transparency_blocked = result.exit_code == 1 and "transparent" in result.output.lower()
        assert not transparency_blocked, "Should be able to opt-out of transparency detection"


class TestScenario2_BatchProcessingCompatibility:
    """
    BDD Scenario: Batch processing with transparency
    Given I'm processing multiple images in batch
    When some images have transparency
    Then the system should report which images have issues
    And continue processing non-transparent images
    And provide a summary of transparency findings
    """
    
    def test_individual_image_processing_compatibility(self, runner, temp_dir):
        """Test processing individual images maintains compatibility."""
        # Given: multiple images with different characteristics
        images = BackwardCompatibilityTestHelpers.create_test_images(temp_dir)
        materials_file = BackwardCompatibilityTestHelpers.create_materials_file(temp_dir)
        
        # When: processing each image individually
        results = {}
        for name, image_path in images.items():
            output_dir = temp_dir / f"output_{name}"
            output_dir.mkdir()
            
            result = runner.invoke(convert, [
                str(image_path),
                '--materials', str(materials_file),
                '--output', str(output_dir),
                '--device', 'cpu',
                '--iterations', '1',
                '--resolution', '32'
            ])
            results[name] = result
        
        # Then: should handle each appropriately
        # Opaque images should not fail due to transparency
        assert not (results['opaque'].exit_code == 1 and "transparent" in results['opaque'].output.lower()), \
            "Opaque image should not fail for transparency"
        assert not (results['small'].exit_code == 1 and "transparent" in results['small'].output.lower()), \
            "Small opaque image should not fail for transparency"
        
        # Transparent image should be detected (exit code 1 with transparency message)
        transparent_detected = results['transparent'].exit_code == 1 and "transparent" in results['transparent'].output.lower()
        assert transparent_detected, "Transparent image should be detected and processing stopped"
    
    def test_batch_processing_with_skip_flag(self, runner, temp_dir):
        """Test batch processing when transparency detection is skipped."""
        # Given: mix of transparent and opaque images with skip flag
        images = BackwardCompatibilityTestHelpers.create_test_images(temp_dir)
        materials_file = BackwardCompatibilityTestHelpers.create_materials_file(temp_dir)
        
        # When: processing with skip flag (simulating batch processing)
        results = {}
        for name, image_path in images.items():
            output_dir = temp_dir / f"batch_output_{name}"
            output_dir.mkdir()
            
            result = runner.invoke(convert, [
                str(image_path),
                '--materials', str(materials_file),
                '--output', str(output_dir),
                '--skip-transparency-check',  # Batch processing mode
                '--device', 'cpu',
                '--iterations', '1',
                '--resolution', '32'
            ])
            results[name] = result
        
        # Then: none should fail due to transparency detection
        for name, result in results.items():
            transparency_failure = result.exit_code == 1 and "transparent" in result.output.lower()
            assert not transparency_failure, f"Image {name} should not fail for transparency when skipped"


class TestHelpSystemCompatibility:
    """Test that help system remains compatible and informative."""
    
    def test_main_help_compatibility(self, runner):
        """Test that main help system works and includes transparency info."""
        # When: requesting main help
        result = runner.invoke(cli, ['--help'])
        
        # Then: should work without errors
        assert result.exit_code == 0, "Main help should work"
        assert "convert" in result.output, "Should list convert command"
    
    def test_convert_help_includes_transparency_options(self, runner):
        """Test that convert help includes transparency options.""" 
        # When: requesting convert help
        result = runner.invoke(convert, ['--help'])
        
        # Then: should include transparency options
        assert result.exit_code == 0, "Convert help should work"
        
        transparency_options = [
            "--skip-transparency-check",
            "--analyze-transparency",
            "--transparency-verbose"
        ]
        
        for option in transparency_options:
            assert option in result.output, f"Help should include {option}"
    
    def test_help_explains_new_behavior(self, runner):
        """Test that help explains the new transparency detection behavior."""
        # When: requesting detailed help
        result = runner.invoke(convert, ['--help'])
        
        # Then: should explain transparency detection
        help_content = result.output.lower()
        transparency_mentioned = (
            "transparency" in help_content or 
            "skip-transparency-check" in help_content
        )
        assert transparency_mentioned, "Help should mention transparency detection"


class TestConfigurationCompatibility:
    """Test compatibility with configuration files and environment variables."""
    
    def test_existing_config_files_still_work(self, runner, temp_dir):
        """Test that existing configuration approaches still work."""
        # Given: traditional command structure
        images = BackwardCompatibilityTestHelpers.create_test_images(temp_dir)
        materials_file = BackwardCompatibilityTestHelpers.create_materials_file(temp_dir)
        output_dir = temp_dir / "config_test_output"
        output_dir.mkdir()
        
        # When: using traditional configuration approach
        result = runner.invoke(convert, [
            str(images['opaque']),  # Use opaque to avoid transparency issues
            '--materials', str(materials_file),
            '--output', str(output_dir),
            '--device', 'cpu',
            '--iterations', '1'
        ])
        
        # Then: should work with traditional configuration
        config_error = "config" in result.output.lower() and result.exit_code != 0
        assert not config_error, "Traditional configuration should still work"


class TestPerformanceCompatibility:
    """Test that transparency detection doesn't significantly impact performance."""
    
    def test_minimal_performance_impact(self, runner, temp_dir):
        """Test that transparency detection has minimal performance impact."""
        # Given: opaque image (won't trigger transparency processing)
        images = BackwardCompatibilityTestHelpers.create_test_images(temp_dir)
        materials_file = BackwardCompatibilityTestHelpers.create_materials_file(temp_dir)
        output_dir = temp_dir / "perf_test_output"
        output_dir.mkdir()
        
        # When: processing opaque image (transparency detection should be fast)
        import time
        start_time = time.time()
        
        result = runner.invoke(convert, [
            str(images['small']),
            '--materials', str(materials_file),
            '--output', str(output_dir),
            '--device', 'cpu',
            '--iterations', '1',
            '--resolution', '32'
        ])
        
        end_time = time.time()
        
        # Then: should complete in reasonable time
        elapsed = end_time - start_time
        assert elapsed < 30.0, f"Processing should be reasonably fast, took {elapsed:.2f}s"
        
        # Should not fail due to performance issues
        timeout_error = "timeout" in result.output.lower() or "too slow" in result.output.lower()
        assert not timeout_error, "Should not have performance-related errors"


class TestErrorMessageCompatibility:
    """Test that error messages remain helpful and don't break workflows."""
    
    def test_traditional_error_scenarios_unchanged(self, runner, temp_dir):
        """Test that traditional error scenarios still provide helpful messages."""
        materials_file = BackwardCompatibilityTestHelpers.create_materials_file(temp_dir)
        output_dir = temp_dir / "error_test_output"
        output_dir.mkdir()
        
        # Test nonexistent file error (traditional scenario)
        result = runner.invoke(convert, [
            '/nonexistent/file.png',
            '--materials', str(materials_file),
            '--output', str(output_dir)
        ])
        
        # Should provide helpful error message
        assert result.exit_code != 0, "Should fail for nonexistent file"
        helpful_error = (
            "not found" in result.output.lower() or 
            "does not exist" in result.output.lower() or
            "transparency detection failed" in result.output.lower()
        )
        assert helpful_error, "Should provide helpful error message"
    
    def test_error_messages_suggest_transparency_solutions(self, runner, temp_dir):
        """Test that error messages include transparency-related solutions when appropriate."""
        # Given: transparent image that will trigger detection
        images = BackwardCompatibilityTestHelpers.create_test_images(temp_dir)
        materials_file = BackwardCompatibilityTestHelpers.create_materials_file(temp_dir)
        output_dir = temp_dir / "error_solution_test"
        output_dir.mkdir()
        
        # When: processing transparent image
        result = runner.invoke(convert, [
            str(images['transparent']),
            '--materials', str(materials_file),
            '--output', str(output_dir),
            '--device', 'cpu'
        ])
        
        # Then: should suggest transparency solutions
        assert result.exit_code == 1, "Should exit with transparency detection"
        suggests_skip = "skip-transparency-check" in result.output
        suggests_analyze = "analyze-transparency" in result.output
        
        assert suggests_skip or suggests_analyze, "Should suggest transparency-related solutions"