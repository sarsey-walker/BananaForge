#!/usr/bin/env python3
"""
TDD/BDD Tests for Story 5.1: CLI Flag for Transparency Behavior

Following BDD Given-When-Then scenarios from tasks1.md:
- Auto-detect transparency (default)
- Skip transparency check
- Force transparency analysis
- Clear flag documentation and help text
"""

import pytest
import subprocess
import sys
import tempfile
from pathlib import Path
from PIL import Image
from click.testing import CliRunner

# Import the CLI components we're testing
from bananaforge.cli import cli, convert


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class CLITestHelpers:
    """Helper class for creating test images and CLI scenarios."""
    
    @staticmethod
    def create_transparent_image(path: Path):
        """Create PNG image with transparency."""
        image = Image.new('RGBA', (100, 100), (255, 255, 255, 255))
        # Add transparent area (50% of pixels)
        for x in range(50, 100):
            for y in range(100):
                image.putpixel((x, y), (255, 0, 0, 0))  # Transparent
        image.save(path, 'PNG')
        return path
    
    @staticmethod  
    def create_opaque_image(path: Path):
        """Create PNG image without transparency."""
        image = Image.new('RGB', (100, 100), (255, 0, 0))  # Opaque red
        image.save(path, 'PNG')
        return path
    
    @staticmethod
    def create_materials_csv(path: Path):
        """Create basic materials CSV for testing."""
        csv_content = """name,hex_color,brand,type
Red PLA,#FF0000,Generic,PLA
Green PLA,#00FF00,Generic,PLA
Blue PLA,#0000FF,Generic,PLA
Black PLA,#000000,Generic,PLA"""
        path.write_text(csv_content)
        return path


class TestScenario1_AutoDetectTransparency:
    """
    BDD Scenario: Auto-detect transparency (default)
    Given I run bananaforge convert without transparency flags
    When the system processes my image
    Then it should automatically detect transparency
    And stop with notification if found
    And proceed normally if no transparency
    """
    
    def test_auto_detect_stops_on_transparency(self, runner, temp_dir):
        """Test that default behavior stops when transparency is detected."""
        # Given: I run bananaforge convert without transparency flags
        transparent_image = temp_dir / "transparent.png"
        CLITestHelpers.create_transparent_image(transparent_image)
        materials_csv = temp_dir / "materials.csv"
        CLITestHelpers.create_materials_csv(materials_csv)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: the system processes my image
        result = runner.invoke(convert, [
            str(transparent_image),
            '--materials', str(materials_csv),
            '--output', str(output_dir),
            '--device', 'cpu',
            '--iterations', '10'  # Minimal iterations for speed
        ])
        
        # Then: it should automatically detect transparency and stop with notification
        assert result.exit_code == 1, "Should exit with error code when transparency detected"
        assert "transparent" in result.output.lower() or "🔍" in result.output, "Should mention transparency detection"
        assert "skip-transparency-check" in result.output, "Should suggest skip flag"
    
    def test_auto_detect_proceeds_without_transparency(self, runner, temp_dir):
        """Test that default behavior proceeds when no transparency is detected."""
        # Given: opaque image without transparency flags
        opaque_image = temp_dir / "opaque.png"
        CLITestHelpers.create_opaque_image(opaque_image)
        materials_csv = temp_dir / "materials.csv"
        CLITestHelpers.create_materials_csv(materials_csv)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: processing opaque image
        result = runner.invoke(convert, [
            str(opaque_image),
            '--materials', str(materials_csv),
            '--output', str(output_dir),
            '--device', 'cpu',
            '--iterations', '2',  # Very minimal for speed
            '--resolution', '32'  # Small resolution for speed
        ])
        
        # Then: should proceed with processing (may fail later due to minimal config, but shouldn't stop for transparency)
        transparency_error = "transparent" in result.output.lower() and result.exit_code == 1
        assert not transparency_error, f"Should not stop for transparency on opaque image. Output: {result.output[:500]}"


class TestScenario2_SkipTransparencyCheck:
    """
    BDD Scenario: Skip transparency check
    Given I use --skip-transparency-check flag
    When I process an image with transparency
    Then the system should proceed with RGB conversion
    And warn about potential quality loss
    And continue with optimization
    """
    
    def test_skip_transparency_check_proceeds(self, runner, temp_dir):
        """Test that --skip-transparency-check bypasses detection."""
        # Given: I use --skip-transparency-check flag with transparent image
        transparent_image = temp_dir / "transparent_skip.png"
        CLITestHelpers.create_transparent_image(transparent_image)
        materials_csv = temp_dir / "materials.csv"
        CLITestHelpers.create_materials_csv(materials_csv)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: I process an image with transparency using skip flag
        result = runner.invoke(convert, [
            str(transparent_image),
            '--materials', str(materials_csv),
            '--output', str(output_dir),
            '--skip-transparency-check',
            '--device', 'cpu',
            '--iterations', '2',
            '--resolution', '32'
        ])
        
        # Then: the system should proceed (not exit with code 1 for transparency)
        transparency_exit = result.exit_code == 1 and "transparent" in result.output.lower()
        assert not transparency_exit, f"Should not exit for transparency when skipped. Exit code: {result.exit_code}, Output: {result.output[:500]}"
    
    def test_skip_flag_with_detection_failure(self, runner, temp_dir):
        """Test that skip flag handles detection failures gracefully."""
        # Given: potentially problematic image with skip flag
        # Create a file that might cause detection issues
        problematic_image = temp_dir / "problematic.png"
        # Create minimal valid PNG
        Image.new('RGB', (10, 10), (255, 0, 0)).save(problematic_image, 'PNG')
        
        materials_csv = temp_dir / "materials.csv"
        CLITestHelpers.create_materials_csv(materials_csv)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: processing with skip flag
        result = runner.invoke(convert, [
            str(problematic_image),
            '--materials', str(materials_csv),
            '--output', str(output_dir),
            '--skip-transparency-check',
            '--device', 'cpu',
            '--iterations', '1'
        ])
        
        # Then: should handle gracefully (may fail for other reasons, but not transparency detection)
        detection_failure_exit = "transparency detection failed" in result.output.lower() and result.exit_code == 1
        assert not detection_failure_exit, "Should not exit for transparency detection failure when skipped"


class TestScenario3_ForceTransparencyAnalysis:
    """
    BDD Scenario: Force transparency analysis
    Given I use --analyze-transparency flag
    When I process any image
    Then the system should provide detailed transparency analysis
    And show statistics even for non-transparent images
    And help me understand the image composition
    """
    
    def test_analyze_transparency_shows_detailed_report(self, runner, temp_dir):
        """Test that --analyze-transparency shows detailed analysis."""
        # Given: I use --analyze-transparency flag
        test_image = temp_dir / "analyze_test.png"
        CLITestHelpers.create_transparent_image(test_image)
        materials_csv = temp_dir / "materials.csv"
        CLITestHelpers.create_materials_csv(materials_csv)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: I process any image with analysis flag
        result = runner.invoke(convert, [
            str(test_image),
            '--materials', str(materials_csv),
            '--output', str(output_dir),
            '--analyze-transparency',
            '--skip-transparency-check',  # Skip stopping so we can see the analysis
            '--device', 'cpu',
            '--iterations', '1'
        ])
        
        # Then: the system should provide detailed transparency analysis
        assert "📊 Transparency Analysis Report" in result.output, "Should show detailed analysis report"
        assert "Basic Statistics:" in result.output, "Should show basic statistics"
        assert "%" in result.output, "Should show percentage statistics"
    
    def test_analyze_transparency_verbose_mode(self, runner, temp_dir):
        """Test verbose transparency analysis."""
        # Given: analysis with verbose flag
        test_image = temp_dir / "verbose_test.png"
        CLITestHelpers.create_transparent_image(test_image)
        materials_csv = temp_dir / "materials.csv"
        CLITestHelpers.create_materials_csv(materials_csv)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: using both analyze and verbose flags
        result = runner.invoke(convert, [
            str(test_image),
            '--materials', str(materials_csv),
            '--output', str(output_dir),
            '--analyze-transparency',
            '--transparency-verbose',
            '--skip-transparency-check',
            '--device', 'cpu',
            '--iterations', '1'
        ])
        
        # Then: should show detailed technical information
        assert "🔬 Detailed Technical Information" in result.output or "Technical Details" in result.output, "Should show verbose technical details"
    
    def test_analyze_transparency_opaque_image(self, runner, temp_dir):
        """Test transparency analysis on opaque images."""
        # Given: opaque image with analysis flag
        opaque_image = temp_dir / "opaque_analyze.png"
        CLITestHelpers.create_opaque_image(opaque_image)
        materials_csv = temp_dir / "materials.csv"
        CLITestHelpers.create_materials_csv(materials_csv)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: analyzing opaque image
        result = runner.invoke(convert, [
            str(opaque_image),
            '--materials', str(materials_csv),
            '--output', str(output_dir),
            '--analyze-transparency',
            '--device', 'cpu',
            '--iterations', '1'
        ])
        
        # Then: should show statistics even for non-transparent images
        assert "📊 Transparency Analysis Report" in result.output, "Should show analysis for opaque image"
        assert "0.0%" in result.output or "0%" in result.output, "Should show 0% transparency"
        assert "✅" in result.output or "ready" in result.output.lower(), "Should indicate readiness"


class TestCLIFlagDocumentation:
    """Test CLI flag documentation and help text."""
    
    def test_transparency_flags_in_help(self, runner):
        """Test that transparency flags appear in help text."""
        # When: requesting help for convert command
        result = runner.invoke(convert, ['--help'])
        
        # Then: should document transparency flags
        expected_flags = [
            "--skip-transparency-check",
            "--analyze-transparency", 
            "--transparency-verbose"
        ]
        
        for flag in expected_flags:
            assert flag in result.output, f"Help should document {flag} flag"
    
    def test_flag_descriptions_are_clear(self, runner):
        """Test that flag descriptions are clear and helpful."""
        # When: requesting help
        result = runner.invoke(convert, ['--help'])
        
        # Then: should have clear descriptions
        flag_descriptions = {
            "--skip-transparency-check": ["skip", "transparency", "detection"],
            "--analyze-transparency": ["detailed", "transparency", "analysis"],
            "--transparency-verbose": ["detailed", "technical", "information"]
        }
        
        for flag, keywords in flag_descriptions.items():
            # Find the flag in help output
            flag_start = result.output.find(flag)
            assert flag_start != -1, f"Should find {flag} in help"
            
            # Check for keywords in the description (next ~100 chars)
            description_section = result.output[flag_start:flag_start + 150].lower()
            for keyword in keywords:
                assert keyword in description_section, f"{flag} description should contain '{keyword}'"


class TestCLIFlagCombinations:
    """Test combinations of CLI flags."""
    
    def test_skip_and_analyze_combination(self, runner, temp_dir):
        """Test combination of skip and analyze flags."""
        # Given: transparent image with both flags
        test_image = temp_dir / "combo_test.png"
        CLITestHelpers.create_transparent_image(test_image)
        materials_csv = temp_dir / "materials.csv"
        CLITestHelpers.create_materials_csv(materials_csv)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: using both skip and analyze flags
        result = runner.invoke(convert, [
            str(test_image),
            '--materials', str(materials_csv),
            '--output', str(output_dir),
            '--skip-transparency-check',
            '--analyze-transparency',
            '--device', 'cpu',
            '--iterations', '1'
        ])
        
        # Then: should show analysis but not stop processing
        assert "📊 Transparency Analysis Report" in result.output, "Should show analysis"
        transparency_stop = result.exit_code == 1 and "🔍" in result.output
        assert not transparency_stop, "Should not stop processing when skip flag is used"
    
    def test_verbose_without_analyze(self, runner, temp_dir):
        """Test that verbose flag requires analyze flag to show details."""
        # Given: image with only verbose flag
        test_image = temp_dir / "verbose_only.png"
        CLITestHelpers.create_transparent_image(test_image)
        materials_csv = temp_dir / "materials.csv"
        CLITestHelpers.create_materials_csv(materials_csv)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: using verbose without analyze (but transparency will be detected)
        result = runner.invoke(convert, [
            str(test_image),
            '--materials', str(materials_csv),
            '--output', str(output_dir),
            '--transparency-verbose',
            '--device', 'cpu',
            '--iterations', '1'
        ])
        
        # Then: should show educational content (verbose affects notification display)
        assert result.exit_code == 1, "Should still stop on transparency detection"
        assert "3D printing" in result.output, "Should show educational content in verbose mode"


class TestErrorHandling:
    """Test error handling for transparency flags."""
    
    def test_nonexistent_image_file(self, runner, temp_dir):
        """Test handling of nonexistent image files."""
        # Given: nonexistent image path
        nonexistent_image = temp_dir / "does_not_exist.png"
        materials_csv = temp_dir / "materials.csv"
        CLITestHelpers.create_materials_csv(materials_csv)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: trying to process nonexistent file
        result = runner.invoke(convert, [
            str(nonexistent_image),
            '--materials', str(materials_csv),
            '--output', str(output_dir),
            '--analyze-transparency'
        ])
        
        # Then: should handle error gracefully
        assert result.exit_code != 0, "Should exit with error for nonexistent file"
        assert "not found" in result.output.lower() or "does not exist" in result.output.lower(), "Should indicate file not found"
    
    def test_corrupted_image_handling(self, runner, temp_dir):
        """Test handling of corrupted image files."""
        # Given: corrupted image file
        corrupted_image = temp_dir / "corrupted.png"
        corrupted_image.write_bytes(b"This is not a valid PNG file")
        materials_csv = temp_dir / "materials.csv"
        CLITestHelpers.create_materials_csv(materials_csv)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: trying to process corrupted file
        result = runner.invoke(convert, [
            str(corrupted_image),
            '--materials', str(materials_csv),
            '--output', str(output_dir),
            '--analyze-transparency'
        ])
        
        # Then: should provide helpful error message
        assert result.exit_code != 0, "Should exit with error for corrupted file"
        detection_failed = "transparency detection failed" in result.output.lower()
        skip_suggestion = "skip-transparency-check" in result.output
        assert detection_failed or skip_suggestion, "Should indicate detection problem or suggest skip flag"


class TestBackwardCompatibility:
    """Test backward compatibility with existing CLI usage."""
    
    def test_existing_commands_still_work(self, runner, temp_dir):
        """Test that existing CLI usage patterns still work."""
        # Given: traditional CLI usage without transparency flags
        opaque_image = temp_dir / "backward_compat.png"
        CLITestHelpers.create_opaque_image(opaque_image)
        materials_csv = temp_dir / "materials.csv"
        CLITestHelpers.create_materials_csv(materials_csv)
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # When: using traditional CLI pattern
        result = runner.invoke(convert, [
            str(opaque_image),
            '--materials', str(materials_csv),
            '--output', str(output_dir),
            '--device', 'cpu',
            '--max-layers', '3',
            '--iterations', '1',
            '--resolution', '32'
        ])
        
        # Then: should work without transparency-related errors
        transparency_error = "transparent" in result.output.lower() and result.exit_code == 1
        assert not transparency_error, "Traditional usage should not fail due to transparency detection on opaque images"
    
    def test_help_includes_new_flags(self, runner):
        """Test that help includes new flags without breaking existing structure."""
        # When: requesting help
        result = runner.invoke(cli, ['--help'])
        convert_help = runner.invoke(convert, ['--help'])
        
        # Then: should include transparency information
        assert result.exit_code == 0, "Main help should work"
        assert convert_help.exit_code == 0, "Convert help should work"
        
        # Should mention transparency capabilities
        transparency_mentioned = (
            "transparency" in convert_help.output.lower() or 
            "skip-transparency-check" in convert_help.output
        )
        assert transparency_mentioned, "Convert help should mention transparency capabilities"