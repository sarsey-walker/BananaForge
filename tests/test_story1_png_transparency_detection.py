#!/usr/bin/env python3
"""
TDD/BDD Tests for Story 1.1: PNG Alpha Channel Detection

Following BDD Given-When-Then scenarios from tasks1.md:
- PNG with full alpha channel (RGBA mode)
- PNG with palette transparency (P mode with transparency key)
- Regular PNG without transparency (RGB mode)
"""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np

# Import the classes we'll create
from bananaforge.image.transparency_detector import TransparencyDetector, TransparencyInfo


@pytest.fixture
def detector():
    """Create a TransparencyDetector instance for testing."""
    return TransparencyDetector()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test images."""
    return tmp_path


class TestPNGTransparencyDetection:
    """BDD test scenarios for PNG transparency detection."""
    
    def create_rgba_png(self, path: Path, width: int = 100, height: int = 100):
        """Helper: Create PNG with full RGBA alpha channel."""
        # Create RGBA image with transparent background and opaque center
        image = Image.new('RGBA', (width, height), (255, 0, 0, 0))  # Transparent red
        
        # Add opaque center square
        for x in range(width // 4, 3 * width // 4):
            for y in range(height // 4, 3 * height // 4):
                image.putpixel((x, y), (0, 255, 0, 255))  # Opaque green
                
        image.save(path, 'PNG')
        return path
    
    def create_palette_transparent_png(self, path: Path, width: int = 100, height: int = 100):
        """Helper: Create PNG with palette mode transparency."""
        # Create palette image with transparency
        # We'll create it directly in P mode to have better control over palette indices
        
        # Create RGB image first
        rgb_image = Image.new('RGB', (width, height), (255, 0, 0))  # Red background
        
        # Add transparent areas (we'll make these index 0 after conversion)
        for x in range(width // 4):  # Left border transparent
            for y in range(height):
                rgb_image.putpixel((x, y), (0, 0, 0))  # Black (will become index 0)
        
        # Add green center
        for x in range(width // 4, 3 * width // 4):
            for y in range(height // 4, 3 * height // 4):
                rgb_image.putpixel((x, y), (0, 255, 0))  # Green center
        
        # Convert to palette mode and set transparency
        palette_image = rgb_image.convert('P')
        palette_image.save(path, 'PNG', transparency=0)  # Make index 0 (black) transparent
        return path
    
    def create_opaque_png(self, path: Path, width: int = 100, height: int = 100):
        """Helper: Create PNG without any transparency."""
        image = Image.new('RGB', (width, height), (255, 0, 0))  # Red background
        
        # Add green center
        for x in range(width // 4, 3 * width // 4):
            for y in range(height // 4, 3 * height // 4):
                image.putpixel((x, y), (0, 255, 0))  # Green center
                
        image.save(path, 'PNG')
        return path


class TestScenario1_PNGWithFullAlphaChannel:
    """
    BDD Scenario: PNG with full alpha channel
    Given a PNG image file with RGBA mode
    When I load the image for processing
    Then the system should detect the alpha channel presence
    And report transparency information to the user
    """
    
    def test_detect_rgba_mode_transparency(self, temp_dir):
        """Test detection of RGBA mode PNG with alpha channel."""
        # Given: a PNG image file with RGBA mode
        png_path = temp_dir / "rgba_transparent.png"
        TestPNGTransparencyDetection().create_rgba_png(png_path)
        
        detector = TransparencyDetector()
        
        # When: I load the image for processing
        result = detector.detect_transparency(str(png_path))
        
        # Then: the system should detect the alpha channel presence
        assert result is not None, "Should return TransparencyInfo object"
        assert result.has_transparency is True, "Should detect transparency in RGBA PNG"
        assert result.format_type == "PNG", "Should identify PNG format"
        assert result.transparency_type == "alpha_channel", "Should identify alpha channel transparency"
        assert result.image_mode == "RGBA", "Should detect RGBA mode"
        
        # And: report transparency information to the user
        assert result.transparent_pixel_count > 0, "Should count transparent pixels"
        assert 0 < result.transparency_percentage < 100, "Should calculate transparency percentage"
        assert len(result.detection_details) > 0, "Should provide detection details"
    
    def test_rgba_transparency_statistics(self, temp_dir):
        """Test detailed transparency statistics for RGBA PNG."""
        # Given: an RGBA PNG with known transparency distribution
        png_path = temp_dir / "rgba_stats.png"
        TestPNGTransparencyDetection().create_rgba_png(png_path, 200, 200)
        
        detector = TransparencyDetector()
        
        # When: analyzing transparency statistics
        result = detector.detect_transparency(str(png_path))
        
        # Then: should provide accurate statistics
        assert result.has_transparency is True
        assert result.total_pixels == 200 * 200, "Should count total pixels correctly"
        assert result.transparent_pixel_count > 0, "Should have transparent pixels"
        assert result.opaque_pixel_count > 0, "Should have opaque pixels"
        assert result.transparent_pixel_count + result.opaque_pixel_count <= result.total_pixels
        

class TestScenario2_PNGWithPaletteTransparency:
    """
    BDD Scenario: PNG with palette transparency
    Given a PNG image with palette mode and transparency info
    When I load the image for processing
    Then the system should detect the transparency key
    And identify it as containing transparent pixels
    """
    
    def test_detect_palette_transparency(self, temp_dir):
        """Test detection of palette-based transparency in PNG."""
        # Given: a PNG image with palette mode and transparency info
        png_path = temp_dir / "palette_transparent.png"
        TestPNGTransparencyDetection().create_palette_transparent_png(png_path)
        
        detector = TransparencyDetector()
        
        # When: I load the image for processing
        result = detector.detect_transparency(str(png_path))
        
        # Then: the system should detect the transparency key
        assert result is not None, "Should return TransparencyInfo object"
        assert result.has_transparency is True, "Should detect palette transparency"
        assert result.format_type == "PNG", "Should identify PNG format"
        assert result.transparency_type == "palette_transparency", "Should identify palette transparency"
        
        # And: identify it as containing transparent pixels
        assert any("transparency" in detail for detail in result.detection_details), "Should mention transparency key in details"
    
    def test_palette_mode_detection(self, temp_dir):
        """Test proper handling of P mode images with transparency."""
        # Given: P mode PNG with transparency key
        png_path = temp_dir / "p_mode_transparent.png"
        TestPNGTransparencyDetection().create_palette_transparent_png(png_path)
        
        detector = TransparencyDetector()
        
        # When: processing the image
        result = detector.detect_transparency(str(png_path))
        
        # Then: should properly identify the mode and transparency
        assert result.image_mode == "P", "Should detect P (palette) mode"
        assert result.has_transparency is True, "Should detect transparency in P mode"


class TestScenario3_RegularPNGWithoutTransparency:
    """
    BDD Scenario: Regular PNG without transparency
    Given a PNG image in RGB mode with no transparency
    When I load the image for processing
    Then the system should report no transparency detected
    And proceed with normal processing
    """
    
    def test_detect_no_transparency_rgb_png(self, temp_dir):
        """Test detection that RGB PNG has no transparency."""
        # Given: a PNG image in RGB mode with no transparency
        png_path = temp_dir / "opaque_rgb.png"
        TestPNGTransparencyDetection().create_opaque_png(png_path)
        
        detector = TransparencyDetector()
        
        # When: I load the image for processing
        result = detector.detect_transparency(str(png_path))
        
        # Then: the system should report no transparency detected
        assert result is not None, "Should return TransparencyInfo object"
        assert result.has_transparency is False, "Should detect no transparency in RGB PNG"
        assert result.format_type == "PNG", "Should identify PNG format"
        assert result.transparency_type == "none", "Should identify no transparency"
        assert result.image_mode == "RGB", "Should detect RGB mode"
        
        # And: should provide appropriate statistics
        assert result.transparent_pixel_count == 0, "Should have no transparent pixels"
        assert result.transparency_percentage == 0.0, "Should have 0% transparency"
        assert result.opaque_pixel_count == result.total_pixels, "All pixels should be opaque"


class TestTransparencyInfoDataStructure:
    """Test the TransparencyInfo data structure requirements."""
    
    def test_transparency_info_structure(self):
        """Test that TransparencyInfo has all required fields."""
        # This test defines the interface we need to implement
        info = TransparencyInfo(
            has_transparency=True,
            format_type="PNG",
            transparency_type="alpha_channel",
            image_mode="RGBA",
            total_pixels=10000,
            transparent_pixel_count=2500,
            opaque_pixel_count=7500,
            transparency_percentage=25.0,
            detection_details=["RGBA mode detected", "Alpha channel present"]
        )
        
        assert info.has_transparency is True
        assert info.format_type == "PNG"
        assert info.transparency_type == "alpha_channel"
        assert info.image_mode == "RGBA"
        assert info.total_pixels == 10000
        assert info.transparent_pixel_count == 2500
        assert info.opaque_pixel_count == 7500
        assert info.transparency_percentage == 25.0
        assert len(info.detection_details) == 2


class TestTransparencyDetectorMethods:
    """Test the TransparencyDetector class methods and interface."""
    
    def test_detector_initialization(self):
        """Test TransparencyDetector can be initialized."""
        detector = TransparencyDetector()
        assert detector is not None
        
    def test_detect_transparency_method_exists(self):
        """Test that detect_transparency method exists and has correct signature."""
        detector = TransparencyDetector()
        assert hasattr(detector, 'detect_transparency')
        
        # Method should accept a file path and return TransparencyInfo
        import inspect
        sig = inspect.signature(detector.detect_transparency)
        params = list(sig.parameters.keys())
        assert 'image_path' in params, "Method should accept image_path parameter"


class TestErrorHandling:
    """Test error handling scenarios for PNG transparency detection."""
    
    def test_nonexistent_file_handling(self, temp_dir):
        """Test graceful handling of nonexistent files."""
        detector = TransparencyDetector()
        nonexistent_path = temp_dir / "does_not_exist.png"
        
        # Should raise appropriate exception, not crash
        with pytest.raises((FileNotFoundError, IOError)):
            detector.detect_transparency(str(nonexistent_path))
    
    def test_invalid_file_handling(self, temp_dir):
        """Test graceful handling of invalid image files."""
        detector = TransparencyDetector()
        
        # Create a fake PNG file
        fake_png = temp_dir / "fake.png"
        fake_png.write_text("This is not a PNG file")
        
        # Should handle gracefully
        with pytest.raises((IOError, ValueError)):
            detector.detect_transparency(str(fake_png))