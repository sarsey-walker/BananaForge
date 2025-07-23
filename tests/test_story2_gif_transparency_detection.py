#!/usr/bin/env python3
"""
TDD/BDD Tests for Story 2.1: GIF Transparency Detection

Following BDD Given-When-Then scenarios from tasks1.md:
- GIF with transparent pixels (transparency index set)
- Animated GIF with transparency (use first frame)
- Palette-based transparency in GIF format
"""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np

# Import the classes we're testing
from bananaforge.image.transparency_detector import TransparencyDetector, TransparencyInfo


@pytest.fixture
def detector():
    """Create a TransparencyDetector instance for testing."""
    return TransparencyDetector()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test images."""
    return tmp_path


class GIFTestHelpers:
    """Helper class for creating various GIF test images."""
    
    @staticmethod
    def create_transparent_gif(path: Path, width: int = 100, height: int = 100, pattern: str = "checkerboard"):
        """Create GIF with transparency using different patterns."""
        # Create image in palette mode
        image = Image.new('P', (width, height))
        
        # Create a palette (256 colors max for GIF)
        palette = []
        for i in range(256):
            palette.extend([
                min(255, i * 3),      # R
                min(255, (i * 2) % 256),  # G  
                min(255, 255 - i)     # B
            ])
        image.putpalette(palette)
        
        if pattern == "checkerboard":
            # Checkerboard pattern with index 0 as transparent
            for x in range(width):
                for y in range(height):
                    if (x + y) % 4 == 0:
                        image.putpixel((x, y), 0)  # Transparent pixels (index 0)
                    elif (x + y) % 4 == 1:
                        image.putpixel((x, y), 1)  # Red pixels
                    elif (x + y) % 4 == 2:
                        image.putpixel((x, y), 2)  # Green pixels
                    else:
                        image.putpixel((x, y), 3)  # Blue pixels
        
        elif pattern == "border":
            # Transparent border, opaque center
            for x in range(width):
                for y in range(height):
                    if x < 10 or x >= width-10 or y < 10 or y >= height-10:
                        image.putpixel((x, y), 0)  # Transparent border
                    else:
                        image.putpixel((x, y), 1 + (x + y) % 10)  # Various colors
        
        elif pattern == "gradient":
            # Gradient with some transparent pixels
            for x in range(width):
                for y in range(height):
                    if x < width // 4:
                        image.putpixel((x, y), 0)  # Transparent section
                    else:
                        color_index = 1 + ((x * y) % 50)
                        image.putpixel((x, y), color_index)
        
        # Save with transparency index 0
        image.save(path, 'GIF', transparency=0)
        return path
    
    @staticmethod
    def create_opaque_gif(path: Path, width: int = 100, height: int = 100):
        """Create GIF without transparency."""
        image = Image.new('P', (width, height))
        
        # Create palette
        palette = []
        for i in range(256):
            palette.extend([i, (i * 2) % 256, 255 - i])
        image.putpalette(palette)
        
        # Fill with non-transparent colors (avoid index 0)
        for x in range(width):
            for y in range(height):
                color_index = 1 + ((x + y) % 100)  # Start from index 1, not 0
                image.putpixel((x, y), color_index)
        
        # Save without transparency
        image.save(path, 'GIF')
        return path
    
    @staticmethod
    def create_animated_gif(path: Path, width: int = 100, height: int = 100, frames: int = 3):
        """Create animated GIF with transparency."""
        images = []
        
        for frame in range(frames):
            image = Image.new('P', (width, height))
            
            # Create palette
            palette = []
            for i in range(256):
                palette.extend([
                    (i + frame * 50) % 256,  # Vary colors per frame
                    (i * 2 + frame * 30) % 256,
                    (255 - i + frame * 20) % 256
                ])
            image.putpalette(palette)
            
            # Create pattern that varies per frame
            for x in range(width):
                for y in range(height):
                    if (x + y + frame) % 6 == 0:
                        image.putpixel((x, y), 0)  # Transparent pixels
                    else:
                        color_index = 1 + ((x + y + frame * 10) % 50)
                        image.putpixel((x, y), color_index)
            
            images.append(image)
        
        # Save as animated GIF with transparency
        images[0].save(
            path, 'GIF',
            save_all=True,
            append_images=images[1:],
            duration=500,  # 500ms per frame
            loop=0,
            transparency=0
        )
        return path
    
    @staticmethod
    def create_gif_with_multiple_transparent_indices(path: Path, width: int = 100, height: int = 100):
        """Create GIF with multiple transparent color indices (advanced case)."""
        image = Image.new('P', (width, height))
        
        # Create palette
        palette = []
        for i in range(256):
            palette.extend([i, (i * 3) % 256, (255 - i * 2) % 256])
        image.putpalette(palette)
        
        # Fill with pattern using indices 0, 1, 2 as potentially transparent
        for x in range(width):
            for y in range(height):
                if (x + y) % 8 == 0:
                    image.putpixel((x, y), 0)  # Will be transparent
                elif (x + y) % 8 == 1:
                    image.putpixel((x, y), 1)  # Will be transparent  
                elif (x + y) % 8 == 2:
                    image.putpixel((x, y), 2)  # Will be transparent
                else:
                    image.putpixel((x, y), 3 + (x * y) % 20)  # Opaque colors
        
        # Note: GIF format typically supports only one transparent index
        # This tests edge case handling
        image.save(path, 'GIF', transparency=0)
        return path


class TestScenario1_GIFWithTransparentPixels:
    """
    BDD Scenario: GIF with transparent pixels
    Given a GIF image with transparency index set
    When I analyze the image for transparency
    Then the system should detect the transparent color index
    And report GIF transparency to the user
    """
    
    def test_detect_gif_checkerboard_transparency(self, detector, temp_dir):
        """Test detection of GIF transparency with checkerboard pattern."""
        # Given: a GIF image with transparency index set
        gif_path = temp_dir / "gif_checkerboard.gif"
        GIFTestHelpers.create_transparent_gif(gif_path, pattern="checkerboard")
        
        # When: I analyze the image for transparency
        result = detector.detect_transparency(str(gif_path))
        
        # Then: the system should detect the transparent color index
        assert result is not None, "Should return TransparencyInfo object"
        assert result.has_transparency is True, "Should detect transparency in GIF"
        assert result.format_type == "GIF", "Should identify GIF format"
        assert result.transparency_type in ["palette_transparency", "gif_transparency"], "Should identify GIF transparency type"
        
        # And: report GIF transparency to the user
        assert result.transparent_pixel_count > 0, "Should count transparent pixels"
        assert 0 < result.transparency_percentage < 100, "Should calculate transparency percentage"
        assert len(result.detection_details) > 0, "Should provide detection details"
        assert any("gif" in detail.lower() or "transparency" in detail.lower() for detail in result.detection_details)
    
    def test_detect_gif_border_transparency(self, detector, temp_dir):
        """Test detection of GIF transparency with border pattern."""
        # Given: a GIF with transparent border
        gif_path = temp_dir / "gif_border.gif"
        GIFTestHelpers.create_transparent_gif(gif_path, pattern="border")
        
        # When: I analyze the image for transparency
        result = detector.detect_transparency(str(gif_path))
        
        # Then: should detect transparency correctly
        assert result.has_transparency is True, "Should detect border transparency"
        assert result.format_type == "GIF", "Should identify GIF format"
        
        # Border pattern should have specific transparency characteristics
        expected_transparent_pixels = 2 * (100 * 10) + 2 * (80 * 10)  # Approximate border pixels
        actual_transparent = result.transparent_pixel_count
        # Allow some tolerance due to pixel counting variations
        assert abs(actual_transparent - expected_transparent_pixels) < 500, f"Expected ~{expected_transparent_pixels} transparent pixels, got {actual_transparent}"
    
    def test_detect_gif_gradient_transparency(self, detector, temp_dir):
        """Test detection of GIF transparency with gradient pattern."""
        # Given: a GIF with gradient transparency
        gif_path = temp_dir / "gif_gradient.gif"
        GIFTestHelpers.create_transparent_gif(gif_path, pattern="gradient")
        
        # When: I analyze the image for transparency
        result = detector.detect_transparency(str(gif_path))
        
        # Then: should detect transparency
        assert result.has_transparency is True, "Should detect gradient transparency"
        assert result.transparency_type in ["palette_transparency", "gif_transparency"], "Should identify appropriate transparency type"
        
        # Gradient pattern should have 1/4 of pixels transparent
        expected_percentage = 25.0  # 25% transparent
        actual_percentage = result.transparency_percentage
        assert abs(actual_percentage - expected_percentage) < 5.0, f"Expected ~25% transparency, got {actual_percentage}%"
    
    def test_gif_transparency_statistics_accuracy(self, detector, temp_dir):
        """Test accuracy of transparency statistics for GIF files."""
        # Given: a GIF with known transparency distribution
        gif_path = temp_dir / "gif_stats_test.gif"
        GIFTestHelpers.create_transparent_gif(gif_path, 200, 200, pattern="checkerboard")
        
        # When: analyzing transparency statistics
        result = detector.detect_transparency(str(gif_path))
        
        # Then: statistics should be accurate
        assert result.total_pixels == 200 * 200, "Should count total pixels correctly"
        assert result.transparent_pixel_count > 0, "Should have transparent pixels"
        assert result.opaque_pixel_count > 0, "Should have opaque pixels"
        assert result.transparent_pixel_count + result.opaque_pixel_count == result.total_pixels, "Pixel counts should sum to total"
        
        # Checkerboard pattern should have roughly 25% transparent pixels
        expected_transparent = result.total_pixels // 4
        actual_transparent = result.transparent_pixel_count
        tolerance = result.total_pixels * 0.05  # 5% tolerance
        assert abs(actual_transparent - expected_transparent) < tolerance, f"Expected ~{expected_transparent} transparent pixels, got {actual_transparent}"


class TestScenario2_AnimatedGIFWithTransparency:
    """
    BDD Scenario: Animated GIF with transparency
    Given an animated GIF with transparent frames
    When I process the image
    Then the system should detect transparency in the frame
    And handle the animation appropriately
    """
    
    def test_detect_animated_gif_transparency(self, detector, temp_dir):
        """Test detection of transparency in animated GIF."""
        # Given: an animated GIF with transparent frames
        gif_path = temp_dir / "animated_transparent.gif"
        GIFTestHelpers.create_animated_gif(gif_path)
        
        # When: I process the image
        result = detector.detect_transparency(str(gif_path))
        
        # Then: the system should detect transparency in the frame
        assert result is not None, "Should return TransparencyInfo object"
        assert result.has_transparency is True, "Should detect transparency in animated GIF"
        assert result.format_type == "GIF", "Should identify GIF format"
        
        # And: handle the animation appropriately
        # The system should analyze the first frame
        assert result.transparent_pixel_count > 0, "Should find transparent pixels in first frame"
        assert "animated" in " ".join(result.detection_details).lower() or "frame" in " ".join(result.detection_details).lower(), "Should mention animation handling in details"
    
    def test_animated_gif_uses_first_frame(self, detector, temp_dir):
        """Test that animated GIF analysis uses the first frame."""
        # Given: an animated GIF where only first frame has specific pattern
        gif_path = temp_dir / "animated_first_frame.gif"
        
        # Create animated GIF manually to control frame contents
        frames = []
        
        # First frame - high transparency
        frame1 = Image.new('P', (100, 100))
        palette = []
        for i in range(256):
            palette.extend([i, i, i])  # Grayscale palette
        frame1.putpalette(palette)
        
        # Fill first frame with 50% transparent pixels
        for x in range(100):
            for y in range(100):
                if (x + y) % 2 == 0:
                    frame1.putpixel((x, y), 0)  # Transparent
                else:
                    frame1.putpixel((x, y), 1)  # Opaque
        frames.append(frame1)
        
        # Second frame - low transparency
        frame2 = Image.new('P', (100, 100))
        frame2.putpalette(palette)
        for x in range(100):
            for y in range(100):
                if (x + y) % 10 == 0:  # Only 10% transparent
                    frame2.putpixel((x, y), 0)
                else:
                    frame2.putpixel((x, y), 2)
        frames.append(frame2)
        
        # Save animated GIF
        frames[0].save(
            gif_path, 'GIF',
            save_all=True,
            append_images=frames[1:],
            duration=500,
            transparency=0
        )
        
        # When: I process the animated image
        result = detector.detect_transparency(str(gif_path))
        
        # Then: should reflect first frame characteristics (50% transparency)
        expected_percentage = 50.0
        actual_percentage = result.transparency_percentage
        assert abs(actual_percentage - expected_percentage) < 10.0, f"Expected ~50% transparency from first frame, got {actual_percentage}%"
    
    def test_animated_gif_performance(self, detector, temp_dir):
        """Test that animated GIF processing is reasonably fast."""
        # Given: a large animated GIF
        gif_path = temp_dir / "large_animated.gif"
        GIFTestHelpers.create_animated_gif(gif_path, 200, 200, frames=10)  # Large with many frames
        
        # When: I process the animated GIF
        import time
        start_time = time.time()
        result = detector.detect_transparency(str(gif_path))
        elapsed_time = time.time() - start_time
        
        # Then: should complete in reasonable time
        assert result is not None, "Should process large animated GIF"
        assert elapsed_time < 3.0, f"Processing should be fast even for large animated GIFs, took {elapsed_time:.2f}s"
        # Should only analyze first frame, not all frames


class TestScenario3_GIFPaletteTransparency:
    """
    BDD Scenario: Palette-based transparency in GIF format
    Given a GIF with palette mode and transparency index
    When I analyze the transparency
    Then the system should detect the palette transparency
    And handle GIF-specific transparency features
    """
    
    def test_detect_gif_palette_transparency_index(self, detector, temp_dir):
        """Test detection of GIF palette transparency index."""
        # Given: a GIF with palette mode and transparency index
        gif_path = temp_dir / "gif_palette.gif"
        GIFTestHelpers.create_transparent_gif(gif_path)
        
        # When: I analyze the transparency
        result = detector.detect_transparency(str(gif_path))
        
        # Then: the system should detect the palette transparency
        assert result.has_transparency is True, "Should detect GIF palette transparency"
        assert result.image_mode == "P", "Should detect P (palette) mode"
        assert result.transparency_type in ["palette_transparency", "gif_transparency"], "Should identify palette transparency"
        
        # And: handle GIF-specific transparency features
        transparency_mentioned = any("transparency" in detail.lower() for detail in result.detection_details)
        assert transparency_mentioned, "Should mention transparency in detection details"
    
    def test_gif_no_transparency_detection(self, detector, temp_dir):
        """Test detection when GIF has no transparency."""
        # Given: a GIF without transparency
        gif_path = temp_dir / "gif_opaque.gif"
        GIFTestHelpers.create_opaque_gif(gif_path)
        
        # When: I analyze the transparency
        result = detector.detect_transparency(str(gif_path))
        
        # Then: should detect no transparency
        assert result.has_transparency is False, "Should detect no transparency in opaque GIF"
        assert result.format_type == "GIF", "Should identify GIF format"
        assert result.transparency_type == "none", "Should identify no transparency"
        assert result.transparent_pixel_count == 0, "Should have no transparent pixels"
        assert result.transparency_percentage == 0.0, "Should have 0% transparency"
    
    def test_gif_transparency_index_edge_cases(self, detector, temp_dir):
        """Test GIF transparency with edge cases in transparency index."""
        # Given: a GIF with multiple potential transparent indices
        gif_path = temp_dir / "gif_multi_transparent.gif"
        GIFTestHelpers.create_gif_with_multiple_transparent_indices(gif_path)
        
        # When: I analyze the transparency
        result = detector.detect_transparency(str(gif_path))
        
        # Then: should handle the edge case appropriately
        assert result is not None, "Should handle multiple transparent indices"
        # GIF format typically supports only one transparent index
        # The system should detect transparency based on the specified index
        if result.has_transparency:
            assert result.transparent_pixel_count > 0, "If transparency detected, should count pixels"


class TestGIFTransparencyIntegration:
    """Test integration of GIF transparency detection with the broader system."""
    
    def test_gif_detection_matches_format(self, detector, temp_dir):
        """Test that GIF files are properly identified as GIF format."""
        # Test various GIF types
        gif_types = [
            ("transparent", lambda p: GIFTestHelpers.create_transparent_gif(p)),
            ("opaque", lambda p: GIFTestHelpers.create_opaque_gif(p)),
            ("animated", lambda p: GIFTestHelpers.create_animated_gif(p, frames=2))
        ]
        
        for gif_type, creator_func in gif_types:
            gif_path = temp_dir / f"format_test_{gif_type}.gif"
            creator_func(gif_path)
            
            result = detector.detect_transparency(str(gif_path))
            
            assert result.format_type == "GIF", f"Should identify {gif_type} as GIF format"
            assert result.image_mode == "P", f"GIF should typically be in P mode for {gif_type}"
    
    def test_gif_vs_png_transparency_differences(self, detector, temp_dir):
        """Test that GIF and PNG transparency are handled differently."""
        # Create similar transparent images in different formats
        gif_path = temp_dir / "compare_transparency.gif"
        png_path = temp_dir / "compare_transparency.png"
        
        # Create transparent GIF
        GIFTestHelpers.create_transparent_gif(gif_path, pattern="checkerboard")
        
        # Create similar transparent PNG
        png_image = Image.new('RGBA', (100, 100), (255, 255, 255, 255))
        for x in range(100):
            for y in range(100):
                if (x + y) % 4 == 0:
                    png_image.putpixel((x, y), (255, 0, 0, 0))  # Transparent
                else:
                    png_image.putpixel((x, y), (0, 255, 0, 255))  # Opaque
        png_image.save(png_path, 'PNG')
        
        # Analyze both
        gif_result = detector.detect_transparency(str(gif_path))
        png_result = detector.detect_transparency(str(png_path))
        
        # Should detect transparency in both but handle differently
        assert gif_result.has_transparency is True, "Should detect GIF transparency"
        assert png_result.has_transparency is True, "Should detect PNG transparency"
        
        assert gif_result.format_type == "GIF", "Should identify GIF format"
        assert png_result.format_type == "PNG", "Should identify PNG format"
        
        # Transparency types should be different
        assert gif_result.transparency_type in ["palette_transparency", "gif_transparency"], "GIF should use palette transparency"
        assert png_result.transparency_type == "alpha_channel", "PNG should use alpha channel"
    
    def test_gif_error_handling(self, detector, temp_dir):
        """Test error handling for problematic GIF files."""
        # Test with corrupted GIF
        corrupted_gif = temp_dir / "corrupted.gif"
        corrupted_gif.write_bytes(b'GIF89a' + b'\x00' * 100)  # Invalid GIF data
        
        with pytest.raises((IOError, ValueError)):
            detector.detect_transparency(str(corrupted_gif))
        
        # Test with file that has .gif extension but isn't actually a GIF
        fake_gif = temp_dir / "fake.gif"
        fake_gif.write_text("This is not a GIF file")
        
        with pytest.raises((IOError, ValueError)):
            detector.detect_transparency(str(fake_gif))