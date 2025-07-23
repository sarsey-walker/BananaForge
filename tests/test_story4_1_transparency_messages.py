#!/usr/bin/env python3
"""
TDD/BDD Tests for Story 4.1: Clear Transparency Detection Messages

Following BDD Given-When-Then scenarios from tasks1.md:
- User receives transparency notification
- User gets educational content
- Clear, friendly error messages with emoji indicators  
- Educational content about 3D printing and transparency
"""

import pytest
import json
from pathlib import Path
from PIL import Image

# Import the classes we're testing
from bananaforge.image.transparency_detector import TransparencyDetector, TransparencyInfo
from bananaforge.image.transparency_notifier import (
    TransparencyNotifier, TransparencyNotification, TransparencyException
)


@pytest.fixture
def notifier():
    """Create a TransparencyNotifier instance for testing."""
    return TransparencyNotifier()


@pytest.fixture
def detector():
    """Create a TransparencyDetector instance for testing."""
    return TransparencyDetector()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test images."""
    return tmp_path


class NotificationTestHelpers:
    """Helper class for creating test scenarios."""
    
    @staticmethod
    def create_high_transparency_image(path: Path):
        """Create image with high transparency (≥50%)."""
        image = Image.new('RGBA', (100, 100), (255, 255, 255, 0))  # Transparent
        # Add some opaque pixels (25%)
        for x in range(25):
            for y in range(100):
                image.putpixel((x, y), (255, 0, 0, 255))  # Opaque red
        image.save(path, 'PNG')
        return path
    
    @staticmethod  
    def create_moderate_transparency_image(path: Path):
        """Create image with moderate transparency (10-49%)."""
        image = Image.new('RGBA', (100, 100), (255, 0, 0, 255))  # Opaque red
        # Add transparent area (25% of pixels)
        for x in range(50, 100):
            for y in range(50, 100):
                image.putpixel((x, y), (0, 255, 0, 0))  # Transparent
        image.save(path, 'PNG')
        return path
    
    @staticmethod
    def create_minimal_transparency_image(path: Path):
        """Create image with minimal transparency (<10%)."""
        image = Image.new('RGBA', (100, 100), (255, 0, 0, 255))  # Opaque red
        # Add tiny transparent area (1% of pixels)
        for x in range(10):
            for y in range(10):
                image.putpixel((x, y), (0, 255, 0, 128))  # Semi-transparent
        image.save(path, 'PNG')
        return path
    
    @staticmethod
    def create_opaque_image(path: Path):
        """Create image without transparency."""
        image = Image.new('RGB', (100, 100), (255, 0, 0))  # Opaque red
        image.save(path, 'PNG')
        return path


class TestScenario1_UserReceivesTransparencyNotification:
    """
    BDD Scenario: User receives transparency notification
    Given I uploaded an image with transparent background
    When the system detects transparency
    Then I should see a clear message: "🔍 Transparent background detected"
    And receive guidance about 3D printing requirements
    And get suggestions for image preparation
    """
    
    def test_high_transparency_notification_message(self, notifier, detector, temp_dir):
        """Test clear message for high transparency detection."""
        # Given: I uploaded an image with transparent background
        high_trans_path = temp_dir / "high_transparency.png"
        NotificationTestHelpers.create_high_transparency_image(high_trans_path)
        
        # When: the system detects transparency
        transparency_info = detector.detect_transparency(str(high_trans_path))
        notification = notifier.create_notification(transparency_info)
        
        # Then: I should see a clear message
        assert "🔍 Transparent Background Detected" in notification.title, "Should have clear title with emoji"
        assert "transparent" in notification.message.lower(), "Should mention transparency"
        assert "3D printing" in notification.message, "Should mention 3D printing context"
        
        # And: receive guidance about 3D printing requirements
        assert "solid colors" in notification.message or "solid" in notification.message, "Should explain solid color requirement"
        
        # And: get suggestions for image preparation
        assert len(notification.suggestions) >= 3, "Should provide multiple suggestions"
        assert any("background" in s.lower() for s in notification.suggestions), "Should suggest adding background"
        assert any("crop" in s.lower() for s in notification.suggestions), "Should suggest cropping"
    
    def test_notification_has_appropriate_severity(self, notifier, detector, temp_dir):
        """Test that notifications have appropriate severity levels."""
        test_cases = [
            ("high_trans.png", NotificationTestHelpers.create_high_transparency_image, "warning"),
            ("mod_trans.png", NotificationTestHelpers.create_moderate_transparency_image, "warning"), 
            ("min_trans.png", NotificationTestHelpers.create_minimal_transparency_image, "info"),
            ("opaque.png", NotificationTestHelpers.create_opaque_image, "info")
        ]
        
        for filename, creator_func, expected_severity in test_cases:
            # Given: image with specific transparency level
            image_path = temp_dir / filename
            creator_func(image_path)
            
            # When: creating notification
            transparency_info = detector.detect_transparency(str(image_path))
            notification = notifier.create_notification(transparency_info)
            
            # Then: should have appropriate severity
            assert notification.severity == expected_severity, f"Expected {expected_severity} severity for {filename}"
    
    def test_notification_includes_technical_details(self, notifier, detector, temp_dir):
        """Test that notifications include helpful technical details."""
        # Given: image with transparency
        trans_path = temp_dir / "technical_test.png"
        NotificationTestHelpers.create_moderate_transparency_image(trans_path)
        
        # When: creating notification
        transparency_info = detector.detect_transparency(str(trans_path))
        notification = notifier.create_notification(transparency_info)
        
        # Then: should include technical details
        assert notification.technical_details is not None, "Should include technical details"
        assert "PNG" in notification.technical_details, "Should mention image format"
        assert "RGBA" in notification.technical_details, "Should mention color mode"
        assert "pixels" in notification.technical_details.lower(), "Should mention pixel counts"


class TestScenario2_UserGetsEducationalContent:
    """
    BDD Scenario: User gets educational content
    Given transparency is detected in my image
    When the notification is displayed
    Then I should receive education about why transparency affects 3D printing
    And get actionable advice for fixing the image
    And understand the implications for print quality
    """
    
    def test_educational_content_explains_3d_printing(self, notifier, detector, temp_dir):
        """Test that educational content explains 3D printing implications."""
        # Given: transparency is detected in my image
        trans_path = temp_dir / "educational_test.png"
        NotificationTestHelpers.create_high_transparency_image(trans_path)
        transparency_info = detector.detect_transparency(str(trans_path))
        
        # When: the notification is displayed
        educational_content = notifier.get_educational_content(transparency_info)
        
        # Then: I should receive education about why transparency affects 3D printing
        assert "3D print" in educational_content, "Should explain 3D printing context"
        assert "solid" in educational_content.lower(), "Should explain solid object requirement"
        assert "physical" in educational_content.lower(), "Should mention physical objects"
        
        # And: get actionable advice for fixing the image
        assert "background" in educational_content.lower(), "Should suggest background solutions"
        assert "crop" in educational_content.lower() or "edit" in educational_content.lower(), "Should suggest editing approaches"
        
        # And: understand the implications for print quality
        assert "color" in educational_content.lower(), "Should mention color implications"
        assert "gap" in educational_content.lower() or "fail" in educational_content.lower(), "Should explain potential issues"
    
    def test_educational_content_includes_statistics(self, notifier, detector, temp_dir):
        """Test that educational content includes transparency statistics."""
        # Given: image with known transparency
        stats_path = temp_dir / "stats_education.png"
        NotificationTestHelpers.create_moderate_transparency_image(stats_path)
        transparency_info = detector.detect_transparency(str(stats_path))
        
        # When: getting educational content
        educational_content = notifier.get_educational_content(transparency_info)
        
        # Then: should include specific statistics
        assert f"{transparency_info.transparency_percentage:.1f}%" in educational_content, "Should include transparency percentage"
        assert transparency_info.format_type in educational_content, "Should mention image format"
        assert transparency_info.image_mode in educational_content, "Should mention color mode"
    
    def test_educational_content_for_no_transparency(self, notifier, detector, temp_dir):
        """Test educational content for images without transparency."""
        # Given: image without transparency
        opaque_path = temp_dir / "opaque_education.png"
        NotificationTestHelpers.create_opaque_image(opaque_path)  
        transparency_info = detector.detect_transparency(str(opaque_path))
        
        # When: getting educational content
        educational_content = notifier.get_educational_content(transparency_info)
        
        # Then: should provide positive educational content
        assert "ready" in educational_content.lower(), "Should indicate readiness"
        assert "suitable" in educational_content.lower(), "Should confirm suitability"
        assert "3D printing" in educational_content, "Should mention 3D printing context"


class TestCLIMessageFormatting:
    """Test CLI message formatting functionality."""
    
    def test_cli_message_formatting_complete(self, notifier, detector, temp_dir):
        """Test complete CLI message formatting with all elements."""
        # Given: transparency notification
        trans_path = temp_dir / "cli_format_test.png"
        NotificationTestHelpers.create_high_transparency_image(trans_path)
        transparency_info = detector.detect_transparency(str(trans_path))
        notification = notifier.create_notification(transparency_info)
        
        # When: formatting for CLI with all options
        cli_message = notifier.format_cli_message(
            notification, 
            show_technical=True, 
            show_suggestions=True
        )
        
        # Then: should include all elements
        assert notification.title in cli_message, "Should include title"
        assert notification.message in cli_message, "Should include main message"
        assert "Recommended Actions:" in cli_message, "Should include suggestions header"
        assert "Technical Details:" in cli_message, "Should include technical details"
        assert all(suggestion in cli_message for suggestion in notification.suggestions), "Should include all suggestions"
    
    def test_cli_message_formatting_minimal(self, notifier, detector, temp_dir):
        """Test minimal CLI message formatting."""
        # Given: transparency notification
        trans_path = temp_dir / "cli_minimal_test.png"
        NotificationTestHelpers.create_minimal_transparency_image(trans_path)
        transparency_info = detector.detect_transparency(str(trans_path))
        notification = notifier.create_notification(transparency_info)
        
        # When: formatting with minimal options
        cli_message = notifier.format_cli_message(
            notification,
            show_technical=False,
            show_suggestions=False
        )
        
        # Then: should include only essential elements
        assert notification.title in cli_message, "Should include title"
        assert notification.message in cli_message, "Should include main message" 
        assert "Technical Details:" not in cli_message, "Should not include technical details"
        assert "Recommended Actions:" not in cli_message, "Should not include suggestions"
    
    def test_json_message_formatting(self, notifier, detector, temp_dir):
        """Test JSON message formatting for API/structured output."""
        # Given: transparency notification
        json_path = temp_dir / "json_test.png"
        NotificationTestHelpers.create_moderate_transparency_image(json_path)
        transparency_info = detector.detect_transparency(str(json_path))
        notification = notifier.create_notification(transparency_info)
        
        # When: formatting as JSON
        json_message = notifier.format_json_message(notification)
        
        # Then: should be valid JSON structure
        assert isinstance(json_message, dict), "Should return dictionary"
        
        required_fields = ["title", "message", "suggestions", "technical_details", "severity"]
        for field in required_fields:
            assert field in json_message, f"Should include {field} field"
        
        # Should be JSON serializable
        json_str = json.dumps(json_message)
        assert isinstance(json_str, str), "Should be JSON serializable"
        
        # Verify content
        assert json_message["title"] == notification.title, "Should preserve title"
        assert json_message["message"] == notification.message, "Should preserve message"
        assert json_message["suggestions"] == notification.suggestions, "Should preserve suggestions"


class TestTransparencyException:
    """Test TransparencyException for stopping processing with user notification."""
    
    def test_transparency_exception_creation(self, notifier, detector, temp_dir):
        """Test creating TransparencyException with notification."""
        # Given: image with transparency
        exception_path = temp_dir / "exception_test.png"
        NotificationTestHelpers.create_high_transparency_image(exception_path)
        transparency_info = detector.detect_transparency(str(exception_path))
        notification = notifier.create_notification(transparency_info)
        
        # When: creating exception
        exception = TransparencyException(notification, transparency_info)
        
        # Then: should carry all necessary information
        assert exception.notification == notification, "Should carry notification"
        assert exception.transparency_info == transparency_info, "Should carry transparency info"
        assert str(exception) == notification.message, "Exception message should match notification"
    
    def test_exception_cli_message_generation(self, notifier, detector, temp_dir):
        """Test that exception can generate CLI messages."""
        # Given: transparency exception
        exc_path = temp_dir / "exception_cli_test.png"
        NotificationTestHelpers.create_moderate_transparency_image(exc_path)
        transparency_info = detector.detect_transparency(str(exc_path))
        notification = notifier.create_notification(transparency_info)
        exception = TransparencyException(notification, transparency_info)
        
        # When: getting CLI message from exception
        cli_message = exception.get_cli_message(show_technical=True, show_suggestions=True)
        
        # Then: should generate proper CLI message
        assert isinstance(cli_message, str), "Should return string message"
        assert notification.title in cli_message, "Should include notification title"
        assert len(cli_message) > 100, "Should be substantial message"
    
    def test_exception_educational_content(self, notifier, detector, temp_dir):
        """Test that exception provides educational content."""
        # Given: transparency exception
        edu_path = temp_dir / "exception_edu_test.png"
        NotificationTestHelpers.create_high_transparency_image(edu_path)
        transparency_info = detector.detect_transparency(str(edu_path))
        notification = notifier.create_notification(transparency_info)
        exception = TransparencyException(notification, transparency_info)
        
        # When: getting educational content
        educational_content = exception.get_educational_content()
        
        # Then: should provide educational information
        assert isinstance(educational_content, str), "Should return string content"
        assert "3D printing" in educational_content, "Should mention 3D printing"
        assert len(educational_content) > 200, "Should be comprehensive content"


class TestNotificationVariations:
    """Test different notification types and variations."""
    
    def test_high_transparency_notification_content(self, notifier):
        """Test content of high transparency notifications."""
        # Given: high transparency info
        high_trans_info = TransparencyInfo(
            has_transparency=True,
            format_type="PNG",
            transparency_type="alpha_channel",
            image_mode="RGBA",
            total_pixels=10000,
            transparent_pixel_count=7500,
            opaque_pixel_count=2500,
            transparency_percentage=75.0,
            detection_details=["RGBA mode detected", "Alpha channel present"]
        )
        
        # When: creating notification
        notification = notifier.create_notification(high_trans_info)
        
        # Then: should be appropriate for high transparency
        assert "🔍 Transparent Background Detected" == notification.title, "Should have specific title"
        assert "significant transparency" in notification.message, "Should mention significance"
        assert "75.0%" in notification.message, "Should include percentage"
        assert notification.severity == "warning", "Should be warning severity"
        assert len(notification.suggestions) >= 4, "Should have multiple suggestions"
    
    def test_minimal_transparency_notification_content(self, notifier):
        """Test content of minimal transparency notifications."""
        # Given: minimal transparency info
        minimal_trans_info = TransparencyInfo(
            has_transparency=True,
            format_type="PNG", 
            transparency_type="alpha_channel",
            image_mode="RGBA",
            total_pixels=10000,
            transparent_pixel_count=50,
            opaque_pixel_count=9950,
            transparency_percentage=0.5,
            detection_details=["RGBA mode detected", "Alpha channel present"]
        )
        
        # When: creating notification
        notification = notifier.create_notification(minimal_trans_info)
        
        # Then: should be appropriate for minimal transparency
        assert "ℹ️ Minor Transparency Detected" == notification.title, "Should have specific title"
        assert "minor transparency" in notification.message, "Should mention it's minor"
        assert "0.5%" in notification.message, "Should include percentage"
        assert notification.severity == "info", "Should be info severity"
        assert any("anti-aliasing" in s for s in notification.suggestions), "Should mention anti-aliasing"
        assert any("skip-transparency-check" in s for s in notification.suggestions), "Should offer skip option"
    
    def test_no_transparency_notification_content(self, notifier):
        """Test content of no transparency notifications."""
        # Given: no transparency info
        no_trans_info = TransparencyInfo(
            has_transparency=False,
            format_type="PNG",
            transparency_type="none", 
            image_mode="RGB",
            total_pixels=10000,
            transparent_pixel_count=0,
            opaque_pixel_count=10000,
            transparency_percentage=0.0,
            detection_details=["RGB mode - no transparency support"]
        )
        
        # When: creating notification
        notification = notifier.create_notification(no_trans_info)
        
        # Then: should be positive and encouraging
        assert "✅ Image Ready for Processing" == notification.title, "Should have positive title"
        assert "ready for 3d printing" in notification.message.lower(), "Should confirm readiness"
        assert notification.severity == "info", "Should be info severity"
        assert any("suitable" in s.lower() for s in notification.suggestions), "Should confirm suitability"


class TestMessageCustomization:
    """Test message customization and internationalization readiness."""
    
    def test_emoji_consistency(self, notifier):
        """Test that emojis are used consistently across notifications."""
        test_infos = [
            # High transparency
            TransparencyInfo(True, "PNG", "alpha_channel", "RGBA", 1000, 800, 200, 80.0, []),
            # Moderate transparency  
            TransparencyInfo(True, "PNG", "alpha_channel", "RGBA", 1000, 300, 700, 30.0, []),
            # Minimal transparency
            TransparencyInfo(True, "PNG", "alpha_channel", "RGBA", 1000, 50, 950, 5.0, []),
            # No transparency
            TransparencyInfo(False, "PNG", "none", "RGB", 1000, 0, 1000, 0.0, [])
        ]
        
        expected_emojis = ["🔍", "⚠️", "ℹ️", "✅"]
        
        for i, info in enumerate(test_infos):
            # When: creating notification
            notification = notifier.create_notification(info)
            
            # Then: should have expected emoji
            assert expected_emojis[i] in notification.title, f"Should have {expected_emojis[i]} emoji for case {i}"
    
    def test_suggestion_actionability(self, notifier):
        """Test that suggestions are actionable and specific."""
        # Given: transparency info
        trans_info = TransparencyInfo(
            has_transparency=True,
            format_type="WEBP",
            transparency_type="alpha_channel", 
            image_mode="RGBA",
            total_pixels=5000,
            transparent_pixel_count=2000,
            opaque_pixel_count=3000,
            transparency_percentage=40.0,
            detection_details=["WEBP format with alpha channel"]
        )
        
        # When: creating notification
        notification = notifier.create_notification(trans_info)
        
        # Then: suggestions should be actionable
        for suggestion in notification.suggestions:
            # Should start with emoji or action word
            action_indicators = ["🎨", "✂️", "🖼️", "💡", "🔍", "✅", "🛠️", "⚡"]
            has_emoji = any(emoji in suggestion for emoji in action_indicators)
            has_action_word = any(word in suggestion.lower() for word in ["add", "crop", "use", "try", "review", "check"])
            
            assert has_emoji or has_action_word, f"Suggestion should be actionable: {suggestion}"
            assert len(suggestion) > 10, f"Suggestion should be descriptive: {suggestion}"