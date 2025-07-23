"""
User notification system for transparency detection.

This module provides clear, friendly messaging when transparency is detected
in images, helping users understand why processing stopped and what to do next.
"""

from dataclasses import dataclass
from typing import List, Optional
from .transparency_detector import TransparencyInfo


@dataclass
class TransparencyNotification:
    """Structured notification about transparency detection."""
    
    title: str
    message: str
    suggestions: List[str]
    technical_details: Optional[str] = None
    severity: str = "warning"  # "warning", "info", "error"


class TransparencyNotifier:
    """
    User notification system for transparency detection results.
    
    Provides clear, educational messaging about transparency detection
    with actionable advice for users.
    """
    
    def __init__(self):
        """Initialize the transparency notifier."""
        pass
    
    def create_notification(self, transparency_info: TransparencyInfo) -> TransparencyNotification:
        """
        Create user notification based on transparency detection results.
        
        Args:
            transparency_info: Results from transparency detection
            
        Returns:
            TransparencyNotification with appropriate messaging
        """
        if not transparency_info.has_transparency:
            return self._create_no_transparency_notification(transparency_info)
        
        # Determine notification type based on transparency characteristics
        if transparency_info.transparency_percentage >= 50:
            return self._create_high_transparency_notification(transparency_info)
        elif transparency_info.transparency_percentage >= 10:
            return self._create_moderate_transparency_notification(transparency_info)
        else:
            return self._create_minimal_transparency_notification(transparency_info)
    
    def _create_no_transparency_notification(self, info: TransparencyInfo) -> TransparencyNotification:
        """Create notification for images without transparency."""
        return TransparencyNotification(
            title="✅ Image Ready for Processing",
            message=f"No transparency detected in {info.format_type} image. Ready for 3D printing optimization.",
            suggestions=[
                "Image is suitable for 3D printing",
                "Proceed with normal processing"
            ],
            technical_details=f"Mode: {info.image_mode}, Total pixels: {info.total_pixels:,}",
            severity="info"
        )
    
    def _create_high_transparency_notification(self, info: TransparencyInfo) -> TransparencyNotification:
        """Create notification for images with high transparency (≥50%)."""
        return TransparencyNotification(
            title="🔍 Transparent Background Detected",
            message=(
                f"Your {info.format_type} image contains significant transparency "
                f"({info.transparency_percentage:.1f}% of pixels). "
                "3D printing requires solid colors without transparent areas."
            ),
            suggestions=[
                "🎨 Add a solid background color to your image",
                "✂️ Crop the image to remove transparent areas", 
                "🖼️ Use an image editor to fill transparent areas",
                "💡 Consider using a different source image without transparency"
            ],
            technical_details=(
                f"Format: {info.format_type} ({info.image_mode}), "
                f"Transparent pixels: {info.transparent_pixel_count:,}/{info.total_pixels:,} "
                f"({info.transparency_percentage:.1f}%)"
            ),
            severity="warning"
        )
    
    def _create_moderate_transparency_notification(self, info: TransparencyInfo) -> TransparencyNotification:
        """Create notification for images with moderate transparency (10-49%)."""
        return TransparencyNotification(
            title="⚠️ Partial Transparency Detected",
            message=(
                f"Your {info.format_type} image contains partial transparency "
                f"({info.transparency_percentage:.1f}% of pixels). "
                "This may affect 3D printing quality and appearance."
            ),
            suggestions=[
                "🎨 Review transparent areas and add background if needed",
                "🔍 Check if transparency is intentional (edges, shadows)",
                "✂️ Crop to focus on opaque regions if possible",
                "🛠️ Use image editing software to adjust transparency"
            ],
            technical_details=(
                f"Format: {info.format_type} ({info.image_mode}), "
                f"Transparent pixels: {info.transparent_pixel_count:,}/{info.total_pixels:,} "
                f"({info.transparency_percentage:.1f}%)"
            ),
            severity="warning"
        )
    
    def _create_minimal_transparency_notification(self, info: TransparencyInfo) -> TransparencyNotification:
        """Create notification for images with minimal transparency (<10%)."""
        return TransparencyNotification(
            title="ℹ️ Minor Transparency Detected",
            message=(
                f"Your {info.format_type} image contains minor transparency "
                f"({info.transparency_percentage:.1f}% of pixels). "
                "This might be from anti-aliasing or edge effects."
            ),
            suggestions=[
                "🔍 Review if transparency is from edge anti-aliasing",
                "✅ May be acceptable for 3D printing depending on content",
                "🎨 Add solid background if sharp edges are important",
                "⚡ Use --skip-transparency-check flag to proceed anyway"
            ],
            technical_details=(
                f"Format: {info.format_type} ({info.image_mode}), "
                f"Transparent pixels: {info.transparent_pixel_count:,}/{info.total_pixels:,} "
                f"({info.transparency_percentage:.1f}%)"
            ),
            severity="info"
        )
    
    def format_cli_message(self, notification: TransparencyNotification, 
                          show_technical: bool = False, 
                          show_suggestions: bool = True) -> str:
        """
        Format notification for CLI display.
        
        Args:
            notification: Notification to format
            show_technical: Whether to include technical details
            show_suggestions: Whether to include suggestions
            
        Returns:
            Formatted message string
        """
        lines = []
        
        # Title and main message
        lines.append(f"\n{notification.title}")
        lines.append("=" * len(notification.title.replace("🔍", "").replace("⚠️", "").replace("ℹ️", "").replace("✅", "").strip()))
        lines.append(f"\n{notification.message}")
        
        # Suggestions
        if show_suggestions and notification.suggestions:
            lines.append("\nRecommended Actions:")
            for suggestion in notification.suggestions:
                lines.append(f"  • {suggestion}")
        
        # Technical details
        if show_technical and notification.technical_details:
            lines.append(f"\nTechnical Details: {notification.technical_details}")
        
        lines.append("")  # Empty line at end
        return "\n".join(lines)
    
    def format_json_message(self, notification: TransparencyNotification) -> dict:
        """
        Format notification as JSON structure.
        
        Args:
            notification: Notification to format
            
        Returns:
            Dictionary suitable for JSON serialization
        """
        return {
            "title": notification.title,
            "message": notification.message,
            "suggestions": notification.suggestions,
            "technical_details": notification.technical_details,
            "severity": notification.severity
        }
    
    def get_educational_content(self, transparency_info: TransparencyInfo) -> str:
        """
        Get educational content about transparency and 3D printing.
        
        Args:
            transparency_info: Transparency detection results
            
        Returns:
            Educational content as formatted string
        """
        if not transparency_info.has_transparency:
            return self._get_no_transparency_education()
        
        return self._get_transparency_education(transparency_info)
    
    def _get_no_transparency_education(self) -> str:
        """Educational content for non-transparent images."""
        return """
💡 About Image Transparency and 3D Printing

Your image is ready for 3D printing! Here's why transparency matters:

🎯 3D Printing Requirements:
  • 3D printers create solid, physical objects
  • Every part of the model needs a defined color/material
  • Transparent areas can't be directly printed

✅ Your Image:
  • Contains no transparent areas
  • All pixels have defined colors
  • Ready for color-to-material conversion
  • Suitable for multi-color 3D printing optimization
"""

    def _get_transparency_education(self, info: TransparencyInfo) -> str:
        """Educational content for transparent images.""" 
        return f"""
💡 About Image Transparency and 3D Printing

Your image contains transparency, which affects 3D printing:

🔬 What We Found:
  • {info.transparency_percentage:.1f}% of pixels are transparent
  • Format: {info.format_type} with {info.image_mode} color mode
  • {info.transparent_pixel_count:,} transparent pixels out of {info.total_pixels:,}

❓ Why Transparency Matters for 3D Printing:
  • 3D printers create solid, physical objects
  • Transparent areas represent "empty space" in your image
  • These areas can't be directly converted to filament colors
  • May result in gaps, unexpected colors, or print failures

🛠️ Common Solutions:
  • Add Background: Fill transparent areas with appropriate colors
  • Crop Image: Focus on the non-transparent subject
  • Layer Approach: Composite image onto colored background
  • Edge Cleanup: Remove anti-aliasing artifacts if minimal

🎨 Image Editing Tips:
  • Use "Flatten Image" or "Merge Visible" in your editor
  • Try "Paint Bucket" tool to fill transparent areas
  • Consider the final print's intended background color
  • Preview how transparency affects your specific design
"""


class TransparencyException(Exception):
    """
    Exception raised when transparency is detected and processing should stop.
    
    This exception carries notification information for user display.
    """
    
    def __init__(self, notification: TransparencyNotification, transparency_info: TransparencyInfo):
        """
        Initialize transparency exception.
        
        Args:
            notification: User notification about transparency
            transparency_info: Technical transparency detection results
        """
        self.notification = notification
        self.transparency_info = transparency_info
        super().__init__(notification.message)
    
    def get_cli_message(self, show_technical: bool = False, 
                       show_suggestions: bool = True) -> str:
        """Get formatted CLI message."""
        notifier = TransparencyNotifier()
        return notifier.format_cli_message(
            self.notification, 
            show_technical=show_technical,
            show_suggestions=show_suggestions
        )
    
    def get_educational_content(self) -> str:
        """Get educational content about the transparency issue."""
        notifier = TransparencyNotifier()
        return notifier.get_educational_content(self.transparency_info)