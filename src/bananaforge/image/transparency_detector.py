"""
Transparency detection for various image formats.

This module implements comprehensive transparency detection for images,
focusing on PNG format support as required by Story 1.1.
"""

from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path
from PIL import Image
import numpy as np


@dataclass
class TransparencyInfo:
    """Information about transparency detection results."""
    
    has_transparency: bool
    format_type: str
    transparency_type: str  # 'alpha_channel', 'palette_transparency', 'none'
    image_mode: str
    total_pixels: int
    transparent_pixel_count: int
    opaque_pixel_count: int
    transparency_percentage: float
    detection_details: List[str]


class TransparencyDetector:
    """
    Detector for transparency in various image formats.
    
    Implements Story 1.1 requirements for PNG alpha channel detection:
    - Detect RGBA and LA image modes
    - Check for transparency key in image info dictionary
    - Handle palette-based transparency (P mode with transparency)
    """
    
    def __init__(self):
        """Initialize the transparency detector."""
        pass
    
    def detect_transparency(self, image_path: Union[str, Path]) -> TransparencyInfo:
        """
        Detect transparency in an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            TransparencyInfo object containing detection results
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            IOError: If the file cannot be opened or is not a valid image
        """
        # Convert to Path object for easier handling
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Open image without converting mode yet
            with Image.open(path) as image:
                # Route to format-specific analysis
                if image.format == 'GIF':
                    return self._analyze_gif_transparency(image)
                elif image.format == 'WEBP':
                    return self._analyze_webp_transparency(image)
                elif image.format == 'TIFF':
                    return self._analyze_tiff_transparency(image)
                else:
                    return self._analyze_png_transparency(image)
                
        except Exception as e:
            raise IOError(f"Cannot open image file {image_path}: {e}")
    
    def _analyze_png_transparency(self, image: Image.Image) -> TransparencyInfo:
        """
        Analyze PNG image for transparency.
        
        This method implements the core logic for Story 1.1:
        - Detect RGBA mode (full alpha channel)
        - Detect palette transparency (P mode with transparency key)
        - Handle RGB mode (no transparency)
        """
        format_type = image.format or "PNG"
        image_mode = image.mode
        width, height = image.size
        total_pixels = width * height
        
        detection_details = []
        has_transparency = False
        transparency_type = "none"
        transparent_pixel_count = 0
        opaque_pixel_count = total_pixels
        
        # Check for RGBA or LA modes (full alpha channel)
        if image_mode in ('RGBA', 'LA'):
            has_transparency, transparent_count, details = self._analyze_alpha_channel(image)
            if has_transparency:
                transparency_type = "alpha_channel"
                transparent_pixel_count = transparent_count
                opaque_pixel_count = total_pixels - transparent_pixel_count
                detection_details.extend(details)
        
        # Check for palette transparency (P mode with transparency key)
        elif image_mode == 'P' and 'transparency' in image.info:
            has_transparency = True
            transparency_type = "palette_transparency"
            detection_details.append("transparency key found in palette mode")
            # For palette transparency, we'd need to analyze the actual pixel values
            # For now, we'll mark it as having transparency
            transparent_pixel_count = self._count_palette_transparent_pixels(image)
            opaque_pixel_count = total_pixels - transparent_pixel_count
        
        # RGB mode should not have transparency
        elif image_mode == 'RGB':
            has_transparency = False
            transparency_type = "none"
            transparent_pixel_count = 0
            opaque_pixel_count = total_pixels
            detection_details.append("RGB mode - no transparency support")
        
        # Other modes
        else:
            detection_details.append(f"Mode {image_mode} - transparency support unknown")
        
        # Calculate transparency percentage
        transparency_percentage = (transparent_pixel_count / total_pixels * 100) if total_pixels > 0 else 0.0
        
        return TransparencyInfo(
            has_transparency=has_transparency,
            format_type=format_type,
            transparency_type=transparency_type,
            image_mode=image_mode,
            total_pixels=total_pixels,
            transparent_pixel_count=transparent_pixel_count,
            opaque_pixel_count=opaque_pixel_count,
            transparency_percentage=transparency_percentage,
            detection_details=detection_details
        )
    
    def _analyze_alpha_channel(self, image: Image.Image) -> tuple[bool, int, List[str]]:
        """
        Analyze alpha channel for transparency.
        
        Returns:
            Tuple of (has_transparency, transparent_pixel_count, details)
        """
        details = []
        
        if image.mode == 'RGBA':
            details.append("RGBA mode detected")
            
            # Convert to numpy array for efficient analysis
            img_array = np.array(image)
            alpha_channel = img_array[:, :, 3]  # Alpha is the 4th channel
            
            # Count pixels with alpha < 255 (not fully opaque)
            transparent_pixels = np.sum(alpha_channel < 255)
            
            if transparent_pixels > 0:
                details.append("Alpha channel present")
                return True, int(transparent_pixels), details
            else:
                details.append("Alpha channel present but all pixels opaque")
                return False, 0, details
                
        elif image.mode == 'LA':
            details.append("LA mode detected")
            
            # Convert to numpy array
            img_array = np.array(image)
            alpha_channel = img_array[:, :, 1]  # Alpha is the 2nd channel in LA
            
            # Count transparent pixels
            transparent_pixels = np.sum(alpha_channel < 255)
            
            if transparent_pixels > 0:
                details.append("Alpha channel present")
                return True, int(transparent_pixels), details
            else:
                details.append("Alpha channel present but all pixels opaque")
                return False, 0, details
        
        return False, 0, details
    
    def _count_palette_transparent_pixels(self, image: Image.Image) -> int:
        """
        Count transparent pixels in a palette mode image.
        
        For palette images with transparency, we need to check which
        palette indices are marked as transparent.
        """
        if 'transparency' not in image.info:
            return 0
        
        transparency_info = image.info['transparency']
        
        # Convert image to array for analysis
        img_array = np.array(image)
        
        if isinstance(transparency_info, int):
            # Single transparent color index
            transparent_pixels = np.sum(img_array == transparency_info)
        elif isinstance(transparency_info, (list, tuple, bytes)):
            # Multiple transparent indices or alpha values
            transparent_pixels = 0
            for i, alpha in enumerate(transparency_info):
                if alpha < 255:  # Semi-transparent or fully transparent
                    transparent_pixels += np.sum(img_array == i)
        else:
            # Unknown transparency format
            transparent_pixels = 0
        
        return int(transparent_pixels)
    
    def _analyze_gif_transparency(self, image: Image.Image) -> TransparencyInfo:
        """
        Analyze GIF image for transparency.
        
        This method implements GIF-specific transparency detection:
        - Handle GIF transparency index
        - Support animated GIFs (analyze first frame)
        - Detect palette-based transparency specific to GIF format
        """
        format_type = "GIF"
        image_mode = image.mode
        width, height = image.size
        total_pixels = width * height
        
        detection_details = []
        has_transparency = False
        transparency_type = "none"
        transparent_pixel_count = 0
        opaque_pixel_count = total_pixels
        
        # Check if this is an animated GIF
        is_animated = getattr(image, 'is_animated', False)
        if is_animated:
            detection_details.append(f"animated GIF detected with {getattr(image, 'n_frames', 1)} frames")
            detection_details.append("analyzing first frame for transparency")
        
        # GIF transparency is always palette-based when present
        if image_mode == 'P' and 'transparency' in image.info:
            has_transparency = True
            transparency_type = "gif_transparency"
            
            # Get transparency information from GIF
            transparency_info = image.info['transparency']
            detection_details.append(f"GIF transparency index: {transparency_info}")
            
            # Count transparent pixels
            transparent_pixel_count = self._count_gif_transparent_pixels(image, transparency_info)
            opaque_pixel_count = total_pixels - transparent_pixel_count
            
            if is_animated:
                detection_details.append("transparency analysis based on first frame of animation")
            
        elif image_mode == 'P':
            # Palette mode but no transparency
            has_transparency = False
            transparency_type = "none"
            transparent_pixel_count = 0
            opaque_pixel_count = total_pixels
            detection_details.append("GIF in palette mode but no transparency index found")
            
        else:
            # Non-palette GIF (unusual but possible)
            has_transparency = False
            transparency_type = "none"
            transparent_pixel_count = 0
            opaque_pixel_count = total_pixels
            detection_details.append(f"GIF in {image_mode} mode - no transparency support")
        
        # Calculate transparency percentage
        transparency_percentage = (transparent_pixel_count / total_pixels * 100) if total_pixels > 0 else 0.0
        
        return TransparencyInfo(
            has_transparency=has_transparency,
            format_type=format_type,
            transparency_type=transparency_type,
            image_mode=image_mode,
            total_pixels=total_pixels,
            transparent_pixel_count=transparent_pixel_count,
            opaque_pixel_count=opaque_pixel_count,
            transparency_percentage=transparency_percentage,
            detection_details=detection_details
        )
    
    def _count_gif_transparent_pixels(self, image: Image.Image, transparency_index: int) -> int:
        """
        Count transparent pixels in a GIF image.
        
        For GIF images, transparency is determined by a single palette index.
        """
        # Convert image to array for analysis
        img_array = np.array(image)
        
        # Count pixels that match the transparency index
        transparent_pixels = np.sum(img_array == transparency_index)
        
        return int(transparent_pixels)
    
    def _analyze_webp_transparency(self, image: Image.Image) -> TransparencyInfo:
        """
        Analyze WebP image for transparency.
        
        WebP supports both lossy and lossless compression with alpha channels.
        This method handles WebP-specific transparency detection and provides
        appropriate statistics for both compression types.
        """
        format_type = "WEBP"
        image_mode = image.mode
        width, height = image.size
        total_pixels = width * height
        
        detection_details = []
        has_transparency = False
        transparency_type = "none"
        transparent_pixel_count = 0
        opaque_pixel_count = total_pixels
        
        # WebP transparency is alpha channel based when present
        if image_mode in ('RGBA', 'LA'):
            has_transparency, transparent_count, details = self._analyze_alpha_channel(image)
            if has_transparency:
                transparency_type = "alpha_channel"
                transparent_pixel_count = transparent_count
                opaque_pixel_count = total_pixels - transparent_pixel_count
                detection_details.extend(details)
                
                # Add WebP-specific details
                if 'lossless' in str(image.info).lower():
                    detection_details.append("WebP lossless format with alpha channel")
                else:
                    detection_details.append("WebP format with alpha channel")
                    
        else:
            # No alpha channel in WebP
            has_transparency = False
            transparency_type = "none"
            transparent_pixel_count = 0
            opaque_pixel_count = total_pixels
            detection_details.append(f"WebP in {image_mode} mode - no alpha channel")
        
        # Check for any WebP-specific transparency information
        if 'transparency' in image.info:
            detection_details.append("WebP transparency metadata found")
        
        # Calculate transparency percentage
        transparency_percentage = (transparent_pixel_count / total_pixels * 100) if total_pixels > 0 else 0.0
        
        return TransparencyInfo(
            has_transparency=has_transparency,
            format_type=format_type,
            transparency_type=transparency_type,
            image_mode=image_mode,
            total_pixels=total_pixels,
            transparent_pixel_count=transparent_pixel_count,
            opaque_pixel_count=opaque_pixel_count,
            transparency_percentage=transparency_percentage,
            detection_details=detection_details
        )
    
    def _analyze_tiff_transparency(self, image: Image.Image) -> TransparencyInfo:
        """
        Analyze TIFF image for transparency.
        
        TIFF supports alpha channels and can store high-quality images
        with transparency. This method handles TIFF-specific transparency
        detection consistently with other formats.
        """
        format_type = "TIFF"
        image_mode = image.mode
        width, height = image.size
        total_pixels = width * height
        
        detection_details = []
        has_transparency = False
        transparency_type = "none"
        transparent_pixel_count = 0
        opaque_pixel_count = total_pixels
        
        # TIFF transparency is alpha channel based when present
        if image_mode in ('RGBA', 'LA'):
            has_transparency, transparent_count, details = self._analyze_alpha_channel(image)
            if has_transparency:
                transparency_type = "alpha_channel"
                transparent_pixel_count = transparent_count
                opaque_pixel_count = total_pixels - transparent_pixel_count
                detection_details.extend(details)
                detection_details.append("TIFF format with alpha channel")
                
        else:
            # No alpha channel in TIFF
            has_transparency = False
            transparency_type = "none"
            transparent_pixel_count = 0
            opaque_pixel_count = total_pixels
            detection_details.append(f"TIFF in {image_mode} mode - no alpha channel")
        
        # Check TIFF-specific transparency information
        if hasattr(image, 'tag') and image.tag:
            # TIFF can have transparency-related tags
            detection_details.append("TIFF metadata analyzed")
        
        # Calculate transparency percentage
        transparency_percentage = (transparent_pixel_count / total_pixels * 100) if total_pixels > 0 else 0.0
        
        return TransparencyInfo(
            has_transparency=has_transparency,
            format_type=format_type,
            transparency_type=transparency_type,
            image_mode=image_mode,
            total_pixels=total_pixels,
            transparent_pixel_count=transparent_pixel_count,
            opaque_pixel_count=opaque_pixel_count,
            transparency_percentage=transparency_percentage,
            detection_details=detection_details
        )