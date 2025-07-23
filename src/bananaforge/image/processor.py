"""Image processing utilities for BananaForge."""

from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from ..utils.color import ColorConverter
from .transparency_detector import TransparencyDetector, TransparencyInfo


class ImageProcessor:
    """Main image processing class for preparing images for optimization."""

    def __init__(self, device: str = "cpu"):
        """Initialize image processor.

        Args:
            device: Device for tensor operations
        """
        self.device = torch.device(device)
        self.color_converter = ColorConverter(device)

        # Standard transforms
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

        # Normalization for neural networks
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        
        # Transparency detection integration
        self.transparency_detector = TransparencyDetector()

    def load_image(
        self,
        image_path: str,
        target_size: Optional[Tuple[int, int]] = None,
        maintain_aspect: bool = True,
    ) -> torch.Tensor:
        """Load and preprocess image from file.

        Args:
            image_path: Path to image file
            target_size: Optional target size (height, width)
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            Preprocessed image tensor (1, 3, H, W)
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Resize if needed
        if target_size is not None:
            if maintain_aspect:
                image = self._resize_with_aspect(image, target_size)
            else:
                image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)

        # Convert to tensor
        tensor = self.to_tensor(image).unsqueeze(0).to(self.device)

        return tensor

    def resize_color_preserving(
        self,
        image: torch.Tensor,
        target_size: Tuple[int, int],
        preserve_edges: bool = True,
    ) -> torch.Tensor:
        """Resize image while preserving color details.

        Uses INTER_AREA for downscaling and INTER_CUBIC for upscaling
        to maintain color fidelity and edge sharpness.

        Args:
            image: Input image tensor (B, 3, H, W)
            target_size: Target size (height, width)
            preserve_edges: Whether to use edge-preserving interpolation

        Returns:
            Resized image tensor
        """
        current_h, current_w = image.shape[-2:]
        target_h, target_w = target_size

        # Convert to numpy for OpenCV processing
        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)

        # Determine if we're upscaling or downscaling
        scale_factor = (target_h * target_w) / (current_h * current_w)

        if scale_factor < 1.0:  # Downscaling
            # Use INTER_AREA for better color preservation when shrinking
            interpolation = cv2.INTER_AREA
        else:  # Upscaling
            # Use INTER_CUBIC for edge preservation when enlarging
            interpolation = cv2.INTER_CUBIC if preserve_edges else cv2.INTER_LINEAR

        # Resize using OpenCV
        resized_np = cv2.resize(
            image_np, (target_w, target_h), interpolation=interpolation
        )

        # Convert back to tensor
        resized_tensor = torch.from_numpy(resized_np).float() / 255.0
        resized_tensor = resized_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        return resized_tensor

    def enhance_saturation(
        self,
        image: torch.Tensor,
        enhancement_factor: float,
        method: str = "hsl",
    ) -> torch.Tensor:
        """Enhance image saturation intelligently.

        Args:
            image: Input image tensor (B, 3, H, W) in range [0, 1]
            enhancement_factor: Enhancement factor (0.0 = no change, 1.0 = double saturation)
            method: Enhancement method ('hsl' or 'lab')

        Returns:
            Saturation-enhanced image tensor
        """
        if method == "lab":
            return self._enhance_saturation_lab(image, enhancement_factor)
        else:
            return self._enhance_saturation_hsl(image, enhancement_factor)

    def _enhance_saturation_hsl(
        self, image: torch.Tensor, enhancement_factor: float
    ) -> torch.Tensor:
        """Enhance saturation using HSL method (faster)."""
        # Calculate luminance weights for grayscale conversion
        weights = torch.tensor(
            [0.2989, 0.5870, 0.1140], device=self.device, dtype=image.dtype
        )

        # Calculate grayscale version
        if image.dim() == 4:  # Batch dimension
            gray = (image * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        else:  # Single image
            gray = (image * weights.view(3, 1, 1)).sum(dim=0, keepdim=True)

        gray = gray.expand_as(image)

        # Enhance saturation by interpolating between grayscale and original
        factor = 1.0 + enhancement_factor
        enhanced = gray + factor * (image - gray)

        # Clamp to valid range
        enhanced = torch.clamp(enhanced, 0.0, 1.0)

        return enhanced

    def _enhance_saturation_lab(
        self, image: torch.Tensor, enhancement_factor: float
    ) -> torch.Tensor:
        """Enhance saturation using LAB color space (more accurate)."""
        # Convert to LAB (expects [0, 255] range)
        image_255 = image * 255.0
        lab_image = self.color_converter.rgb_to_lab(image_255)

        # Enhance saturation in LAB space
        enhanced_lab = self.color_converter.enhance_saturation_lab(
            lab_image, enhancement_factor
        )

        # Convert back to RGB
        enhanced_rgb = self.color_converter.lab_to_rgb(enhanced_lab)

        # Normalize back to [0, 1] range
        enhanced = enhanced_rgb / 255.0
        enhanced = torch.clamp(enhanced, 0.0, 1.0)

        return enhanced

    def load_and_process_enhanced(
        self,
        image_path: str,
        target_size: Tuple[int, int],
        enable_lab_conversion: bool = True,
        saturation_enhancement: float = 0.0,
        use_color_preserving_resize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Load and process image with all enhancements.

        Complete pipeline implementation for Feature 1.

        Args:
            image_path: Path to image file
            target_size: Target size (height, width)
            enable_lab_conversion: Whether to provide LAB conversion
            saturation_enhancement: Saturation enhancement factor
            use_color_preserving_resize: Whether to use enhanced resizing

        Returns:
            Dictionary containing processed images and metadata
        """
        # Load original image
        original_image = self.load_image(image_path)

        # Apply saturation enhancement if requested
        if saturation_enhancement > 0:
            enhanced_image = self.enhance_saturation(
                original_image, saturation_enhancement, method="lab"
            )
        else:
            enhanced_image = original_image

        # Resize using color-preserving method if requested
        if use_color_preserving_resize:
            resized_image = self.resize_color_preserving(enhanced_image, target_size)
        else:
            resized_image = F.interpolate(
                enhanced_image, size=target_size, mode="bilinear", align_corners=False
            )

        result = {
            "rgb_image": resized_image,
            "original_size": original_image.shape[-2:],
            "target_size": target_size,
            "saturation_enhanced": saturation_enhancement > 0,
            "color_preserving_resize": use_color_preserving_resize,
        }

        # Add LAB conversion if requested
        if enable_lab_conversion:
            lab_image = self.color_converter.rgb_to_lab(resized_image * 255.0)
            result["lab_image"] = lab_image

        return result

    def _resize_with_aspect(
        self, image: Image.Image, target_size: Tuple[int, int]
    ) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        target_h, target_w = target_size
        orig_w, orig_h = image.size

        # Calculate scaling factor
        scale = min(target_w / orig_w, target_h / orig_h)

        # Calculate new size
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize image
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Pad to target size
        if new_w != target_w or new_h != target_h:
            # Create padded image
            padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))

            # Calculate padding offset
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2

            # Paste resized image
            padded.paste(image, (x_offset, y_offset))
            image = padded

        return image

    def preprocess_for_optimization(
        self, image: torch.Tensor, target_resolution: int = 256
    ) -> torch.Tensor:
        """Preprocess image for optimization.

        Args:
            image: Input image tensor (1, 3, H, W)
            target_resolution: Target resolution for optimization

        Returns:
            Preprocessed image tensor
        """
        # Resize to target resolution
        if image.shape[-1] != target_resolution or image.shape[-2] != target_resolution:
            image = F.interpolate(
                image,
                size=(target_resolution, target_resolution),
                mode="bilinear",
                align_corners=False,
            )

        # Ensure values are in [0, 1]
        image = torch.clamp(image, 0, 1)

        return image

    def enhance_contrast(
        self, image: torch.Tensor, factor: float = 1.2
    ) -> torch.Tensor:
        """Enhance image contrast.

        Args:
            image: Input image tensor (1, 3, H, W)
            factor: Contrast enhancement factor

        Returns:
            Contrast-enhanced image
        """
        # Convert to grayscale for mean calculation
        gray = torch.mean(image, dim=1, keepdim=True)
        mean_val = torch.mean(gray)

        # Apply contrast enhancement
        enhanced = (image - mean_val) * factor + mean_val

        return torch.clamp(enhanced, 0, 1)

    def adjust_gamma(self, image: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        """Apply gamma correction.

        Args:
            image: Input image tensor (1, 3, H, W)
            gamma: Gamma value (< 1 brightens, > 1 darkens)

        Returns:
            Gamma-corrected image
        """
        return torch.pow(image, gamma)

    def apply_bilateral_filter(
        self,
        image: torch.Tensor,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75,
    ) -> torch.Tensor:
        """Apply bilateral filter for edge-preserving smoothing.

        Args:
            image: Input image tensor (1, 3, H, W)
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space

        Returns:
            Filtered image tensor
        """
        # Convert to numpy for OpenCV
        np_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        np_image = (np_image * 255).astype(np.uint8)

        # Apply bilateral filter
        filtered = cv2.bilateralFilter(np_image, d, sigma_color, sigma_space)

        # Convert back to tensor
        filtered_tensor = torch.from_numpy(filtered).float() / 255.0
        filtered_tensor = filtered_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        return filtered_tensor

    def extract_edges(
        self, image: torch.Tensor, threshold1: float = 50, threshold2: float = 150
    ) -> torch.Tensor:
        """Extract edges using Canny edge detection.

        Args:
            image: Input image tensor (1, 3, H, W)
            threshold1: First threshold for edge linking
            threshold2: Second threshold for edge linking

        Returns:
            Edge map tensor (1, 1, H, W)
        """
        # Convert to grayscale
        gray = torch.mean(image, dim=1, keepdim=True)

        # Convert to numpy for OpenCV
        np_gray = gray.squeeze().cpu().numpy()
        np_gray = (np_gray * 255).astype(np.uint8)

        # Apply Canny edge detection
        edges = cv2.Canny(np_gray, threshold1, threshold2)

        # Convert back to tensor
        edge_tensor = torch.from_numpy(edges).float() / 255.0
        edge_tensor = edge_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        return edge_tensor

    def create_distance_transform(self, edge_map: torch.Tensor) -> torch.Tensor:
        """Create distance transform from edge map.

        Args:
            edge_map: Binary edge map (1, 1, H, W)

        Returns:
            Distance transform (1, 1, H, W)
        """
        # Convert to numpy
        edges_np = edge_map.squeeze().cpu().numpy()
        edges_np = (edges_np > 0.5).astype(np.uint8)

        # Compute distance transform
        dist_transform = cv2.distanceTransform(
            1 - edges_np, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )

        # Normalize
        if dist_transform.max() > 0:
            dist_transform = dist_transform / dist_transform.max()

        # Convert back to tensor
        dist_tensor = torch.from_numpy(dist_transform).float()
        dist_tensor = dist_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        return dist_tensor

    def segment_colors(
        self, image: torch.Tensor, num_colors: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Segment image into dominant colors using K-means.

        Args:
            image: Input image tensor (1, 3, H, W)
            num_colors: Number of color clusters

        Returns:
            Tuple of (segmented_image, color_centers)
        """
        # Reshape for clustering
        h, w = image.shape[-2:]
        pixels = image.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        pixels_np = pixels.cpu().numpy()

        # K-means clustering
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels_np)
        centers = kmeans.cluster_centers_

        # Create segmented image
        segmented_pixels = centers[labels]
        segmented_image = torch.from_numpy(segmented_pixels).float()
        segmented_image = segmented_image.reshape(h, w, 3).permute(2, 0, 1).unsqueeze(0)
        segmented_image = segmented_image.to(self.device)

        # Color centers
        color_centers = torch.from_numpy(centers).float().to(self.device)

        return segmented_image, color_centers

    def create_depth_cues(self, image: torch.Tensor) -> torch.Tensor:
        """Create depth cues from image characteristics.

        Args:
            image: Input image tensor (1, 3, H, W)

        Returns:
            Depth cue map (1, 1, H, W)
        """
        # Convert to grayscale
        gray = torch.mean(image, dim=1, keepdim=True)

        # Brightness-based depth (darker = further)
        brightness_depth = 1.0 - gray

        # Blur-based depth (more blurred = further)
        blur_kernel = torch.ones(1, 1, 5, 5, device=self.device) / 25.0
        blurred = F.conv2d(gray, blur_kernel, padding=2)
        blur_diff = torch.abs(gray - blurred)
        blur_depth = 1.0 - blur_diff / (blur_diff.max() + 1e-6)

        # Edge-based depth (fewer edges = further)
        edges = self.extract_edges(image)
        edge_depth = 1.0 - edges

        # Combine depth cues
        depth_cues = 0.4 * brightness_depth + 0.3 * blur_depth + 0.3 * edge_depth

        return depth_cues

    def save_tensor_as_image(self, tensor: torch.Tensor, filepath: str) -> None:
        """Save tensor as image file.

        Args:
            tensor: Image tensor (1, 3, H, W) or (3, H, W)
            filepath: Output file path
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Clamp and convert to PIL
        tensor = torch.clamp(tensor, 0, 1)
        pil_image = self.to_pil(tensor.cpu())
        pil_image.save(filepath)

    def create_thumbnail(
        self, image: torch.Tensor, size: Tuple[int, int] = (128, 128)
    ) -> torch.Tensor:
        """Create thumbnail of image.

        Args:
            image: Input image tensor (1, 3, H, W)
            size: Thumbnail size (height, width)

        Returns:
            Thumbnail tensor
        """
        return F.interpolate(image, size=size, mode="bilinear", align_corners=False)
    
    # Transparency detection integration methods
    def detect_transparency(self, image_path: Union[str, Path]) -> TransparencyInfo:
        """
        Detect transparency in an image file.
        
        Integrates transparency detection cleanly with ImageProcessor while
        maintaining separation of concerns. Detection does not modify the image
        or affect processing operations.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            TransparencyInfo object containing detection results
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            IOError: If the file cannot be opened or is not a valid image
        """
        return self.transparency_detector.detect_transparency(image_path)
    
    def load_image_with_transparency(
        self, 
        image_path: Union[str, Path], 
        target_size: Optional[Tuple[int, int]] = None,
        maintain_aspect: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image with optional alpha channel preservation.
        
        This method provides alpha channel preservation while maintaining
        backward compatibility. The RGB processing follows the same pipeline
        as the standard load_image method.
        
        Args:
            image_path: Path to image file
            target_size: Optional target size (height, width)
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Tuple of (rgb_tensor, alpha_mask)
            - rgb_tensor: RGB image tensor (1, 3, H, W)
            - alpha_mask: Alpha mask tensor (1, 1, H, W)
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            IOError: If the file cannot be opened or is not a valid image
        """
        from pathlib import Path
        
        # Convert to Path object for consistent handling
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Open image preserving original mode
            with Image.open(path) as image:
                original_mode = image.mode
                
                # Handle different image modes for alpha extraction
                alpha_mask = None
                
                if original_mode == 'RGBA':
                    # Extract alpha channel before RGB conversion
                    alpha_channel = image.split()[-1]  # Alpha is last channel
                    alpha_array = np.array(alpha_channel)
                    alpha_mask = torch.from_numpy(alpha_array).float() / 255.0
                    alpha_mask = alpha_mask.unsqueeze(0).unsqueeze(0).to(self.device)
                    
                elif original_mode == 'LA':
                    # Grayscale with alpha
                    alpha_channel = image.split()[-1]
                    alpha_array = np.array(alpha_channel)
                    alpha_mask = torch.from_numpy(alpha_array).float() / 255.0
                    alpha_mask = alpha_mask.unsqueeze(0).unsqueeze(0).to(self.device)
                    
                elif original_mode == 'P' and 'transparency' in image.info:
                    # Palette with transparency key
                    # Convert to RGBA first to extract alpha properly
                    rgba_image = image.convert('RGBA')
                    alpha_channel = rgba_image.split()[-1]
                    alpha_array = np.array(alpha_channel)
                    alpha_mask = torch.from_numpy(alpha_array).float() / 255.0
                    alpha_mask = alpha_mask.unsqueeze(0).unsqueeze(0).to(self.device)
                    
                else:
                    # No transparency - create opaque alpha mask
                    width, height = image.size
                    alpha_mask = torch.ones(1, 1, height, width, device=self.device)
                
                # Now process RGB using existing pipeline (ensures backward compatibility)
                rgb_image = image.convert("RGB")
                
                # Resize if needed (same logic as load_image)
                if target_size is not None:
                    if maintain_aspect:
                        rgb_image = self._resize_with_aspect(rgb_image, target_size)
                    else:
                        rgb_image = rgb_image.resize((target_size[1], target_size[0]), Image.LANCZOS)
                    
                    # Also resize alpha mask to match
                    if alpha_mask is not None:
                        target_h, target_w = target_size
                        alpha_mask = F.interpolate(
                            alpha_mask, size=(target_h, target_w), 
                            mode='bilinear', align_corners=False
                        )
                
                # Convert RGB to tensor (same as load_image)
                rgb_tensor = self.to_tensor(rgb_image).unsqueeze(0).to(self.device)
                
                return rgb_tensor, alpha_mask
                
        except Exception as e:
            raise IOError(f"Cannot open image file {image_path}: {e}")


class BatchImageProcessor:
    """Batch processing utilities for multiple images."""

    def __init__(self, device: str = "cpu"):
        """Initialize batch processor."""
        self.processor = ImageProcessor(device)
        self.device = device

    def process_batch(
        self, image_paths: list, target_size: Tuple[int, int], batch_size: int = 4
    ) -> torch.Tensor:
        """Process multiple images in batches.

        Args:
            image_paths: List of image file paths
            target_size: Target size for all images
            batch_size: Batch size for processing

        Returns:
            Batched image tensor (N, 3, H, W)
        """
        processed_images = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []

            for path in batch_paths:
                img = self.processor.load_image(path, target_size)
                batch_images.append(img.squeeze(0))

            if batch_images:
                batch_tensor = torch.stack(batch_images, dim=0)
                processed_images.append(batch_tensor)

        return (
            torch.cat(processed_images, dim=0) if processed_images else torch.empty(0)
        )

    def apply_augmentations(
        self, images: torch.Tensor, augment_prob: float = 0.5
    ) -> torch.Tensor:
        """Apply random augmentations to batch of images.

        Args:
            images: Batch of images (N, 3, H, W)
            augment_prob: Probability of applying each augmentation

        Returns:
            Augmented images
        """
        augmented = images.clone()

        for i in range(images.shape[0]):
            img = images[i : i + 1]

            # Random contrast adjustment
            if torch.rand(1) < augment_prob:
                factor = 0.8 + torch.rand(1) * 0.4  # 0.8 to 1.2
                img = self.processor.enhance_contrast(img, factor.item())

            # Random gamma correction
            if torch.rand(1) < augment_prob:
                gamma = 0.8 + torch.rand(1) * 0.4  # 0.8 to 1.2
                img = self.processor.adjust_gamma(img, gamma.item())

            augmented[i] = img.squeeze(0)

        return augmented
