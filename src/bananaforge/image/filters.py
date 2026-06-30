"""Advanced filtering and analysis utilities for image processing."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class EdgeDetector:
    """Advanced edge detection algorithms."""

    def __init__(self, device: str = "cpu"):
        """Initialize edge detector.

        Args:
            device: Device for computations
        """
        self.device = torch.device(device)

    def multi_scale_edges(
        self, image: torch.Tensor, scales: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Detect edges at multiple scales.

        Args:
            image: Input image (1, 3, H, W)
            scales: List of scales for edge detection

        Returns:
            Multi-scale edge map (1, 1, H, W)
        """
        if scales is None:
            scales = [1.0, 2.0, 4.0]

        gray = torch.mean(image, dim=1, keepdim=True)
        edge_maps = []

        for scale in scales:
            # Apply Gaussian smoothing at different scales
            smoothed = self._gaussian_blur(gray, sigma=scale)

            # Sobel edge detection
            edges = self._sobel_edges(smoothed)
            edge_maps.append(edges)

        # Combine edge maps
        combined_edges = torch.stack(edge_maps, dim=0).max(dim=0)[0]

        return combined_edges

    def canny_edges(
        self,
        image: torch.Tensor,
        low_threshold: float = 50,
        high_threshold: float = 150,
        kernel_size: int = 3,
    ) -> torch.Tensor:
        """Canny edge detection.

        Args:
            image: Input image (1, 3, H, W)
            low_threshold: Low threshold for edge linking
            high_threshold: High threshold for edge linking
            kernel_size: Gaussian blur kernel size

        Returns:
            Canny edge map (1, 1, H, W)
        """
        # Convert to grayscale and numpy
        gray = torch.mean(image, dim=1, keepdim=True)
        gray_np = gray.squeeze().cpu().numpy()
        gray_np = (gray_np * 255).astype(np.uint8)

        # Apply Canny edge detection
        edges = cv2.Canny(
            gray_np, low_threshold, high_threshold, apertureSize=kernel_size
        )

        # Convert back to tensor
        edge_tensor = torch.from_numpy(edges).float() / 255.0
        edge_tensor = edge_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        return edge_tensor

    def oriented_edges(
        self, image: torch.Tensor, num_orientations: int = 8
    ) -> torch.Tensor:
        """Detect edges at specific orientations.

        Args:
            image: Input image (1, 3, H, W)
            num_orientations: Number of orientation bins

        Returns:
            Orientation-specific edge map (1, num_orientations, H, W)
        """
        gray = torch.mean(image, dim=1, keepdim=True)
        oriented_edges = []

        for i in range(num_orientations):
            angle = i * 180.0 / num_orientations

            # Create oriented filter
            kernel = self._create_oriented_kernel(angle, size=7)
            kernel = kernel.to(self.device).unsqueeze(0).unsqueeze(0)

            # Apply filter
            response = torch.abs(F.conv2d(gray, kernel, padding=3))
            oriented_edges.append(response)

        return torch.cat(oriented_edges, dim=1)

    def _sobel_edges(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Sobel edge detection."""
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=self.device,
        )

        grad_x = F.conv2d(image, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        grad_y = F.conv2d(image, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)

        return torch.sqrt(grad_x**2 + grad_y**2)

    def _gaussian_blur(self, image: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur."""
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create Gaussian kernel
        x = torch.arange(
            -kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32
        )
        kernel = torch.exp(-(x**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        # Separable convolution
        kernel_x = kernel.view(1, 1, 1, -1).to(self.device)
        kernel_y = kernel.view(1, 1, -1, 1).to(self.device)

        blurred = F.conv2d(image, kernel_x, padding=(0, kernel_size // 2))
        blurred = F.conv2d(blurred, kernel_y, padding=(kernel_size // 2, 0))

        return blurred

    def _create_oriented_kernel(self, angle: float, size: int = 7) -> torch.Tensor:
        """Create oriented edge detection kernel."""
        # Create coordinate grids
        x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
        y = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # Rotate coordinates
        angle_rad = np.radians(angle)
        X_rot = X * np.cos(angle_rad) + Y * np.sin(angle_rad)

        # Create oriented derivative kernel
        kernel = -X_rot * torch.exp(-(X**2 + Y**2) / (2 * 1.0**2))

        # Normalize
        kernel = kernel - kernel.mean()
        kernel = kernel / (kernel.abs().sum() + 1e-6)

        return kernel


class ColorAnalyzer:
    """Advanced color analysis and manipulation."""

    def __init__(self, device: str = "cpu"):
        """Initialize color analyzer."""
        self.device = torch.device(device)

    def extract_dominant_colors(
        self, image: torch.Tensor, num_colors: int = 8, method: str = "kmeans"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract dominant colors from image.

        Args:
            image: Input image (1, 3, H, W)
            num_colors: Number of colors to extract
            method: Color extraction method ("kmeans" or "median_cut")

        Returns:
            Tuple of (color_palette, color_weights)
        """
        if method == "kmeans":
            return self._kmeans_colors(image, num_colors)
        elif method == "median_cut":
            return self._median_cut_colors(image, num_colors)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _kmeans_colors(
        self, image: torch.Tensor, num_colors: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract colors using K-means clustering."""
        # Reshape image for clustering
        h, w = image.shape[-2:]
        pixels = image.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        pixels_np = pixels.cpu().numpy()

        # K-means clustering
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels_np)

        # Get color centers and weights
        centers = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)

        # Calculate color weights
        unique_labels, counts = np.unique(labels, return_counts=True)
        weights = torch.from_numpy(counts.astype(np.float32)).to(self.device)
        weights = weights / weights.sum()

        return centers, weights

    def _median_cut_colors(
        self, image: torch.Tensor, num_colors: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract colors using median cut algorithm."""
        # Convert to PIL for median cut
        from PIL import Image as PILImage

        # Convert tensor to PIL
        img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img_np)

        # Quantize using median cut
        quantized = pil_img.quantize(colors=num_colors, method=PILImage.MEDIANCUT)
        palette = quantized.getpalette()

        # Reshape palette
        palette_rgb = np.array(palette[: num_colors * 3]).reshape(-1, 3)
        centers = torch.from_numpy(palette_rgb).float().to(self.device) / 255.0

        # Calculate weights (uniform for median cut)
        weights = torch.ones(num_colors, device=self.device) / num_colors

        return centers, weights

    def analyze_color_harmony(self, colors: torch.Tensor) -> dict:
        """Analyze color harmony relationships.

        Args:
            colors: Color palette (num_colors, 3)

        Returns:
            Dictionary of harmony metrics
        """
        # Convert to HSV for better analysis
        hsv_colors = self._rgb_to_hsv(colors)

        hues = hsv_colors[:, 0]
        saturations = hsv_colors[:, 1]
        values = hsv_colors[:, 2]

        # Analyze harmony types
        metrics = {
            "hue_variance": torch.var(hues).item(),
            "saturation_variance": torch.var(saturations).item(),
            "value_variance": torch.var(values).item(),
            "complementary_score": self._complementary_score(hues),
            "analogous_score": self._analogous_score(hues),
            "triadic_score": self._triadic_score(hues),
        }

        return metrics

    def _rgb_to_hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to HSV color space."""
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

        max_rgb = torch.max(torch.max(r, g), b)
        min_rgb = torch.min(torch.min(r, g), b)
        delta = max_rgb - min_rgb

        # Value
        v = max_rgb

        # Saturation
        s = torch.where(max_rgb != 0, delta / max_rgb, torch.zeros_like(max_rgb))

        # Hue
        h = torch.zeros_like(max_rgb)

        # Red is max
        mask = (max_rgb == r) & (delta != 0)
        h[mask] = (60 * ((g[mask] - b[mask]) / delta[mask]) + 360) % 360

        # Green is max
        mask = (max_rgb == g) & (delta != 0)
        h[mask] = (60 * ((b[mask] - r[mask]) / delta[mask]) + 120) % 360

        # Blue is max
        mask = (max_rgb == b) & (delta != 0)
        h[mask] = (60 * ((r[mask] - g[mask]) / delta[mask]) + 240) % 360

        return torch.stack([h / 360.0, s, v], dim=1)

    def _complementary_score(self, hues: torch.Tensor) -> float:
        """Calculate complementary color harmony score."""
        # Look for hues that are ~180 degrees apart
        scores = []
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                diff = abs(hues[i] - hues[j])
                diff = min(diff, 1.0 - diff)  # Handle wraparound
                if 0.4 < diff < 0.6:  # Around 180 degrees
                    scores.append(1.0 - abs(diff - 0.5) * 2)

        return np.mean(scores) if scores else 0.0

    def _analogous_score(self, hues: torch.Tensor) -> float:
        """Calculate analogous color harmony score."""
        # Look for hues that are close together
        scores = []
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                diff = abs(hues[i] - hues[j])
                diff = min(diff, 1.0 - diff)  # Handle wraparound
                if diff < 0.17:  # Within ~60 degrees
                    scores.append(1.0 - diff / 0.17)

        return np.mean(scores) if scores else 0.0

    def _triadic_score(self, hues: torch.Tensor) -> float:
        """Calculate triadic color harmony score."""
        if len(hues) < 3:
            return 0.0

        # Look for three hues ~120 degrees apart
        best_score = 0.0
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                for k in range(j + 1, len(hues)):
                    h1, h2, h3 = hues[i], hues[j], hues[k]

                    # Sort hues
                    sorted_hues = torch.sort(torch.stack([h1, h2, h3]))[0]

                    # Calculate differences
                    diff1 = sorted_hues[1] - sorted_hues[0]
                    diff2 = sorted_hues[2] - sorted_hues[1]
                    diff3 = (1.0 + sorted_hues[0]) - sorted_hues[2]

                    # Check if close to 120 degrees each
                    target = 1.0 / 3.0
                    score = 1.0 - (
                        abs(diff1 - target) + abs(diff2 - target) + abs(diff3 - target)
                    )
                    best_score = max(best_score, score)

        return best_score

    def create_color_map(
        self, image: torch.Tensor, target_colors: torch.Tensor
    ) -> torch.Tensor:
        """Map image colors to target color palette.

        Args:
            image: Input image (1, 3, H, W)
            target_colors: Target color palette (num_colors, 3)

        Returns:
            Color-mapped image (1, 3, H, W)
        """
        h, w = image.shape[-2:]
        pixels = image.squeeze(0).permute(1, 2, 0).reshape(-1, 3)

        # Find closest target color for each pixel
        distances = torch.cdist(pixels, target_colors)
        closest_indices = torch.argmin(distances, dim=1)

        # Map to target colors
        mapped_pixels = target_colors[closest_indices]
        mapped_image = mapped_pixels.reshape(h, w, 3).permute(2, 0, 1).unsqueeze(0)

        return mapped_image

    def enhance_color_separation(
        self, image: torch.Tensor, factor: float = 1.5
    ) -> torch.Tensor:
        """Enhance color separation for better material distinction.

        Args:
            image: Input image (1, 3, H, W)
            factor: Enhancement factor

        Returns:
            Enhanced image (1, 3, H, W)
        """
        # Convert to LAB color space for better perceptual enhancement
        lab_image = self._rgb_to_lab(image)

        # Enhance A and B channels (color channels)
        enhanced_lab = lab_image.clone()
        enhanced_lab[:, 1:] = (enhanced_lab[:, 1:] - 0.5) * factor + 0.5
        enhanced_lab = torch.clamp(enhanced_lab, 0, 1)

        # Convert back to RGB
        enhanced_rgb = self._lab_to_rgb(enhanced_lab)

        return enhanced_rgb

    def _rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to LAB color space (simplified)."""
        # Simplified RGB to LAB conversion
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        # Approximate L*a*b* conversion
        lightness = 0.299 * r + 0.587 * g + 0.114 * b
        a = 0.5 + 0.5 * (r - g)
        b_comp = 0.5 + 0.25 * (g + r - 2 * b)

        return torch.cat([lightness, a, b_comp], dim=1)

    def _lab_to_rgb(self, lab: torch.Tensor) -> torch.Tensor:
        """Convert LAB to RGB color space (simplified)."""
        l, a, b_comp = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]

        # Approximate LAB to RGB conversion
        r = l + (a - 0.5)
        g = l - (a - 0.5)
        b = l - 2 * (b_comp - 0.5)

        return torch.clamp(torch.cat([r, g, b], dim=1), 0, 1)
