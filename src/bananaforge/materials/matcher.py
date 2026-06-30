"""Advanced color matching algorithms for material selection."""

from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from .database import MaterialDatabase
from .transparency_mixer import TransparencyColorMixer


class ColorMatcher:
    """Advanced color matching for material selection."""

    def __init__(
        self,
        material_db: MaterialDatabase,
        device: str = "cuda",
        enable_transparency: bool = False,
    ):
        """Initialize color matcher.

        Args:
            material_db: Material database
            device: Device for computations
            enable_transparency: Enable transparency-based color mixing
        """
        self.material_db = material_db
        self.device = torch.device(device)
        self.material_colors = material_db.get_color_palette(device)
        self.material_ids = material_db.get_material_ids()
        self.enable_transparency = enable_transparency

        # Initialize transparency mixer if enabled
        if enable_transparency:
            self.transparency_mixer = TransparencyColorMixer(device=device)
            self._achievable_colors = None
            self._achievable_colors_info = None

    def match_image_colors(
        self, image: torch.Tensor, max_materials: int = 4, method: str = "perceptual"
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Match image colors to available materials.

        Args:
            image: Input image (1, 3, H, W)
            max_materials: Maximum number of materials to use
            method: Matching method ("perceptual", "euclidean", "lab")

        Returns:
            Tuple of (selected_material_ids, selected_colors, color_mapping)
        """
        # Check if image is grayscale and handle appropriately
        if self._is_grayscale_image(image):
            return self._grayscale_matching(image, max_materials)

        if method == "perceptual":
            return self._perceptual_matching(image, max_materials)
        elif method == "euclidean":
            return self._euclidean_matching(image, max_materials)
        elif method == "lab":
            return self._lab_matching(image, max_materials)
        else:
            raise ValueError(f"Unknown matching method: {method}")

    def _perceptual_matching(
        self, image: torch.Tensor, max_materials: int
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Perceptual color matching using LAB color space."""
        # Extract dominant colors from image
        dominant_colors = self._extract_dominant_colors(image, max_materials * 2)

        # If transparency is enabled, consider achievable colors through mixing
        if self.enable_transparency and hasattr(self, "transparency_mixer"):
            return self._transparency_aware_matching(
                image, dominant_colors, max_materials
            )

        # Standard perceptual matching
        # Convert to LAB color space
        image_lab = self._rgb_to_lab(dominant_colors)
        material_lab = self._rgb_to_lab(self.material_colors)

        # Find best matches using perceptual distance
        selected_indices = []
        selected_colors = []

        for img_color_lab in image_lab:
            # Calculate perceptual distances
            distances = torch.norm(material_lab - img_color_lab.unsqueeze(0), dim=1)

            # Find closest material not already selected
            sorted_indices = torch.argsort(distances)
            for idx in sorted_indices:
                if idx.item() not in selected_indices:
                    selected_indices.append(idx.item())
                    selected_colors.append(self.material_colors[idx])
                    break

            if len(selected_indices) >= max_materials:
                break

        # Create color mapping
        selected_tensor = (
            torch.stack(selected_colors) if selected_colors else torch.empty(0, 3)
        )
        color_mapping = self._create_color_mapping(image, selected_tensor)

        selected_ids = [self.material_ids[i] for i in selected_indices]

        return selected_ids, selected_tensor, color_mapping

    def _euclidean_matching(
        self, image: torch.Tensor, max_materials: int
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Simple Euclidean distance matching in RGB space."""
        # Extract dominant colors
        dominant_colors = self._extract_dominant_colors(image, max_materials)

        # Find closest materials
        selected_indices = []
        selected_colors = []

        for img_color in dominant_colors:
            distances = torch.norm(self.material_colors - img_color.unsqueeze(0), dim=1)
            closest_idx = torch.argmin(distances).item()

            if closest_idx not in selected_indices:
                selected_indices.append(closest_idx)
                selected_colors.append(self.material_colors[closest_idx])

        selected_tensor = (
            torch.stack(selected_colors) if selected_colors else torch.empty(0, 3)
        )
        color_mapping = self._create_color_mapping(image, selected_tensor)

        selected_ids = [self.material_ids[i] for i in selected_indices]

        return selected_ids, selected_tensor, color_mapping

    def _lab_matching(
        self, image: torch.Tensor, max_materials: int
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """LAB color space matching with optimal assignment."""
        # Extract dominant colors
        dominant_colors = self._extract_dominant_colors(image, max_materials)

        # Convert to LAB
        image_lab = self._rgb_to_lab(dominant_colors)
        material_lab = self._rgb_to_lab(self.material_colors)

        # Create cost matrix
        cost_matrix = torch.cdist(image_lab, material_lab).cpu().numpy()

        # Solve assignment problem
        if len(dominant_colors) <= len(self.material_colors):
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            selected_indices = col_indices[:max_materials]
        else:
            # More image colors than materials - select best materials
            row_indices, col_indices = linear_sum_assignment(cost_matrix.T)
            selected_indices = col_indices[:max_materials]

        selected_colors = self.material_colors[selected_indices]
        color_mapping = self._create_color_mapping(image, selected_colors)

        selected_ids = [self.material_ids[i] for i in selected_indices]

        return selected_ids, selected_colors, color_mapping

    def _extract_dominant_colors(
        self, image: torch.Tensor, num_colors: int
    ) -> torch.Tensor:
        """Extract dominant colors using K-means clustering."""
        # Reshape image for clustering
        h, w = image.shape[-2:]
        pixels = image.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        pixels_np = pixels.cpu().numpy()

        # K-means clustering
        n_clusters = min(num_colors, len(pixels_np))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels_np)

        # Get cluster centers
        centers = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)

        return centers

    def _create_color_mapping(
        self, image: torch.Tensor, target_colors: torch.Tensor
    ) -> torch.Tensor:
        """Create pixel-wise color mapping.

        Args:
            image: Input image (1, 3, H, W)
            target_colors: Target material colors (num_materials, 3)

        Returns:
            Color mapping indices (1, H, W)
        """
        if len(target_colors) == 0:
            return torch.zeros(
                1, *image.shape[-2:], dtype=torch.long, device=self.device
            )

        h, w = image.shape[-2:]
        pixels = image.squeeze(0).permute(1, 2, 0).reshape(-1, 3)

        # Find closest target color for each pixel
        distances = torch.cdist(pixels, target_colors)
        closest_indices = torch.argmin(distances, dim=1)

        # Reshape to image dimensions
        color_mapping = closest_indices.reshape(1, h, w)

        return color_mapping

    def optimize_material_selection(
        self,
        image: torch.Tensor,
        max_materials: int = 4,
        weight_coverage: float = 0.6,
        weight_accuracy: float = 0.4,
    ) -> Tuple[List[str], torch.Tensor, float]:
        """Optimize material selection balancing coverage and accuracy.

        Args:
            image: Input image (1, 3, H, W)
            max_materials: Maximum number of materials
            weight_coverage: Weight for color coverage
            weight_accuracy: Weight for color accuracy

        Returns:
            Tuple of (material_ids, colors, optimization_score)
        """
        best_score = -1
        best_materials = []
        best_colors = None

        # Try different selection strategies
        strategies = ["perceptual", "euclidean", "lab"]

        for strategy in strategies:
            materials, colors, mapping = self.match_image_colors(
                image, max_materials, strategy
            )

            if len(colors) == 0:
                continue

            # Calculate coverage score
            coverage_score = self._calculate_coverage_score(image, colors, mapping)

            # Calculate accuracy score
            accuracy_score = self._calculate_accuracy_score(image, colors, mapping)

            # Combined score
            total_score = (
                weight_coverage * coverage_score + weight_accuracy * accuracy_score
            )

            if total_score > best_score:
                best_score = total_score
                best_materials = materials
                best_colors = colors

        return best_materials, best_colors, best_score

    def _calculate_coverage_score(
        self, image: torch.Tensor, colors: torch.Tensor, mapping: torch.Tensor
    ) -> float:
        """Calculate how well the colors cover the image color space."""
        # Extract image colors
        h, w = image.shape[-2:]
        pixels = image.squeeze(0).permute(1, 2, 0).reshape(-1, 3)

        # Calculate coverage in color space
        # Use the ratio of covered color space volume
        image_color_range = pixels.max(dim=0)[0] - pixels.min(dim=0)[0]
        material_color_range = colors.max(dim=0)[0] - colors.min(dim=0)[0]

        coverage_ratio = material_color_range / (image_color_range + 1e-6)
        coverage_score = torch.min(coverage_ratio).item()

        return min(coverage_score, 1.0)

    def _calculate_accuracy_score(
        self, image: torch.Tensor, colors: torch.Tensor, mapping: torch.Tensor
    ) -> float:
        """Calculate color accuracy of the mapping."""
        # Reconstruct image using selected colors
        h, w = image.shape[-2:]
        reconstructed = torch.zeros_like(image)

        for i in range(len(colors)):
            mask = (mapping == i).float()
            color = colors[i].view(1, 3, 1, 1)
            reconstructed += mask.unsqueeze(1) * color

        # Calculate MSE between original and reconstructed
        mse = torch.mean((image - reconstructed) ** 2).item()
        accuracy_score = 1.0 / (1.0 + mse)

        return accuracy_score

    def refine_color_selection(
        self, image: torch.Tensor, initial_materials: List[str], iterations: int = 5
    ) -> Tuple[List[str], torch.Tensor]:
        """Refine color selection using iterative improvement.

        Args:
            image: Input image (1, 3, H, W)
            initial_materials: Initial material selection
            iterations: Number of refinement iterations

        Returns:
            Tuple of (refined_material_ids, refined_colors)
        """
        current_materials = initial_materials.copy()
        current_colors = torch.stack(
            [
                self.material_db.get_material(mid).color_tensor
                for mid in current_materials
            ]
        ).to(self.device)

        best_score = self._evaluate_selection(image, current_colors)

        for _ in range(iterations):
            # Try swapping each material with alternatives
            for i in range(len(current_materials)):
                original_color = current_colors[i]

                # Find similar materials to try
                similar_materials = self._find_similar_materials(
                    original_color, exclude=current_materials
                )

                for alt_material_id in similar_materials[:3]:  # Try top 3
                    alt_material = self.material_db.get_material(alt_material_id)
                    alt_color = alt_material.color_tensor.to(self.device)

                    # Temporarily replace
                    test_colors = current_colors.clone()
                    test_colors[i] = alt_color

                    # Evaluate
                    score = self._evaluate_selection(image, test_colors)

                    if score > best_score:
                        best_score = score
                        current_materials[i] = alt_material_id
                        current_colors[i] = alt_color

        return current_materials, current_colors

    def _find_similar_materials(
        self, target_color: torch.Tensor, exclude: List[str], num_similar: int = 5
    ) -> List[str]:
        """Find materials with similar colors."""
        distances = torch.norm(self.material_colors - target_color.unsqueeze(0), dim=1)
        sorted_indices = torch.argsort(distances)

        similar_ids = []
        for idx in sorted_indices:
            material_id = self.material_ids[idx.item()]
            if material_id not in exclude:
                similar_ids.append(material_id)
                if len(similar_ids) >= num_similar:
                    break

        return similar_ids

    def _evaluate_selection(self, image: torch.Tensor, colors: torch.Tensor) -> float:
        """Evaluate quality of color selection."""
        mapping = self._create_color_mapping(image, colors)
        coverage = self._calculate_coverage_score(image, colors, mapping)
        accuracy = self._calculate_accuracy_score(image, colors, mapping)
        return 0.6 * coverage + 0.4 * accuracy

    def _rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to LAB color space."""
        # Simplified RGB to LAB conversion
        # For more accuracy, use proper color space conversion
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        # Approximate L*a*b* conversion
        lightness = 0.299 * r + 0.587 * g + 0.114 * b
        a = 0.5 + 0.5 * (r - g)
        b_comp = 0.5 + 0.25 * (g + r - 2 * b)

        return torch.stack([lightness, a, b_comp], dim=-1)

    def create_transition_map(
        self,
        image: torch.Tensor,
        selected_colors: torch.Tensor,
        smoothing_factor: float = 1.0,
    ) -> torch.Tensor:
        """Create smooth transitions between material regions.

        Args:
            image: Input image (1, 3, H, W)
            selected_colors: Selected material colors (num_materials, 3)
            smoothing_factor: Amount of smoothing to apply

        Returns:
            Smooth material probability map (1, num_materials, H, W)
        """
        if len(selected_colors) == 0:
            return torch.zeros(1, 1, *image.shape[-2:], device=self.device)

        h, w = image.shape[-2:]
        pixels = image.squeeze(0).permute(1, 2, 0).reshape(-1, 3)

        # Calculate distances to each material
        distances = torch.cdist(pixels, selected_colors)  # (num_pixels, num_materials)

        # Convert distances to probabilities using softmax
        probabilities = torch.softmax(-distances / smoothing_factor, dim=1)

        # Reshape to image dimensions
        prob_maps = probabilities.t().reshape(len(selected_colors), h, w)

        return prob_maps.unsqueeze(0)

    def _is_grayscale_image(self, image: torch.Tensor, threshold: float = 0.02) -> bool:
        """Check if image is effectively grayscale.

        Args:
            image: Input image (1, 3, H, W)
            threshold: Color variation threshold for grayscale detection

        Returns:
            True if image is grayscale, False otherwise
        """
        # Extract RGB channels
        r, g, b = image.squeeze(0)

        # Calculate differences between channels
        rg_diff = torch.abs(r - g).mean().item()
        rb_diff = torch.abs(r - b).mean().item()
        gb_diff = torch.abs(g - b).mean().item()

        # Check if all channel differences are below threshold
        max_diff = max(rg_diff, rb_diff, gb_diff)
        return max_diff < threshold

    def _grayscale_matching(
        self, image: torch.Tensor, max_materials: int
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Match grayscale image to appropriate grayscale materials.

        Args:
            image: Grayscale input image (1, 3, H, W)
            max_materials: Maximum number of materials to use

        Returns:
            Tuple of (selected_material_ids, selected_colors, color_mapping)
        """
        # Get available grayscale materials
        grayscale_materials = self._get_grayscale_materials()

        if not grayscale_materials:
            # Fallback to regular matching if no grayscale materials
            return self._perceptual_matching(image, max_materials)

        # Select materials based on luminance levels
        selected_materials = []
        selected_colors = []

        # Always include the darkest available material
        darkest_mat = min(
            grayscale_materials, key=lambda m: m.color_tensor.mean().item()
        )
        selected_materials.append(darkest_mat)
        selected_colors.append(darkest_mat.color_tensor)

        # Always include the lightest available material if we have space
        if max_materials > 1:
            lightest_mat = max(
                grayscale_materials, key=lambda m: m.color_tensor.mean().item()
            )
            if lightest_mat.id != darkest_mat.id:
                selected_materials.append(lightest_mat)
                selected_colors.append(lightest_mat.color_tensor)

        # Add intermediate materials if needed
        if max_materials > 2 and len(grayscale_materials) > 2:
            # Find materials with intermediate luminance
            remaining_materials = [
                m
                for m in grayscale_materials
                if m.id not in [darkest_mat.id, lightest_mat.id]
            ]

            # Sort by luminance
            remaining_materials.sort(key=lambda m: m.color_tensor.mean().item())

            # Add up to remaining slots
            for i, mat in enumerate(remaining_materials):
                if len(selected_materials) >= max_materials:
                    break
                selected_materials.append(mat)
                selected_colors.append(mat.color_tensor)

        # Convert to tensors
        selected_tensor = torch.stack(selected_colors).to(self.device)
        selected_ids = [mat.id for mat in selected_materials]

        # Create color mapping based on luminance
        color_mapping = self._create_luminance_mapping(image, selected_tensor)

        return selected_ids, selected_tensor, color_mapping

    def _get_grayscale_materials(self) -> List:
        """Get materials that are suitable for grayscale images.

        Returns:
            List of grayscale-appropriate materials
        """
        grayscale_materials = []

        for i, material_id in enumerate(self.material_ids):
            material = self.material_db.get_material(material_id)
            if material and material.available:
                color = material.color_tensor

                # Check if material is grayscale (R≈G≈B)
                r, g, b = color[0].item(), color[1].item(), color[2].item()
                color_diff = max(abs(r - g), abs(r - b), abs(g - b))

                # Consider it grayscale if color channels are very similar
                if color_diff < 0.1:
                    grayscale_materials.append(material)

        return grayscale_materials

    def _create_luminance_mapping(
        self, image: torch.Tensor, target_colors: torch.Tensor
    ) -> torch.Tensor:
        """Create pixel-wise color mapping based on luminance.

        Args:
            image: Input image (1, 3, H, W)
            target_colors: Target material colors (num_materials, 3)

        Returns:
            Color mapping indices (1, H, W)
        """
        if len(target_colors) == 0:
            return torch.zeros(
                1, *image.shape[-2:], dtype=torch.long, device=self.device
            )

        h, w = image.shape[-2:]

        # Convert image to luminance
        luminance = (
            0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        ).flatten()

        # Convert target colors to luminance
        target_luminance = (
            0.299 * target_colors[:, 0]
            + 0.587 * target_colors[:, 1]
            + 0.114 * target_colors[:, 2]
        )

        # Find closest target luminance for each pixel
        distances = torch.abs(luminance.unsqueeze(1) - target_luminance.unsqueeze(0))
        closest_indices = torch.argmin(distances, dim=1)

        # Reshape to image dimensions
        color_mapping = closest_indices.reshape(1, h, w)

        return color_mapping

    def _transparency_aware_matching(
        self, image: torch.Tensor, dominant_colors: torch.Tensor, max_materials: int
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Color matching that considers achievable colors through transparency mixing.

        This method finds the optimal set of base materials that can create the target
        colors through transparency mixing (e.g., red + white = pink).
        """
        # Get all achievable colors through transparency mixing
        filament_colors = [
            self.material_colors[i] for i in range(len(self.material_colors))
        ]
        achievable_combinations = self.transparency_mixer.compute_achievable_colors(
            filament_colors, max_layers=3
        )

        # Convert to tensors for efficient processing
        achievable_colors = torch.stack(
            [combo["color"] for combo in achievable_combinations]
        ).to(self.device)

        # Convert dominant colors and achievable colors to LAB space for perceptual matching
        dominant_lab = self._rgb_to_lab(dominant_colors)
        achievable_lab = self._rgb_to_lab(achievable_colors)

        # Find best matches for each dominant color
        best_combinations = []
        selected_material_indices = set()

        for dominant_color_lab in dominant_lab:
            # Calculate perceptual distances to all achievable colors
            distances = torch.norm(
                achievable_lab - dominant_color_lab.unsqueeze(0), dim=1
            )

            # Find best matches that don't exceed our material budget
            sorted_indices = torch.argsort(distances)

            for idx in sorted_indices:
                combo = achievable_combinations[idx.item()]
                base_idx = combo["base_material"]
                overlay_idx = combo["overlay_material"]

                # Check if adding this combination would exceed material budget
                required_materials = {base_idx, overlay_idx}
                if len(selected_material_indices | required_materials) <= max_materials:
                    best_combinations.append(combo)
                    selected_material_indices.update(required_materials)
                    break

            if len(best_combinations) >= max_materials:
                break

        # Extract unique materials needed
        final_material_indices = list(selected_material_indices)[:max_materials]
        final_material_ids = [self.material_ids[i] for i in final_material_indices]
        final_colors = torch.stack(
            [self.material_colors[i] for i in final_material_indices]
        )

        # Create color mapping using the final selected materials
        color_mapping = self._create_color_mapping(image, final_colors)

        return final_material_ids, final_colors, color_mapping


class AdaptiveMatcher:
    """Adaptive color matching that learns from user preferences."""

    def __init__(self, material_db: MaterialDatabase, device: str = "cuda"):
        """Initialize adaptive matcher."""
        self.base_matcher = ColorMatcher(material_db, device)
        self.user_preferences = {}  # Store user preference history

    def match_with_preferences(
        self, image: torch.Tensor, max_materials: int = 4, user_id: Optional[str] = None
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Match colors considering user preferences.

        Args:
            image: Input image (1, 3, H, W)
            max_materials: Maximum number of materials
            user_id: Optional user ID for personalization

        Returns:
            Tuple of (material_ids, colors, mapping)
        """
        # Get base matching
        base_materials, base_colors, base_mapping = (
            self.base_matcher.match_image_colors(image, max_materials, "perceptual")
        )

        # Apply user preferences if available
        if user_id and user_id in self.user_preferences:
            adjusted_materials = self._apply_user_preferences(base_materials, user_id)

            # Recalculate colors and mapping
            adjusted_colors = torch.stack(
                [
                    self.base_matcher.material_db.get_material(mid).color_tensor
                    for mid in adjusted_materials
                ]
            ).to(self.base_matcher.device)

            adjusted_mapping = self.base_matcher._create_color_mapping(
                image, adjusted_colors
            )

            return adjusted_materials, adjusted_colors, adjusted_mapping

        return base_materials, base_colors, base_mapping

    def record_user_feedback(
        self, user_id: str, selected_materials: List[str], rating: float
    ) -> None:
        """Record user feedback for learning.

        Args:
            user_id: User identifier
            selected_materials: Materials that were selected
            rating: User rating (0-1 scale)
        """
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "material_scores": {},
                "brand_preferences": {},
                "color_preferences": {},
            }

        # Update material scores
        for material_id in selected_materials:
            if material_id not in self.user_preferences[user_id]["material_scores"]:
                self.user_preferences[user_id]["material_scores"][material_id] = []
            self.user_preferences[user_id]["material_scores"][material_id].append(
                rating
            )

    def _apply_user_preferences(
        self, base_materials: List[str], user_id: str
    ) -> List[str]:
        """Apply user preferences to material selection."""
        preferences = self.user_preferences[user_id]
        material_scores = preferences["material_scores"]

        # Calculate preference scores for materials
        scored_materials = []
        for material_id in base_materials:
            if material_id in material_scores:
                avg_score = np.mean(material_scores[material_id])
            else:
                avg_score = 0.5  # Neutral score for unknown materials

            scored_materials.append((material_id, avg_score))

        # Sort by preference score
        scored_materials.sort(key=lambda x: x[1], reverse=True)

        return [material_id for material_id, _ in scored_materials]
