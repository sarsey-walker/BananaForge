"""Core optimization engine for multi-layer 3D printing."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .gumbel import GumbelSoftmax, TemperatureScheduler
from .loss import CombinedLoss


@dataclass
class OptimizationConfig:
    """Configuration for layer optimization."""

    iterations: int = 6000
    learning_rate: float = 0.01
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    temperature_decay: str = "linear"
    layer_height: float = 0.08
    max_layers: int = 15
    device: str = "cuda"
    early_stopping_patience: int = 1000
    loss_weights: Optional[Dict[str, float]] = None


class LayerOptimizer(nn.Module):
    """Core optimization engine for multi-layer 3D printing.

    Uses differentiable optimization with Gumbel softmax to jointly optimize
    height maps and material assignments for each layer.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        num_materials: int,
        config: OptimizationConfig,
        target_image: Optional[torch.Tensor] = None,
        initial_height_logits: Optional[torch.Tensor] = None,
        initial_global_logits: Optional[torch.Tensor] = None,
    ):
        """Initialize layer optimizer.

        Args:
            image_size: (height, width) of input image
            num_materials: Number of available materials
            config: Optimization configuration
            target_image: Target image for height map initialization
            initial_height_logits: Pre-initialized height map logits
            initial_global_logits: Pre-initialized global logits
        """
        super().__init__()
        self.config = config
        self.image_size = image_size
        self.num_materials = num_materials
        self.device = torch.device(config.device)

        if initial_height_logits is not None:
            self.height_logits = nn.Parameter(initial_height_logits.to(self.device))
        else:
            # Initialize height map with intelligent clustering
            if target_image is not None:
                height_init = self._initialize_height_map_from_image(target_image)
            else:
                # Fallback to simple initialization
                height_init = torch.randn(*image_size, device=self.device)
            self.height_logits = nn.Parameter(height_init * 0.1)

        # Initialize global material logits (per layer, not per pixel)
        if initial_global_logits is not None:
            # Check if the initial global logits have the right dimensions
            init_layers, init_materials = initial_global_logits.shape
            if init_layers != config.max_layers or init_materials != num_materials:
                # Resize the global logits to match expected dimensions
                resized_global_logits = torch.zeros(
                    config.max_layers, num_materials, device=self.device
                )

                # Copy available data
                copy_layers = min(init_layers, config.max_layers)
                copy_materials = min(init_materials, num_materials)
                resized_global_logits[:copy_layers, :copy_materials] = (
                    initial_global_logits[:copy_layers, :copy_materials]
                )

                # Initialize remaining layers with cycling pattern
                for i in range(copy_layers, config.max_layers):
                    resized_global_logits[i, i % num_materials] = 1.0

                self.global_logits = nn.Parameter(resized_global_logits.to(self.device))
            else:
                self.global_logits = nn.Parameter(initial_global_logits.to(self.device))
        else:
            self.global_logits = nn.Parameter(
                torch.randn(config.max_layers, num_materials, device=self.device) * 0.5
            )

        # Initialize global logits with cycling pattern (only if not already initialized)
        if initial_global_logits is None:
            for i in range(config.max_layers):
                self.global_logits.data[i, i % num_materials] = 1.0

        # Temperature scheduling parameters
        self.current_temperature = config.initial_temperature

        # Gumbel softmax for material selection
        self.gumbel_softmax = GumbelSoftmax(temperature=config.initial_temperature)

        # Temperature scheduler
        self.temp_scheduler = TemperatureScheduler(
            initial_temp=config.initial_temperature,
            final_temp=config.final_temperature,
            decay_type=config.temperature_decay,
        )

        # Loss function
        loss_weights = config.loss_weights or {}
        self.loss_fn = CombinedLoss(
            perceptual_weight=loss_weights.get("perceptual", 1.0),
            color_weight=loss_weights.get("color", 1.0),
            smoothness_weight=loss_weights.get("smoothness", 0.1),
            consistency_weight=loss_weights.get("consistency", 0.5),
            device=config.device,
        )

        # Optimizer with improved settings - only optimize global logits
        self.optimizer = optim.Adam(
            [self.global_logits],  # Only optimize global logits, not pixel heights
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,  # Small regularization
        )

        # Optimization state
        self.current_step = 0
        self.best_loss = float("inf")
        self.patience_counter = 0

    def forward(
        self, material_colors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to generate composite image using compositing.

        Args:
            material_colors: RGB colors for each material (num_materials, 3)

        Returns:
            Tuple of (composite_image, height_map)
        """
        # Get current height map from logits
        height_map = (
            torch.sigmoid(self.height_logits) * self.config.max_layers
        )  # [H, W]
        # Clamp to max_layers to ensure we don't exceed the limit
        height_map = torch.clamp(height_map, 0, self.config.max_layers)

        # Get material probabilities for each layer using Gumbel softmax
        material_probs = self.gumbel_softmax(
            self.global_logits
        )  # [max_layers, num_materials]

        # Compute layer colors by multiplying probabilities with material colors
        layer_colors = material_probs @ material_colors  # [max_layers, 3]

        # Create layer masks for each height level
        layer_idx = torch.arange(
            self.config.max_layers, dtype=torch.float32, device=self.device
        ).view(
            -1, 1, 1
        )  # [L, 1, 1]

        # Soft print mask - which pixels are printed at each layer
        eps = 1e-8
        scale = 10.0 / (
            self.current_temperature + eps
        )  # Temperature-dependent sharpness
        print_mask = torch.sigmoid(
            (height_map.unsqueeze(0) - (layer_idx + 0.5)) * scale
        )  # [max_layers, H, W]

        # Physics-based opacity calculation
        layer_height = self.config.layer_height
        eff_thickness = torch.clamp(print_mask, 0.0, 1.0) * layer_height

        # Default transmission distances (TD values) - simplified
        TDs = (
            torch.ones(self.config.max_layers, device=self.device) * 4.0
        )  # Typical PLA TD
        thick_ratio = eff_thickness / TDs.view(-1, 1, 1)

        # Opacity formula: o + A*log1p(k*ratio) + b*ratio
        o, A, k, b = -1.2416557e-02, 9.6407950e-01, 3.4103447e01, -4.1554203e00
        opacity = o + (A * torch.log1p(k * thick_ratio) + b * thick_ratio)
        opacity = torch.clamp(opacity, 0.0, 1.0)  # [max_layers, H, W]

        # Composite layers from top to bottom
        opacity_flipped = torch.flip(opacity, dims=[0])  # Top to bottom
        colors_flipped = torch.flip(layer_colors, dims=[0])  # Top to bottom

        # Compute remaining light after each layer
        transparency = 1.0 - opacity_flipped
        remaining = torch.ones_like(transparency[:1])  # Start with full light

        composite = torch.zeros(*self.image_size, 3, device=self.device)

        for i in range(self.config.max_layers):
            # Add contribution from this layer
            layer_opacity = opacity_flipped[i]  # [H, W]
            layer_color = colors_flipped[i]  # [3]

            # Expand layer color to match spatial dimensions
            layer_color_expanded = layer_color.view(1, 1, 3).expand(
                *self.image_size, 3
            )  # [H, W, 3]

            layer_contrib = (
                remaining[0].unsqueeze(-1)
                * layer_opacity.unsqueeze(-1)
                * layer_color_expanded
            )
            composite += layer_contrib

            # Update remaining light
            if i < self.config.max_layers - 1:
                remaining = remaining * transparency[i : i + 1]

        # Add white background
        background_contrib = remaining[0].unsqueeze(-1) * torch.ones(
            3, device=self.device
        ).view(1, 1, 3)
        composite += background_contrib

        # Reshape to match expected output format [1, 3, H, W]
        composite_image = composite.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        height_map = height_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        return composite_image, height_map

    def optimize(
        self,
        target_image: torch.Tensor,
        material_colors: torch.Tensor,
        callback: Optional[callable] = None,
    ) -> Dict[str, List[float]]:
        """Optimize height map and material assignments.

        Args:
            target_image: Target image to match (1, 3, H, W)
            material_colors: Available material colors (num_materials, 3)
            callback: Optional callback for progress updates

        Returns:
            Dictionary of loss histories
        """
        target_image = target_image.to(self.device)
        material_colors = material_colors.to(self.device)

        # Initialize loss history
        loss_history = {
            "total": [],
            "perceptual": [],
            "color": [],
            "smoothness": [],
            "consistency": [],
        }

        self.train()

        for step in range(self.config.iterations):
            self.current_step = step

            # Update temperature with proper scheduling
            warmup_fraction = 0.1  # 10% warmup
            warmup_steps = int(warmup_fraction * self.config.iterations)

            if step < warmup_steps:
                self.current_temperature = self.config.initial_temperature
            else:
                # Linear decay after warmup
                progress = (step - warmup_steps) / (
                    self.config.iterations - warmup_steps
                )
                self.current_temperature = (
                    self.config.initial_temperature
                    + progress
                    * (self.config.final_temperature - self.config.initial_temperature)
                )

            self.gumbel_softmax.set_temperature(self.current_temperature)

            # Forward pass
            pred_image, height_map = self.forward(material_colors)

            # Get material probabilities for consistency loss from global logits
            material_probs = torch.softmax(
                self.global_logits, dim=1
            )  # [max_layers, num_materials]

            # Compute loss
            total_loss, loss_dict = self.loss_fn(
                pred_image, target_image, height_map, material_probs
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # More conservative gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)

            self.optimizer.step()

            # Record losses
            for key, value in loss_dict.items():
                loss_history[key].append(value.item())

            # Early stopping check
            if total_loss.item() < self.best_loss:
                self.best_loss = total_loss.item()
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at step {step}")
                break

            # Progress callback
            if callback and step % 10 == 0:
                callback(step, loss_dict, pred_image, height_map)

        return loss_history

    def _initialize_height_map_from_image(
        self, target_image: torch.Tensor
    ) -> torch.Tensor:
        """Initialize height map using image clustering.

        Args:
            target_image: Target image tensor [1, 3, H, W]

        Returns:
            Height logits tensor [H, W]
        """
        # Convert to numpy for clustering
        image = target_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        H, W = image.shape[:2]

        # Simple k-means clustering to create height layers
        from sklearn.cluster import KMeans

        # Reshape for clustering
        pixels = image.reshape(-1, 3)  # [H*W, 3]

        # Cluster into layers
        n_clusters = min(self.config.max_layers, 8)  # Reasonable number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)

        # Reshape back to image
        label_image = labels.reshape(H, W)

        # Convert cluster labels to height values
        # Sort clusters by brightness (darker = lower layers)
        cluster_brightness = []
        for i in range(n_clusters):
            mask = label_image == i
            if mask.sum() > 0:
                avg_brightness = image[mask].mean()
                cluster_brightness.append((i, avg_brightness))

        # Sort by brightness (dark to light = bottom to top)
        cluster_brightness.sort(key=lambda x: x[1])

        # Create height mapping
        height_map = torch.zeros(H, W, device=self.device)
        for height_level, (cluster_id, _) in enumerate(cluster_brightness):
            mask = label_image == cluster_id
            height_map[mask] = height_level

        # Normalize to [0, max_layers] range
        height_map = height_map * (self.config.max_layers - 1) / (n_clusters - 1)

        return height_map

    def get_final_results(
        self, material_colors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get final optimized results.

        Args:
            material_colors: Material colors (num_materials, 3)

        Returns:
            Tuple of (composite_image, height_map, material_assignments)
        """
        self.eval()
        with torch.no_grad():
            # Set temperature to final value for sharp assignments
            self.gumbel_softmax.set_temperature(self.config.final_temperature)

            # Get final composite image and height map
            composite_image, height_map = self.forward(material_colors)

            # Get discrete material assignments from global logits
            material_assignments = torch.argmax(
                self.global_logits, dim=1
            )  # [max_layers]

            # Expand to per-pixel assignments for compatibility
            expanded_assignments = torch.zeros(
                self.config.max_layers,
                *self.image_size,
                device=self.device,
                dtype=torch.long,
            )

            for layer_idx in range(self.config.max_layers):
                expanded_assignments[layer_idx] = material_assignments[layer_idx]

        return composite_image, height_map, expanded_assignments

    def save_checkpoint(self, filepath: str) -> None:
        """Save optimization checkpoint."""
        checkpoint = {
            "height_logits": self.height_logits.data,
            "global_logits": self.global_logits.data,
            "optimizer_state": self.optimizer.state_dict(),
            "current_step": self.current_step,
            "best_loss": self.best_loss,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load optimization checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.height_logits.data = checkpoint["height_logits"]
        self.global_logits.data = checkpoint["global_logits"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.current_step = checkpoint["current_step"]
        self.best_loss = checkpoint["best_loss"]


class HeightMapInitializer:
    """Utilities for intelligent height map initialization."""

    @staticmethod
    def depth_based_init(
        image: torch.Tensor, max_layers: int, device: str = "cpu"
    ) -> torch.Tensor:
        """Initialize height map based on image depth cues.

        Args:
            image: Input image (3, H, W)
            max_layers: Maximum number of layers
            device: Device for computation

        Returns:
            Initial height map logits
        """
        # Convert to grayscale for depth estimation
        gray = torch.mean(image, dim=0, keepdim=True)

        # Simple depth heuristic (darker = further/lower)
        depth = 1.0 - gray

        # Smooth depth map
        kernel_size = 5
        padding = kernel_size // 2
        depth = torch.nn.functional.avg_pool2d(
            depth.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=padding
        ).squeeze(0)

        # Convert to logits
        height_logits = torch.logit(
            depth * 0.8 + 0.1, eps=1e-6  # Scale to avoid extremes
        )

        return height_logits.unsqueeze(0).to(device)

    @staticmethod
    def gradient_based_init(
        image: torch.Tensor, max_layers: int, device: str = "cpu"
    ) -> torch.Tensor:
        """Initialize height map based on image gradients.

        Args:
            image: Input image (3, H, W)
            max_layers: Maximum number of layers
            device: Device for computation

        Returns:
            Initial height map logits
        """
        # Compute image gradients
        gray = torch.mean(image, dim=0)

        # Sobel edge detection
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device
        )

        grad_x = torch.nn.functional.conv2d(
            gray.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1
        ).squeeze()

        grad_y = torch.nn.functional.conv2d(
            gray.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1
        ).squeeze()

        # Edge magnitude
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

        # Use edges to define height variation
        height_map = 1.0 - edge_magnitude / (edge_magnitude.max() + 1e-6)

        # Convert to logits
        height_logits = torch.logit(height_map * 0.8 + 0.1, eps=1e-6)

        return height_logits.unsqueeze(0).unsqueeze(0).to(device)
