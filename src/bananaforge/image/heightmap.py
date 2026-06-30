"""Height map generation algorithms for BananaForge."""

from typing import Tuple

import numpy as np
import torch

from ..utils.config import Config
from .christofides import run_init_threads


class HeightMapGenerator:
    """Advanced height map generation using Christofides algorithm."""

    def __init__(self, config: Config, device: str = "cpu"):
        """Initialize height map generator.
        Args:
            config: Configuration object
            device: Device for computations
        """
        self.config = config
        self.device = torch.device(device)

    def generate(
        self,
        image: np.ndarray,
        background_tuple: tuple[int, int, int],
        material_colors_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate height map using Christofides algorithm.
        Args:
            image: Input image (H, W, 3)
            background_tuple: RGB tuple for background color
            material_colors_np: Numpy array of material colors
        Returns:
            Tuple of (height_map, global_logits, labels)
        """
        height_map, global_logits, labels = run_init_threads(
            target=image,
            max_layers=self.config.max_layers,
            h=self.config.layer_height,
            background_tuple=background_tuple,
            random_seed=self.config.random_seed,
            num_threads=self.config.num_init_rounds,
            init_method="kmeans",
            cluster_layers=self.config.num_init_cluster_layers,
            material_colors=material_colors_np,
        )

        return height_map, global_logits, labels
