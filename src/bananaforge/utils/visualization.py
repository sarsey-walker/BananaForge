"""Visualization utilities for BananaForge."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap


class Visualizer:
    """Visualization utilities for optimization results and analysis."""

    def __init__(self, style: str = "default", dpi: int = 150):
        """Initialize visualizer.

        Args:
            style: Matplotlib style to use
            dpi: DPI for output images
        """
        plt.style.use(style)
        self.dpi = dpi

    def plot_optimization_progress(
        self,
        loss_history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot optimization loss curves.

        Args:
            loss_history: Dictionary of loss histories
            save_path: Optional path to save plot
            show: Whether to display plot

        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Optimization Progress", fontsize=16)

        # Total loss
        if "total" in loss_history and loss_history["total"]:
            axes[0, 0].plot(loss_history["total"], "b-", linewidth=2)
            axes[0, 0].set_title("Total Loss")
            axes[0, 0].set_xlabel("Iteration")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].grid(True, alpha=0.3)

        # Individual loss components
        loss_components = ["perceptual", "color", "smoothness", "consistency"]
        colors = ["red", "green", "orange", "purple"]

        for i, (component, color) in enumerate(zip(loss_components, colors)):
            if component in loss_history and loss_history[component]:
                row, col = divmod(i + 1, 2)
                if row < 2 and col < 2:
                    axes[row, col].plot(
                        loss_history[component], color=color, linewidth=2
                    )
                    axes[row, col].set_title(f"{component.title()} Loss")
                    axes[row, col].set_xlabel("Iteration")
                    axes[row, col].set_ylabel("Loss")
                    axes[row, col].grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(len(loss_components) + 1, 4):
            row, col = divmod(i, 2)
            if row < 2 and col < 2:
                axes[row, col].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_material_comparison(
        self,
        original_image: torch.Tensor,
        optimized_image: torch.Tensor,
        material_colors: torch.Tensor,
        material_names: List[str],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Compare original and optimized images with material palette.

        Args:
            original_image: Original input image (3, H, W)
            optimized_image: Optimized output image (3, H, W)
            material_colors: Material color palette (num_materials, 3)
            material_names: List of material names
            save_path: Optional save path
            show: Whether to display plot

        Returns:
            Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Convert tensors to numpy for plotting
        if isinstance(original_image, torch.Tensor):
            orig_np = original_image.permute(1, 2, 0).cpu().numpy()
        else:
            orig_np = original_image

        if isinstance(optimized_image, torch.Tensor):
            opt_np = optimized_image.permute(1, 2, 0).cpu().numpy()
        else:
            opt_np = optimized_image

        # Original image
        axes[0].imshow(np.clip(orig_np, 0, 1))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Optimized image
        axes[1].imshow(np.clip(opt_np, 0, 1))
        axes[1].set_title("Optimized Result")
        axes[1].axis("off")

        # Material palette
        self._plot_material_palette(axes[2], material_colors, material_names)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def _plot_material_palette(
        self, ax: plt.Axes, material_colors: torch.Tensor, material_names: List[str]
    ) -> None:
        """Plot material color palette."""
        colors_np = material_colors.cpu().numpy()
        num_materials = len(colors_np)

        # Create color swatches
        swatch_height = 1.0 / num_materials

        for i, (color, name) in enumerate(zip(colors_np, material_names)):
            y_pos = 1.0 - (i + 1) * swatch_height

            # Color swatch
            rect = patches.Rectangle(
                (0, y_pos),
                0.3,
                swatch_height,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

            # Material name
            ax.text(
                0.35,
                y_pos + swatch_height / 2,
                name,
                verticalalignment="center",
                fontsize=10,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Material Palette")
        ax.axis("off")

    def plot_height_map(
        self,
        height_map: torch.Tensor,
        material_assignments: Optional[torch.Tensor] = None,
        material_colors: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Visualize height map and material assignments.

        Args:
            height_map: Height map tensor (H, W)
            material_assignments: Optional material assignments (num_layers, H, W)
            material_colors: Optional material colors for visualization
            save_path: Optional save path
            show: Whether to display plot

        Returns:
            Figure object
        """
        if material_assignments is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes = [axes[0], axes[1], None]

        height_np = height_map.squeeze().cpu().numpy()

        # Height map
        im1 = axes[0].imshow(height_np, cmap="viridis", aspect="equal")
        axes[0].set_title("Height Map")
        axes[0].axis("off")
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="Height (layers)")

        # 3D visualization
        self._plot_height_map_3d(axes[1], height_np)

        # Material assignments if provided
        if material_assignments is not None and axes[2] is not None:
            # Show top layer material assignment
            top_layer_assignment = material_assignments[-1].cpu().numpy()

            if material_colors is not None:
                # Create custom colormap from material colors
                colors_np = material_colors.cpu().numpy()
                cmap = ListedColormap(colors_np)
                im3 = axes[2].imshow(top_layer_assignment, cmap=cmap, aspect="equal")
            else:
                im3 = axes[2].imshow(top_layer_assignment, cmap="tab10", aspect="equal")

            axes[2].set_title("Material Assignment (Top Layer)")
            axes[2].axis("off")
            plt.colorbar(
                im3, ax=axes[2], fraction=0.046, pad=0.04, label="Material Index"
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def _plot_height_map_3d(self, ax: plt.Axes, height_map: np.ndarray) -> None:
        """Create 3D visualization of height map."""
        # Remove old axes and create 3D axes
        fig = ax.figure
        ax.remove()
        ax = fig.add_subplot(1, 3, 2, projection="3d")

        h, w = height_map.shape
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)

        # Downsample for performance if too large
        if h > 100 or w > 100:
            step = max(h // 50, w // 50, 1)
            X = X[::step, ::step]
            Y = Y[::step, ::step]
            height_map = height_map[::step, ::step]

        ax.plot_surface(
            X, Y, height_map, cmap="viridis", alpha=0.8, antialiased=True
        )

        ax.set_title("3D Height Map")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Height")

    def plot_color_analysis(
        self,
        image: torch.Tensor,
        dominant_colors: torch.Tensor,
        selected_materials: List[str],
        material_colors: torch.Tensor,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Visualize color analysis and material matching.

        Args:
            image: Original image
            dominant_colors: Extracted dominant colors
            selected_materials: Selected material names
            material_colors: Selected material colors
            save_path: Optional save path
            show: Whether to display

        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Original image
        img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        axes[0, 0].imshow(np.clip(img_np, 0, 1))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # Dominant colors
        self._plot_color_swatches(
            axes[0, 1], dominant_colors.cpu().numpy(), "Dominant Colors"
        )

        # Selected material colors
        self._plot_color_swatches(
            axes[1, 0],
            material_colors.cpu().numpy(),
            "Selected Materials",
            selected_materials,
        )

        # Color space comparison
        self._plot_color_space_comparison(axes[1, 1], dominant_colors, material_colors)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def _plot_color_swatches(
        self,
        ax: plt.Axes,
        colors: np.ndarray,
        title: str,
        labels: Optional[List[str]] = None,
    ) -> None:
        """Plot color swatches."""
        num_colors = len(colors)
        swatch_width = 1.0 / num_colors

        for i, color in enumerate(colors):
            x_pos = i * swatch_width
            rect = patches.Rectangle(
                (x_pos, 0),
                swatch_width,
                1,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

            if labels and i < len(labels):
                ax.text(
                    x_pos + swatch_width / 2,
                    0.5,
                    labels[i],
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.axis("off")

    def _plot_color_space_comparison(
        self, ax: plt.Axes, dominant_colors: torch.Tensor, material_colors: torch.Tensor
    ) -> None:
        """Plot colors in RGB space."""
        dom_colors_np = dominant_colors.cpu().numpy()
        mat_colors_np = material_colors.cpu().numpy()

        # Plot in 3D RGB space
        ax.remove()
        fig = ax.figure
        ax = fig.add_subplot(2, 2, 4, projection="3d")

        # Dominant colors
        ax.scatter(
            dom_colors_np[:, 0],
            dom_colors_np[:, 1],
            dom_colors_np[:, 2],
            c=dom_colors_np,
            s=100,
            alpha=0.8,
            label="Image Colors",
            marker="o",
        )

        # Material colors
        ax.scatter(
            mat_colors_np[:, 0],
            mat_colors_np[:, 1],
            mat_colors_np[:, 2],
            c=mat_colors_np,
            s=100,
            alpha=0.8,
            label="Materials",
            marker="^",
        )

        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")
        ax.set_title("RGB Color Space")
        ax.legend()

    def create_print_preview(
        self,
        layers: List[np.ndarray],
        layer_heights: List[float],
        material_colors: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Create print preview showing individual layers.

        Args:
            layers: List of layer images
            layer_heights: Heights of each layer
            material_colors: Colors for each material
            save_path: Optional save path
            show: Whether to display

        Returns:
            Figure object
        """
        num_layers = min(len(layers), 9)  # Show max 9 layers
        cols = 3
        rows = (num_layers + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle("Layer Preview", fontsize=16)

        for i in range(num_layers):
            row, col = divmod(i, cols)
            ax = axes[row, col]

            # Show layer
            ax.imshow(layers[i], cmap="gray", aspect="equal")
            ax.set_title(f"Layer {i+1}\nHeight: {layer_heights[i]:.2f}mm")
            ax.axis("off")

        # Hide empty subplots
        for i in range(num_layers, rows * cols):
            row, col = divmod(i, cols)
            axes[row, col].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_cost_analysis(
        self,
        cost_data: Dict[str, Dict],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Visualize cost analysis.

        Args:
            cost_data: Cost data dictionary
            save_path: Optional save path
            show: Whether to display

        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        materials = list(cost_data.keys())
        weights = [data["weight_grams"] for data in cost_data.values()]
        costs = [data["cost_usd"] for data in cost_data.values()]
        colors = [data.get("color", "#888888") for data in cost_data.values()]

        # Weight distribution
        axes[0, 0].pie(weights, labels=materials, colors=colors, autopct="%1.1f%%")
        axes[0, 0].set_title("Material Weight Distribution")

        # Cost distribution
        axes[0, 1].pie(costs, labels=materials, colors=colors, autopct="$%.2f")
        axes[0, 1].set_title("Cost Distribution")

        # Weight comparison
        axes[1, 0].bar(materials, weights, color=colors)
        axes[1, 0].set_title("Material Usage (grams)")
        axes[1, 0].set_ylabel("Weight (g)")
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

        # Cost comparison
        axes[1, 1].bar(materials, costs, color=colors)
        axes[1, 1].set_title("Material Costs (USD)")
        axes[1, 1].set_ylabel("Cost ($)")
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def save_all_visualizations(self, output_dir: Path, **kwargs) -> Dict[str, str]:
        """Save all available visualizations.

        Args:
            output_dir: Output directory
            **kwargs: Visualization data

        Returns:
            Dictionary mapping visualization type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Optimization progress
        if "loss_history" in kwargs:
            path = output_dir / "optimization_progress.png"
            self.plot_optimization_progress(
                kwargs["loss_history"], str(path), show=False
            )
            saved_files["optimization_progress"] = str(path)

        # Material comparison
        if all(
            k in kwargs
            for k in [
                "original_image",
                "optimized_image",
                "material_colors",
                "material_names",
            ]
        ):
            path = output_dir / "material_comparison.png"
            self.plot_material_comparison(
                kwargs["original_image"],
                kwargs["optimized_image"],
                kwargs["material_colors"],
                kwargs["material_names"],
                str(path),
                show=False,
            )
            saved_files["material_comparison"] = str(path)

        # Height map
        if "height_map" in kwargs:
            path = output_dir / "height_map.png"
            self.plot_height_map(
                kwargs["height_map"],
                kwargs.get("material_assignments"),
                kwargs.get("material_colors"),
                str(path),
                show=False,
            )
            saved_files["height_map"] = str(path)

        # Cost analysis
        if "cost_data" in kwargs:
            path = output_dir / "cost_analysis.png"
            self.plot_cost_analysis(kwargs["cost_data"], str(path), show=False)
            saved_files["cost_analysis"] = str(path)

        return saved_files
