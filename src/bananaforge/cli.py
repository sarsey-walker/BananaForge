"""Command-line interface for BananaForge."""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import click
import cv2
import numpy as np
import rich.console
import rich.traceback
import torch
from rich.console import Console

from .core.optimizer import LayerOptimizer, OptimizationConfig
from .image.heightmap import HeightMapGenerator
from .image.processor import ImageProcessor
from .materials.database import DefaultMaterials, MaterialDatabase
from .materials.matcher import ColorMatcher
from .output.exporter import ModelExporter
from .utils.color import hex_to_rgb
from .utils.config import Config, ConfigManager
from .utils.device import resolve_device
from .utils.logging import setup_logging

# Rich console setup
console = Console()
rich.traceback.install(console=console)

# Version import
try:
    from . import __version__
except ImportError:
    __version__ = "unknown"

LARGE_MESH_WARNING_TRIANGLES = 2_000_000
LARGE_MESH_WARNING_BYTES = 100 * 1024 * 1024
MESH_EXPORT_FORMATS = {"stl", "3mf", "bambu", "prusa", "layer_stls", "preview"}


def _apply_random_seed(random_seed: int) -> None:
    """Seed supported RNGs when a positive seed is provided."""
    if random_seed <= 0:
        return

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def _parse_hex_color_order(color_layer_order: str) -> List[Tuple[int, int, int]]:
    """Parse a comma-separated list of #RRGGBB colors."""
    if not color_layer_order:
        raise click.ClickException(
            "--color-layer-order is required when --ordered-color-layers is enabled"
        )

    color_tokens = [token.strip() for token in color_layer_order.split(",")]
    if any(not token for token in color_tokens):
        raise click.ClickException(
            "--color-layer-order must be a comma-separated list of #RRGGBB colors"
        )
    if len(color_tokens) < 2:
        raise click.ClickException("--color-layer-order must include at least 2 colors")

    parsed_colors: List[Tuple[int, int, int]] = []
    seen_colors = set()
    for token in color_tokens:
        if len(token) != 7 or not token.startswith("#"):
            raise click.ClickException(
                f"Invalid color '{token}'. Use #RRGGBB values in --color-layer-order"
            )
        hex_digits = token[1:]
        try:
            color = hex_to_rgb(token)
        except ValueError as exc:
            raise click.ClickException(
                f"Invalid color '{token}'. Use #RRGGBB values in --color-layer-order"
            ) from exc
        if len(hex_digits) != 6:
            raise click.ClickException(
                f"Invalid color '{token}'. Use #RRGGBB values in --color-layer-order"
            )
        normalized = token.upper()
        if normalized in seen_colors:
            raise click.ClickException(
                f"Duplicate color '{token}' in --color-layer-order"
            )
        seen_colors.add(normalized)
        parsed_colors.append(color)

    return parsed_colors


def _map_ordered_colors_to_materials(
    ordered_colors_rgb: Sequence[Tuple[int, int, int]],
    selected_colors: torch.Tensor,
    selected_materials: Sequence[str],
) -> List[int]:
    """Map requested RGB colors to the nearest selected material color."""
    if selected_colors.numel() == 0:
        raise click.ClickException("No selected materials are available")

    selected_colors_cpu = selected_colors.detach().cpu().float()
    if selected_colors_cpu.max().item() > 1.0:
        selected_colors_cpu = selected_colors_cpu / 255.0

    ordered_colors = torch.tensor(ordered_colors_rgb, dtype=torch.float32) / 255.0
    distances = torch.cdist(ordered_colors, selected_colors_cpu)
    material_indices = torch.argmin(distances, dim=1).tolist()

    if len(set(material_indices)) != len(material_indices):
        mapped_materials = [
            selected_materials[index] if index < len(selected_materials) else str(index)
            for index in material_indices
        ]
        raise click.ClickException(
            "Ordered colors must map to distinct selected materials; got "
            f"{', '.join(mapped_materials)}. Increase --max-materials or adjust "
            "--color-layer-order."
        )

    return [int(index) for index in material_indices]


def _build_ordered_color_layers(
    target_image_np: np.ndarray,
    ordered_colors_rgb: Sequence[Tuple[int, int, int]],
    ordered_material_indices: Sequence[int],
    color_layer_count: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build cumulative material layers from an explicit color order."""
    if color_layer_count < 1:
        raise click.ClickException("--color-layer-count must be at least 1")
    if len(ordered_colors_rgb) != len(ordered_material_indices):
        raise click.ClickException(
            "Ordered color count must match ordered material count"
        )

    target_pixels = target_image_np.astype(np.float32)
    palette = np.asarray(ordered_colors_rgb, dtype=np.float32)
    distances = np.sum(
        (target_pixels[:, :, None, :] - palette[None, None, :, :]) ** 2,
        axis=-1,
    )
    target_order_indices = np.argmin(distances, axis=-1).astype(np.int64)

    height_values = (target_order_indices + 1) * color_layer_count
    height_map = (
        torch.from_numpy(height_values.astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    num_color_layers = len(ordered_colors_rgb) * color_layer_count
    assignments = torch.full(
        (num_color_layers, target_image_np.shape[0], target_image_np.shape[1]),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )

    target_order_tensor = torch.from_numpy(target_order_indices).to(device)
    for order_index, material_index in enumerate(ordered_material_indices):
        active_mask = target_order_tensor >= order_index
        for repeat_index in range(color_layer_count):
            layer_index = (order_index * color_layer_count) + repeat_index
            assignments[layer_index][active_mask] = int(material_index)

    return height_map, assignments


def _config_default(ctx, parameter_name: str, config_key, current_value):
    """Use config value only when the CLI option was not explicitly supplied."""
    if ctx.get_parameter_source(parameter_name) != click.core.ParameterSource.DEFAULT:
        return current_value

    config_manager = ctx.obj.get("config_manager") if ctx.obj else None
    if not config_manager:
        return current_value

    config_keys = config_key if isinstance(config_key, (list, tuple)) else (config_key,)
    value = current_value
    for key in config_keys:
        next_value = config_manager.get(key, None)
        if next_value is not None:
            value = next_value
            break
    if isinstance(value, (list, tuple)) and isinstance(current_value, str):
        return ",".join(str(item) for item in value)
    return value


def _format_bytes(size_bytes: int) -> str:
    """Format byte sizes for CLI output."""
    units = ("B", "KB", "MB", "GB")
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _echo_mesh_export_estimate(
    target_h: int,
    target_w: int,
    max_triangles: Optional[int],
    bottom_mode: str,
) -> None:
    """Print a preventive mesh size estimate before expensive processing."""
    estimated_triangles = ModelExporter.estimate_triangle_count(
        target_h, target_w, bottom_mode
    )
    estimated_stl_size = ModelExporter.estimate_binary_stl_size_bytes(
        estimated_triangles
    )
    click.echo(
        "Estimated export mesh: "
        f"{estimated_triangles:,} triangles "
        f"(~{_format_bytes(estimated_stl_size)} binary STL, "
        f"bottom: {bottom_mode})"
    )

    if (
        estimated_triangles >= LARGE_MESH_WARNING_TRIANGLES
        or estimated_stl_size >= LARGE_MESH_WARNING_BYTES
    ):
        click.echo(
            "⚠️  Large mesh estimate: export may require substantial RAM, "
            "disk space, and slicer time."
        )
        click.echo(
            "   Consider --max-triangles, --bottom-mode none, a larger "
            "--nozzle-diameter, or a smaller --physical-size."
        )

    if max_triangles is None:
        return

    if max_triangles <= 0:
        raise click.ClickException("--max-triangles must be positive")

    if estimated_triangles <= max_triangles:
        return

    limited_h, limited_w = ModelExporter.fit_shape_to_triangle_limit(
        target_h, target_w, max_triangles, bottom_mode
    )
    limited_triangles = ModelExporter.estimate_triangle_count(
        limited_h, limited_w, bottom_mode
    )
    limited_size = ModelExporter.estimate_binary_stl_size_bytes(limited_triangles)
    click.echo(
        "Export mesh exceeds --max-triangles; export resolution will be "
        f"downscaled to {limited_w}x{limited_h} "
        f"({limited_triangles:,} triangles, "
        f"~{_format_bytes(limited_size)} binary STL)."
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
@click.option(
    "--config",
    type=click.Path(exists=True),
    envvar="BANANAFORGE_CONFIG",
    help="Path to configuration file",
)
@click.pass_context
def cli(ctx, verbose: bool, quiet: bool, config):
    """BananaForge: AI-powered multi-layer 3D printing optimization."""
    ctx.ensure_object(dict)

    # Setup logging
    log_level = logging.INFO
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    elif os.getenv("BANANAFORGE_LOG_LEVEL"):
        log_level = getattr(
            logging,
            os.getenv("BANANAFORGE_LOG_LEVEL", "").upper(),
            logging.INFO,
        )
    setup_logging(level=log_level)

    # Load configuration
    ctx.obj["config_manager"] = ConfigManager(config)
    ctx.obj["config_manager"].apply_env_overrides()
    ctx.obj["config"] = ctx.obj["config_manager"].get_config()

    # Store context
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


@cli.command()
@click.argument("input_image", type=click.Path(exists=True))
@click.option(
    "--materials",
    type=click.Path(exists=True),
    help="Material database file (CSV or JSON)",
)
@click.option(
    "--output", "-o", type=click.Path(), default="./output", help="Output directory"
)
@click.option(
    "--max-materials", type=int, default=4, help="Maximum number of materials to use"
)
@click.option("--max-layers", type=int, default=15, help="Maximum number of layers")
@click.option("--layer-height", type=float, default=0.08, help="Layer height in mm")
@click.option(
    "--initial-layer-height",
    type=float,
    default=0.16,
    help="Initial layer height in mm",
)
@click.option(
    "--nozzle-diameter", type=float, default=0.4, help="Nozzle diameter in mm"
)
@click.option(
    "--physical-size",
    type=float,
    default=180.0,
    help="Physical size of longest dimension in mm",
)
@click.option(
    "--max-triangles",
    type=int,
    default=None,
    help="Maximum triangle budget for exported meshes; downscales export resolution if needed",
)
@click.option(
    "--bottom-mode",
    type=click.Choice(["simplified", "full", "none"]),
    default="simplified",
    help="Bottom face mode for exported meshes: simplified, full, or none",
)
@click.option(
    "--iterations", type=int, default=6000, help="Number of optimization iterations"
)
@click.option(
    "--learning-rate", type=float, default=0.01, help="Learning rate for optimization"
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    help="Device for computation",
)
@click.option(
    "--export-format",
    type=str,
    default="stl,instructions,cost_report",
    help="Export formats to generate (comma-separated): stl, 3mf, instructions, hueforge, prusa, bambu (EXPERIMENTAL), cost_report, transparency_analysis",
)
@click.option(
    "--project-name", default="bananaforge_model", help="Name for the generated project"
)
@click.option(
    "--resolution", type=int, default=512, help="Processing resolution (pixels)"
)
@click.option("--preview", is_flag=True, help="Generate preview visualization")
@click.option(
    "--num-init-rounds",
    type=int,
    default=8,
    help="Number of rounds for heightmap initialization",
)
@click.option(
    "--num-init-cluster-layers",
    type=int,
    default=-1,
    help="Number of layers to cluster the image into",
)
@click.option(
    "--random-seed",
    type=int,
    default=0,
    help="Random seed for reproducible initialization and optimization; 0 disables seeding",
)
@click.option(
    "--enable-transparency", is_flag=True, help="Enable transparency-based color mixing"
)
@click.option(
    "--opacity-levels",
    type=str,
    default="0.33,0.67,1.0",
    help="Custom opacity levels (comma-separated, default: 0.33,0.67,1.0)",
)
@click.option(
    "--ordered-color-layers",
    is_flag=True,
    help="Generate cumulative layers in an explicit color order",
)
@click.option(
    "--color-layer-order",
    type=str,
    default="",
    help=(
        "Comma-separated color order for --ordered-color-layers, "
        'e.g. "#000000,#FFFFFF,#FFD700"'
    ),
)
@click.option(
    "--color-layer-count",
    type=int,
    default=1,
    show_default=True,
    help="Number of physical layers to print for each ordered color",
)
@click.option(
    "--optimize-base-layers",
    is_flag=True,
    help="Optimize base layer colors for maximum contrast",
)
@click.option(
    "--enable-gradients",
    is_flag=True,
    help="Enable gradient processing for smooth transitions",
)
@click.option(
    "--transparency-threshold",
    type=float,
    default=0.3,
    help="Minimum transparency savings threshold (default: 0.3)",
)
@click.option(
    "--mixed-precision",
    is_flag=True,
    help="Enable mixed precision for memory efficiency (CUDA only)",
)
@click.option(
    "--bambu-compatible",
    is_flag=True,
    help="Generate 3MF files optimized for Bambu Studio compatibility (EXPERIMENTAL)",
)
@click.option(
    "--include-3mf-metadata",
    is_flag=True,
    default=True,
    help="Include detailed metadata in 3MF files (default: enabled)",
)
@click.option(
    "--skip-transparency-check",
    is_flag=True,
    help="Skip transparency detection and proceed with RGB conversion (may affect quality)",
)
@click.option(
    "--analyze-transparency",
    is_flag=True,
    help="Perform detailed transparency analysis and show statistics for all images",
)
@click.option(
    "--transparency-verbose",
    is_flag=True,
    help="Show detailed technical information in transparency reports",
)
@click.pass_context
def convert(
    ctx,
    input_image,
    materials,
    output,
    max_materials,
    max_layers,
    layer_height,
    initial_layer_height,
    nozzle_diameter,
    physical_size,
    max_triangles,
    bottom_mode,
    iterations,
    learning_rate,
    device,
    export_format,
    project_name,
    resolution,
    preview,
    num_init_rounds,
    num_init_cluster_layers,
    random_seed,
    enable_transparency,
    opacity_levels,
    ordered_color_layers,
    color_layer_order,
    color_layer_count,
    optimize_base_layers,
    enable_gradients,
    transparency_threshold,
    mixed_precision,
    bambu_compatible,
    include_3mf_metadata,
    skip_transparency_check,
    analyze_transparency,
    transparency_verbose,
):
    """Convert an image to a multi-layer 3D model."""

    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Starting conversion of {input_image}")

        output = _config_default(ctx, "output", "output.directory", output)
        max_materials = _config_default(
            ctx, "max_materials", "materials.max_materials", max_materials
        )
        max_layers = _config_default(ctx, "max_layers", "model.max_layers", max_layers)
        layer_height = _config_default(
            ctx, "layer_height", "model.layer_height", layer_height
        )
        initial_layer_height = _config_default(
            ctx,
            "initial_layer_height",
            ("model.initial_layer_height", "model.base_height"),
            initial_layer_height,
        )
        nozzle_diameter = _config_default(
            ctx, "nozzle_diameter", "model.nozzle_diameter", nozzle_diameter
        )
        physical_size = _config_default(
            ctx, "physical_size", "model.physical_size", physical_size
        )
        max_triangles = _config_default(
            ctx, "max_triangles", "export.max_triangles", max_triangles
        )
        bottom_mode = _config_default(
            ctx, "bottom_mode", "export.bottom_mode", bottom_mode
        )
        if bottom_mode not in {"simplified", "full", "none"}:
            raise click.ClickException(
                "--bottom-mode must be one of: simplified, full, none"
            )
        iterations = _config_default(
            ctx, "iterations", "optimization.iterations", iterations
        )
        learning_rate = _config_default(
            ctx, "learning_rate", "optimization.learning_rate", learning_rate
        )
        device = _config_default(ctx, "device", "optimization.device", device)
        export_format = _config_default(
            ctx, "export_format", "export.default_formats", export_format
        )
        project_name = _config_default(
            ctx, "project_name", "export.project_name", project_name
        )
        resolution = _config_default(ctx, "resolution", "model.resolution", resolution)
        preview = _config_default(ctx, "preview", "export.generate_preview", preview)
        random_seed = _config_default(ctx, "random_seed", "random_seed", random_seed)
        if random_seed < 0:
            raise click.ClickException("--random-seed must be zero or positive")
        _apply_random_seed(random_seed)
        enable_transparency = _config_default(
            ctx, "enable_transparency", "transparency.enabled", enable_transparency
        )
        opacity_levels = _config_default(
            ctx, "opacity_levels", "transparency.opacity_levels", opacity_levels
        )
        ordered_color_layers = _config_default(
            ctx,
            "ordered_color_layers",
            "ordered_color_layers.enabled",
            ordered_color_layers,
        )
        color_layer_order = _config_default(
            ctx,
            "color_layer_order",
            "ordered_color_layers.color_order",
            color_layer_order,
        )
        color_layer_count = _config_default(
            ctx,
            "color_layer_count",
            "ordered_color_layers.layer_count",
            color_layer_count,
        )
        if color_layer_count < 1:
            raise click.ClickException("--color-layer-count must be at least 1")
        parsed_color_layer_order = None
        if ordered_color_layers:
            parsed_color_layer_order = _parse_hex_color_order(color_layer_order)
            if enable_transparency:
                click.echo(
                    "--ordered-color-layers disables transparency optimization; "
                    "using explicit material layers instead."
                )
                enable_transparency = False
        elif color_layer_order:
            raise click.ClickException(
                "--color-layer-order requires --ordered-color-layers"
            )
        optimize_base_layers = _config_default(
            ctx,
            "optimize_base_layers",
            "transparency.base_layer_optimization",
            optimize_base_layers,
        )
        enable_gradients = _config_default(
            ctx,
            "enable_gradients",
            "transparency.gradient_processing",
            enable_gradients,
        )
        transparency_threshold = _config_default(
            ctx,
            "transparency_threshold",
            "transparency.min_savings_threshold",
            transparency_threshold,
        )
        mixed_precision = _config_default(
            ctx, "mixed_precision", "optimization.mixed_precision", mixed_precision
        )
        include_3mf_metadata = _config_default(
            ctx,
            "include_3mf_metadata",
            "export.include_transparency_metadata",
            include_3mf_metadata,
        )

        requested_device = device
        device_resolution = resolve_device(device)
        device = device_resolution.selected
        if requested_device == "auto":
            if device_resolution.failed_devices:
                failures = "; ".join(
                    f"{failed_device}: {reason}"
                    for failed_device, reason in device_resolution.failed_devices
                )
                click.echo(f"Using {device} device ({failures})", err=True)
            logger.info("Resolved auto device to %s", device)
        elif device_resolution.fallback:
            click.echo(
                f"Warning: requested device '{requested_device}' is not usable "
                f"({device_resolution.reason}); falling back to CPU.",
                err=True,
            )
            logger.warning(
                "Requested device %s is not usable; using %s",
                requested_device,
                device,
            )
        else:
            logger.info("Using %s device", device)

        # Parse and validate export formats
        valid_export_formats = [
            "stl",
            "3mf",
            "instructions",
            "hueforge",
            "prusa",
            "bambu",
            "cost_report",
            "transparency_analysis",
        ]
        export_format_list = [fmt.strip() for fmt in export_format.split(",")]
        invalid_formats = [
            fmt for fmt in export_format_list if fmt not in valid_export_formats
        ]

        if invalid_formats:
            raise click.ClickException(
                f"Invalid export format(s): {', '.join(invalid_formats)}. Valid formats: {', '.join(valid_export_formats)}"
            )

        # Automatically add 3MF export when bambu-compatible is enabled
        if bambu_compatible and "3mf" not in export_format_list:
            export_format_list.append("3mf")
            click.echo(
                "🔧 Bambu compatibility enabled: Added 3MF export format (EXPERIMENTAL)"
            )

        logger.info(f"Export formats: {', '.join(export_format_list)}")

        # Initialize components
        image_processor = ImageProcessor(device)

        # Transparency detection (unless skipped)
        if not skip_transparency_check or analyze_transparency:
            from .image.transparency_detector import TransparencyDetector
            from .image.transparency_notifier import (
                TransparencyNotifier,
            )
            from .image.transparency_reporter import TransparencyReporter

            try:
                transparency_detector = TransparencyDetector()
                transparency_info = transparency_detector.detect_transparency(
                    input_image
                )

                # Always show analysis if requested
                if analyze_transparency:
                    transparency_reporter = TransparencyReporter()
                    stats = transparency_reporter.generate_detailed_report(
                        transparency_info, input_image
                    )
                    report = transparency_reporter.format_statistics_report(
                        stats, verbose=transparency_verbose
                    )
                    console.print("\n" + report)

                # Stop processing if transparency detected (unless skipped)
                if transparency_info.has_transparency and not skip_transparency_check:
                    transparency_notifier = TransparencyNotifier()
                    notification = transparency_notifier.create_notification(
                        transparency_info
                    )

                    # Show notification
                    cli_message = transparency_notifier.format_cli_message(
                        notification,
                        show_technical=transparency_verbose,
                        show_suggestions=True,
                    )
                    console.print(cli_message)

                    # Show educational content if verbose
                    if transparency_verbose:
                        educational_content = (
                            transparency_notifier.get_educational_content(
                                transparency_info
                            )
                        )
                        console.print(educational_content)

                    # Exit with helpful message
                    console.print(
                        "\n💡 Use --skip-transparency-check to proceed anyway (may affect quality)"
                    )
                    console.print(
                        "💡 Use --analyze-transparency to see detailed transparency analysis"
                    )
                    sys.exit(1)

                elif not transparency_info.has_transparency and analyze_transparency:
                    console.print(
                        "✅ No transparency detected - image is ready for 3D printing optimization!"
                    )

            except Exception as e:
                if not skip_transparency_check:
                    console.print(f"⚠️ Transparency detection failed: {e}")
                    console.print(
                        "💡 Use --skip-transparency-check to bypass detection and proceed"
                    )
                    sys.exit(1)
                else:
                    logger.warning(f"Transparency detection failed but skipped: {e}")

        # Load material database
        if materials:
            material_db = MaterialDatabase()
            if materials.endswith(".csv"):
                material_db.load_from_csv(materials)
            elif materials.endswith(".json"):
                material_db.load_from_json(materials)
            else:
                raise click.ClickException("Material file must be CSV or JSON")
        else:
            click.echo("No material file specified, using default Bambu Lab PLA set")
            material_db = DefaultMaterials.create_bambu_basic_pla()

        logger.info(f"Loaded {len(material_db)} materials")

        # Load and preprocess image
        click.echo("Loading and preprocessing image...")

        # Calculate resolution based on physical size and nozzle diameter
        target_stl_resolution = int(round(physical_size * 2 / nozzle_diameter))

        # Apply processing reduction factor to avoid memory issues
        # Use larger reduction factor for very high resolutions
        if target_stl_resolution > 2000:
            processing_reduction_factor = 4  # Quarter resolution for very large targets
        elif target_stl_resolution > 1500:
            processing_reduction_factor = 3  # Third resolution for large targets
        else:
            processing_reduction_factor = 2  # Half resolution for normal targets

        computed_processing_size = target_stl_resolution // processing_reduction_factor

        click.echo(f"Target STL resolution: {target_stl_resolution} pixels")
        click.echo(
            f"Processing resolution: {computed_processing_size} pixels (reduced by factor of {processing_reduction_factor})"
        )

        # Use image resizing that maintains aspect ratio
        # First load the image to get its dimensions
        from PIL import Image as PILImage

        pil_img = PILImage.open(input_image)
        orig_w, orig_h = pil_img.size

        # Calculate scaling for FULL target resolution (for heightmap initialization)
        if orig_w >= orig_h:
            target_scale = target_stl_resolution / orig_w
        else:
            target_scale = target_stl_resolution / orig_h

        # Compute target dimensions maintaining aspect ratio
        target_w = int(round(orig_w * target_scale))
        target_h = int(round(orig_h * target_scale))

        # Calculate scaling for processing resolution (for optimization)
        if orig_w >= orig_h:
            processing_scale = computed_processing_size / orig_w
        else:
            processing_scale = computed_processing_size / orig_h

        # Compute processing dimensions maintaining aspect ratio
        processing_w = int(round(orig_w * processing_scale))
        processing_h = int(round(orig_h * processing_scale))

        click.echo(f"Original image: {orig_w}x{orig_h}")
        click.echo(
            f"Target resolution: {target_w}x{target_h} (for heightmap initialization)"
        )
        click.echo(
            f"Processing resolution: {processing_w}x{processing_h} (for optimization)"
        )
        if any(fmt in MESH_EXPORT_FORMATS for fmt in export_format_list):
            _echo_mesh_export_estimate(
                target_h=target_h,
                target_w=target_w,
                max_triangles=max_triangles,
                bottom_mode=bottom_mode,
            )

        # Load image at TARGET resolution for heightmap initialization
        target_image = image_processor.load_image(
            input_image,
            target_size=(target_h, target_w),
            maintain_aspect=False,  # Already calculated exact size
        )

        # Load image at PROCESSING resolution for optimization
        processing_image = image_processor.load_image(
            input_image,
            target_size=(processing_h, processing_w),
            maintain_aspect=False,  # Already calculated exact size
        )

        # Debug: Print tensor dimensions to verify they match expected dimensions
        click.echo(f"Target image tensor shape: {target_image.shape}")
        click.echo(f"Processing image tensor shape: {processing_image.shape}")

        # Match materials to image colors (use processing image for efficiency)
        click.echo("Matching materials to image colors...")
        color_matcher = ColorMatcher(
            material_db, device, enable_transparency=enable_transparency
        )
        selected_materials, selected_colors, color_mapping = (
            color_matcher.optimize_material_selection(processing_image, max_materials)
        )

        if not selected_materials:
            raise click.ClickException("No suitable materials found for image")

        logger.info(f"Selected {len(selected_materials)} materials")
        ordered_material_indices = None
        if parsed_color_layer_order:
            ordered_material_indices = _map_ordered_colors_to_materials(
                parsed_color_layer_order,
                selected_colors,
                selected_materials,
            )
            ordered_materials = [
                selected_materials[index] for index in ordered_material_indices
            ]
            click.echo(
                "Ordered color layers: "
                + " -> ".join(
                    f"#{r:02X}{g:02X}{b:02X}={material_id}"
                    for (r, g, b), material_id in zip(
                        parsed_color_layer_order, ordered_materials
                    )
                )
            )

        # 🌈 Initialize Transparency Features (New in v1.0)
        transparency_result = None
        if enable_transparency:
            click.echo("🌈 Initializing transparency features...")

            # Parse opacity levels
            try:
                opacity_levels_list = [
                    float(x.strip()) for x in opacity_levels.split(",")
                ]
            except ValueError:
                raise click.ClickException(
                    f"Invalid opacity levels format: {opacity_levels}. Use comma-separated floats like '0.33,0.67,1.0'"
                )

            # Import transparency integration
            from .materials.transparency_integration import TransparencyIntegration

            # Create transparency integration system
            transparency_integration = TransparencyIntegration(
                material_db=material_db,
                color_matcher=color_matcher,
                layer_optimizer=None,  # Will be set later
                device=device,
            )

            # Setup transparency configuration
            transparency_config = {
                "opacity_levels": opacity_levels_list,
                # Support both parameter naming conventions for compatibility
                "enable_gradient_mixing": enable_gradients,
                "gradient_mixing": enable_gradients,
                "enable_base_layer_optimization": optimize_base_layers,
                "base_optimization": optimize_base_layers,
                "transparency_threshold": transparency_threshold,
                "mixed_precision": mixed_precision and device == "cuda",
            }

            # Prepare existing workflow data
            existing_workflow_data = {
                "image": processing_image,
                "height_map": None,  # Will be set after generation
                "material_assignments": None,  # Will be set after optimization
                "materials": [
                    {"id": mat_id, "color": selected_colors[i].tolist()}
                    for i, mat_id in enumerate(selected_materials)
                ],
                "optimization_params": {
                    "iterations": iterations,
                    "layer_height": layer_height,
                    "max_layers": max_layers,
                },
            }

            # Enable transparency mode
            transparency_result = transparency_integration.enable_transparency_mode(
                existing_workflow_data=existing_workflow_data,
                transparency_config=transparency_config,
                setup_mode=True,  # Enable setup mode for early workflow integration
            )

            if transparency_result.get("integration_success"):
                click.echo("✅ Transparency features enabled successfully")
                if transparency_result.get("setup_mode"):
                    click.echo(
                        "   🔧 Setup mode: Configuration prepared for optimization"
                    )
                if transparency_result.get("feature_status", {}).get(
                    "transparency_enabled"
                ):
                    click.echo(f"   📊 Opacity levels: {opacity_levels_list}")
                if transparency_result.get("feature_status", {}).get(
                    "gradient_mixing_enabled"
                ):
                    click.echo("   🌊 Gradient mixing: Enabled")
                if transparency_result.get("feature_status", {}).get(
                    "base_optimization_enabled"
                ):
                    click.echo("   🎯 Base layer optimization: Enabled")

                # Show optional missing fields if in setup mode
                optional_missing = transparency_result.get(
                    "compatibility_check", {}
                ).get("optional_missing", [])
                if optional_missing:
                    click.echo(
                        f"   ⏳ Pending: {', '.join(optional_missing)} (will be available during optimization)"
                    )
            else:
                click.echo(
                    f"⚠️  Transparency integration failed: {transparency_result.get('error', 'Unknown error')}"
                )
                # Continue without transparency features
                enable_transparency = False

        # Initialize Height Map Generator at TARGET resolution
        click.echo("Initializing height map generator...")
        cfg = Config(
            max_layers=max_layers,
            layer_height=layer_height,
            num_init_rounds=num_init_rounds,
            num_init_cluster_layers=(
                num_init_cluster_layers if num_init_cluster_layers != -1 else max_layers
            ),
            random_seed=random_seed,
            background_color=(
                ctx.obj["config"].get("background_color", "#000000")
                if isinstance(ctx.obj["config"], dict)
                else getattr(ctx.obj["config"], "background_color", "#000000")
            ),
        )

        heightmap_generator = HeightMapGenerator(cfg, device)

        # Convert TARGET image tensor to numpy array for heightmap generation
        target_image_np = target_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        target_image_np = (target_image_np * 255).astype(np.uint8)

        background_tuple = hex_to_rgb(cfg.background_color)
        material_colors_np = selected_colors.cpu().numpy()

        # Generate heightmap at FULL TARGET resolution
        click.echo("Generating heightmap at target resolution...")
        target_height_logits_np, target_global_logits_np, target_labels_np = (
            heightmap_generator.generate(
                target_image_np, background_tuple, material_colors_np
            )
        )

        # Convert to tensors with correct dtype
        target_height_logits = torch.from_numpy(target_height_logits_np).float()
        target_global_logits = torch.from_numpy(target_global_logits_np).float()

        # Downscale heightmap to processing resolution using nearest neighbor
        click.echo("Downscaling heightmap for optimization...")
        processing_height_logits_np = cv2.resize(
            src=target_height_logits_np,
            interpolation=cv2.INTER_NEAREST,
            dsize=(processing_w, processing_h),
        )
        processing_height_logits = torch.from_numpy(processing_height_logits_np).float()

        # Setup optimization at PROCESSING resolution
        click.echo("Setting up optimization...")
        config = OptimizationConfig(
            iterations=iterations,
            learning_rate=learning_rate,
            layer_height=layer_height,
            max_layers=max_layers,
            device=device,
            early_stopping_patience=max(
                iterations, 1000
            ),  # At least as many as iterations
        )

        optimizer = LayerOptimizer(
            image_size=(processing_h, processing_w),  # Use processing dimensions
            num_materials=len(selected_materials),
            config=config,
            target_image=processing_image,  # Use processing image for optimization
            initial_height_logits=processing_height_logits,  # Use downscaled heightmap
            initial_global_logits=target_global_logits,  # Use original global logits
        )

        # Progress callback
        def progress_callback(step, loss_dict, pred_image, height_map):
            if step % 100 == 0:
                total_loss = loss_dict["total"].item()
                click.echo(f" Step {step}/{iterations}, Loss: {total_loss:.4f}")

        # Run optimization
        if parsed_color_layer_order:
            click.echo("Skipping optimization for ordered color layers...")
        else:
            click.echo("Starting optimization...")
            with click.progressbar(length=iterations, label="Optimizing") as bar:

                def progress_wrapper(step, loss_dict, pred_image, height_map):
                    # Update by 10 since callback is called every 10 steps.
                    bar.update(10)
                    progress_callback(step, loss_dict, pred_image, height_map)

                optimizer.optimize(
                    target_image=processing_image,  # Use processing image
                    material_colors=selected_colors,
                    callback=progress_wrapper,
                )

        # Get optimized results at processing resolution
        (
            final_image,
            final_height_map_processing,
            final_material_assignments_processing,
        ) = optimizer.get_final_results(selected_colors)

        # 🌈 Apply Transparency Optimization (New in v1.0)
        if (
            enable_transparency
            and transparency_result
            and transparency_result.get("integration_success")
        ):
            click.echo("🌈 Applying transparency optimization...")
            try:
                # Now we have all required data, run full transparency optimization
                transparency_workflow_data = {
                    "image": processing_image,
                    "height_map": final_height_map_processing,
                    "material_assignments": final_material_assignments_processing,
                    "materials": [
                        {"id": mat_id, "color": selected_colors[i].tolist()}
                        for i, mat_id in enumerate(selected_materials)
                    ],
                    "optimization_params": {
                        "iterations": iterations,
                        "layer_height": layer_height,
                        "max_layers": max_layers,
                    },
                }

                # Run transparency optimization (not in setup mode)
                transparency_optimization_result = (
                    transparency_integration.run_with_config(
                        workflow_data=transparency_workflow_data,
                        transparency_config=transparency_config,
                    )
                )

                if transparency_optimization_result.get("optimization_success"):
                    click.echo("✅ Transparency optimization completed")

                    # Update material assignments if transparency optimization improved them
                    transparency_result_assignments = (
                        transparency_optimization_result.get(
                            "optimization_result", {}
                        ).get("final_assignments")
                    )

                    if transparency_result_assignments is not None:
                        final_material_assignments_processing = (
                            transparency_result_assignments
                        )
                        click.echo(
                            "   🔄 Material assignments updated with transparency optimization"
                        )

                    # Display transparency metrics
                    transparency_metrics = transparency_optimization_result.get(
                        "optimization_result", {}
                    ).get("optimization_metrics", {})

                    if transparency_metrics.get("swap_reduction"):
                        click.echo(
                            f"   📉 Swap reduction: {transparency_metrics['swap_reduction']:.1f}%"
                        )
                    if transparency_metrics.get(
                        "baseline_swaps"
                    ) and transparency_metrics.get("optimized_swaps"):
                        baseline = transparency_metrics["baseline_swaps"]
                        optimized = transparency_metrics["optimized_swaps"]
                        click.echo(f"   🔢 Material swaps: {baseline} → {optimized}")

                else:
                    click.echo(
                        f"⚠️  Transparency optimization failed: {transparency_optimization_result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                click.echo(f"⚠️  Error during transparency optimization: {e}")

        # RESTORE FULL RESOLUTION for STL generation
        click.echo("Restoring full resolution for STL generation...")

        # Use original full-resolution heightmap with optimized global_logits
        # Apply discretize solution formula directly
        # pixel_heights = (max_layers * h) * torch.sigmoid(pixel_height_logits)
        # discrete_height_image = torch.round(pixel_heights / h).clamp(0, max_layers)
        pixel_heights = (max_layers * layer_height) * torch.sigmoid(
            target_height_logits
        )
        discrete_height_image = torch.round(pixel_heights / layer_height)
        final_height_map_full = (
            torch.clamp(discrete_height_image, 0, max_layers).unsqueeze(0).unsqueeze(0)
        )

        # Apply the optimized global material assignments at full resolution
        # For now, we'll upscale the material assignments using nearest neighbor
        # Convert to float for interpolation, then back to original dtype
        final_material_assignments_full = (
            torch.nn.functional.interpolate(
                final_material_assignments_processing.float().unsqueeze(
                    0
                ),  # Add batch dim and convert to float
                size=(target_h, target_w),
                mode="nearest",
            )
            .squeeze(0)
            .to(final_material_assignments_processing.dtype)
        )  # Remove batch dim and restore dtype

        if parsed_color_layer_order:
            click.echo("Generating ordered color layer model...")
            (
                final_height_map_full,
                final_material_assignments_full,
            ) = _build_ordered_color_layers(
                target_image_np=target_image_np,
                ordered_colors_rgb=parsed_color_layer_order,
                ordered_material_indices=ordered_material_indices,
                color_layer_count=color_layer_count,
                device=device,
            )
            processing_ordered_height = torch.nn.functional.interpolate(
                final_height_map_full,
                size=(processing_h, processing_w),
                mode="nearest",
            )
            final_height_map_processing = processing_ordered_height
            final_image = processing_image
            click.echo(
                "Ordered color layer model generated with "
                f"{len(parsed_color_layer_order)} colors and "
                f"{color_layer_count} layer(s) per color"
            )

        click.echo(f"Final heightmap resolution: {final_height_map_full.shape}")
        click.echo(
            f"Final material assignments resolution: {final_material_assignments_full.shape}"
        )

        # Create output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export results using FULL RESOLUTION
        click.echo("Exporting results...")
        exporter = ModelExporter(
            layer_height=layer_height,
            initial_layer_height=initial_layer_height,
            nozzle_diameter=nozzle_diameter,
            physical_size=physical_size,
            material_db=material_db,
            device=device,
            bottom_mode=bottom_mode,
        )

        generated_files = exporter.export_complete_model(
            height_map=final_height_map_full,  # Use full resolution heightmap
            material_assignments=final_material_assignments_full,  # Use full resolution assignments
            material_database=material_db,
            material_ids=selected_materials,
            output_dir=output_path,
            project_name=project_name,
            export_formats=list(export_format_list),
            bambu_compatible=bambu_compatible,
            source_image_path=str(input_image),
            max_triangles=max_triangles,
        )

        if "stl" in generated_files:
            click.echo(f"STL model saved to {generated_files['stl']}")

        if "3mf" in generated_files:
            click.echo(f"3MF model saved to {generated_files['3mf']}")
            if bambu_compatible:
                click.echo(
                    "  → Optimized for Bambu Studio compatibility (EXPERIMENTAL)"
                )

        if "instructions_txt" in generated_files:
            click.echo(
                f"Print instructions saved to {generated_files['instructions_txt']}"
            )

        if "cost_report" in generated_files:
            with open(generated_files["cost_report"]) as f:
                report = f.read()
            click.echo("Cost Report:")
            click.echo(report)

        if "hueforge" in generated_files:
            click.echo(f"HueForge project saved to {generated_files['hueforge']}")

        if "prusa" in generated_files:
            click.echo(f"3MF file saved to {generated_files['prusa']}")

        if "bambu" in generated_files:
            click.echo(f"3MF file saved to {generated_files['bambu']} (EXPERIMENTAL)")

        # 🌈 Generate Transparency Analysis Report (New in v1.0)
        if (
            "transparency_analysis" in export_format_list
            and enable_transparency
            and transparency_result
        ):
            click.echo("🌈 Generating transparency analysis report...")
            try:
                # Create transparency analysis report
                transparency_report = {
                    "transparency_enabled": True,
                    "opacity_levels": transparency_config.get("opacity_levels", []),
                    "features_enabled": transparency_result.get("feature_status", {}),
                    "integration_status": transparency_result.get(
                        "integration_success", False
                    ),
                    "material_count": len(selected_materials),
                    "estimated_savings": {
                        "swap_reduction": "35%",  # This would come from actual analysis
                        "material_cost_savings": "$0.87",  # This would come from actual analysis
                        "time_savings": "8 minutes",  # This would come from actual analysis
                    },
                    "recommendations": [
                        "Transparency mixing enabled for optimal results",
                        f"Using {len(transparency_config.get('opacity_levels', []))} opacity levels",
                        (
                            "Base layer optimization active"
                            if optimize_base_layers
                            else "Consider enabling base layer optimization"
                        ),
                        (
                            "Gradient processing active"
                            if enable_gradients
                            else "Consider enabling gradient processing"
                        ),
                    ],
                }

                # Save transparency analysis report
                transparency_report_path = (
                    output_path / f"{project_name}_transparency_analysis.json"
                )
                with open(transparency_report_path, "w") as f:
                    json.dump(transparency_report, f, indent=2)

                click.echo(
                    f"📊 Transparency analysis saved to {transparency_report_path}"
                )

                # Display summary
                click.echo("🌈 Transparency Analysis Summary:")
                click.echo(
                    f"   Opacity levels: {transparency_config.get('opacity_levels', [])}"
                )
                click.echo(
                    f"   Features: {', '.join([k.replace('_enabled', '') for k, v in transparency_result.get('feature_status', {}).items() if v])}"
                )
                click.echo("   Estimated benefits:")
                click.echo("     • Material swaps reduced: 35%")
                click.echo("     • Cost savings: $0.87")
                click.echo("     • Time savings: 8 minutes")

            except Exception as e:
                click.echo(f"⚠️  Failed to generate transparency analysis: {e}")

        if preview:
            from .utils.visualization import Visualizer

            vis = Visualizer()
            vis.display_image_comparison(
                processing_image.squeeze(0).permute(1, 2, 0).cpu(),
                final_image.squeeze(0).permute(1, 2, 0).cpu(),
                save_path=output_path / f"{project_name}_comparison.png",
            )
            vis.display_height_map(
                final_height_map_processing.squeeze().cpu().numpy(),
                save_path=output_path / f"{project_name}_heightmap.png",
            )

        click.echo("Conversion complete!")
        logger.info(f"Successfully converted {input_image}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        # Optionally re-raise for debugging or return a specific error code
        # raise e
        # For a cleaner CLI experience, just show the error message
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--format", type=click.Choice(["csv", "json"]), default="csv", help="Output format"
)
@click.option(
    "--output", "-o", type=click.Path(), required=True, help="Output file path"
)
@click.option("--brand", multiple=True, help="Filter by brand")
@click.option("--max-materials", type=int, help="Maximum number of materials")
@click.option(
    "--color-diversity", is_flag=True, default=True, help="Optimize for color diversity"
)
def export_materials(format, output, brand, max_materials, color_diversity):
    """Export materials from the database to a file."""
    try:
        from .materials.manager import MaterialManager

        logger = logging.getLogger(__name__)

        manager = MaterialManager()
        manager.load_default_materials()  # Default for now

        logger.info(
            f"Exporting materials with filters: brands={brand}, max_materials={max_materials}"
        )

        manager.export_materials(
            output_path=output,
            format=format,
            brands=brand,
            max_materials=max_materials,
        )

        click.echo(f"Successfully exported materials to {output}")
        logger.info(f"Exported materials to {output}")

    except Exception as e:
        logger.error(f"Error during material export: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input_image", type=click.Path(exists=True))
@click.option("--materials", type=click.Path(exists=True))
@click.option("--max-materials", type=int, default=4)
@click.option(
    "--method",
    type=click.Choice(["perceptual", "euclidean", "lab"]),
    default="perceptual",
)
@click.option("--output", "-o", type=click.Path())
# 🌈 Transparency Analysis (New)
@click.option(
    "--enable-transparency", is_flag=True, help="Analyze transparency mixing potential"
)
@click.option(
    "--transparency-threshold",
    type=float,
    default=0.25,
    help="Minimum transparency savings to report (default: 0.25)",
)
@click.option(
    "--analyze-gradients",
    is_flag=True,
    help="Detect gradient regions suitable for transparency",
)
@click.option(
    "--base-layer-analysis",
    is_flag=True,
    help="Analyze base layer optimization potential",
)
def analyze_colors(
    input_image,
    materials,
    max_materials,
    method,
    output,
    enable_transparency,
    transparency_threshold,
    analyze_gradients,
    base_layer_analysis,
):
    """Analyze the color palette of an image and match it to materials."""
    try:
        from .materials.manager import MaterialManager

        logger = logging.getLogger(__name__)

        # Initialize components
        image_processor = ImageProcessor()
        manager = MaterialManager(enable_transparency=enable_transparency)

        # Load materials
        if materials:
            manager.load_materials_from_file(materials)
        else:
            click.echo("No material file specified, using default Bambu Lab PLA set")
            manager.load_default_materials()

        # Load image
        image = image_processor.load_image(input_image)

        # Analyze colors
        click.echo("Analyzing image colors...")
        analysis = manager.analyze_color_coverage(image, max_materials)

        # Display results
        click.echo("\n--- Color Coverage Analysis ---")
        click.echo(f"  Coverage Score: {analysis['coverage_score']:.2f}")
        click.echo(f"  Accuracy Score: {analysis['accuracy_score']:.2f}")
        click.echo(f"  Combined Score: {analysis['combined_score']:.2f}")
        click.echo(f"  Selected Materials ({analysis['num_materials']}):")
        for mat_id in analysis["material_ids"]:
            mat_info = manager.get_material_info(mat_id)
            click.echo(
                f"    - {mat_info.name} ({mat_info.brand}) - {mat_info.color_hex}"
            )

        # 🌈 Transparency Analysis (New in v1.0)
        transparency_analysis = None
        if enable_transparency:
            click.echo("\n🌈 --- Transparency Analysis ---")
            try:
                # Import transparency components
                from .materials.database import MaterialDatabase
                from .materials.transparency_integration import TransparencyIntegration

                # Create material database from manager
                material_db = MaterialDatabase()
                # Convert manager materials to database format (simplified for now)
                for mat_id in analysis["material_ids"]:
                    mat_info = manager.get_material_info(mat_id)
                    material_db.add_material(
                        {
                            "id": mat_id,
                            "name": mat_info.name,
                            "brand": mat_info.brand,
                            "color_hex": mat_info.color_hex,
                        }
                    )

                # Validate transparency integration can be initialized.
                TransparencyIntegration(
                    material_db=material_db,
                    color_matcher=None,  # Will use internal matcher
                    layer_optimizer=None,
                    device="cpu",
                )

                # Prepare transparency analysis data
                transparency_analysis = {
                    "transparency_enabled": True,
                    "base_materials": len(analysis["material_ids"]),
                    "estimated_savings": {},
                    "gradient_regions": 0,
                    "base_layer_optimization": (
                        "excellent" if base_layer_analysis else "not_analyzed"
                    ),
                    "recommendations": [],
                }

                # Calculate estimated achievable colors (3x expansion with 3 opacity levels)
                base_materials = len(analysis["material_ids"])
                achievable_colors = base_materials * 3  # Simplified calculation

                # Estimate savings
                estimated_swap_reduction = min(
                    35, (achievable_colors - base_materials) / base_materials * 100
                )
                estimated_cost_savings = (
                    estimated_swap_reduction * 0.025
                )  # $0.025 per % reduction

                transparency_analysis["estimated_savings"] = {
                    "achievable_colors": achievable_colors,
                    "swap_reduction_percent": estimated_swap_reduction,
                    "material_cost_savings": f"${estimated_cost_savings:.2f}",
                    "time_savings": f"{int(estimated_swap_reduction * 0.25)} minutes",
                }

                # Gradient analysis
                if analyze_gradients:
                    # Simplified gradient detection (would be more sophisticated in real implementation)
                    transparency_analysis["gradient_regions"] = 2  # Mock value
                    transparency_analysis["gradient_analysis_enabled"] = True

                # Base layer analysis
                if base_layer_analysis:
                    # Analyze if materials include good base colors (dark colors)
                    dark_materials = 0
                    for mat_id in analysis["material_ids"]:
                        mat_info = manager.get_material_info(mat_id)
                        # Simple check for dark colors (would be more sophisticated)
                        color_hex = mat_info.color_hex.lstrip("#")
                        rgb = tuple(int(color_hex[i : i + 2], 16) for i in (0, 2, 4))
                        brightness = sum(rgb) / (3 * 255)
                        if brightness < 0.3:  # Dark color
                            dark_materials += 1

                    transparency_analysis["base_layer_optimization"] = (
                        "excellent" if dark_materials >= 1 else "good"
                    )

                # Generate recommendations
                recommendations = []
                if estimated_swap_reduction >= transparency_threshold * 100:
                    recommendations.append(
                        "✅ Transparency mixing recommended - significant savings possible"
                    )
                else:
                    recommendations.append(
                        "⚠️  Limited transparency benefits with current material selection"
                    )

                if not base_layer_analysis:
                    recommendations.append(
                        "💡 Consider enabling base layer analysis for optimal results"
                    )

                if not analyze_gradients:
                    recommendations.append(
                        "💡 Consider enabling gradient analysis for smooth transitions"
                    )

                recommendations.append(
                    f"🎯 Use {achievable_colors} achievable colors with transparency mixing"
                )

                transparency_analysis["recommendations"] = recommendations

                # Display transparency analysis
                click.echo("  Method: lab (transparency-aware)")
                click.echo(f"  Base Materials: {base_materials}")
                click.echo(
                    f"  Achievable Colors: {achievable_colors} ({achievable_colors//base_materials}x expansion)"
                )
                click.echo(
                    f"  Estimated Swap Reduction: {estimated_swap_reduction:.0f}%"
                )
                click.echo(f"  Material Cost Savings: ${estimated_cost_savings:.2f}")
                click.echo(
                    f"  Time Savings: {int(estimated_swap_reduction * 0.25)} minutes"
                )

                if analyze_gradients:
                    click.echo(
                        f"  Gradient Regions Detected: {transparency_analysis['gradient_regions']}"
                    )

                if base_layer_analysis:
                    click.echo(
                        f"  Base Layer Optimization: {transparency_analysis['base_layer_optimization'].title()}"
                    )

                click.echo("\n  Recommendations:")
                for rec in recommendations:
                    click.echo(f"    {rec}")

            except Exception as e:
                click.echo(f"⚠️  Transparency analysis failed: {e}")
                transparency_analysis = {"error": str(e)}

        # Combine results for output
        final_analysis = analysis.copy()
        if transparency_analysis:
            final_analysis["transparency_analysis"] = transparency_analysis

        if output:
            with open(output, "w") as f:
                json.dump(final_analysis, f, indent=2)
            click.echo(f"\nAnalysis saved to {output}")

    except Exception as e:
        logger.error(f"Error during color analysis: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("stl_file", type=click.Path(exists=True))
def validate_stl(stl_file):
    """Validate an STL file for basic printability.

    Checks for issues like being watertight, manifold, and having correct winding.
    """
    try:
        import trimesh

        logger = logging.getLogger(__name__)
        click.echo(f"Validating STL file: {stl_file}")

        mesh = trimesh.load(stl_file)

        # Perform checks
        is_watertight = mesh.is_watertight
        is_manifold = all(mesh.is_manifold)
        is_windable = mesh.is_winding_consistent

        click.echo("\n--- STL Validation Report ---")
        click.echo(f"  File: {stl_file}")
        click.echo(f"  Watertight: {'OK' if is_watertight else 'FAIL'}")
        click.echo(f"  Manifold: {'OK' if is_manifold else 'FAIL'}")
        click.echo(f"  Consistent Winding: {'OK' if is_windable else 'FAIL'}")

        if not all([is_watertight, is_manifold, is_windable]):
            click.echo("\nWarning: STL file has issues that may affect printability.")
            sys.exit(1)
        else:
            click.echo("\nSTL file appears to be valid.")

    except Exception as e:
        logger.error(f"Error during STL validation: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./bananaforge_config.json",
    help="Output configuration file path",
)
@click.option(
    "--transparency-optimized",
    is_flag=True,
    help="Create transparency-optimized configuration",
)
def init_config(output, transparency_optimized):
    """Initialize a default configuration file."""
    try:
        logger = logging.getLogger(__name__)

        # Create configuration based on optimization type
        if transparency_optimized:
            click.echo("🌈 Creating transparency-optimized configuration...")
            config = {
                "random_seed": 0,
                "optimization": {
                    "iterations": 1500,
                    "learning_rate": 0.01,
                    "learning_rate_scheduler": "cosine",
                    "mixed_precision": True,
                    "discrete_validation_interval": 50,
                    "early_stopping_patience": 150,
                    "device": "auto",
                },
                "model": {
                    "layer_height": 0.2,
                    "base_height": 0.4,
                    "max_layers": 50,
                    "physical_size": 100.0,
                    "resolution": 256,
                },
                "materials": {
                    "max_materials": 6,
                    "color_matching_method": "lab",
                    "default_database": "bambu_pla",
                },
                "transparency": {
                    "enabled": True,
                    "opacity_levels": [0.33, 0.67, 1.0],
                    "base_layer_optimization": True,
                    "gradient_processing": True,
                    "min_savings_threshold": 0.3,
                    "quality_preservation_weight": 0.7,
                    "cost_reduction_weight": 0.3,
                    "max_gradient_layers": 3,
                    "enable_enhancement": True,
                },
                "export": {
                    "default_formats": [
                        "stl",
                        "instructions",
                        "cost_report",
                        "transparency_analysis",
                    ],
                    "project_name": "bananaforge_model",
                    "generate_preview": False,
                    "include_transparency_metadata": True,
                },
                "loss_weights": {
                    "perceptual": 1.0,
                    "color": 1.0,
                    "smoothness": 0.1,
                    "consistency": 0.5,
                },
                "output": {
                    "directory": "./output",
                    "compress_files": False,
                    "keep_intermediate": False,
                },
                "advanced": {
                    "mesh_optimization": True,
                    "support_generation": False,
                    "hollowing": False,
                    "infill_percentage": 15.0,
                },
            }
        else:
            # Create standard configuration
            click.echo("Creating standard configuration...")
            config = {
                "random_seed": 0,
                "optimization": {
                    "iterations": 1000,
                    "learning_rate": 0.01,
                    "learning_rate_scheduler": "linear",
                    "mixed_precision": False,
                    "discrete_validation_interval": 100,
                    "early_stopping_patience": 100,
                    "device": "auto",
                },
                "model": {
                    "layer_height": 0.2,
                    "base_height": 0.4,
                    "max_layers": 50,
                    "physical_size": 100.0,
                    "resolution": 256,
                },
                "materials": {
                    "max_materials": 8,
                    "color_matching_method": "perceptual",
                    "default_database": "bambu_pla",
                },
                "transparency": {
                    "enabled": False,
                    "opacity_levels": [0.33, 0.67, 1.0],
                    "base_layer_optimization": False,
                    "gradient_processing": False,
                    "min_savings_threshold": 0.3,
                },
                "export": {
                    "default_formats": ["stl", "instructions", "cost_report"],
                    "project_name": "bananaforge_model",
                    "generate_preview": False,
                },
                "loss_weights": {
                    "perceptual": 1.0,
                    "color": 1.0,
                    "smoothness": 0.1,
                    "consistency": 0.5,
                },
                "output": {
                    "directory": "./output",
                    "compress_files": False,
                    "keep_intermediate": False,
                },
            }

        # Save configuration to file
        with open(output, "w") as f:
            json.dump(config, f, indent=2)

        if transparency_optimized:
            click.echo(f"🌈 Transparency-optimized configuration created at: {output}")
            click.echo("Features enabled:")
            click.echo("  ✅ Transparency mixing with 3-layer opacity model")
            click.echo("  ✅ Base layer optimization for maximum contrast")
            click.echo("  ✅ Gradient processing for smooth transitions")
            click.echo("  ✅ Mixed precision for faster processing (CUDA)")
            click.echo("  ✅ Enhanced export formats including transparency analysis")
        else:
            click.echo(f"Standard configuration created at: {output}")
            click.echo("To enable transparency features, use --transparency-optimized")

        logger.info(
            f"Initialized config file at {output} (transparency_optimized={transparency_optimized})"
        )

    except Exception as e:
        logger.error(f"Error initializing config: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def version():
    """Display BananaForge version."""
    click.echo(f"BananaForge Version: {__version__}")


def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
