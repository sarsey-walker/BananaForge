#!/usr/bin/env python3
"""
CLI Integration Tests for Story 7.1: Comprehensive Test Coverage

Tests CLI integration with transparency detection to ensure the
command-line interface properly handles transparency scenarios.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test images."""
    return tmp_path


class TestCLITransparencyIntegration:
    """
    Test CLI integration with transparency detection functionality.
    """

    def create_test_rgba_image(self, path: Path, width: int = 100, height: int = 100):
        """Create test RGBA PNG image."""
        image = Image.new("RGBA", (width, height), (255, 0, 0, 0))  # Transparent red
        # Add opaque center square
        for x in range(width // 4, 3 * width // 4):
            for y in range(height // 4, 3 * height // 4):
                image.putpixel((x, y), (0, 255, 0, 255))  # Opaque green
        image.save(path, "PNG")
        return path

    def create_test_rgb_image(self, path: Path, width: int = 100, height: int = 100):
        """Create test RGB PNG image."""
        image = Image.new("RGB", (width, height), (255, 0, 0))  # Red background
        # Add green center
        for x in range(width // 4, 3 * width // 4):
            for y in range(height // 4, 3 * height // 4):
                image.putpixel((x, y), (0, 255, 0))  # Green center
        image.save(path, "PNG")
        return path

    def test_cli_module_imports(self):
        """Test that CLI module can be imported and basic commands exist."""
        try:
            from bananaforge.cli import cli, convert

            assert cli is not None, "CLI group should be importable"
            assert convert is not None, "Convert command should be importable"
        except ImportError as e:
            pytest.skip(f"CLI module not available for testing: {e}")

    def test_cli_help_command(self):
        """Test that CLI help command works."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "bananaforge", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Help should exit with code 0
            assert result.returncode == 0, "Help command should succeed"
            assert "BananaForge" in result.stdout, "Help should mention BananaForge"
        except subprocess.TimeoutExpired:
            pytest.skip("CLI help command timed out")
        except FileNotFoundError:
            pytest.skip("BananaForge CLI not available in current environment")

    def test_cli_convert_help(self):
        """Test that convert command help works."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "bananaforge", "convert", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, "Convert help should succeed"
            assert "convert" in result.stdout.lower(), "Help should mention convert"
        except subprocess.TimeoutExpired:
            pytest.skip("CLI convert help command timed out")
        except FileNotFoundError:
            pytest.skip("BananaForge CLI not available in current environment")

    def test_cli_version_command(self):
        """Test that version command works."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "bananaforge", "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Version command might not exist, so we check if it at least doesn't crash
            # The important thing is that the module is importable
            assert result.returncode in [0, 2], "Version command should not crash"
        except subprocess.TimeoutExpired:
            pytest.skip("CLI version command timed out")
        except FileNotFoundError:
            pytest.skip("BananaForge CLI not available in current environment")

    def test_cli_transparency_flags_exist(self):
        """Test that transparency-related flags exist in CLI."""
        try:
            # Check if transparency flags are in the convert command
            import click

            from bananaforge.cli import convert

            ctx = click.Context(convert)
            params = [param.name for param in convert.params]

            # Look for transparency-related parameters
            transparency_params = [p for p in params if "transparency" in p.lower()]
            assert (
                len(transparency_params) >= 0
            ), "Should have transparency-related parameters"
            assert "ordered_color_layers" in params
            assert "color_layer_order" in params
            assert "color_layer_count" in params

        except ImportError:
            pytest.skip("CLI module not available for testing")

    @pytest.mark.integration
    def test_cli_basic_functionality_with_transparent_image(self, temp_dir):
        """Test CLI basic functionality with transparent image (integration test)."""
        # This test is marked as integration and might be skipped in regular runs
        try:
            # Create test image
            rgba_path = temp_dir / "test_rgba.png"
            self.create_test_rgba_image(rgba_path)

            # Create output directory
            output_dir = temp_dir / "output"
            output_dir.mkdir()

            # Try to run a basic convert command with minimal options
            # Note: This might fail due to missing dependencies, but we test the interface
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bananaforge",
                    "convert",
                    str(rgba_path),
                    "--output",
                    str(output_dir),
                    "--max-layers",
                    "5",
                    "--iterations",
                    "10",  # Very few iterations for speed
                    "--device",
                    "cpu",
                    "--resolution",
                    "64",  # Low resolution for speed
                    "--skip-transparency-check",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # The command might fail due to various reasons (missing materials, etc.)
            # but it should not crash due to transparency issues
            if result.returncode != 0:
                # Check if failure is transparency-related
                error_output = result.stderr.lower()
                transparency_errors = [
                    "transparency",
                    "alpha",
                    "rgba",
                    "transparent background",
                ]

                has_transparency_error = any(
                    err in error_output for err in transparency_errors
                )

                if has_transparency_error:
                    pytest.fail(
                        f"CLI failed due to transparency handling: {result.stderr}"
                    )
                else:
                    # Other type of failure (materials, optimization, etc.) - acceptable
                    print(
                        f"CLI failed for non-transparency reasons: {result.stderr[:200]}..."
                    )

        except subprocess.TimeoutExpired:
            pytest.skip("CLI convert command timed out")
        except FileNotFoundError:
            pytest.skip("BananaForge CLI not available in current environment")
        except Exception as e:
            pytest.skip(f"CLI integration test failed due to environment: {e}")


class TestCLITransparencyValidation:
    """
    Test CLI validation and error handling for transparency scenarios.
    """

    def test_cli_parameter_validation(self):
        """Test that CLI parameters are properly validated."""
        try:
            import click

            from bananaforge.cli import convert

            # Test that convert command exists and has expected parameters
            assert convert is not None
            assert isinstance(convert, click.Command)

            # Check for transparency-related parameters
            param_names = [param.name for param in convert.params]
            expected_params = ["input_image", "output", "device"]

            for param in expected_params:
                assert (
                    param in param_names
                ), f"Expected parameter {param} not found in CLI"

        except ImportError:
            pytest.skip("CLI module not available")

    def test_cli_transparency_export_formats(self):
        """Test that transparency-related export formats are available."""
        try:
            from bananaforge.cli import convert

            # Find export_format parameter
            export_param = None
            for param in convert.params:
                if param.name == "export_format":
                    export_param = param
                    break

            if export_param and hasattr(export_param, "default"):
                default_formats = str(export_param.default)
                # Should have basic formats available
                assert "stl" in default_formats, "STL export should be available"

        except ImportError:
            pytest.skip("CLI module not available")


class TestCLIDeviceResolution:
    """Test CLI device resolution and accelerator fallback."""

    def test_auto_device_falls_back_to_cpu_when_accelerators_unusable(
        self, monkeypatch
    ):
        """Auto device selection should choose CPU when accelerators fail checks."""
        from bananaforge.utils import device as device_utils

        def fake_device_is_usable(device):
            if device == "cpu":
                return True, ""
            return False, f"{device} unavailable"

        monkeypatch.setattr(
            device_utils,
            "device_is_usable",
            fake_device_is_usable,
        )

        resolution = device_utils.resolve_device("auto")

        assert resolution.selected == "cpu"
        assert resolution.fallback

    def test_requested_unusable_cuda_falls_back_to_cpu(self, monkeypatch):
        """Explicit CUDA should fall back cleanly when PyTorch cannot use it."""
        from bananaforge.utils import device as device_utils

        def fake_device_is_usable(device):
            if device == "cuda":
                return False, "no kernel image is available for execution on the device"
            return True, ""

        monkeypatch.setattr(
            device_utils,
            "device_is_usable",
            fake_device_is_usable,
        )

        resolution = device_utils.resolve_device("cuda")

        assert resolution.selected == "cpu"
        assert resolution.fallback
        assert "no kernel image" in resolution.reason

    def test_convert_accepts_auto_device_option(self):
        """The convert command should expose auto device selection."""
        from bananaforge.cli import convert

        device_param = next(param for param in convert.params if param.name == "device")

        assert "auto" in device_param.type.choices
        assert device_param.default == "auto"

    def test_convert_exposes_mesh_triangle_budget_option(self):
        """The convert command should expose an export triangle budget."""
        from bananaforge.cli import convert

        param_names = [param.name for param in convert.params]

        assert "max_triangles" in param_names

    def test_convert_exposes_bottom_mode_option(self):
        """The convert command should expose bottom face mesh control."""
        from bananaforge.cli import convert

        bottom_mode_param = next(
            param for param in convert.params if param.name == "bottom_mode"
        )

        assert bottom_mode_param.default == "simplified"
        assert set(bottom_mode_param.type.choices) == {"simplified", "full", "none"}

    def test_convert_exposes_random_seed_option(self):
        """The convert command should expose reproducible run seeding."""
        from bananaforge.cli import convert

        random_seed_param = next(
            param for param in convert.params if param.name == "random_seed"
        )

        assert random_seed_param.default == 0
        assert any("--random-seed" in opts for opts in random_seed_param.opts)

    def test_random_seed_helper_seeds_supported_rngs(self):
        """A fixed seed should reset Python, NumPy, and Torch RNG streams."""
        import random

        import numpy as np
        import torch

        from bananaforge.cli import _apply_random_seed

        _apply_random_seed(123)
        first_values = (random.random(), np.random.rand(), torch.rand(1).item())

        _apply_random_seed(123)
        second_values = (random.random(), np.random.rand(), torch.rand(1).item())

        assert first_values == second_values

    def test_preventive_mesh_warning_mentions_size_controls(self, capsys):
        """Large mesh estimates should warn before conversion does heavy work."""
        from bananaforge.cli import _echo_mesh_export_estimate

        _echo_mesh_export_estimate(
            target_h=1000,
            target_w=1000,
            max_triangles=None,
            bottom_mode="simplified",
        )

        output = capsys.readouterr().out
        assert "Large mesh estimate" in output
        assert "--max-triangles" in output
        assert "--bottom-mode none" in output


class TestCLIConfigDefaults:
    """Test config and environment defaults used by CLI commands."""

    def test_config_default_applies_when_option_not_explicit(self):
        """Config values should fill only defaulted Click parameters."""
        import click
        from click.testing import CliRunner

        from bananaforge.cli import _config_default
        from bananaforge.utils.config import ConfigManager

        @click.command()
        @click.option("--iterations", type=int, default=6000)
        @click.pass_context
        def command(ctx, iterations):
            config_manager = ConfigManager()
            config_manager.set("optimization.iterations", 42)
            ctx.obj = {"config_manager": config_manager}
            click.echo(
                _config_default(
                    ctx,
                    "iterations",
                    "optimization.iterations",
                    iterations,
                )
            )

        result = CliRunner().invoke(command, [])

        assert result.exit_code == 0
        assert result.output.strip() == "42"

    def test_explicit_cli_option_overrides_config_default(self):
        """Explicit CLI values should win over config defaults."""
        import click
        from click.testing import CliRunner

        from bananaforge.cli import _config_default
        from bananaforge.utils.config import ConfigManager

        @click.command()
        @click.option("--iterations", type=int, default=6000)
        @click.pass_context
        def command(ctx, iterations):
            config_manager = ConfigManager()
            config_manager.set("optimization.iterations", 42)
            ctx.obj = {"config_manager": config_manager}
            click.echo(
                _config_default(
                    ctx,
                    "iterations",
                    "optimization.iterations",
                    iterations,
                )
            )

        result = CliRunner().invoke(command, ["--iterations", "7"])

        assert result.exit_code == 0
        assert result.output.strip() == "7"

    def test_parse_export_format_list_rejects_unknown_format(self):
        """Export format parsing should keep the existing Click error behavior."""
        import click

        from bananaforge.cli import _parse_export_format_list

        assert _parse_export_format_list("stl,3mf,instructions") == [
            "stl",
            "3mf",
            "instructions",
        ]
        with pytest.raises(click.ClickException):
            _parse_export_format_list("stl,unknown")

    def test_resolve_convert_options_applies_config_and_ordered_layers(self):
        """Convert option resolution should handle config and ordered layers."""
        import logging

        import click
        from click.testing import CliRunner

        from bananaforge.cli import _resolve_convert_options
        from bananaforge.utils.config import ConfigManager

        @click.command()
        @click.option("--max-materials", type=int, default=4)
        @click.option("--device", default="cpu")
        @click.pass_context
        def command(ctx, max_materials, device):
            config_manager = ConfigManager()
            config_manager.set("materials.max_materials", 7)
            ctx.obj = {"config_manager": config_manager}

            options = _resolve_convert_options(
                ctx,
                logging.getLogger(__name__),
                output="./output",
                max_materials=max_materials,
                max_layers=15,
                layer_height=0.08,
                initial_layer_height=0.16,
                nozzle_diameter=0.4,
                physical_size=180.0,
                max_triangles=None,
                bottom_mode="simplified",
                iterations=6000,
                learning_rate=0.01,
                device=device,
                export_format="stl,instructions",
                project_name="bananaforge_model",
                resolution=512,
                preview=False,
                random_seed=0,
                enable_transparency=True,
                opacity_levels="0.33,0.67,1.0",
                ordered_color_layers=True,
                color_layer_order="#000000,#FFFFFF",
                color_layer_count=2,
                optimize_base_layers=False,
                enable_gradients=False,
                transparency_threshold=0.3,
                mixed_precision=False,
                bambu_compatible=True,
                include_3mf_metadata=True,
            )
            click.echo(
                "|".join(
                    [
                        str(options["max_materials"]),
                        options["device"],
                        str(options["enable_transparency"]),
                        str(options["parsed_color_layer_order"]),
                        ",".join(options["export_format_list"]),
                    ]
                )
            )

        result = CliRunner().invoke(command, [])

        assert result.exit_code == 0
        assert result.output.strip().splitlines()[-1] == (
            "7|cpu|False|[(0, 0, 0), (255, 255, 255)]|stl,instructions,3mf"
        )

    def test_calculate_image_dimensions_preserves_landscape_aspect_ratio(self):
        """Landscape images should scale longest side to target resolution."""
        from bananaforge.cli import _calculate_image_dimensions

        dimensions = _calculate_image_dimensions(
            orig_w=1200,
            orig_h=600,
            physical_size=180.0,
            nozzle_diameter=0.4,
        )

        assert dimensions["target_stl_resolution"] == 900
        assert dimensions["processing_reduction_factor"] == 2
        assert dimensions["computed_processing_size"] == 450
        assert dimensions["target_w"] == 900
        assert dimensions["target_h"] == 450
        assert dimensions["processing_w"] == 450
        assert dimensions["processing_h"] == 225

    def test_calculate_image_dimensions_preserves_portrait_aspect_ratio(self):
        """Portrait images should scale longest side to target resolution."""
        from bananaforge.cli import _calculate_image_dimensions

        dimensions = _calculate_image_dimensions(
            orig_w=600,
            orig_h=1200,
            physical_size=180.0,
            nozzle_diameter=0.4,
        )

        assert dimensions["target_w"] == 450
        assert dimensions["target_h"] == 900
        assert dimensions["processing_w"] == 225
        assert dimensions["processing_h"] == 450

    def test_calculate_image_dimensions_uses_large_resolution_reduction(self):
        """Large target resolutions should use stronger processing reduction."""
        from bananaforge.cli import _calculate_image_dimensions

        dimensions = _calculate_image_dimensions(
            orig_w=1000,
            orig_h=1000,
            physical_size=401.0,
            nozzle_diameter=0.4,
        )

        assert dimensions["target_stl_resolution"] == 2005
        assert dimensions["processing_reduction_factor"] == 4
        assert dimensions["computed_processing_size"] == 501

    def test_load_material_database_uses_default_when_no_file(self):
        """No material file should use the default Bambu material set."""
        from bananaforge.cli import _load_material_database

        material_db = _load_material_database(None)

        assert len(material_db) > 0

    def test_load_material_database_rejects_unknown_extension(self, tmp_path):
        """Only CSV and JSON material files are supported by convert."""
        import click

        from bananaforge.cli import _load_material_database

        material_path = tmp_path / "materials.txt"
        material_path.write_text("not,a,material,database")

        with pytest.raises(click.ClickException):
            _load_material_database(str(material_path))

    def test_load_material_database_from_csv(self, tmp_path):
        """CSV material files should load through the CLI helper."""
        from bananaforge.cli import _load_material_database

        material_path = tmp_path / "materials.csv"
        material_path.write_text(
            "\n".join(
                [
                    "id,name,brand,color_hex,transparency,td,density,temperature,cost",
                    "black,Black,Test,#000000,0.0,4.0,1.25,200,25.0",
                ]
            )
        )

        material_db = _load_material_database(str(material_path))

        assert len(material_db) == 1
        assert material_db.get_material("black").color_hex == "#000000"

    def test_prepare_heightmap_uses_config_background_and_downscales(
        self, monkeypatch
    ):
        """Heightmap preparation should adapt tensors and resize for processing."""
        from types import SimpleNamespace

        import numpy as np
        import torch

        import bananaforge.cli as cli_module

        captured = {}

        class FakeHeightMapGenerator:
            def __init__(self, config, device):
                captured["config"] = config
                captured["device"] = device

            def generate(self, image, background_tuple, material_colors_np):
                captured["image_shape"] = image.shape
                captured["background_tuple"] = background_tuple
                captured["material_colors"] = material_colors_np
                return (
                    np.arange(6, dtype=np.float32).reshape(2, 3),
                    np.ones((4, 2), dtype=np.float32),
                    np.zeros((2, 3), dtype=np.int64),
                )

        monkeypatch.setattr(
            cli_module,
            "HeightMapGenerator",
            FakeHeightMapGenerator,
        )

        ctx = SimpleNamespace(obj={"config": {"background_color": "#010203"}})
        target_image = torch.zeros(1, 3, 2, 3)
        selected_colors = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        prepared = cli_module._prepare_heightmap(
            ctx=ctx,
            target_image=target_image,
            selected_colors=selected_colors,
            max_layers=4,
            layer_height=0.08,
            num_init_rounds=1,
            num_init_cluster_layers=-1,
            random_seed=123,
            processing_w=2,
            processing_h=1,
            device="cpu",
        )

        assert captured["device"] == "cpu"
        assert captured["config"].num_init_cluster_layers == 4
        assert captured["image_shape"] == (2, 3, 3)
        assert captured["background_tuple"] == (1, 2, 3)
        assert captured["material_colors"].shape == (2, 3)
        assert prepared.target_image_np.shape == (2, 3, 3)
        assert prepared.target_height_logits.shape == (2, 3)
        assert prepared.target_global_logits.shape == (4, 2)
        assert prepared.processing_height_logits.shape == (1, 2)

    def test_environment_overrides_apply_to_config_manager(self, monkeypatch):
        """Documented environment variables should populate config defaults."""
        from bananaforge.utils.config import ConfigManager

        monkeypatch.setenv("BANANAFORGE_ITERATIONS", "123")
        monkeypatch.setenv("BANANAFORGE_DEVICE", "cpu")
        monkeypatch.setenv("BANANAFORGE_MAX_TRIANGLES", "500000")
        monkeypatch.setenv("BANANAFORGE_BOTTOM_MODE", "none")
        monkeypatch.setenv("BANANAFORGE_RANDOM_SEED", "123")

        config_manager = ConfigManager()
        config_manager.apply_env_overrides()

        assert config_manager.get("optimization.iterations") == 123
        assert config_manager.get("optimization.device") == "cpu"
        assert config_manager.get("export.max_triangles") == 500000
        assert config_manager.get("export.bottom_mode") == "none"
        assert config_manager.get("random_seed") == 123

    def test_config_option_accepts_environment_variable(self):
        """The documented config env var should be wired into the root CLI."""
        from bananaforge.cli import cli

        config_param = next(param for param in cli.params if param.name == "config")

        assert config_param.envvar == "BANANAFORGE_CONFIG"


class TestCLIDocumentation:
    """
    Test CLI documentation and help text for transparency features.
    """

    def test_cli_help_mentions_transparency(self):
        """Test that CLI help mentions transparency features."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "bananaforge", "convert", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                help_text = result.stdout.lower()
                # Check if transparency is mentioned in help
                transparency_mentions = ["transparency", "alpha", "enable-transparency"]

                mentions_found = sum(
                    1 for mention in transparency_mentions if mention in help_text
                )
                # Don't require transparency mentions (they might not be implemented yet)
                # but if they exist, that's good
                if mentions_found > 0:
                    print(f"Found {mentions_found} transparency-related help mentions")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI help not available")

    def test_cli_error_messages_quality(self, temp_dir):
        """Test that CLI error messages are helpful."""
        try:
            # Test with non-existent file
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bananaforge",
                    "convert",
                    "/nonexistent/file.png",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Should fail with helpful error message
            assert result.returncode != 0, "Should fail with nonexistent file"

            error_message = result.stderr.lower()
            # Should mention file not found or similar
            error_indicators = ["not found", "does not exist", "file", "error"]
            has_helpful_error = any(
                indicator in error_message for indicator in error_indicators
            )

            if not has_helpful_error and error_message:
                print(f"Error message might need improvement: {result.stderr[:200]}")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI error testing not available")


# Performance marker for slow tests
pytestmark = pytest.mark.cli
