"""Configuration management for BananaForge."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel

from .logging import get_logger

logger = get_logger(__name__)


class Config(BaseModel):
    """Manage configuration settings for BananaForge."""

    max_layers: int
    layer_height: float
    num_init_rounds: int
    num_init_cluster_layers: int
    random_seed: int
    background_color: str


class ConfigManager:
    """Manage configuration settings for BananaForge."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self._config = self.get_default_config()

        if self.config_path and self.config_path.exists():
            self.load_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "optimization": {
                "iterations": 6000,
                "learning_rate": 0.01,
                "initial_temperature": 1.0,
                "final_temperature": 0.1,
                "temperature_decay": "linear",
                "early_stopping_patience": 6000,
                "device": "auto",
            },
            "model": {
                "layer_height": 0.08,
                "initial_layer_height": 0.16,
                "base_height": 0.16,
                "max_layers": 15,
                "physical_size": 180.0,
                "resolution": 512,
                "nozzle_diameter": 0.4,
            },
            "materials": {
                "max_materials": 4,
                "color_matching_method": "perceptual",
                "default_database": "bambu_pla",
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
            "advanced": {
                "mesh_optimization": True,
                "support_generation": False,
                "hollowing": False,
                "infill_percentage": 15.0,
            },
        }

    def load_config(self) -> None:
        """Load configuration from file."""
        if not self.config_path or not self.config_path.exists():
            return

        try:
            with open(self.config_path, "r") as f:
                if (
                    self.config_path.suffix.lower() == ".yaml"
                    or self.config_path.suffix.lower() == ".yml"
                ):
                    loaded_config = yaml.safe_load(f)
                else:
                    loaded_config = json.load(f)

            # Merge with defaults
            self._config = self._deep_merge(self._config, loaded_config)

        except Exception as e:
            raise ValueError(
                f"Failed to load configuration from {self.config_path}: {e}"
            )

    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file.

        Args:
            output_path: Optional output path, defaults to current config_path
        """
        save_path = Path(output_path) if output_path else self.config_path

        if not save_path:
            raise ValueError("No output path specified")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            if (
                save_path.suffix.lower() == ".yaml"
                or save_path.suffix.lower() == ".yml"
            ):
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            else:
                json.dump(self._config, f, indent=2)

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set value
        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with dictionary of changes.

        Args:
            updates: Dictionary of configuration updates
        """
        self._config = self._deep_merge(self._config, updates)

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def get_optimization_config(self):
        """Get optimization configuration object."""
        from ..core.optimizer import OptimizationConfig

        opt_config = self.get("optimization", {})
        loss_weights = self.get("loss_weights", {})

        return OptimizationConfig(
            iterations=opt_config.get("iterations", 6000),
            learning_rate=opt_config.get("learning_rate", 0.01),
            initial_temperature=opt_config.get("initial_temperature", 1.0),
            final_temperature=opt_config.get("final_temperature", 0.1),
            temperature_decay=opt_config.get("temperature_decay", "linear"),
            layer_height=self.get("model.layer_height", 0.08),
            max_layers=self.get("model.max_layers", 15),
            device=opt_config.get("device", "auto"),
            early_stopping_patience=opt_config.get("early_stopping_patience", 100),
            loss_weights=loss_weights,
        )

    @classmethod
    def from_env(cls) -> "ConfigManager":
        """Create configuration manager from environment variables."""
        config_manager = cls()
        config_manager.apply_env_overrides()
        return config_manager

    def apply_env_overrides(self) -> None:
        """Apply supported environment variable overrides to this config."""
        # Override with environment variables
        env_mappings = {
            "BANANAFORGE_DEVICE": "optimization.device",
            "BANANAFORGE_ITERATIONS": "optimization.iterations",
            "BANANAFORGE_LEARNING_RATE": "optimization.learning_rate",
            "BANANAFORGE_MAX_MATERIALS": "materials.max_materials",
            "BANANAFORGE_PHYSICAL_SIZE": "model.physical_size",
            "BANANAFORGE_LAYER_HEIGHT": "model.layer_height",
            "BANANAFORGE_OUTPUT_DIR": "output.directory",
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Try to convert to appropriate type
                try:
                    if "." in value:
                        value = float(value)
                    elif value.isdigit():
                        value = int(value)
                    elif value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                except ValueError:
                    pass  # Keep as string

                self.set(config_key, value)

    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate current configuration.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Validate optimization settings
        opt_config = self.get("optimization", {})

        if opt_config.get("iterations", 0) <= 0:
            errors.append("optimization.iterations must be positive")

        if opt_config.get("learning_rate", 0) <= 0:
            errors.append("optimization.learning_rate must be positive")

        if not (0 < opt_config.get("initial_temperature", 1) <= 10):
            errors.append("optimization.initial_temperature must be between 0 and 10")

        if not (0 < opt_config.get("final_temperature", 0.1) <= 1):
            errors.append("optimization.final_temperature must be between 0 and 1")

        # Validate model settings
        model_config = self.get("model", {})

        if model_config.get("layer_height", 0) <= 0:
            errors.append("model.layer_height must be positive")

        if model_config.get("max_layers", 0) <= 0:
            errors.append("model.max_layers must be positive")

        if model_config.get("physical_size", 0) <= 0:
            errors.append("model.physical_size must be positive")

        # Validate materials settings
        materials_config = self.get("materials", {})

        if materials_config.get("max_materials", 0) <= 0:
            errors.append("materials.max_materials must be positive")

        # Validate loss weights
        loss_weights = self.get("loss_weights", {})
        for weight_name, weight_value in loss_weights.items():
            if not isinstance(weight_value, (int, float)) or weight_value < 0:
                errors.append(f"loss_weights.{weight_name} must be non-negative number")

        return len(errors) == 0, errors

    def get_profile_configs(self) -> Dict[str, Dict]:
        """Get predefined configuration profiles."""
        return {
            "fast": {
                "optimization": {
                    "iterations": 500,
                    "learning_rate": 0.02,
                    "early_stopping_patience": 50,
                },
                "model": {"resolution": 256, "max_layers": 25},
            },
            "balanced": {
                "optimization": {
                    "iterations": 6000,
                    "learning_rate": 0.01,
                    "early_stopping_patience": 100,
                },
                "model": {"resolution": 512, "max_layers": 15},
            },
            "quality": {
                "optimization": {
                    "iterations": 2000,
                    "learning_rate": 0.005,
                    "early_stopping_patience": 200,
                },
                "model": {"resolution": 512, "max_layers": 75},
                "loss_weights": {
                    "perceptual": 1.5,
                    "color": 1.2,
                    "smoothness": 0.2,
                    "consistency": 0.8,
                },
            },
            "prototype": {
                "optimization": {
                    "iterations": 200,
                    "learning_rate": 0.05,
                    "early_stopping_patience": 20,
                },
                "model": {"resolution": 64, "max_layers": 15},
                "materials": {"max_materials": 4},
            },
        }

    def apply_profile(self, profile_name: str) -> None:
        """Apply a predefined configuration profile.

        Args:
            profile_name: Name of profile to apply
        """
        profiles = self.get_profile_configs()

        if profile_name not in profiles:
            raise ValueError(
                f"Unknown profile: {profile_name}. Available: {list(profiles.keys())}"
            )

        profile_config = profiles[profile_name]
        self.update(profile_config)
